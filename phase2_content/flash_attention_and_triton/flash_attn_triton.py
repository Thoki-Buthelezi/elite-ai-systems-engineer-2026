"""
flash_attn_triton.py
--------------------
FlashAttention-style fused attention kernel implemented in Triton.

Implements the tiled / online-softmax algorithm from:
  Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
  with IO-Awareness", NeurIPS 2022.

Key ideas
---------
1. Tile Q, K, V along the sequence dimension so each tile fits in SRAM.
2. Compute softmax incrementally (online rescaling) to avoid materialising
   the full N×N attention matrix in HBM.
3. Accumulate the weighted sum of V in the same pass -> single fused kernel.

Peak HBM memory: O(N) instead of O(N^2).

Usage
-----
    from flash_attn_triton import flash_attention_triton
    out = flash_attention_triton(q, k, v)   # (B, H, N, D)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,   # tile size along Q (rows)
    BLOCK_N: tl.constexpr,   # tile size along K/V (cols)
    scale: tl.constexpr,
):
    """
    One program instance handles one (batch, head) pair and one Q-tile.

    Grid: (num_q_tiles, batch * n_heads)
    """
    # identify this program
    tile_m   = tl.program_id(0)          # which Q tile
    bh_idx   = tl.program_id(1)          # flattened (batch, head) index
    b_idx    = bh_idx // tl.num_programs(2) if False else (bh_idx // 1)
    # We fold batch and head into axis-1 of the grid so we need to decode them.
    # Actually handled by strides below — bh_idx maps to a unique (b, h) pair.

    # pointers for this (b, h)
    Q_bh = Q_ptr   + bh_idx * stride_qh
    K_bh = K_ptr   + bh_idx * stride_kh
    V_bh = V_ptr   + bh_idx * stride_vh
    O_bh = Out_ptr + bh_idx * stride_oh

    # row offsets for this Q tile
    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_d = tl.arange(0, D_HEAD)                         # [D_HEAD]

    # Load Q tile: [BLOCK_M, D_HEAD]
    q_ptrs = Q_bh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # online-softmax accumulators 
    # m_i: running max  [BLOCK_M]
    # l_i: running sum  [BLOCK_M]
    # acc:  running weighted-V sum  [BLOCK_M, D_HEAD]
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    # iterate over K/V tiles
    num_kv_tiles = tl.cdiv(N_CTX, BLOCK_N)
    for tile_n in range(num_kv_tiles):
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

        # Load K tile: [D_HEAD, BLOCK_N]  (transposed for matmul)
        k_ptrs = K_bh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k_mask = offs_n[None, :] < N_CTX
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # QK^T: [BLOCK_M, BLOCK_N]
        s = tl.dot(q, k) * scale

        # Mask out-of-bounds positions to -inf
        s = tl.where(offs_n[None, :] < N_CTX, s, -float("inf"))

        # online softmax update
        m_new = tl.maximum(m_i, tl.max(s, axis=1))            # new row max
        alpha  = tl.exp(m_i - m_new)                           # rescale factor
        p      = tl.exp(s - m_new[:, None])                    # [BLOCK_M, BLOCK_N]

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V tile: [BLOCK_N, D_HEAD]
        v_ptrs = V_bh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        acc = acc + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # normalise and store
    acc = acc / l_i[:, None]

    o_ptrs = O_bh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def flash_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """
    FlashAttention forward pass (Triton implementation).

    Args
    ----
    q, k, v : torch.Tensor  shape (B, H, N, D),  dtype float16,  contiguous,  on CUDA
    block_m  : Q tile size along sequence dimension
    block_n  : K/V tile size along sequence dimension

    Returns
    -------
    out : torch.Tensor  shape (B, H, N, D),  dtype float16
    """
    assert q.is_cuda and q.dtype == torch.float16, "q must be float16 on CUDA"
    B, H, N, D = q.shape
    scale = D ** -0.5

    out = torch.empty_like(q)

    # Grid: (num_Q_tiles, B*H)
    grid = (triton.cdiv(N, block_m), B * H)

    _flash_attn_fwd_kernel[grid](
        q, k, v, out,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # K strides
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # V strides
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # Out strides
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N_CTX=N,
        D_HEAD=D,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        scale=scale,
        num_warps=4,
        num_stages=2,
    )
    return out


# ---------------------------------------------------------------------------
# Naive PyTorch reference (materialises full N×N matrix)
# ---------------------------------------------------------------------------

def naive_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Standard scaled dot-product attention.
    Materialises the full (B, H, N, N) attention matrix in HBM.

    Args
    ----
    q, k, v : (B, H, N, D)  float16 on CUDA

    Returns
    -------
    out : (B, H, N, D)  float16
    """
    scale = q.shape[-1] ** -0.5
    # (B, H, N, N)
    attn = torch.softmax(
        torch.matmul(q * scale, k.transpose(-2, -1)).float(),
        dim=-1,
    ).half()
    return torch.matmul(attn, v)

# Quick correctness smoke-test
if __name__ == "__main__":
    torch.manual_seed(42)
    B, H, N, D = 2, 4, 512, 64

    q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
    k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
    v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")

    ref = naive_attention_pytorch(q, k, v)
    out = flash_attention_triton(q, k, v)

    max_err = (out - ref).abs().max().item()
    print(f"Max absolute error (Triton vs PyTorch): {max_err:.6f}")
    assert max_err < 0.05, f"Correctness check FAILED: max_err={max_err:.4f}"
    print("Correctness check PASSED")