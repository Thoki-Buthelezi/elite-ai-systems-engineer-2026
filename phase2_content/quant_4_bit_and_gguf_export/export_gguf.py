import struct
import torch

from gguf_utils import (
    GGML_TYPE_F32,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_KM,
    write_header,
    write_metadata_kv_string,
    write_metadata_kv_uint32,
    write_tensor_info,
)

from quantize import (
    quantize_q8_0,
    quantize_q4_k_m,
    expected_q8_0_bytes,
    expected_q4_k_m_bytes,
)


ALIGNMENT = 32


def align_offset(offset: int) -> int:
    return (offset + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def export_gguf(
    checkpoint_path: str,
    output_path: str,
    quant_type: str = "f32",
):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove tril buffers
    tensors = [
        (name, tensor.contiguous())
        for name, tensor in state_dict.items()
        if "tril" not in name
    ]

    # Decide tensor formats + compute offsets
    tensor_infos = []
    current_offset = 0

    for name, tensor in tensors:
        numel = tensor.numel()
        keep_fp32 = "ln" in name or "embedding" in name

        if keep_fp32 or quant_type == "f32":
            ggml_type = GGML_TYPE_F32
            byte_size = numel * 4
        elif quant_type == "q8_0":
            ggml_type = GGML_TYPE_Q8_0
            byte_size = expected_q8_0_bytes(numel)
        elif quant_type == "q4_k_m":
            ggml_type = GGML_TYPE_Q4_KM
            byte_size = expected_q4_k_m_bytes(numel)
        else:
            raise ValueError(f"Unknown quantization type: {quant_type}")

        current_offset = align_offset(current_offset)
        tensor_infos.append({
            "name":      name,
            "tensor":    tensor,
            "shape":     list(tensor.shape),
            "ggml_type": ggml_type,
            "offset":    current_offset,
            "size":      byte_size,
        })
        current_offset += byte_size

    # Write file
    with open(output_path, "wb") as f:

        write_header(f, tensor_count=len(tensor_infos), metadata_kv_count=7)

        write_metadata_kv_string(f, "general.architecture",      "nanogpt")
        write_metadata_kv_string(f, "general.name",              "nanoGPT-phase1")
        write_metadata_kv_uint32(f, "nanogpt.context_length",    64)
        write_metadata_kv_uint32(f, "nanogpt.embedding_length",  128)
        write_metadata_kv_uint32(f, "nanogpt.block_count",       4)
        write_metadata_kv_uint32(f, "nanogpt.attention.head_count", 4)
        write_metadata_kv_uint32(f, "nanogpt.vocab_size",        65)

        for info in tensor_infos:
            write_tensor_info(f, info["name"], info["shape"], info["ggml_type"], info["offset"])

        # Align start of tensor data section to 32 bytes
        pad = align_offset(f.tell()) - f.tell()
        if pad:
            f.write(b"\x00" * pad)

        data_start = f.tell()

        # Write tensor data
        for info in tensor_infos:
            desired = data_start + info["offset"]
            current = f.tell()
            if current < desired:
                f.write(b"\x00" * (desired - current))

            tensor = info["tensor"]
            if info["ggml_type"] == GGML_TYPE_F32:
                f.write(tensor.to(torch.float32).contiguous().numpy().tobytes())
            elif info["ggml_type"] == GGML_TYPE_Q8_0:
                f.write(quantize_q8_0(tensor))
            elif info["ggml_type"] == GGML_TYPE_Q4_KM:
                f.write(quantize_q4_k_m(tensor))
            else:
                raise RuntimeError("Unsupported GGML type.")

    print(f"Exported {len(tensor_infos)} tensors to {output_path}")


if __name__ == "__main__":
    import sys
    import os

    print("inside export gguf")
    checkpoint = sys.argv[1]
    quant_type = sys.argv[2] if len(sys.argv) > 2 else "f32"
    output = f"models/nanogpt_{quant_type}.gguf"

    os.makedirs("models", exist_ok=True)
    export_gguf(checkpoint, output, quant_type)

    size_mb = os.path.getsize(output) / 1024 / 1024
    print(f"File size: {size_mb:.3f} MB")
    print("done")