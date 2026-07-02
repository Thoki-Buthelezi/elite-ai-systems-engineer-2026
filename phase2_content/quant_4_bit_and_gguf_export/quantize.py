import struct
import torch

# Q8_0 constants
Q8_BLOCK_SIZE = 32
BLOCK_SIZE = 32

# Q4_K_M constants
SUPER_BLOCK_SIZE = 256
SUB_BLOCK_SIZE   = 32
NUM_SUB_BLOCKS   = 8


def quantize_q8_0(tensor: torch.Tensor) -> bytes:
    """
    Quantize a tensor into GGUF Q8_0 block format.
    Each block contains:
        - float32 scale  (4 bytes)
        - 32 int8 values (32 bytes)
    Total: 36 bytes per block
    """
    weights = tensor.flatten().to(torch.float32)

    remainder = weights.numel() % Q8_BLOCK_SIZE
    if remainder != 0:
        pad = Q8_BLOCK_SIZE - remainder
        weights = torch.cat([weights, torch.zeros(pad)])

    blocks = weights.view(-1, Q8_BLOCK_SIZE)
    output = bytearray()

    for block in blocks:
        max_abs = torch.max(torch.abs(block)).item()
        if max_abs == 0.0:
            scale = 1.0
            quantized = torch.zeros(Q8_BLOCK_SIZE, dtype=torch.int8)
        else:
            scale = max_abs / 127.0
            quantized = torch.round(block / scale)
            quantized = torch.clamp(quantized, -127, 127).to(torch.int8)

        output.extend(struct.pack('<f', scale))
        output.extend(struct.pack('<32b', *quantized.tolist()))

    return bytes(output)


def quantize_q4_k_m(tensor: torch.Tensor) -> bytes:
    """
    Quantize a tensor into an approximate GGUF Q4_K_M layout.
    Layout per super-block:
        - 8 x float16 scales (16 bytes)
        - 256 4-bit weights packed into 128 bytes
    Total: 144 bytes per 256-weight super-block
    """
    weights = tensor.flatten().to(torch.float32)

    remainder = weights.numel() % SUPER_BLOCK_SIZE
    if remainder != 0:
        pad = SUPER_BLOCK_SIZE - remainder
        weights = torch.cat([weights, torch.zeros(pad)])

    super_blocks = weights.view(-1, SUPER_BLOCK_SIZE)
    output = bytearray()

    for super_block in super_blocks:
        sub_blocks    = super_block.view(NUM_SUB_BLOCKS, SUB_BLOCK_SIZE)
        packed_scales = bytearray()
        packed_weights = bytearray()

        for sub_block in sub_blocks:
            max_abs = torch.max(torch.abs(sub_block)).item()
            if max_abs == 0.0:
                scale = 1.0
                quantized = torch.full((SUB_BLOCK_SIZE,), 8, dtype=torch.uint8)
            else:
                scale = max_abs / 7.0
                quantized = torch.round(sub_block / scale + 8)
                quantized = torch.clamp(quantized, 0, 15).to(torch.uint8)

            packed_scales.extend(struct.pack('<e', scale))

            q = quantized.tolist()
            for i in range(0, SUB_BLOCK_SIZE, 2):
                byte = (q[i] & 0xF) | ((q[i + 1] & 0xF) << 4)
                packed_weights.append(byte)

        output.extend(packed_scales)
        output.extend(packed_weights)

    return bytes(output)


def expected_q8_0_bytes(numel: int) -> int:
    import math
    num_blocks = math.ceil(numel / Q8_BLOCK_SIZE)
    return num_blocks * 36   # 4 (scale) + 32 (int8s)


def expected_q4_k_m_bytes(numel: int) -> int:
    import math
    num_super_blocks = math.ceil(numel / SUPER_BLOCK_SIZE)
    return num_super_blocks * 144   # 16 (scales) + 128 (packed 4-bit)


def dequantize_q8_0(data: bytes, shape: tuple) -> torch.Tensor:
    """
    Reverse quantize_q8_0().

    Each block:
        float32 scale (4 bytes)
        32 int8 values (32 bytes)
    """

    block_bytes = 4 + BLOCK_SIZE
    num_blocks = len(data) // block_bytes

    weights = []

    offset = 0

    for _ in range(num_blocks):
        # Read scale
        scale = struct.unpack_from("<f", data, offset)[0]
        offset += 4

        # Read 32 int8 values
        q = struct.unpack_from("<32b", data, offset)
        offset += BLOCK_SIZE

        # Dequantize
        block = torch.tensor(q, dtype=torch.float32) * scale
        weights.append(block)

    weights = torch.cat(weights)

    # Remove any padding added during quantization
    numel = 1
    for dim in shape:
        numel *= dim

    weights = weights[:numel]

    return weights.reshape(shape)

def dequantize_q4_k_m(data: bytes, shape: tuple) -> torch.Tensor:
    """
    Reverse quantize_q4_k_m().

    Per super-block:
        8 float16 scales (16 bytes)
        128 packed bytes (256 four-bit values)
    """

    SUPER_BLOCK_BYTES = 16 + 128

    num_super_blocks = len(data) // SUPER_BLOCK_BYTES

    weights = []

    offset = 0

    for _ in range(num_super_blocks):

        # Read eight float16 scales
        scales = []

        for _ in range(NUM_SUB_BLOCKS):
            scale = struct.unpack_from("<e", data, offset)[0]
            offset += 2
            scales.append(scale)

        # Reconstruct each 32-weight sub-block
        for scale in scales:

            block = []

            for _ in range(SUB_BLOCK_SIZE // 2):

                packed = data[offset]
                offset += 1

                # Extract nibbles
                low = packed & 0x0F
                high = (packed >> 4) & 0x0F

                # Undo zero-point shift
                block.append((low - 8) * scale)
                block.append((high - 8) * scale)

            weights.append(torch.tensor(block, dtype=torch.float32))

    weights = torch.cat(weights)

    # Remove padding
    numel = 1
    for dim in shape:
        numel *= dim

    weights = weights[:numel]

    return weights.reshape(shape)