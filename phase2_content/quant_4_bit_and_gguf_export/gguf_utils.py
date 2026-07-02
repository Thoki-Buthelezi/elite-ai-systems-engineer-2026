import struct


# GGML type constants
GGML_TYPE_F32   = 0
GGML_TYPE_F16   = 1
GGML_TYPE_Q8_0  = 8
GGML_TYPE_Q4_K  = 12   # Q4_K_S
GGML_TYPE_Q4_KM = 15   # Q4_K_M


def write_header(f, tensor_count, metadata_kv_count):
    GGUF_MAGIC = b'GGUF'
    GGUF_VERSION = 3
    header_bytes = struct.pack('<4sIQQ', GGUF_MAGIC, GGUF_VERSION, tensor_count, metadata_kv_count)
    f.write(header_bytes)


def write_metadata_kv_string(f, key, value):
    key_bytes = key.encode('utf-8')
    value_bytes = value.encode('utf-8')
    f.write(struct.pack('<Q', len(key_bytes)))    # key length as uint64
    f.write(key_bytes)                             # key
    f.write(struct.pack('<I', 8))                  # value_type = 8 (string)
    f.write(struct.pack('<Q', len(value_bytes)))   # value length as uint64
    f.write(value_bytes)                           # value


def write_metadata_kv_uint32(f, key, value):
    key_bytes = key.encode('utf-8')
    f.write(struct.pack('<Q', len(key_bytes)))     # key length
    f.write(key_bytes)                             # key
    f.write(struct.pack('<I', 4))                  # value_type = 4 (uint32)
    f.write(struct.pack('<I', value))              # value, no length prefix needed


def write_metadata_kv_float32(f, key, value):
    key_bytes = key.encode('utf-8')
    f.write(struct.pack('<Q', len(key_bytes)))     # key length
    f.write(key_bytes)                             # key
    f.write(struct.pack('<I', 6))                  # value_type = 6 (float32)
    f.write(struct.pack('<f', value))              # value, no length prefix needed


def write_tensor_info(f, name, shape, ggml_type, offset):
    name_bytes = name.encode('utf-8')
    f.write(struct.pack('<Q', len(name_bytes)))    # name length
    f.write(name_bytes)                            # name
    f.write(struct.pack('<I', len(shape)))         # n_dimensions
    for dim in shape:
        f.write(struct.pack('<Q', dim))            # each dimension as uint64
    f.write(struct.pack('<I', ggml_type))          # ggml_type
    f.write(struct.pack('<Q', offset))             # byte offset