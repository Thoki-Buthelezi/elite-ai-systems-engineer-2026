from collections import deque
from dataclasses import dataclass
import math

# a single fixed-size slot in the GPU KV cache pool
@dataclass
class PhysicalBlock:
    block_id : int
    ref_count : int = 0 # how many sequences currently point to this block

# per-sequence mapping, logical block index -> physical block id
# (this IS the page table analogy)
BlockTable = list[int]

class BlockManager:
    def __init__(self, num_gpu_blocks, block_size):
        self.block_size = block_size
        self.free_blocks = deque(range(num_gpu_blocks))  # free list
        self.physical_blocks = {i: PhysicalBlock(block_id=i, ref_count=0) for i in range(num_gpu_blocks)}
        self.block_tables = {}  # seq_id -> BlockTable

    def can_allocate(self, seq) -> bool:
        needed = math.ceil(seq.prompt_len / self.block_size)
        return len(self.free_blocks) >= needed

    def allocate(self, seq):
        needed = math.ceil(seq.prompt_len / self.block_size)
        table = []
        for _ in range(needed):
            if not self.free_blocks:
                raise RuntimeError("Out of GPU KV cache blocks")
            block_id = self.free_blocks.pop()
            self.physical_blocks[block_id].ref_count = 1
            table.append(block_id)
        self.block_tables[seq.id] = table

    def needs_new_block(self, seq) -> bool:
        table = self.block_tables[seq.id]
        last_block = table[-1]
        if seq.num_tokens % self.block_size == 0:
            return True
        if self.physical_blocks[last_block].ref_count > 1:
            return True
        return False

    def can_append_slot(self, seq) -> bool:
        # does the next decode step for this sequence need a block we don't have?
        if not self.needs_new_block(seq):
            return True
        return len(self.free_blocks) > 0

    def append_slot(self, seq):
        table = self.block_tables[seq.id]
        last_block = table[-1]
        if seq.num_tokens % self.block_size == 0:
            if not self.free_blocks:
                raise RuntimeError("Out of GPU KV cache blocks")
            new_block = self.free_blocks.pop()
            self.physical_blocks[new_block].ref_count = 1
            table.append(new_block)
        elif self.physical_blocks[last_block].ref_count > 1:
            if not self.free_blocks:
                raise RuntimeError("Out of GPU KV cache blocks")
            new_block = self.free_blocks.pop()
            copy_kv_data(src=last_block, dst=new_block)
            self.physical_blocks[last_block].ref_count -= 1
            self.physical_blocks[new_block].ref_count = 1
            table[-1] = new_block

    def fork(self, parent_seq, child_seq):
        parent_table = self.block_tables[parent_seq.id]
        self.block_tables[child_seq.id] = list(parent_table)
        for block_id in parent_table:
            self.physical_blocks[block_id].ref_count += 1

    def free(self, seq):
        table = self.block_tables.pop(seq.id)
        for block_id in table:
            self.physical_blocks[block_id].ref_count -= 1
            if self.physical_blocks[block_id].ref_count == 0:
                self.free_blocks.append(block_id)

    def num_used_blocks(self):
        return len(self.physical_blocks) - len(self.free_blocks)


def copy_kv_data(src, dst):
    pass