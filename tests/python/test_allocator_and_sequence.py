"""Tests for BlockAllocator and Sequence bindings."""

import gc

import pytest

import radix_kv as rk


def test_allocator_reference_counting() -> None:
    pool = rk.BlockAllocator(num_blocks=4, block_size=16)
    assert (pool.num_blocks, pool.block_size, pool.num_free) == (4, 16, 4)
    block = pool.allocate()
    pool.retain(block)
    assert pool.ref_count(block) == 2
    assert pool.release(block) == 1
    assert pool.release(block) == 0
    assert not pool.is_allocated(block)
    assert pool.blocks_for(17) == 2
    assert pool.peak_used == 1


def test_allocator_errors_map_to_python_exceptions() -> None:
    pool = rk.BlockAllocator(1, 4)
    pool.allocate()
    with pytest.raises(rk.OutOfBlocksError, match="out_of_blocks"):
        pool.allocate()
    with pytest.raises(rk.RadixKvError):  # OutOfBlocksError derives from RadixKvError
        pool.allocate()
    with pytest.raises(ValueError, match="invalid_block"):
        pool.retain(7)
    with pytest.raises(ValueError, match="block_not_allocated"):
        rk.BlockAllocator(2, 4).release(0)
    with pytest.raises(ValueError, match="positive"):
        rk.BlockAllocator(0, 4)


def test_sequence_append_extend_and_block_table() -> None:
    pool = rk.BlockAllocator(8, 4)
    seq = rk.Sequence(pool)
    result = seq.extend([1, 2, 3, 4, 5])
    assert result.new_blocks == 2
    assert result.copy is None
    assert seq.append(6).new_blocks == 0
    assert seq.tokens == [1, 2, 3, 4, 5, 6]
    assert seq.block_table == [0, 1]
    assert (len(seq), seq.num_blocks, seq.num_full_blocks) == (6, 2, 1)


def test_fork_copies_on_write() -> None:
    pool = rk.BlockAllocator(8, 4)
    parent = rk.Sequence(pool)
    parent.extend([1, 2, 3, 4, 5, 6])
    child = parent.fork()
    assert child.block_table == parent.block_table
    result = child.append(99)
    assert result.copy is not None
    assert (result.copy.src, result.copy.dst) == (1, 2)
    assert parent.block_table == [0, 1]
    assert child.block_table == [0, 2]
    assert pool.ref_count(0) == 2


def test_failed_extend_changes_nothing() -> None:
    pool = rk.BlockAllocator(2, 4)
    seq = rk.Sequence(pool)
    seq.extend([1] * 8)
    with pytest.raises(rk.OutOfBlocksError):
        seq.extend([2])
    assert len(seq) == 8
    assert pool.num_free == 0


def test_truncate_release_and_assign_prefix() -> None:
    pool = rk.BlockAllocator(8, 2)
    seq = rk.Sequence(pool)
    seq.extend([1, 2, 3, 4, 5])
    seq.truncate(2)
    assert seq.tokens == [1, 2]
    assert pool.num_used == 1
    with pytest.raises(ValueError, match="invalid_argument"):
        seq.truncate(3)

    other = rk.Sequence(pool)
    other.assign_prefix([1, 2], seq.block_table)
    assert pool.ref_count(seq.block_table[0]) == 2
    with pytest.raises(ValueError, match="invalid_argument"):
        other.assign_prefix([1, 2], seq.block_table)  # no longer empty

    seq.release()
    other.release()
    assert pool.num_free == pool.num_blocks


def test_sequences_keep_their_allocator_alive() -> None:
    def make_fork() -> rk.Sequence:
        pool = rk.BlockAllocator(4, 2)
        seq = rk.Sequence(pool)
        seq.extend([1, 2, 3])
        return seq.fork()  # neither the pool nor the parent is referenced afterwards

    fork = make_fork()
    gc.collect()
    fork.append(4)
    assert fork.tokens == [1, 2, 3, 4]


def test_dropping_a_sequence_releases_its_blocks() -> None:
    pool = rk.BlockAllocator(4, 2)
    seq = rk.Sequence(pool)
    seq.extend([1, 2, 3])
    del seq
    gc.collect()
    assert pool.num_free == 4
