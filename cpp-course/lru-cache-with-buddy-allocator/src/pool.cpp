#include "pool.h"

PoolAllocator::PoolAllocator(const unsigned int N, const unsigned int M)
    : unit_size(N)
    , pool(1 << M)
    , avail(N, M)
    , blocks(1 << M, 1 << N)
{
    push_empty(M, Pointer(0));
}

void * PoolAllocator::allocate(const size_t n)
{
    size_t required_level = std::max(get_level(n), unit_size);
    size_t level = required_level - 1;
    do {
        if (++level >= avail.size()) {
            throw std::bad_alloc{};
        }
    } while (avail[level].empty());
    Pointer ptr = split(level, pop_empty(level), required_level);
    return &pool[ptr];
}

void PoolAllocator::deallocate(const void * address)
{
    Pointer ptr = static_cast<const std::byte *>(address) - &pool.front();
    deallocate(ptr, blocks[ptr].size);
}

size_t PoolAllocator::get_level(const size_t size)
{
    return static_cast<size_t>(std::ceil(std::log2(std::max(std::size_t(1), size))));
}

void PoolAllocator::push_empty(const size_t level, PoolAllocator::Pointer ptr)
{
    avail[level].push_front(ptr);
    size_t size = 1 << level;
    blocks[ptr] = Block(size, avail[level].begin());
}

PoolAllocator::Pointer PoolAllocator::pop_empty(const size_t level)
{
    Pointer ptr = avail[level].back();
    avail[level].pop_back();
    blocks[ptr].iterator = Link();
    return ptr;
}

PoolAllocator::Pointer PoolAllocator::split(
        size_t level, PoolAllocator::Pointer ptr, size_t required_level)
{
    if (level == required_level) {
        return ptr;
    }
    size_t size = 1 << (level - 1);
    std::pair<Pointer, Pointer> new_blocks{ptr, ptr + size};
    blocks[new_blocks.first].size = blocks[new_blocks.second].size = size;
    push_empty(level - 1, new_blocks.first);
    return split(level - 1, new_blocks.second, required_level);
}

void PoolAllocator::reset(PoolAllocator::Pointer ptr)
{
    Block & current = blocks[ptr];
    if (current.iterator != Link()) {
        size_t level = get_level(current.size);
        avail[level].erase(current.iterator);
    }
    blocks.reset(ptr);
}

void PoolAllocator::deallocate(PoolAllocator::Pointer ptr, size_t size)
{
    size_t level = get_level(size);
    Pointer buddy = ptr ^ size;

    if (buddy < pool.size() && blocks[buddy].size == size && blocks[buddy].iterator != Link()) {
        reset(buddy);
        reset(ptr);
        deallocate(std::min(ptr, buddy), size * 2);
    }
    else {
        push_empty(level, ptr);
    }
}

PoolAllocator::Block::Block()
    : size(0)
    , iterator(Link())
{
}

PoolAllocator::Block::Block(size_t size, const PoolAllocator::Link & iterator)
    : size(size)
    , iterator(iterator)
{
}

PoolAllocator::BlocksList::BlocksList(const size_t memory_size, const size_t unit_size)
    : unit_size(unit_size)
    , blocks(memory_size / unit_size, MARKED)
{
}

void PoolAllocator::BlocksList::reset(const PoolAllocator::Pointer ptr)
{
    blocks[ptr / unit_size] = MARKED;
}

PoolAllocator::Block & PoolAllocator::BlocksList::operator[](const size_t ptr)
{
    return blocks[ptr / unit_size];
}

PoolAllocator::EmptyBlocksList::EmptyBlocksList(const size_t N, const size_t M)
    : unit_size(N)
    , avail(M - N + 1)
{
}

std::list<PoolAllocator::Pointer> & PoolAllocator::EmptyBlocksList::operator[](const size_t level)
{
    return avail[level - unit_size];
}

size_t PoolAllocator::EmptyBlocksList::size()
{
    return avail.size() + unit_size;
}
