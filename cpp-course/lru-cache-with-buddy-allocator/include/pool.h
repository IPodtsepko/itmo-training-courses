#pragma once

#include <cmath>
#include <list>
#include <vector>

class PoolAllocator
{
public:
    PoolAllocator(unsigned, unsigned);
    void * allocate(size_t);
    void deallocate(const void *);

private:
    using Pointer = size_t;
    using Link = std::list<Pointer>::iterator;
    using Pool = std::vector<std::byte>;

    struct Block
    {
        size_t size;
        Link iterator;

        Block();
        Block(size_t, const Link &);
    };

    class BlocksList
    {
    public:
        BlocksList(size_t, size_t);
        void reset(Pointer);
        Block & operator[](size_t);

    private:
        const Block MARKED = Block();
        const size_t unit_size;
        std::vector<Block> blocks;
    };

    class EmptyBlocksList
    {
    public:
        EmptyBlocksList(size_t, size_t);
        std::list<Pointer> & operator[](size_t);
        size_t size();

    private:
        size_t unit_size;
        std::vector<std::list<Pointer>> avail;
    };

    size_t unit_size;

    Pool pool;
    EmptyBlocksList avail;
    BlocksList blocks;

    static size_t get_level(size_t);
    void push_empty(size_t, Pointer);
    Pointer pop_empty(size_t);
    Pointer split(size_t, Pointer, size_t);
    void reset(Pointer);
    void deallocate(Pointer, size_t);
};
