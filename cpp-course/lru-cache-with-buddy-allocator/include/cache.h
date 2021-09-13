#pragma once

#include <algorithm>
#include <cstddef>
#include <new>
#include <ostream>

template <class Key, class KeyProvider, class Allocator>
class Cache
{
public:
    template <class... AllocArgs>
    Cache(const std::size_t cache_size, AllocArgs &&... alloc_args)
        : max_top_size(cache_size)
        , max_low_size(cache_size)
        , alloc(std::forward<AllocArgs>(alloc_args)...)
    {
    }

    std::size_t size() const
    {
        return priority.size() + regular.size();
    }

    bool empty() const
    {
        return priority.empty() && regular.empty();
    }

    template <class T>
    T & get(const Key & e);

    std::ostream & print(std::ostream & strm) const;

    friend std::ostream & operator<<(std::ostream & strm, const Cache & cache)
    {
        return cache.print(strm);
    }

private:
    const std::size_t max_top_size;
    const std::size_t max_low_size;

    std::list<KeyProvider *> priority; // LRU
    std::list<KeyProvider *> regular;  // FIFO

    Allocator alloc;

    void free_up_regular();
    static std::ostream & print_list(std::ostream &, const std::list<KeyProvider *> &);
};

template <class Key, class KeyProvider, class Allocator>
void Cache<Key, KeyProvider, Allocator>::free_up_regular()
{
    if (regular.size() == max_low_size) {
        alloc.template destroy<KeyProvider>(regular.back());
        regular.pop_back();
    }
}

template <class Key, class KeyProvider, class Allocator>
template <class T>
inline T & Cache<Key, KeyProvider, Allocator>::get(const Key & key)
{
    auto comparator = [&key](KeyProvider * e) { return *e == key; };

    auto i_priority = std::find_if(priority.begin(), priority.end(), comparator);
    if (i_priority != priority.end()) {
        priority.splice(priority.begin(), priority, i_priority);
        return *static_cast<T *>(priority.front());
    }

    auto i_regular = std::find_if(regular.begin(), regular.end(), comparator);
    if (i_regular != regular.end()) {
        KeyProvider * e = *i_regular;
        regular.erase(i_regular);

        if (priority.size() == max_top_size) {
            free_up_regular();
            regular.splice(regular.begin(), priority, --priority.end());
        }
        priority.push_front(e);

        return *static_cast<T *>(priority.front());
    }

    free_up_regular();
    regular.push_front(alloc.template create<T>(key));
    return *static_cast<T *>(regular.front());
}

template <class Key, class KeyProvider, class Allocator>
std::ostream & Cache<Key, KeyProvider, Allocator>::print_list(
        std::ostream & stream, const std::list<KeyProvider *> & queue)
{
    if (queue.empty()) {
        stream << "<empty>\n";
    }
    else {
        stream << "{ ";
        for (const auto x : queue) {
            stream << *x << " ";
        }
        stream << "}\n";
    }
    return stream;
}

template <class Key, class KeyProvider, class Allocator>
std::ostream & Cache<Key, KeyProvider, Allocator>::print(std::ostream & strm) const
{
    strm << "Priority: ";
    print_list(strm, priority);
    strm << "Regular: ";
    return print_list(strm, regular);
}