#ifndef B_PLUS_TREE_ITERATOR_H
#define B_PLUS_TREE_ITERATOR_H

#include "nodes/leaf_node.h"

template <class Leaf, class ValueType>
class BPTreeIterator
{
private:
    static_assert(
            std::is_same<typename Leaf::value_type, typename std::remove_const<ValueType>::type>(),
            "LeafType::value_type must be equal to (const) ValueType");
    using iterator = BPTreeIterator<Leaf, ValueType>;

    Leaf * current;
    std::size_t position;

public:
    using difference_type = std::ptrdiff_t;
    using value_type = ValueType;
    using pointer = value_type *;
    using reference = value_type &;
    using iterator_category = std::forward_iterator_tag;

    BPTreeIterator()
        : current(nullptr)
        , position(0)
    {
    }

    BPTreeIterator(Leaf * current, std::size_t position)
        : current(current)
        , position(position)
    {
    }

    BPTreeIterator(const BPTreeIterator & other)
        : current(other.current)
        , position(other.position)
    {
    }

    BPTreeIterator & operator=(const iterator & other)
    {
        this->current = other.current;
        this->position = other.position;
        return *this;
    }

    reference operator*() const
    {
        return current->get(position);
    }

    pointer operator->() const
    {
        return &current->get(position);
    }

    iterator & operator++()
    {
        if (!current) {
            return *this;
        }
        if (position++ == current->size() - 1) {
            current = dynamic_cast<Leaf *>(current->right);
            position = 0;
        }
        return *this;
    }
    iterator operator++(int)
    {
        auto tmp = this;
        operator++();
        return *tmp;
    }

    friend bool operator==(const iterator & lhs, const iterator & rhs)
    {
        return lhs.current == rhs.current && lhs.position == rhs.position;
    }
    friend bool operator!=(const iterator & lhs, const iterator & rhs)
    {
        return !(lhs == rhs);
    }
};

#endif //B_PLUS_TREE_ITERATOR_H