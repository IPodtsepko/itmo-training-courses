#ifndef B_PLUS_TREE_ITERATOR_H
#define B_PLUS_TREE_ITERATOR_H

#pragma once

#include "nodes/leaf_node.h"

template <typename LeafType, typename ValueType>
class Iterator
{
    static_assert(
            std::is_same<typename LeafType::value_type, typename std::remove_cv<ValueType>::type>(),
            "LeafType::value_type must be equal to (const) ValueType");

    using iterator = Iterator<LeafType, ValueType>;

private:
    LeafType * current;
    std::size_t position;

public:
    using difference_type = std::ptrdiff_t;
    using value_type = ValueType;
    using pointer = value_type *;
    using reference = value_type &;
    using iterator_category = std::forward_iterator_tag;

    Iterator()
        : current(nullptr)
        , position(0)
    {
    }

    Iterator<LeafType, ValueType>(LeafType * current, std::size_t position)
        : current(current)
        , position(position)
    {
    }

    Iterator<LeafType, ValueType>(const Iterator<LeafType, ValueType> & other)
        : current(other.current)
        , position(other.position)
    {
    }

    Iterator<LeafType, ValueType> operator=(const Iterator<LeafType, ValueType> & other)
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
            current = dynamic_cast<LeafType *>(current->right);
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

    //    template <class T1, class T2>
    //    friend bool operator==(const Iterator<LeafType, T1> & lhs, const Iterator<LeafType, T2> rhs) {
    //        return lhs.current == rhs.current && lhs.position == rhs.position;
    //    }

    friend bool operator==(const iterator & lhs, const iterator & rhs)
    {
        return lhs.current == rhs.current && lhs.position == lhs.position;
    }

    //    friend bool operator==(const Iterator<LeafType, std::remove_const<ValueType>> lhs, const Iterator<LeafType, const std::remove_const<ValueType>> & rhs)
    //    {
    //        return lhs.current == rhs.current && lhs.position == rhs.position;
    //    }

    friend bool operator!=(const iterator & lhs, const iterator & rhs)
    {
        return lhs.current != rhs.current || lhs.position != rhs.position;
    }
};

#endif //B_PLUS_TREE_ITERATOR_H