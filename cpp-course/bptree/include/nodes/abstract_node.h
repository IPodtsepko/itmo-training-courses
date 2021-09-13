#ifndef B_PLUS_TREE_ABSTRACT_NODE_H
#define B_PLUS_TREE_ABSTRACT_NODE_H

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

template <class T>
std::vector<T> split_vector(std::vector<T> & source, std::size_t i)
{
    std::vector<T> right_part = std::vector<T>(source.begin() + i, source.end());
    source.erase(source.begin() + i, source.end());
    return right_part;
}

template <class Key, class Value, class Less>
class AbstractNode : Less
{
protected:
    static constexpr std::size_t block_size = 4096;
    static constexpr std::size_t node_capacity = block_size / sizeof(Key);
    using node_type = AbstractNode<Key, Value, Less>;

public:
    node_type * parent = nullptr;
    node_type * left = nullptr;
    node_type * right = nullptr;

    virtual const Key & min() const = 0;
    virtual std::size_t size() const = 0;

    void set_parent(node_type * new_parent)
    {
        parent = new_parent;
    }

    node_type * get_parent()
    {
        return parent;
    }

    bool full()
    {
        return size() == node_capacity;
    }
    bool small()
    {
        return size() < node_capacity / 2 - 1;
    }
    bool can_give()
    {
        return size() >= node_capacity / 2;
    }
    bool empty()
    {
        return size() == 0;
    }

    virtual node_type * search_leaf(const Key & key) = 0;

    virtual node_type * get_right_part() = 0;
    node_type * split()
    {
        auto right_part = get_right_part();

        right_part->right = right;
        if (right) {
            right->left = right_part;
        }

        right = right_part;
        right_part->left = this;

        right_part->parent = parent;

        return right_part;
    }

    virtual void merge_implementation(node_type & node) = 0;
    void merge(node_type & node)
    {
        merge_implementation(node);
        right = node.right;
        if (right) {
            node.right->left = this;
        }
    }

    virtual std::size_t lower_bound(const Key & key) const = 0;

    virtual void print() = 0;

    virtual bool remove(const Key & key) = 0;
    virtual void ask_left() = 0;
    virtual void ask_right() = 0;

    virtual void update() = 0;

    virtual ~AbstractNode() = default;
};

#endif //B_PLUS_TREE_ABSTRACT_NODE_H