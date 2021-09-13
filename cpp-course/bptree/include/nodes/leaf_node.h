#ifndef B_PLUS_TREE_LEAF_NODE_H
#define B_PLUS_TREE_LEAF_NODE_H

#include "abstract_node.h"

#include <fstream>
#include <functional>

template <class Key, class Value, class Less>
class LeafNode : public AbstractNode<Key, Value, Less>
{
    using node_type = AbstractNode<Key, Value, Less>;
    using leaf_node_type = LeafNode<Key, Value, Less>;

public:
    using value_type = std::pair<Key, Value>;

    std::vector<value_type> values{};

    const Key & min() const override
    {
        return values.front().first;
    }

    size_t size() const override
    {
        return values.size();
    }

    node_type * get_right_part() override
    {
        auto right_part = new leaf_node_type();
        right_part->values = split_vector(values, node_type::node_capacity / 2);
        return right_part;
    }

    void merge_implementation(node_type & node) override
    {
        auto source = static_cast<leaf_node_type *>(&node);
        for (const auto & value : source->values) {
            values.push_back(value);
        }
    }

    node_type * search_leaf(const Key &) override
    {
        return this;
    }

    std::size_t lower_bound(const Key & key) const override
    {
        return std::lower_bound(values.begin(), values.end(), key, [](value_type x, Key key) {
                   return Less{}(x.first, key);
               }) -
                values.begin();
    }

    std::size_t upper_bound(const Key & key) const
    {
        return std::upper_bound(values.begin(), values.end(), key, [](Key key, value_type x) {
                   return Less{}(key, x.first);
               }) -
                values.begin();
    }

    bool insert(const Key & key, const Value & value)
    {
        std::size_t i = lower_bound(key);
        if (i < values.size() && values[i].first == key) {
            return false;
        }
        values.insert(values.begin() + i, {key, value});
        return true;
    }

    void print() override
    {
        std::cout << "(";
        for (std::size_t i = 0; i < size(); i++) {
            std::cout << values[i].first << '*';
            if (i != size() - 1) {
                std::cout << ' ';
            }
        }
        std::cout << ")";
    }

    bool remove(const Key & key) override
    {
        std::size_t i = lower_bound(key);
        if (i == values.size() || values[i].first != key) {
            return false;
        }
        values.erase(values.begin() + i);
        return true;
    }

    void ask_left() override
    {
        auto left = dynamic_cast<leaf_node_type *>(this->left);
        values.insert(values.begin(), left->values.back());
        left->values.pop_back();
    }

    void ask_right() override
    {
        auto right = dynamic_cast<leaf_node_type *>(this->right);
        values.push_back(right->values.front());
        right->values.erase(right->values.begin());
    }

    value_type & get(std::size_t position)
    {
        return values[position];
    }

    const value_type & get(std::size_t position) const
    {
        return values[position];
    }

    bool contains(const Key & key)
    {
        return index_of(key) != size();
    }

    std::size_t index_of(const Key & key) const
    {
        std::size_t i = lower_bound(key);
        if (i == size() || values[i].first != key) {
            return size();
        }
        return i;
    }

    void update() override
    {
    }

    ~LeafNode() = default;
};

#endif //B_PLUS_TREE_LEAF_NODE_H
