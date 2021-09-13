#ifndef B_PLUS_TREE_INTERNAL_NODE_H
#define B_PLUS_TREE_INTERNAL_NODE_H

#include "abstract_node.h"

#include <cassert>
#include <optional>

template <class Key, class Value, class Less>
class InternalNode : public AbstractNode<Key, Value, Less>
{
    using node_type = AbstractNode<Key, Value, Less>;
    using internal_node_type = InternalNode<Key, Value, Less>;

    Key subtree_minimum;
    std::vector<Key> keys{};
    std::vector<node_type *> children{};

public:
    InternalNode() = default;

    InternalNode(node_type * left, node_type * right)
        : children({left, right})
    {
        left->set_parent(this);
        right->set_parent(this);
        update();
    }

    const Key & min() const override
    {
        return subtree_minimum;
    }

    size_t size() const override
    {
        return keys.size();
    }

    node_type * search_leaf(const Key & key) override
    {
        for (std::size_t i = 0; i <= size(); i++) {
            if (i == size() || Less{}(key, keys[i])) {
                return children[i]->search_leaf(key);
            }
        }
        return nullptr;
    }

    void update_min()
    {
        subtree_minimum = children[0]->min();
    }

    void update_keys()
    {
        keys = std::vector<Key>();
        for (std::size_t i = 1; i < children.size(); i++) {
            keys.push_back(children[i]->min());
        }
    }

    void update() override
    {
        update_min();
        update_keys();
    }

    void insert(node_type * child)
    {
        if (child) {
            std::size_t i = lower_bound(child->min());
            children.insert(children.begin() + i + 1, child);
            child->set_parent(this);
            update();
        }
    }

    node_type * get_right_part() override
    {
        auto right_part = new internal_node_type();

        std::size_t t = node_type::node_capacity / 2;

        right_part->keys = split_vector(this->keys, t + 1);
        this->keys.pop_back();

        right_part->children = split_vector(this->children, t + 1);

        for (auto child : right_part->children) {
            child->set_parent(right_part);
        }

        right_part->update_min();

        return right_part;
    }

    std::size_t lower_bound(const Key & key) const override
    {
        return std::lower_bound(keys.begin(), keys.end(), key, Less{}) - keys.begin();
    }

    void merge_implementation(node_type & other) override
    {
        auto source = static_cast<internal_node_type *>(&other);
        for (auto child : source->children) {
            children.push_back(child);
            child->set_parent(this);
        }
        source->children = {};
        update_keys();
    }

    void print() override
    {
        std::cout << "(";
        for (std::size_t i = 0; i < size(); i++) {
            children[i]->print();
            std::cout << ' ' << keys[i] << ' ';
        }
        children[size()]->print();
        std::cout << ")";
    }

    bool remove(const Key & key) override
    {
        std::size_t i = lower_bound(key);
        if (i == keys.size() || keys[i] != key) {
            return false;
        }
        keys.erase(keys.begin() + i);
        children.erase(children.begin() + i + 1);
        return true;
    }

    std::optional<node_type *> update_root()
    {
        if (this->empty()) {
            node_type * new_root = children[0];
            children = {};
            return new_root;
        }
        return {};
    }

    void ask_left() override
    {
        auto left = dynamic_cast<internal_node_type *>(this->left);

        children.insert(children.begin(), left->children.back());
        left->children.pop_back();
        children[0]->parent = this;

        update();
        left->update();
    }

    void ask_right() override
    {
        auto right = dynamic_cast<internal_node_type *>(this->right);

        children.push_back(right->children.front());
        right->children.erase(right->children.begin());
        children.back()->parent = this;

        update();
        right->update();
    }

    void cut()
    {
        children = {};
    }

    ~InternalNode()
    {
        for (auto child : children) {
            delete child;
        }
    }
};

#endif //B_PLUS_TREE_INTERNAL_NODE_H
