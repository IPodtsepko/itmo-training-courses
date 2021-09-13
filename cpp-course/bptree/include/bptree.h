#ifndef B_PLUS_TREE_H
#define B_PLUS_TREE_H

#include "bptree_iterator.h"
#include "nodes/abstract_node.h"
#include "nodes/internal_node.h"
#include "nodes/leaf_node.h"

#include <algorithm>
#include <cassert>
#include <functional>

template <class Key, class Value, class Less = std::less<Key>>
class BPTree : Less
{
    using node_type = AbstractNode<Key, Value, Less>;
    using internal_node_type = InternalNode<Key, Value, Less>;
    using leaf_node_type = LeafNode<Key, Value, Less>;

public:
    using key_type = Key;
    using mapped_type = Value;
    using value_type = std::pair<Key, Value>;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using size_type = std::size_t;

    using iterator = BPTreeIterator<leaf_node_type, value_type>;
    using const_iterator = BPTreeIterator<leaf_node_type, const value_type>;

    BPTree()
        : root(new leaf_node_type())
    {
    }
    BPTree(std::initializer_list<value_type> values)
        : root(new leaf_node_type())
    {
        for (auto value : values) {
            insert(value.first, value.second);
        }
    }
    BPTree(const BPTree & other)
        : root(new leaf_node_type())
    {
        for (const auto & element : other) {
            insert(element.first, element.second);
        }
    }
    BPTree(BPTree && other)
        : root(new leaf_node_type())
    {
        for (const auto & element : other) {
            insert(element.first, element.second);
        }
    }
    ~BPTree()
    {
        delete root;
        /*
         * Очистит дерево рекурсивно:
         * 1) Крайний случай - лист
         * 2) Удаление внутреннего узла вызовет удаление каждого из его детей
         * => все элементы в дереве удалятся
         * */
    }

    iterator begin()
    {
        if (tree_size) {
            return iterator(search_leaf(this->root->min()), 0);
        }
        return end();
    };
    const_iterator cbegin() const
    {
        if (tree_size) {
            return const_iterator(search_leaf(this->root->min()), 0);
        }
        return cend();
    };
    const_iterator begin() const
    {
        if (tree_size) {
            return const_iterator(search_leaf(this->root->min()), 0);
        }
        return end();
    };
    const_iterator cend() const
    {
        return const_iterator();
    };
    const_iterator end() const
    {
        return cend();
    };
    iterator end()
    {
        return iterator();
    };

    bool empty() const
    {
        return tree_size == 0;
    }
    size_type size() const
    {
        return tree_size;
    }
    void clear()
    {
        delete root;
        tree_size = 0;
        root = new leaf_node_type();
    };

    size_type count(const Key & key) const
    {
        return static_cast<size_type>(contains(key));
    }
    bool contains(const Key & key) const
    {
        leaf_node_type * leaf = search_leaf(key);
        assert(leaf);
        return leaf->contains(key);
    };
    std::pair<iterator, iterator> equal_range(const Key & key)
    {
        return equal_range<iterator>(key);
    }
    std::pair<const_iterator, const_iterator> equal_range(const Key & key) const
    {
        return equal_range<const_iterator>(key);
    }
    iterator lower_bound(const Key & key)
    {
        return lower_bound<iterator>(key);
    }
    const_iterator lower_bound(const Key & key) const
    {
        return this->lower_bound<const_iterator>(key);
    };
    iterator upper_bound(const Key & key)
    {
        return upper_bound<iterator>(key);
    }
    const_iterator upper_bound(const Key & key) const
    {
        return upper_bound<const_iterator>(key);
    };

    iterator find(const Key & key)
    {
        return find<iterator>(key);
    }
    const_iterator find(const Key & key) const
    {
        return find<const_iterator>(key);
    }

    Value & at(const Key & key)
    {
        return at_implementation(key);
    };
    const Value & at(const Key & key) const
    {
        return at_implementation(key);
    };

    Value & operator[](const Key & key)
    {
        return insert(key, Value()).first->second;
    };
    Value & operator[](Key && key)
    {
        return operator[](key);
    };

    std::pair<iterator, bool> insert(const Key & key, const Value & value)
    {
        leaf_node_type * leaf = search_leaf(key);
        if (leaf->insert(key, value)) {
            tree_size++;
            if (leaf->full()) {
                split(*leaf);
                auto right = as_leaf(leaf->right);
                if (right && right->contains(key)) {
                    return {iterator(right, right->lower_bound(key)), true};
                }
            }
            return {iterator(leaf, leaf->lower_bound(key)), true};
        }
        return {iterator(leaf, leaf->lower_bound(key)), false};
    }
    std::pair<iterator, bool> insert(const Key & key, Value && value)
    {
        return insert(key, value);
    }
    std::pair<iterator, bool> insert(Key && key, Value && value)
    {
        return insert(key, value);
    }
    template <class ForwardIt>
    void insert(ForwardIt begin, ForwardIt end)
    {
        for (ForwardIt it = begin; it != end; it++) {
            insert(it->first, it->second);
        }
    }
    void insert(std::initializer_list<value_type> pairs)
    {
        for (auto pair : pairs) {
            insert(pair.first, pair.second);
        }
    }

    iterator erase(const_iterator it)
    {
        Key key = it->first;
        remove(key);
        return lower_bound(key);
    };
    iterator erase(iterator it)
    {
        Key key = it->first;
        remove(key);
        return lower_bound(key);
    };
    iterator erase(const_iterator first, const_iterator last)
    {
        std::vector<Key> keys;
        for (const_iterator it = first; it != last; it++) {
            keys.push_back(it->first);
        }
        for (const auto & key : keys) {
            remove(key);
        }
        return lower_bound(keys.back());
    }
    size_type erase(const Key & key)
    {
        return static_cast<size_type>(remove(key));
    }

    bool remove(const Key & key)
    {
        node_type * leaf = search_leaf(key);
        if (delete_in_node(*leaf, key)) {
            tree_size--;
            return true;
        }
        return false;
    }

    void print()
    {
        this->root->print();
        std::cout << '\n';
    }

private:
    size_type tree_size = 0;
    node_type * root;

    leaf_node_type * search_leaf(const Key & key) const
    {
        assert(this->root);
        auto leaf = as_leaf(this->root->search_leaf(key));
        assert(leaf);
        return leaf;
    }

    Value & at_implementation(const Key & key) const
    {
        leaf_node_type * leaf = search_leaf(key);
        std::size_t i = leaf->index_of(key);
        if (i == leaf->size()) {
            throw std::out_of_range{"No such key"};
        }
        return leaf->get(i).second;
    }

    void split(node_type & node)
    {
        node_type * right_part = node.split();

        if (&node == root) {
            root = new internal_node_type(&node, right_part);
        }
        else {
            auto parent = as_internal(node.get_parent());
            assert(parent);
            parent->insert(right_part);
            if (parent->full()) {
                split(*parent);
            }
        }
    }

    void update_parent(node_type * parent)
    {
        if (!parent) {
            return;
        }
        parent->update();
        if (parent != this->root) {
            update_parent(parent->get_parent());
        }
    }
    void update_root()
    {
        auto internal = as_internal(root);
        if (internal) {
            auto updated = internal->update_root();
            if (updated) {
                delete root;
                root = updated.value();
                root->parent = root->left = root->right = nullptr;
            }
        }
    }

    bool delete_in_node(node_type & node, const Key & key)
    {
        const Key delimiter = node.min();
        if (!node.remove(key)) {
            return false;
        }

        node_type * require_update = &node;
        bool is_root = &node == root;
        if (node.small()) {
            node_type * left = node.left;
            node_type * right = node.right;

            if (left && left->can_give()) {
                node.ask_left();
            }
            else if (right && right->can_give()) {
                node.ask_right();
                if (right->get_parent() != node.get_parent()) {
                    update_parent(right->get_parent());
                }
            }
            else if (left && left->get_parent() == node.get_parent()) {
                left->merge(node);
                delete_in_node(*node.parent, delimiter);
                delete &node;
                require_update = left;
            }
            else if (right) {
                node.merge(*right);
                delete_in_node(*right->parent, right->min());
                delete right;
            }
            update_root();
        }
        if (!is_root) {
            update_parent(require_update->get_parent());
        }
        return true;
    }

    template <class IteratorType>
    IteratorType process_end_of_leaf(leaf_node_type * leaf, std::size_t position) const
    {
        if (leaf && position == leaf->size()) {
            return IteratorType(as_leaf(leaf->right), 0);
        }
        return IteratorType(leaf, position);
    }
    leaf_node_type * as_leaf(node_type * leaf) const
    {
        /*
         * Использовал dynamic, чтобы иметь возможность проверять,
         * что узел именно лист. Так как не использую её в дальнейшем,
         * заменил на static.
         * */
        return static_cast<leaf_node_type *>(leaf);
    }
    internal_node_type * as_internal(node_type * internal_node) const
    {
        return dynamic_cast<internal_node_type *>(internal_node);
    }

    template <class IteratorType>
    std::pair<IteratorType, IteratorType> equal_range(const Key & key) const
    {
        leaf_node_type * leaf = search_leaf(key);
        std::size_t i = leaf->index_of(key);
        if (i == leaf->size()) {
            return {IteratorType(), IteratorType()};
        }
        IteratorType it(leaf, i);
        return {IteratorType(it), it++};
    }
    template <class IteratorType>
    IteratorType find(const Key & key) const
    {
        IteratorType founded = lower_bound<IteratorType>(key);
        if (founded == IteratorType() || founded->first == key) {
            return founded;
        }
        return IteratorType();
    }
    template <class IteratorType>
    IteratorType lower_bound(const Key & key) const
    {
        leaf_node_type * leaf = search_leaf(key);
        return process_end_of_leaf<IteratorType>(leaf, leaf->lower_bound(key));
    }
    template <class IteratorType>
    IteratorType upper_bound(const Key & key) const
    {
        leaf_node_type * leaf = search_leaf(key);
        return process_end_of_leaf<IteratorType>(leaf, leaf->upper_bound(key));
    }
};

#endif //B_PLUS_TREE_H
