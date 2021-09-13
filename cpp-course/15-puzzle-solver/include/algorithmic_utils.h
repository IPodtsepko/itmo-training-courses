#pragma once

#include <vector>

inline bool permutation_odd(const std::vector<unsigned> & a)
{
    bool odd = false;
    std::vector<bool> used(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::size_t j = i;
        odd = (odd == used[j]);
        while (!used[j]) {
            used[j] = true;
            j = a[j] - 1;
            odd = !odd;
        }
    }
    return odd;
}