#include "board.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <queue>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

class Solver
{
private:
    struct Path
    {
        std::shared_ptr<Path> m_prefix;
        std::unique_ptr<AbstractBoard> m_board_ptr;

        Path(std::unique_ptr<AbstractBoard>);
        Path(const std::shared_ptr<Path> &, std::unique_ptr<AbstractBoard>);
    };

    static unsigned potential(const std::shared_ptr<Path> & move);
    static constexpr auto comp = [](const std::shared_ptr<Path> & lhs, const std::shared_ptr<Path> & rhs) {
        return potential(lhs) > potential(rhs);
    };

public:
    class Solution
    {
    public:
        Solution() = default;
        explicit Solution(const Board & solution)
            : m_moves({solution})
        {
        }
        explicit Solution(const std::shared_ptr<Path> & best);

        std::size_t moves() const;

        using const_iterator = std::vector<Board>::const_iterator;
        const_iterator begin() const { return m_moves.begin(); }
        const_iterator end() const { return m_moves.end(); }

    private:
        std::vector<Board> m_moves;
    };
    static Solution solve(const Board & initial);
};
