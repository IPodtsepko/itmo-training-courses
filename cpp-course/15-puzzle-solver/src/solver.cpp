#include "solver.h"

Solver::Path::Path(std::unique_ptr<AbstractBoard> board_ptr)
    : m_board_ptr(std::move(board_ptr))
{
}

Solver::Path::Path(const std::shared_ptr<Path> & prefix, std::unique_ptr<AbstractBoard> board_ptr)
    : m_prefix(prefix)
    , m_board_ptr(std::move(board_ptr))
{
}

unsigned Solver::potential(const std::shared_ptr<Path> & move)
{
    return move->m_board_ptr->hamming() + move->m_board_ptr->manhattan();
}

std::size_t Solver::Solution::moves() const
{
    return !m_moves.empty() ? m_moves.size() - 1 : 0;
}

Solver::Solution::Solution(const std::shared_ptr<Path> & best)
{
    Path * current = best.get();
    while (current) {
        auto not_reduced = dynamic_cast<Board *>(current->m_board_ptr.get());
        if (not_reduced != nullptr) {
            m_moves.emplace_back(*not_reduced);
        }
        else {
            m_moves.emplace_back(Board(current->m_board_ptr->get_data()));
        }
        current = current->m_prefix.get();
    }
    std::reverse(m_moves.begin(), m_moves.end());
}

Solver::Solution Solver::solve(const Board & initial)
{
    if (initial.is_goal()) {
        return Solution(initial);
    }

    if (!initial.is_solvable()) {
        return Solution();
    }

    std::unique_ptr<AbstractBoard> reduced_initial;

    if (initial.size() < 5) {
        reduced_initial = std::make_unique<ReducedBoard>(initial.get_data());
    }
    else {
        reduced_initial = std::make_unique<Board>(initial);
    }

    std::unordered_set<std::size_t> completed{reduced_initial->hash()};
    std::priority_queue<std::shared_ptr<Path>, std::vector<std::shared_ptr<Path>>, decltype(comp)> solutions(comp);
    solutions.push(std::make_shared<Path>(std::move(reduced_initial)));

    while (true) {
        std::shared_ptr<Path> best = solutions.top();
        solutions.pop();

        if (best->m_board_ptr->is_goal()) {
            Solution answer(best);
            return answer;
        }

        auto states = best->m_board_ptr->get_next_states();
        for (std::size_t i = 0; i < states.size(); ++i) {
            if (states[i] == nullptr) {
                continue;
            }
            std::size_t hash = states[i]->hash();
            if (completed.count(hash) == 0) {
                solutions.push(std::make_shared<Path>(best, std::move(states[i])));
                completed.insert(hash);
            }
        }
    }
}