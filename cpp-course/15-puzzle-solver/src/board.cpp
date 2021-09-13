#include "board.h"

#include "algorithmic_utils.h"

#include <cassert>
#include <iostream>
#include <random>
#include <sstream>

namespace {
std::vector<std::vector<unsigned>> goal(const unsigned n)
{
    const unsigned cell_count = n * n;
    auto board = std::vector<std::vector<unsigned>>(n);
    for (unsigned i = 0; i < n; i++) {
        board[i] = std::vector<unsigned>(n);
        for (unsigned j = 0; j < n; j++) {
            board[i][j] = (i * n + j + 1) % cell_count;
        }
    }
    return board;
}
} // anonymous namespace

// < ABSTRACT BOARD METHODS IMPLEMENTATION >
std::unique_ptr<AbstractBoard> AbstractBoard::next_for_shift(int di, int dj) const
{
    if (m_zero.i + di >= m_size || m_zero.j + dj >= m_size) {
        return nullptr;
    }
    std::unique_ptr<AbstractBoard> next_state = copy();
    next_state->move(m_zero.i + di, m_zero.j + dj);
    return next_state;
}

unsigned AbstractBoard::manhattan_in_cell(int i, int j) const
{
    const unsigned value = get(i, j) - 1;
    const int expected_i = value / size();
    const int expected_j = value - expected_i * size();
    return std::abs(i - expected_i) + std::abs(j - expected_j);
}

std::vector<unsigned> AbstractBoard::get_linearized() const
{
    std::vector<unsigned> linearized;
    for (unsigned i = 0; i < size(); i++) {
        for (unsigned j = 0; j < size(); j++) {
            if (get(i, j)) {
                linearized.push_back(get(i, j));
            }
        }
    }
    return linearized;
}

Board Board::create_goal(const unsigned n)
{
    return Board(goal(n));
}

Board Board::create_random(const unsigned n)
{
    std::random_device rd;
    std::mt19937 mersenne(rd());

    std::vector<std::vector<unsigned>> board = goal(n);

    // gen random permutation
    for (unsigned i = n * n; i > 1; --i) {
        std::uniform_int_distribution distribution(0, static_cast<int>(i - 1));
        unsigned j = distribution(mersenne);
        std::swap(board[(i - 1) / n][(i - 1) % n], board[j / n][j % n]);
    }

    return Board(board);
}

void AbstractBoard::precalculate_distances()
{
    for (std::size_t i = 0; i < m_size; i++) {
        for (std::size_t j = 0; j < m_size; j++) {
            if (get(i, j) == 0) {
                m_zero = {i, j};
            }
            else {
                m_manhattan_distance += manhattan_in_cell(i, j);
            }
            if (get(i, j) != (i * m_size + j + 1) % (m_size * m_size)) {
                m_hamming_distance++;
            }
        }
    }
}

unsigned AbstractBoard::hamming() const
{
    return m_hamming_distance;
}

unsigned AbstractBoard::manhattan() const
{
    return m_manhattan_distance;
}

bool AbstractBoard::is_goal() const
{
    return m_hamming_distance == 0;
}

bool AbstractBoard::is_solvable() const
{
    if (m_size == 0) {
        return true;
    }
    std::vector<unsigned> linearized;
    for (unsigned i = 0; i < size(); i++) {
        for (unsigned j = 0; j < size(); j++) {
            auto value = get(i, j);
            if (value) {
                linearized.push_back(value);
            }
        }
    }
    return permutation_odd(linearized) != (m_zero.i % 2 == 1 || m_size % 2 == 1);
}

std::size_t AbstractBoard::size() const
{
    return m_size;
}

std::string AbstractBoard::to_string() const
{
    std::stringstream string_builder;
    const unsigned n = size();
    for (unsigned i = 0; i < n; i++) {
        for (unsigned j = 0; j < n; j++) {
            string_builder << get(i, j) << (j < n - 1 ? " " : "");
        }
        if (i < n - 1) {
            string_builder << '\n';
        }
    }
    return string_builder.str();
}

bool AbstractBoard::move(unsigned int i, unsigned int j)
{
    if (i >= size() || j >= size()) {
        return false;
    }
    const unsigned value = get(i, j);
    m_hamming_distance += (static_cast<unsigned>(m_zero.i == size() - 1 && m_zero.j == size() - 1) +
                           static_cast<unsigned>(value == i * size() + j + 1) -
                           static_cast<unsigned>(value == m_zero.i * size() + m_zero.j + 1) -
                           static_cast<unsigned>(i == size() - 1 && j == size() - 1));

    m_manhattan_distance -= manhattan_in_cell(i, j);

    set(m_zero.i, m_zero.j, value);
    set(i, j, 0);

    m_manhattan_distance += manhattan_in_cell(m_zero.i, m_zero.j);

    m_zero = {i, j};
    return true;
}

std::vector<std::unique_ptr<AbstractBoard>> AbstractBoard::get_next_states() const
{
    std::vector<std::unique_ptr<AbstractBoard>> states;

    states.push_back(next_for_shift(0, -1));
    states.push_back(next_for_shift(-1, 0));
    states.push_back(next_for_shift(0, 1));
    states.push_back(next_for_shift(1, 0));

    return states;
}

const std::vector<unsigned> & Board::operator[](std::size_t idx) const
{
    return m_data[idx];
}

std::ostream & operator<<(std::ostream & out, const AbstractBoard & board)
{
    return out << board.to_string() << '\n';
}

// < GENERAL BOARD IMPLEMENTATION >
std::unique_ptr<AbstractBoard> Board::copy() const
{
    return std::make_unique<Board>(*this);
}

size_t Board::hash() const
{
    std::size_t h = 0;
    for (const auto & line : m_data) {
        for (const auto & cell : line) {
            h = h * 31 + cell;
        }
    }
    return h;
}

unsigned int Board::get(std::size_t i, std::size_t j) const
{
    return m_data[i][j];
}

void Board::set(std::size_t i, std::size_t j, unsigned value)
{
    m_data[i][j] = value;
}

bool operator==(const Board & lhs, const Board & rhs)
{
    return lhs.m_data == rhs.m_data;
}

bool operator!=(const Board & lhs, const Board & rhs)
{
    return !(lhs == rhs);
}

// < REDUCED BOARD IMPLEMENTATION >
ReducedBoard::ReducedBoard(const std::vector<std::vector<unsigned int>> & data)
    : AbstractBoard(data.size())
{
    for (int i = size() - 1; i >= 0; --i) {
        for (int j = size() - 1; j >= 0; --j) {
            m_data = (m_data << 4) | data[i][j];
        }
    }
    precalculate_distances();
}

std::unique_ptr<AbstractBoard> ReducedBoard::copy() const
{
    return std::make_unique<ReducedBoard>(*this);
}

size_t ReducedBoard::hash() const
{
    return m_data;
}

unsigned int ReducedBoard::get(std::size_t i, std::size_t j) const
{
    unsigned shift = 4 * (i * size() + j);
    return (m_data >> shift) & 0b1111;
}

void ReducedBoard::set(std::size_t i, std::size_t j, unsigned value)
{
    std::size_t shift = 4 * (i * size() + j); // cell position
    m_data &= ~(0b1111ull << shift);          // turn off cell
    m_data |= static_cast<std::size_t>(value) << shift;
}

bool operator==(const ReducedBoard & lhs, const ReducedBoard & rhs)
{
    return lhs.m_data == rhs.m_data;
}

bool operator!=(const ReducedBoard & lhs, const ReducedBoard & rhs)
{
    return !(lhs == rhs);
}
