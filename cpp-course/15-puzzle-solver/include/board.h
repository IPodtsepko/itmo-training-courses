#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

class AbstractBoard
{
private:
    struct Point
    {
        std::size_t i, j;
    };

    std::size_t m_size = 0;
    Point m_zero;

    unsigned m_hamming_distance = 0;
    unsigned m_manhattan_distance = 0;

    std::unique_ptr<AbstractBoard> next_for_shift(int, int) const;

    bool move(unsigned i, unsigned j);

    unsigned manhattan_in_cell(int, int) const;

    std::vector<unsigned> get_linearized() const;

    virtual void set(std::size_t, std::size_t, unsigned) = 0;

protected:
    AbstractBoard() = default;
    explicit AbstractBoard(std::size_t size)
        : m_size(size)
    {
    }

    void precalculate_distances();

    virtual unsigned get(std::size_t, std::size_t) const = 0;

public:
    unsigned hamming() const;

    unsigned manhattan() const;

    bool is_goal() const;

    bool is_solvable() const;

    std::size_t size() const;

    std::string to_string() const;

    std::vector<std::unique_ptr<AbstractBoard>> get_next_states() const;

    std::vector<std::vector<unsigned>> get_data() const
    {
        std::vector<std::vector<unsigned>> data(m_size);
        for (std::size_t i = 0; i < m_size; ++i) {
            data[i] = std::vector<unsigned>(m_size);
            for (std::size_t j = 0; j < m_size; ++j) {
                data[i][j] = get(i, j);
            }
        }
        return data;
    };

    virtual std::size_t hash() const = 0;

    virtual std::unique_ptr<AbstractBoard> copy() const = 0;

    virtual ~AbstractBoard() = default;

    friend std::ostream & operator<<(std::ostream & out, const AbstractBoard & board);
};

class Board : public AbstractBoard
{
private:
    std::vector<std::vector<unsigned>> m_data{};

    void set(std::size_t, std::size_t, unsigned) override;
    unsigned int get(std::size_t, std::size_t) const override;

public:
    Board() = default;

    explicit Board(std::vector<std::vector<unsigned>> data)
        : AbstractBoard(data.size())
        , m_data(std::move(data))
    {
        precalculate_distances();
    }

    static Board create_goal(unsigned);

    static Board create_random(unsigned);

    std::unique_ptr<AbstractBoard> copy() const override;

    size_t hash() const override;

    const std::vector<unsigned> & operator[](std::size_t) const;

    friend bool operator==(const Board & lhs, const Board & rhs);

    friend bool operator!=(const Board & lhs, const Board & rhs);
};

class ReducedBoard : public AbstractBoard
{
private:
    std::size_t m_data = 0;

    void set(std::size_t, std::size_t, unsigned int) override;
    unsigned int get(std::size_t, std::size_t) const override;

public:
    explicit ReducedBoard(const std::vector<std::vector<unsigned>> & data);

    std::unique_ptr<AbstractBoard> copy() const override;

    size_t hash() const override;

    friend bool operator==(const ReducedBoard & lhs, const ReducedBoard & rhs);

    friend bool operator!=(const ReducedBoard & lhs, const ReducedBoard & rhs);
};