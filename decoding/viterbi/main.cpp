#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>

static std::mt19937 engine{std::random_device{}()};

class Encoder
{
public:
    Encoder(const std::size_t n, const std::size_t k, std::vector<std::vector<int>> data)
        : m_n(n)
        , m_k(k)
        , m_data(std::move(data))
    {
    }

    std::vector<int> encode(const std::vector<int> & input) const
    {
        std::vector<int> word(m_n, 0);
        for (std::size_t i = 0; i < m_n; ++i) {
            for (std::size_t j = 0; j < m_k; ++j) {
                word[i] ^= input[j] * m_data[j][i];
            }
        }
        return word;
    }

private:
    const std::size_t m_n;
    const std::size_t m_k;
    const std::vector<std::vector<int>> m_data;
};

std::vector<std::vector<int>> to_min_span_form(std::vector<std::vector<int>> data, const std::size_t n, const std::size_t k)
{
    const int ZERO = 0;

    // Make step form the left side of matrix
    for (std::size_t steps_found = 0, column = 0; column < n && steps_found < k; ++column) {
        std::size_t row = steps_found;
        while (row < k && data[row][column] == ZERO) {
            ++row;
        }
        if (row == k) {
            continue;
        }

        if (steps_found != row) {
            std::swap(data[steps_found], data[row]);
            row = steps_found;
        }

        for (std::size_t i = row + 1; i < k; ++i) {
            if (data[i][column] != ZERO) {
                for (std::size_t k = 0; k < n; ++k) {
                    data[i][k] ^= data[row][k];
                }
            }
        }
        steps_found += 1;
    }

    // Make ends different
    std::vector<std::size_t> ends(k, n);
    for (int column = n - 1; column > -1; --column) {
        int row = k - 1;
        bool all_ends_found = true;
        while (row > -1 && (ends[row] < n || data[row][column] == ZERO)) {
            all_ends_found &= ends[row] < n;
            --row;
        }
        if (row < 0) {
            if (all_ends_found) {
                break;
            }
            continue;
        }

        ends[row] = static_cast<std::size_t>(column);
        for (std::size_t i = 0; i < row; ++i) {
            if (ends[i] < n || data[i][column] == 0) {
                continue;
            }
            for (std::size_t k = 0; k < n; ++k) {
                data[i][k] ^= data[row][k];
            }
        }
    }

    return std::move(data);
}

class State
{
public:
    State(const std::vector<std::size_t> & begins, const std::vector<std::size_t> & ends)
        : m_index(0)
        , m_begins(begins)
        , m_ends(ends)
    {
    }

    State(const State & other)
        : m_index(0)
        , m_begins(other.m_begins)
        , m_ends(other.m_ends)
    {
    }

    void next()
    {
        ++m_index;
    }

    std::vector<std::size_t> get_active() const
    {
        std::vector<std::size_t> result;
        for (std::size_t i = 0; i < m_begins.size(); ++i) {
            if (m_begins[i] <= m_index && m_index < m_ends[i]) {
                result.push_back(i);
            }
        }
        return result;
    }

    std::size_t get_active_count() const
    {
        std::size_t result = 0;
        for (std::size_t i = 0; i < m_begins.size(); ++i) {
            if (m_begins[i] <= m_index && m_index < m_ends[i]) {
                ++result;
            }
        }
        return result;
    }

private:
    std::size_t m_index;
    const std::vector<std::size_t> & m_begins;
    const std::vector<std::size_t> & m_ends;
};

std::vector<std::size_t>
get_begins(const std::size_t k, const std::vector<std::vector<int>> & data)
{
    std::vector<std::size_t> begins(k, 0);
    for (std::size_t i = 0; i < k; ++i) {
        auto & j = begins[i];
        while (data[i][j] == 0) {
            ++j;
        }
    }
    return begins;
}

std::vector<std::size_t>
get_ends(const std::size_t n, const std::size_t k, const std::vector<std::vector<int>> & data)
{
    std::vector<std::size_t> ends(k, n - 1);
    for (std::size_t i = 0; i < k; ++i) {
        auto & j = ends[i];
        while (data[i][j] == 0) {
            --j;
        }
    }
    return ends;
}

std::vector<std::size_t>
get_trellis_profile(
        const std::size_t n,
        const std::vector<std::vector<int>> & min_span_form,
        State state)
{
    std::vector<std::size_t> profile = {1};
    for (std::size_t i = 0; i < n; ++i, state.next()) {
        const auto active_count = state.get_active_count();
        const auto level_size = 1 << active_count;
        profile.push_back(level_size);
    }

    return profile;
}

struct Node
{
    int m_0 = -1;
    int m_1 = -1;
};

struct Trellis
{
    std::vector<std::size_t> m_profile;
    std::vector<Node> m_nodes;
};

Trellis build_trellis(const std::size_t n, const std::size_t k, const std::vector<std::vector<int>> & data)
{
    const auto min_span_form = to_min_span_form(data, n, k);
    const auto begins = get_begins(k, min_span_form);
    const auto ends = get_ends(n, k, min_span_form);
    State state(begins, ends);

    const auto trellis_profile = get_trellis_profile(n, min_span_form, state);
    std::size_t size = 0;
    for (const auto & layer_size : trellis_profile) {
        size += layer_size;
    }

    std::vector<Node> nodes{size};

    std::size_t last_level_marker = 0;
    std::size_t last_level_size = 1;
    std::vector<std::size_t> last_active_lines{};
    for (std::size_t level = 1; level < trellis_profile.size(); ++level, state.next()) {
        const auto level_marker = last_level_marker + last_level_size;
        const auto level_size = trellis_profile[level];

        auto active_lines = state.get_active();

        for (std::size_t i = 0; i < last_level_size; ++i) {
            for (std::size_t j = 0; j < level_size; ++j) {
                auto & edge_begin = nodes[last_level_marker + i];
                const auto edge_end = level_marker + j;

                bool edge_exists = true;
                int label = 0;

                for (std::size_t line = 0, lhs = 0, rhs = 0; line < k; ++line) {
                    const auto bit_in_min_span_form = min_span_form[line][level - 1];
                    const auto begin_bit = (i >> lhs) & 1;
                    const auto end_bit = (j >> rhs) & 1;

                    const auto line_active_on_last_layer = lhs < last_active_lines.size() && last_active_lines[lhs] == line;
                    if (line_active_on_last_layer) {
                        ++lhs;
                    }
                    const auto line_active_on_current_layer = rhs < active_lines.size() && active_lines[rhs] == line;
                    if (line_active_on_current_layer) {
                        ++rhs;
                    }

                    if (line_active_on_last_layer) {
                        if (line_active_on_current_layer && begin_bit != end_bit) {
                            edge_exists = false;
                            break;
                        }
                        label ^= (bit_in_min_span_form & begin_bit);
                    }
                    else if (line_active_on_current_layer) {
                        label ^= (bit_in_min_span_form & end_bit);
                    }
                }

                if (edge_exists) {
                    (label == 0 ? edge_begin.m_0 : edge_begin.m_1) = edge_end;
                }
            }
        }

        last_level_marker = level_marker;
        last_level_size = level_size;
        std::swap(last_active_lines, active_lines);
    }

    return {trellis_profile, nodes};
}

class Decoder
{
public:
    Decoder(const std::size_t n, const std::size_t k, const std::vector<std::vector<int>> & data)
        : m_trellis(build_trellis(n, k, data))
    {
    }

private:
    struct Record
    {
        long double m_value = std::numeric_limits<long double>::infinity();
        int m_symbol = -1;
        int m_best_parent = -1;
    };

public:
    const std::vector<std::size_t> & get_trellis_profile() const
    {
        return m_trellis.m_profile;
    }

    std::vector<int> decode(const std::vector<long double> & llhrs) const
    {
        const auto & nodes = m_trellis.m_nodes;
        std::vector<Record> records{nodes.size()};

        std::size_t i = 0;
        records[i].m_value = 0.0;
        const auto & profile = m_trellis.m_profile;
        for (std::size_t level = 0; level < profile.size() - 1; ++level) {
            const auto level_size = profile[level];
            for (std::size_t j = 0; j < level_size; ++i, ++j) {
                const auto & record = records[i];
                const auto & node = nodes[i];
                if (node.m_0 != -1) {
                    relax(records[node.m_0], record.m_value - llhrs[level], 0, i);
                }
                if (node.m_1 != -1) {
                    relax(records[node.m_1], record.m_value + llhrs[level], 1, i);
                }
            }
        }

        std::vector<int> result;
        result.reserve(llhrs.size());
        i = records.size() - 1;
        while (true) {
            const auto & record = records[i];
            if (record.m_best_parent == -1) {
                break;
            }
            result.push_back(record.m_symbol);
            i = record.m_best_parent;
        }
        std::reverse(result.begin(), result.end());

        return result;
    }

private:
    static void relax(Record & next, const long double weight, const int symbol, const int parent)
    {
        if (next.m_value <= weight) {
            return;
        }
        next.m_value = weight;
        next.m_symbol = symbol;
        next.m_best_parent = parent;
    }

private:
    const Trellis m_trellis;
};

class Simulator
{
public:
    Simulator(const Encoder & encoder, const Decoder & decoder)
        : m_encoder(encoder)
        , m_decoder(decoder)
    {
    }

    long double simulate(
            const std::size_t n,
            const std::size_t k,
            const long double noise_level,
            const std::size_t num_of_iterations,
            const std::size_t max_errors) const
    {
        std::size_t j = 0;
        std::size_t errors_count = 0;
        for (; j < num_of_iterations && errors_count < max_errors; ++j) {
            static std::uniform_int_distribution<> uniform_int_distribution(0, 1);

            std::vector<int> input;
            input.reserve(k);

            for (std::size_t i = 0; i < k; ++i) {
                input.push_back(uniform_int_distribution(engine));
            }

            const auto encoded = m_encoder.encode(input);

            std::normal_distribution<> normal_distribution(0, std::sqrt(0.5 * pow(10, (-noise_level) / 10) * (static_cast<long double>(n) / k)));
            std::vector<long double> noised;
            noised.reserve(encoded.size());
            for (std::size_t i = 0; i < encoded.size(); ++i) {
                noised.push_back(1 - 2 * encoded[i] + normal_distribution(engine));
            }

            const auto decoded = m_decoder.decode(noised);
            bool decoding_invalid = false;
            for (std::size_t i = 0; i < n; ++i) {
                if (encoded[i] != decoded[i]) {
                    decoding_invalid = true;
                    break;
                }
            }

            if (decoding_invalid) {
                ++errors_count;
            }
        }

        return static_cast<long double>(errors_count) / j;
    }

private:
    const Encoder & m_encoder;
    const Decoder & m_decoder;
};

template <typename T>
void print(std::ostream & output, const std::vector<T> & values)
{
    for (std::size_t i = 0; i < values.size(); ++i) {
        output << values[i];
        if (i + 1 < values.size()) {
            output << " ";
        }
    }
    output << "\n";
}

int main()
{
    engine.seed(3);
    std::ifstream input_file("input.txt");

    if (!input_file.is_open()) {
        return 1;
    }

    std::size_t n;
    std::size_t k;

    input_file >> n >> k;

    std::vector<std::vector<int>> data(k);
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            int value;
            input_file >> value;
            data[i].push_back(value);
        }
    }

    Encoder encoder{n, k, data};
    Decoder decoder{n, k, data};
    Simulator simulator{encoder, decoder};

    std::ofstream output_file("output.txt");
    print(output_file, decoder.get_trellis_profile());
    std::string command;
    while (input_file >> command) {
        if (command == "Encode") {
            std::vector<int> input(k);
            for (auto & value : input) {
                input_file >> value;
            }
            print(output_file, encoder.encode(input));
        }
        else if (command == "Decode") {
            std::vector<long double> input(n);
            for (auto & value : input) {
                input_file >> value;
            }
            print(output_file, decoder.decode(input));
        }
        else if (command == "Simulate") {
            long double noise_level;
            std::size_t num_of_iterations;
            std::size_t max_errors;
            input_file >> noise_level >> num_of_iterations >> max_errors;
            output_file << std::scientific << simulator.simulate(n, k, noise_level, num_of_iterations, max_errors) << '\n';
        }
        else {
            output_file << "FAILURE\n";
        }
    }

    return 0;
}
