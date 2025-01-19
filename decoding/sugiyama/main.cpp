#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

static std::mt19937 engine{std::random_device{}()};

template <typename T>
void print(std::ostream & output, const std::vector<T> & values, const std::string & name = "")
{
    if (!name.empty()) {
        output << name << ": ";
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
        output << values[i];
        if (i + 1 < values.size()) {
            output << " ";
        }
    }
    output << "\n";
}

void remove_leading_zeros(std::vector<std::size_t> & polynomial)
{
    while (!polynomial.empty() && polynomial.back() == 0) {
        polynomial.pop_back();
    }
}

class GF
{
public:
    GF(const std::size_t primitive_poly_mask)
    {
        const auto primitive_poly_degree = get_poly_degree(primitive_poly_mask);
        const auto field_size = (1 << primitive_poly_degree) - 1;

        m_masks = {1};
        m_degrees = {0};

        m_masks.resize(field_size);
        m_degrees.resize(field_size);

        for (std::size_t i = 1; i < field_size; ++i) {
            auto mask = m_masks[i - 1] << 1;
            if (mask > field_size) {
                mask ^= primitive_poly_mask;
            }
            m_masks[i] = mask;
            m_degrees[mask - 1] = i;
        }
    }

    std::vector<std::size_t> get_generating_polynomial(const std::size_t delta)
    {
        std::vector<std::size_t> result = {1};

        std::vector<std::vector<std::size_t>> classes;
        const auto field_size = m_masks.size();
        std::vector<bool> class_found(field_size, false);
        for (std::size_t i = 1; i < delta; ++i) {
            if (class_found[i]) {
                continue;
            }

            auto j = i;
            while (!class_found[j]) {
                class_found[j] = true;

                std::vector<std::size_t> product_prefix = {0};
                product_prefix.reserve(result.size() + 1);
                product_prefix.insert(product_prefix.end(), result.begin(), result.end());

                for (std::size_t k = 0; k < result.size(); ++k) {
                    if (result[k] == 0) {
                        continue;
                    }
                    product_prefix[k] ^= get_mask(get_degree(result[k]) + j);
                }

                result = std::move(product_prefix);

                j = (j << 1) % field_size;
            }
        }

        return result;
    }

    std::vector<std::size_t> sum_polynomials(
            const std::vector<std::size_t> & lhs,
            const std::vector<std::size_t> & rhs) const
    {
        std::vector<std::size_t> result(std::max(lhs.size(), rhs.size()), 0);
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            result[i] = lhs[i];
        }
        for (std::size_t i = 0; i < rhs.size(); ++i) {
            result[i] ^= rhs[i];
        }
        remove_leading_zeros(result);
        return result;
    }

    std::vector<std::size_t> multiply_polinomials(
            const std::vector<std::size_t> & lhs,
            const std::vector<std::size_t> & rhs) const
    {
        std::vector<std::size_t> result(lhs.size() + rhs.size() - 1);
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            for (std::size_t j = 0; j < rhs.size(); ++j) {
                result[i + j] ^= get_masks_mul(lhs[i], rhs[j]);
            }
        }
        remove_leading_zeros(result);
        return result;
    }

    void divide_polynomials(
            const std::vector<std::size_t> & lhs,
            const std::vector<std::size_t> & rhs,
            std::vector<std::size_t> & mod,
            std::vector<std::size_t> & div) const
    {
        auto divisor = rhs;
        remove_leading_zeros(divisor);
        const auto k = divisor.size() - 1;

        mod = lhs;
        remove_leading_zeros(mod);

        if (mod.size() < divisor.size()) {
            return;
        }

        div = std::vector<std::size_t>(mod.size() - k, 0);

        while (mod.size() > k) {
            const auto multiplier_degree = mod.size() - k - 1;
            const auto multiplier = get_masks_div(mod.back(), divisor.back());

            div[multiplier_degree] = multiplier;

            for (std::size_t i = multiplier_degree; i < mod.size(); ++i) {
                const auto divisor_coefficient = divisor[i - multiplier_degree];
                mod[i] ^= get_masks_mul(divisor_coefficient, multiplier);
            }

            remove_leading_zeros(mod);
        }
    }

    std::size_t get_max_degree() const
    {
        return m_masks.size();
    }

    std::size_t get_mask(const std::size_t degree) const
    {
        return m_masks[degree % m_masks.size()];
    }

    std::size_t get_degree(const std::size_t mask) const
    {
        assert(mask != 0 && mask - 1 < m_degrees.size());
        return m_degrees[mask - 1];
    }

    std::size_t get_masks_sum(const std::size_t lhs, const std::size_t rhs) const
    {
        return lhs ^ rhs;
    }

    std::size_t get_degrees_sum(const std::size_t lhs, const std::size_t rhs) const
    {
        return get_degree(get_masks_sum(get_mask(lhs), get_mask(rhs)));
    }

    std::size_t get_degrees_mul(const std::size_t lhs, const std::size_t rhs) const
    {
        return (lhs + rhs) % m_masks.size();
    }

    std::size_t get_masks_mul(const std::size_t lhs, const std::size_t rhs) const
    {
        if (lhs == 0 || rhs == 0) {
            return 0;
        }
        return get_mask(get_degrees_mul(get_degree(lhs), get_degree(rhs)));
    }

    std::size_t get_degree_inv(const std::size_t degree) const
    {
        return (m_masks.size() - degree % m_masks.size()) % m_masks.size();
    }

    std::size_t get_mask_inv(const std::size_t mask) const
    {
        return get_mask(get_degree_inv(get_degree(mask)));
    }

    std::size_t get_degree_div(const std::size_t lhs, const std::size_t rhs) const
    {
        return (lhs + get_degree_inv(rhs)) % m_masks.size();
    }

    std::size_t get_masks_div(const std::size_t lhs, const std::size_t rhs) const
    {
        if (lhs == 0 || rhs == 0) {
            return 0;
        }
        return get_mask(get_degree_div(get_degree(lhs), get_degree(rhs)));
    }

private:
    static std::size_t get_poly_degree(std::size_t poly)
    {
        poly >>= 1;
        std::size_t degree = 0;
        while (poly > 0) {
            ++degree;
            poly >>= 1;
        }
        return degree;
    }

private:
    /**
     * @brief A list of degrees of polynomials.
     *
     * Cell i contains the degree of the polynomial with bitmask equals i.
     */
    std::vector<int> m_degrees;
    /**
     * @brief A list of bitmasks of polynomials.
     *
     * Cell i contains the bit mask of the polynomial a^i.
     */
    std::vector<int> m_masks;
};

void divide_polynomials(
        const std::vector<std::size_t> & lhs,
        const std::vector<std::size_t> & rhs,
        std::vector<std::size_t> & mod,
        std::vector<std::size_t> & div)
{
    auto divisor = rhs;
    remove_leading_zeros(divisor);
    const auto k = divisor.size() - 1;

    mod = lhs;
    remove_leading_zeros(mod);

    if (mod.size() < divisor.size()) {
        return;
    }

    div = std::vector<std::size_t>(mod.size() - k, 0);

    while (mod.size() > k) {
        const auto multiplier_degree = mod.size() - k - 1;
        div[multiplier_degree] = 1;

        for (std::size_t i = multiplier_degree; i < mod.size(); ++i) {
            mod[i] ^= divisor[i - multiplier_degree];
        }

        remove_leading_zeros(mod);
    }
}

class Encoder
{
public:
    Encoder(std::vector<std::size_t> generating_polynomial)
        : m_generating_polynomial(std::move(generating_polynomial))
    {
    }

    std::vector<std::size_t> encode(const std::vector<std::size_t> & input) const
    {
        std::vector<std::size_t> result(m_generating_polynomial.size() - 1 + input.size(), 0);
        for (std::size_t i = 0; i < input.size(); ++i) {
            result[m_generating_polynomial.size() - 1 + i] = input[i];
        }

        std::vector<std::size_t> mod;
        std::vector<std::size_t> div;
        divide_polynomials(result, m_generating_polynomial, mod, div);
        for (std::size_t i = 0; i < mod.size(); ++i) {
            result[i] = mod[i];
        }

        return result;
    }

private:
    std::vector<std::size_t> m_generating_polynomial;
};

class Decoder
{
public:
    Decoder(const GF & field, const std::size_t delta)
        : m_field(field)
        , m_delta(delta)
    {
    }

    std::vector<std::size_t> decode(std::vector<std::size_t> y) const
    {
        std::vector<std::size_t> u = {1};
        std::vector<std::size_t> u_;

        std::vector<std::size_t> b(m_delta, 0);
        b[m_delta - 1] = 1;

        std::vector<std::size_t> a;
        a.reserve(m_delta - 1);
        for (std::size_t i = 1; i < m_delta; ++i) {
            std::size_t syndrome = 0;
            for (std::size_t j = 0; j < y.size(); ++j) {
                syndrome ^= y[j] * m_field.get_mask(i * j);
            }
            a.push_back(syndrome);
        }

        remove_leading_zeros(a);
        if (a.empty()) {
            return y;
        }

        while (2 * a.size() > m_delta) {
            std::vector<std::size_t> q;
            std::vector<std::size_t> r;

            m_field.divide_polynomials(b, a, r, q);
            const auto v = m_field.sum_polynomials(m_field.multiply_polinomials(q, u), u_);

            u_ = u;
            u = v;

            b = a;
            a = r;
        }

        for (std::size_t i = 0; i < m_field.get_max_degree(); ++i) {
            std::size_t u_evaluation_result = 0;
            for (std::size_t j = 0; j < u.size(); ++j) {
                if (u[j] != 0) {
                    u_evaluation_result ^= m_field.get_mask(m_field.get_degree(u[j]) + i * j);
                }
            }
            if (u_evaluation_result == 0) {
                y[m_field.get_degree_inv(i)] ^= 1;
            }
        }

        return y;
    }

private:
    const GF m_field;
    const std::size_t m_delta;
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
        std::size_t iterations_count = 0;
        std::size_t errors_count = 0;
        for (; iterations_count < num_of_iterations && errors_count < max_errors; ++iterations_count) {
            static std::uniform_int_distribution<> uniform_int_distribution(0, 1);

            std::vector<std::size_t> input;
            input.reserve(k);

            for (std::size_t i = 0; i < k; ++i) {
                input.push_back(uniform_int_distribution(engine));
            }

            const auto encoded = m_encoder.encode(input);

            auto noised = encoded;
            std::uniform_real_distribution<> uniform_real_distribution{0.0, 1.0};
            for (auto & bit : noised) {
                if (uniform_real_distribution(engine) < noise_level) {
                    bit ^= 1;
                }
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

        return static_cast<long double>(errors_count) / iterations_count;
    }

private:
    const Encoder & m_encoder;
    const Decoder & m_decoder;
};

int main()
{
    engine.seed(42);
    std::ifstream input_file("input.txt");

    if (!input_file.is_open()) {
        return 1;
    }

    std::size_t n; // 2^m - 1
    std::size_t primitive_poly_mask;
    std::size_t delta;

    input_file >> n >> primitive_poly_mask >> delta;

    GF field(primitive_poly_mask);

    std::vector<std::size_t> mod;
    std::vector<std::size_t> div;

    const auto generating_polynomial = field.get_generating_polynomial(delta);

    std::size_t k = n - generating_polynomial.size() + 1;

    std::ofstream output_file("output.txt");
    output_file << k << '\n';
    print(output_file, generating_polynomial);

    Encoder encoder{generating_polynomial};
    Decoder decoder{field, delta};
    Simulator simulator{encoder, decoder};

    std::string command;
    while (input_file >> command) {
        if (command == "Encode") {
            std::vector<std::size_t> input(k);
            for (auto & value : input) {
                input_file >> value;
            }
            print(output_file, encoder.encode(input));
        }
        else if (command == "Decode") {
            std::vector<std::size_t> input(n);
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
            output_file << simulator.simulate(n, k, noise_level, num_of_iterations, max_errors) << '\n';
        }
        else {
            output_file << "FAILURE\n";
        }
    }

    return 0;
}
