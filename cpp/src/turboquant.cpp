#include "turboquant/turboquant.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llama_lab {

namespace {

constexpr double kPi = 3.14159265358979323846;

static float dot_product(const std::vector<float> & a, const std::vector<float> & b) {
    float acc = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

static float l2_norm(const std::vector<float> & x) {
    return std::sqrt(std::max(0.0f, dot_product(x, x)));
}

static std::vector<float> normalize_vector(const std::vector<float> & x, float & norm_out) {
    std::vector<float> y = x;
    norm_out = l2_norm(y);
    if (norm_out <= 0.0f) {
        norm_out = 1.0f;
        return y;
    }
    for (float & v : y) {
        v /= norm_out;
    }
    return y;
}

static std::vector<float> matvec_rows(const std::vector<float> & m, int rows, int cols, const std::vector<float> & x) {
    std::vector<float> out(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        float acc = 0.0f;
        const int base = r * cols;
        for (int c = 0; c < cols; ++c) {
            acc += m[base + c] * x[c];
        }
        out[r] = acc;
    }
    return out;
}

static std::vector<float> matvec_cols(const std::vector<float> & m, int rows, int cols, const std::vector<float> & x) {
    std::vector<float> out(cols, 0.0f);
    for (int r = 0; r < rows; ++r) {
        const float xr = x[r];
        const int base = r * cols;
        for (int c = 0; c < cols; ++c) {
            out[c] += xr * m[base + c];
        }
    }
    return out;
}

static std::vector<float> build_rotation_matrix(int dim, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    std::vector<float> q(dim * dim, 0.0f);

    for (int r = 0; r < dim; ++r) {
        for (int c = 0; c < dim; ++c) {
            q[r * dim + c] = normal(rng);
        }
    }

    // Modified Gram-Schmidt over rows.
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < i; ++j) {
            float proj = 0.0f;
            for (int c = 0; c < dim; ++c) {
                proj += q[i * dim + c] * q[j * dim + c];
            }
            for (int c = 0; c < dim; ++c) {
                q[i * dim + c] -= proj * q[j * dim + c];
            }
        }
        float nrm = 0.0f;
        for (int c = 0; c < dim; ++c) {
            nrm += q[i * dim + c] * q[i * dim + c];
        }
        nrm = std::sqrt(std::max(nrm, 1e-12f));
        for (int c = 0; c < dim; ++c) {
            q[i * dim + c] /= nrm;
        }
    }

    return q;
}

static std::vector<float> build_gaussian_matrix(int rows, int cols, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> m(rows * cols);
    for (float & v : m) {
        v = normal(rng);
    }
    return m;
}

static std::vector<float> solve_lloyd_max_centroids(int dim, int bits, int iterations, int grid_size) {
    if (bits < 1 || bits > 15) {
        throw std::invalid_argument("TurboQuant bits must be between 1 and 15");
    }
    const int n_levels = 1 << bits;
    const double sigma = 1.0 / std::sqrt(std::max(dim, 1));
    const double lo = -6.0 * sigma;
    const double hi = 6.0 * sigma;
    const double step = (hi - lo) / std::max(grid_size - 1, 1);

    std::vector<double> xs(grid_size);
    std::vector<double> pdf(grid_size);
    const double coeff = 1.0 / (std::sqrt(2.0 * kPi) * sigma);
    for (int i = 0; i < grid_size; ++i) {
        const double x = lo + step * i;
        xs[i] = x;
        pdf[i] = coeff * std::exp(-(x * x) / (2.0 * sigma * sigma));
    }

    std::vector<double> centroids(n_levels);
    for (int i = 0; i < n_levels; ++i) {
        centroids[i] = lo + (hi - lo) * (double(i) + 0.5) / double(n_levels);
    }

    std::vector<double> boundaries(std::max(0, n_levels - 1));
    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 0; i + 1 < n_levels; ++i) {
            boundaries[i] = 0.5 * (centroids[i] + centroids[i + 1]);
        }

        std::vector<double> next = centroids;
        for (int level = 0; level < n_levels; ++level) {
            const double a = (level == 0) ? lo : boundaries[level - 1];
            const double b = (level + 1 == n_levels) ? hi : boundaries[level];
            double num = 0.0;
            double den = 0.0;
            for (int i = 0; i < grid_size; ++i) {
                const double x = xs[i];
                if (x < a || x > b) {
                    continue;
                }
                const double w = pdf[i];
                num += x * w;
                den += w;
            }
            if (den > 1e-12) {
                next[level] = num / den;
            }
        }

        double max_delta = 0.0;
        for (int i = 0; i < n_levels; ++i) {
            max_delta = std::max(max_delta, std::abs(next[i] - centroids[i]));
        }
        centroids.swap(next);
        if (max_delta < 1e-10) {
            break;
        }
    }

    std::vector<float> out(n_levels);
    for (int i = 0; i < n_levels; ++i) {
        out[i] = static_cast<float>(centroids[i]);
    }
    return out;
}

static std::vector<float> make_boundaries(const std::vector<float> & centroids) {
    std::vector<float> boundaries;
    boundaries.reserve(centroids.size() > 1 ? centroids.size() - 1 : 0);
    for (size_t i = 0; i + 1 < centroids.size(); ++i) {
        boundaries.push_back(0.5f * (centroids[i] + centroids[i + 1]));
    }
    return boundaries;
}

static uint16_t quantize_coordinate(float value, const std::vector<float> & boundaries) {
    return static_cast<uint16_t>(std::upper_bound(boundaries.begin(), boundaries.end(), value) - boundaries.begin());
}

static turboquant_metrics finalize_ip_metrics(
        double mse_sum,
        double mse_max,
        size_t mse_count,
        double bias_sum,
        double abs_sum,
        double sq_sum,
        size_t ip_count) {
    turboquant_metrics metrics;
    if (mse_count > 0) {
        metrics.mse_mean = mse_sum / double(mse_count);
        metrics.mse_max = mse_max;
    }
    if (ip_count == 0) {
        return metrics;
    }
    metrics.ip_bias_mean = bias_sum / double(ip_count);
    metrics.ip_rmse = std::sqrt(sq_sum / double(ip_count));
    metrics.ip_mae = abs_sum / double(ip_count);
    return metrics;
}

static turboquant_init_params make_prod_params(const turboquant_init_params & params) {
    turboquant_init_params out = params;
    out.qjl_dim = out.qjl_dim > 0 ? out.qjl_dim : out.dim;
    return out;
}

static turboquant_init_params make_prod_mse_params(const turboquant_init_params & params) {
    turboquant_init_params out = make_prod_params(params);
    out.bits = std::max(1, params.bits - 1);
    out.qjl_dim = 0;
    return out;
}

static std::vector<uint16_t> sanitize_forced_indices(
        const std::vector<uint16_t> & forced_outlier_indices,
        int dim,
        int n_outliers) {
    std::vector<uint16_t> indices;
    indices.reserve(std::min<int>(n_outliers, forced_outlier_indices.size()));
    std::vector<uint8_t> seen((size_t) dim, 0);
    for (uint16_t idx : forced_outlier_indices) {
        if ((int) idx >= dim || seen[(size_t) idx]) {
            continue;
        }
        indices.push_back(idx);
        seen[(size_t) idx] = 1;
        if ((int) indices.size() >= n_outliers) {
            break;
        }
    }
    std::sort(indices.begin(), indices.end());
    return indices;
}

} // namespace

turboquant_mse::turboquant_mse(const turboquant_init_params & params) : params_(params) {
    if (params_.dim <= 0) {
        throw std::invalid_argument("TurboQuant dimension must be positive");
    }
    if (params_.bits <= 0) {
        throw std::invalid_argument("TurboQuant bits must be positive");
    }
    rotation_ = build_rotation_matrix(params_.dim, params_.seed);
    centroids_ = solve_lloyd_max_centroids(params_.dim, params_.bits, params_.codebook_iters, params_.codebook_grid);
    boundaries_ = make_boundaries(centroids_);
}

int turboquant_mse::dim() const {
    return params_.dim;
}

int turboquant_mse::bits() const {
    return params_.bits;
}

std::vector<float> turboquant_mse::rotate_unit(const std::vector<float> & x) const {
    return matvec_rows(rotation_, params_.dim, params_.dim, x);
}

std::vector<float> turboquant_mse::unrotate_unit(const std::vector<float> & y) const {
    return matvec_cols(rotation_, params_.dim, params_.dim, y);
}

turboquant_mse_compressed turboquant_mse::quantize(const std::vector<float> & x) const {
    if ((int) x.size() != params_.dim) {
        throw std::invalid_argument("TurboQuant input dimension mismatch");
    }
    turboquant_mse_compressed out;
    std::vector<float> x_unit = normalize_vector(x, out.vec_norm);
    std::vector<float> rotated = rotate_unit(x_unit);

    out.indices.resize(params_.dim);
    for (int i = 0; i < params_.dim; ++i) {
        out.indices[i] = quantize_coordinate(rotated[i], boundaries_);
    }
    return out;
}

std::vector<float> turboquant_mse::dequantize(const turboquant_mse_compressed & compressed) const {
    if ((int) compressed.indices.size() != params_.dim) {
        throw std::invalid_argument("TurboQuant compressed dimension mismatch");
    }
    std::vector<float> rotated(params_.dim, 0.0f);
    for (int i = 0; i < params_.dim; ++i) {
        rotated[i] = centroids_.at(compressed.indices[i]);
    }
    std::vector<float> x = unrotate_unit(rotated);
    for (float & v : x) {
        v *= compressed.vec_norm;
    }
    return x;
}

const std::vector<float> & turboquant_mse::centroids() const {
    return centroids_;
}

const std::vector<float> & turboquant_mse::boundaries() const {
    return boundaries_;
}

turboquant_prod::turboquant_prod(const turboquant_init_params & params)
    : params_(make_prod_params(params)),
      mse_(make_prod_mse_params(params)) {
    qjl_matrix_ = build_gaussian_matrix(params_.qjl_dim, params_.dim, params_.seed + 1);
}

int turboquant_prod::dim() const {
    return params_.dim;
}

int turboquant_prod::bits() const {
    return params_.bits;
}

int turboquant_prod::qjl_dim() const {
    return params_.qjl_dim;
}

int turboquant_prod::outlier_channels() const {
    return std::max(0, params_.outlier_channels);
}

turboquant_prod_compressed turboquant_prod::quantize(const std::vector<float> & x) const {
    return quantize(x, {});
}

turboquant_prod_compressed turboquant_prod::quantize(
        const std::vector<float> & x,
        const std::vector<uint16_t> & forced_outlier_indices) const {
    turboquant_prod_compressed out;
    out.mse = mse_.quantize(x);
    const std::vector<float> x_mse = mse_.dequantize(out.mse);
    std::vector<float> residual(params_.dim, 0.0f);
    for (int i = 0; i < params_.dim; ++i) {
        residual[i] = x[i] - x_mse[i];
    }

    const int n_outliers = std::min(params_.dim, std::max(0, params_.outlier_channels));
    if (n_outliers > 0) {
        std::vector<uint16_t> selected = sanitize_forced_indices(forced_outlier_indices, params_.dim, n_outliers);
        if ((int) selected.size() < n_outliers) {
            std::vector<std::pair<float, int>> ranked;
            ranked.reserve(params_.dim);
            for (int i = 0; i < params_.dim; ++i) {
                ranked.emplace_back(std::fabs(residual[i]), i);
            }
            std::partial_sort(ranked.begin(), ranked.begin() + n_outliers, ranked.end(),
                    [](const auto & a, const auto & b) {
                        if (a.first != b.first) {
                            return a.first > b.first;
                        }
                        return a.second < b.second;
                    });
            selected.resize((size_t) n_outliers);
            for (int i = 0; i < n_outliers; ++i) {
                selected[(size_t) i] = static_cast<uint16_t>(ranked[(size_t) i].second);
            }
            std::sort(selected.begin(), selected.end());
        }
        out.outlier_indices.resize(n_outliers);
        out.outlier_values.resize(n_outliers);
        for (int i = 0; i < n_outliers; ++i) {
            const int idx = selected[(size_t) i];
            out.outlier_indices[i] = static_cast<uint16_t>(idx);
            out.outlier_values[i] = residual[idx];
            residual[idx] = 0.0f;
        }
    }

    out.residual_norm = l2_norm(residual);
    std::vector<float> projected = matvec_rows(qjl_matrix_, params_.qjl_dim, params_.dim, residual);
    out.qjl_signs.resize(params_.qjl_dim);
    for (int i = 0; i < params_.qjl_dim; ++i) {
        out.qjl_signs[i] = projected[i] >= 0.0f ? int8_t(1) : int8_t(-1);
    }
    return out;
}

std::vector<float> turboquant_prod::dequantize(const turboquant_prod_compressed & compressed) const {
    return mse_.dequantize(compressed.mse);
}

double turboquant_prod::inner_product(const std::vector<float> & q, const turboquant_prod_compressed & compressed) const {
    if ((int) q.size() != params_.dim) {
        throw std::invalid_argument("TurboQuant query dimension mismatch");
    }
    const std::vector<float> x_mse = mse_.dequantize(compressed.mse);
    const double term1 = double(dot_product(q, x_mse));
    double outlier_term = 0.0;
    const size_t n_outliers = std::min(compressed.outlier_indices.size(), compressed.outlier_values.size());
    for (size_t i = 0; i < n_outliers; ++i) {
        const uint16_t idx = compressed.outlier_indices[i];
        if (idx < q.size()) {
            outlier_term += double(q[idx]) * double(compressed.outlier_values[i]);
        }
    }
    const std::vector<float> projected = matvec_rows(qjl_matrix_, params_.qjl_dim, params_.dim, q);
    double qjl_ip = 0.0;
    for (int i = 0; i < params_.qjl_dim; ++i) {
        qjl_ip += double(projected[i]) * double(compressed.qjl_signs[i]);
    }
    const double correction_scale = std::sqrt(kPi / 2.0) / double(params_.qjl_dim);
    const double term2 = double(compressed.residual_norm) * correction_scale * qjl_ip;
    return term1 + outlier_term + term2;
}

const turboquant_mse & turboquant_prod::mse_quantizer() const {
    return mse_;
}

const std::vector<float> & turboquant_prod::qjl_matrix() const {
    return qjl_matrix_;
}

std::vector<float> turboquant_make_unit_vector(int dim, uint64_t seed, uint64_t sample_idx) {
    std::mt19937_64 rng(seed + 0x9e3779b97f4a7c15ULL * (sample_idx + 1));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x(dim);
    for (float & v : x) {
        v = normal(rng);
    }
    float norm = 1.0f;
    return normalize_vector(x, norm);
}

turboquant_metrics turboquant_score_mse(
        const turboquant_mse & quantizer,
        const std::vector<std::vector<float>> & dataset) {
    double mse_sum = 0.0;
    double mse_max = 0.0;
    for (const auto & x : dataset) {
        const auto compressed = quantizer.quantize(x);
        const auto x_hat = quantizer.dequantize(compressed);
        double err = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            const double d = double(x[i]) - double(x_hat[i]);
            err += d * d;
        }
        err /= double(std::max<size_t>(1, x.size()));
        mse_sum += err;
        mse_max = std::max(mse_max, err);
    }
    turboquant_metrics metrics;
    if (!dataset.empty()) {
        metrics.mse_mean = mse_sum / double(dataset.size());
        metrics.mse_max = mse_max;
    }
    return metrics;
}

turboquant_metrics turboquant_score_mse_inner_products(
        const turboquant_mse & quantizer,
        const std::vector<std::vector<float>> & dataset,
        const std::vector<std::vector<float>> & queries) {
    std::vector<std::vector<float>> reconstructed;
    reconstructed.reserve(dataset.size());

    double mse_sum = 0.0;
    double mse_max = 0.0;
    for (const auto & x : dataset) {
        const auto compressed = quantizer.quantize(x);
        reconstructed.push_back(quantizer.dequantize(compressed));
        double err = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            const double d = double(x[i]) - double(reconstructed.back()[i]);
            err += d * d;
        }
        err /= double(std::max<size_t>(1, x.size()));
        mse_sum += err;
        mse_max = std::max(mse_max, err);
    }

    double bias_sum = 0.0;
    double abs_sum = 0.0;
    double sq_sum = 0.0;
    size_t ip_count = 0;
    for (const auto & q : queries) {
        for (size_t i = 0; i < dataset.size(); ++i) {
            const double truth = double(dot_product(q, dataset[i]));
            const double estimate = double(dot_product(q, reconstructed[i]));
            const double delta = estimate - truth;
            bias_sum += delta;
            abs_sum += std::abs(delta);
            sq_sum += delta * delta;
            ip_count++;
        }
    }

    return finalize_ip_metrics(mse_sum, mse_max, dataset.size(), bias_sum, abs_sum, sq_sum, ip_count);
}

turboquant_metrics turboquant_score_prod(
        const turboquant_prod & quantizer,
        const std::vector<std::vector<float>> & dataset,
        const std::vector<std::vector<float>> & queries) {
    std::vector<turboquant_prod_compressed> compressed;
    compressed.reserve(dataset.size());
    for (const auto & x : dataset) {
        compressed.push_back(quantizer.quantize(x));
    }

    double mse_sum = 0.0;
    double mse_max = 0.0;
    double bias_sum = 0.0;
    double abs_sum = 0.0;
    double sq_sum = 0.0;
    size_t count = 0;

    for (size_t i = 0; i < dataset.size(); ++i) {
        const auto x_hat = quantizer.dequantize(compressed[i]);
        double err = 0.0;
        for (size_t j = 0; j < dataset[i].size(); ++j) {
            const double d = double(dataset[i][j]) - double(x_hat[j]);
            err += d * d;
        }
        err /= double(std::max<size_t>(1, dataset[i].size()));
        mse_sum += err;
        mse_max = std::max(mse_max, err);
    }

    for (const auto & q : queries) {
        for (size_t i = 0; i < dataset.size(); ++i) {
            const double truth = double(dot_product(q, dataset[i]));
            const double estimate = quantizer.inner_product(q, compressed[i]);
            const double delta = estimate - truth;
            bias_sum += delta;
            abs_sum += std::abs(delta);
            sq_sum += delta * delta;
            count++;
        }
    }

    return finalize_ip_metrics(mse_sum, mse_max, dataset.size(), bias_sum, abs_sum, sq_sum, count);
}

std::string turboquant_metrics_json(const turboquant_metrics & metrics) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(8)
        << "{"
        << "\"mse_mean\":" << metrics.mse_mean << ","
        << "\"mse_max\":" << metrics.mse_max << ","
        << "\"ip_bias_mean\":" << metrics.ip_bias_mean << ","
        << "\"ip_rmse\":" << metrics.ip_rmse << ","
        << "\"ip_mae\":" << metrics.ip_mae
        << "}";
    return out.str();
}

turboquant_recipe_config turboquant_recipe_for_name(const std::string & name, int dim) {
    turboquant_recipe_config recipe;
    recipe.bits = 4;
    recipe.qjl_dim = std::max(dim, dim * 2);
    recipe.outlier_channels = std::max(0, dim / 32);

    std::string normalized = name;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (normalized.empty() || normalized == "manual") {
        recipe.name = "manual";
        return recipe;
    }
    if (normalized == "turboquant25" || normalized == "tq25" || normalized == "25") {
        recipe.name = "turboquant25";
        recipe.bits = 3;
        recipe.qjl_dim = std::max(dim, dim * 2);
        recipe.outlier_channels = std::max(0, dim / 4);
        return recipe;
    }
    if (normalized == "turboquant35" || normalized == "tq35" || normalized == "35") {
        recipe.name = "turboquant35";
        recipe.bits = 4;
        recipe.qjl_dim = std::max(dim, dim * 4);
        recipe.outlier_channels = std::max(0, dim / 2);
        return recipe;
    }
    if (normalized == "best" || normalized == "lab_best" || normalized == "speed") {
        recipe.name = "lab_best";
        recipe.bits = 4;
        recipe.qjl_dim = std::max(dim, dim * 4);
        recipe.outlier_channels = std::max(4, dim / 32);
        return recipe;
    }
    if (normalized == "lab_context" || normalized == "context" || normalized == "aggressive") {
        recipe.name = "lab_context";
        recipe.bits = 4;
        recipe.qjl_dim = std::max(dim, dim * 2);
        recipe.outlier_channels = std::max(2, dim / 64);
        return recipe;
    }
    if (normalized == "lab_context_fast" || normalized == "context_fast" || normalized == "aggressive_fast") {
        recipe.name = "lab_context_fast";
        recipe.bits = 4;
        recipe.qjl_dim = std::max(dim / 2, dim);
        recipe.outlier_channels = std::max(1, dim / 128);
        return recipe;
    }
    if (normalized == "lab_context_ultra" || normalized == "context_ultra" || normalized == "aggressive_ultra") {
        recipe.name = "lab_context_ultra";
        recipe.bits = 4;
        recipe.qjl_dim = std::max(64, dim / 2);
        recipe.outlier_channels = 1;
        return recipe;
    }
    if (normalized == "memory_max" || normalized == "memmax" || normalized == "max_memory" || normalized == "memory-first") {
        recipe.name = "memory_max";
        recipe.bits = 3;
        recipe.qjl_dim = std::max(64, dim / 2);
        recipe.outlier_channels = 0;
        return recipe;
    }

    recipe.name = normalized;
    return recipe;
}

} // namespace llama_lab
