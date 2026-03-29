#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llama_lab {

struct turboquant_recipe_config {
    std::string name = "manual";
    int bits = 4;
    int qjl_dim = 0;
    int outlier_channels = 0;
};

struct turboquant_init_params {
    int dim = 128;
    int bits = 4;
    int qjl_dim = 0;
    int outlier_channels = 0;
    uint64_t seed = 42;
    int codebook_iters = 96;
    int codebook_grid = 4096;
};

struct turboquant_mse_compressed {
    std::vector<uint16_t> indices;
    float vec_norm = 1.0f;
};

struct turboquant_prod_compressed {
    turboquant_mse_compressed mse;
    std::vector<int8_t> qjl_signs;
    float residual_norm = 0.0f;
    std::vector<uint16_t> outlier_indices;
    std::vector<float> outlier_values;
};

struct turboquant_metrics {
    double mse_mean = 0.0;
    double mse_max = 0.0;
    double ip_bias_mean = 0.0;
    double ip_rmse = 0.0;
    double ip_mae = 0.0;
};

class turboquant_mse {
public:
    explicit turboquant_mse(const turboquant_init_params & params);

    int dim() const;
    int bits() const;

    turboquant_mse_compressed quantize(const std::vector<float> & x) const;
    std::vector<float> dequantize(const turboquant_mse_compressed & compressed) const;

    const std::vector<float> & centroids() const;
    const std::vector<float> & boundaries() const;

private:
    turboquant_init_params params_;
    std::vector<float> rotation_;
    std::vector<float> centroids_;
    std::vector<float> boundaries_;

    std::vector<float> rotate_unit(const std::vector<float> & x) const;
    std::vector<float> unrotate_unit(const std::vector<float> & y) const;
};

class turboquant_prod {
public:
    explicit turboquant_prod(const turboquant_init_params & params);

    int dim() const;
    int bits() const;
    int qjl_dim() const;
    int outlier_channels() const;

    turboquant_prod_compressed quantize(const std::vector<float> & x) const;
    turboquant_prod_compressed quantize(
            const std::vector<float> & x,
            const std::vector<uint16_t> & forced_outlier_indices) const;
    std::vector<float> dequantize(const turboquant_prod_compressed & compressed) const;
    double inner_product(const std::vector<float> & q, const turboquant_prod_compressed & compressed) const;
    const turboquant_mse & mse_quantizer() const;
    const std::vector<float> & qjl_matrix() const;

private:
    turboquant_init_params params_;
    turboquant_mse mse_;
    std::vector<float> qjl_matrix_;
};

std::vector<float> turboquant_make_unit_vector(int dim, uint64_t seed, uint64_t sample_idx);
turboquant_metrics turboquant_score_mse(
        const turboquant_mse & quantizer,
        const std::vector<std::vector<float>> & dataset);
turboquant_metrics turboquant_score_mse_inner_products(
        const turboquant_mse & quantizer,
        const std::vector<std::vector<float>> & dataset,
        const std::vector<std::vector<float>> & queries);
turboquant_metrics turboquant_score_prod(
        const turboquant_prod & quantizer,
        const std::vector<std::vector<float>> & dataset,
        const std::vector<std::vector<float>> & queries);
std::string turboquant_metrics_json(const turboquant_metrics & metrics);
turboquant_recipe_config turboquant_recipe_for_name(const std::string & name, int dim);

} // namespace llama_lab
