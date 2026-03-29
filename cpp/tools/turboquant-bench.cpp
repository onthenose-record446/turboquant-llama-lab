#include "turboquant/turboquant.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace llama_lab;

namespace {

struct bench_args {
    int dim = 128;
    int bits = 4;
    int qjl_dim = 0;
    int samples = 256;
    int queries = 64;
    int seed = 42;
    int codebook_iters = 96;
    int codebook_grid = 4096;
};

void print_usage() {
    std::cout
        << "usage: llama-turboquant-bench [options]\n"
        << "  --dim N              vector dimension (default: 128)\n"
        << "  --bits N             total TurboQuant bit budget (default: 4)\n"
        << "  --qjl-dim N          QJL projection dimension (default: dim)\n"
        << "  --samples N          dataset vectors (default: 256)\n"
        << "  --queries N          query vectors (default: 64)\n"
        << "  --seed N             RNG seed (default: 42)\n"
        << "  --codebook-iters N   Lloyd-Max iterations (default: 96)\n"
        << "  --codebook-grid N    Gaussian grid points (default: 4096)\n";
}

bool parse_int_arg(const char * value, int & out) {
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

bench_args parse_args(int argc, char ** argv) {
    bench_args args;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto need_value = [&](int & slot) {
            if (i + 1 >= argc || !parse_int_arg(argv[++i], slot)) {
                throw std::runtime_error("invalid value for " + flag);
            }
        };

        if (flag == "-h" || flag == "--help") {
            print_usage();
            std::exit(0);
        } else if (flag == "--dim") {
            need_value(args.dim);
        } else if (flag == "--bits") {
            need_value(args.bits);
        } else if (flag == "--qjl-dim") {
            need_value(args.qjl_dim);
        } else if (flag == "--samples") {
            need_value(args.samples);
        } else if (flag == "--queries") {
            need_value(args.queries);
        } else if (flag == "--seed") {
            need_value(args.seed);
        } else if (flag == "--codebook-iters") {
            need_value(args.codebook_iters);
        } else if (flag == "--codebook-grid") {
            need_value(args.codebook_grid);
        } else {
            throw std::runtime_error("unknown argument: " + flag);
        }
    }

    if (args.dim <= 0 || args.bits <= 1 || args.samples <= 0 || args.queries <= 0) {
        throw std::runtime_error("invalid benchmark parameters");
    }
    return args;
}

std::vector<std::vector<float>> make_vectors(int count, int dim, uint64_t seed) {
    std::vector<std::vector<float>> out;
    out.reserve(count);
    for (int i = 0; i < count; ++i) {
        out.push_back(turboquant_make_unit_vector(dim, seed, static_cast<uint64_t>(i)));
    }
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const bench_args args = parse_args(argc, argv);

        turboquant_init_params params;
        params.dim = args.dim;
        params.bits = args.bits;
        params.qjl_dim = args.qjl_dim;
        params.seed = static_cast<uint64_t>(args.seed);
        params.codebook_iters = args.codebook_iters;
        params.codebook_grid = args.codebook_grid;

        const auto dataset = make_vectors(args.samples, args.dim, params.seed + 1);
        const auto queries = make_vectors(args.queries, args.dim, params.seed + 2);

        turboquant_mse mse_quant(params);
        turboquant_init_params prod_mse_params = params;
        prod_mse_params.bits = std::max(1, args.bits - 1);
        turboquant_mse prod_mse_quant(prod_mse_params);
        turboquant_prod prod_quant(params);

        const auto mse_metrics = turboquant_score_mse(mse_quant, dataset);
        const auto prod_mse_metrics = turboquant_score_mse(prod_mse_quant, dataset);
        const auto prod_reference_ip_metrics = turboquant_score_mse_inner_products(prod_mse_quant, dataset, queries);
        const auto prod_metrics = turboquant_score_prod(prod_quant, dataset, queries);

        std::cout
            << "{"
            << "\"dim\":" << args.dim << ","
            << "\"bits\":" << args.bits << ","
            << "\"prod_mse_bits\":" << std::max(1, args.bits - 1) << ","
            << "\"qjl_dim\":" << (args.qjl_dim > 0 ? args.qjl_dim : args.dim) << ","
            << "\"samples\":" << args.samples << ","
            << "\"queries\":" << args.queries << ","
            << "\"mse_stage\":" << turboquant_metrics_json(mse_metrics) << ","
            << "\"prod_reference_mse_stage\":" << turboquant_metrics_json(prod_mse_metrics) << ","
            << "\"prod_reference_ip_stage\":" << turboquant_metrics_json(prod_reference_ip_metrics) << ","
            << "\"prod_stage\":" << turboquant_metrics_json(prod_metrics)
            << "}\n";

        return 0;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "llama-turboquant-bench: %s\n", exc.what());
        return 1;
    }
}
