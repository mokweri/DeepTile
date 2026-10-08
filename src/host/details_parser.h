/**
 * @file details_parser.h
 * @brief Parser for legacy single-layer details.txt fixtures.
 * @ingroup deeptile_host
 */

#ifndef DETAILS_PARSER_H
#define DETAILS_PARSER_H

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <iostream>

static inline std::string ltrim(const std::string& s) {
    size_t i = 0; while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i; return s.substr(i);
}
static inline std::string rtrim(const std::string& s) {
    if (s.empty()) return s;
    size_t i = s.size();
    while (i > 0 && std::isspace(static_cast<unsigned char>(s[i-1]))) { --i; }
    return s.substr(0, i);
}
static inline std::string trim(const std::string& s) { return rtrim(ltrim(s)); }

static inline bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}

static inline std::vector<int> parse_csv_ints(const std::string& v) {
    std::vector<int> out; std::stringstream ss(v); std::string item;
    while (std::getline(ss, item, ',')) { item = trim(item); if (!item.empty()) out.push_back(std::stoi(item)); }
    return out;
}

/** Parsed single-layer fixture metadata. */
struct DetailsConfig {
    std::string op_type = "Conv2d";

    // Shapes and layouts.
    std::string activation_layout = "CHW";
    std::string weights_layout = "OIHW";
    int C_in = 0, H_in = 0, W_in = 0;
    int C_out = 0, kH = 0, kW = 0;
    int stride_h = 1, stride_w = 1;
    int pad_h = 0, pad_w = 0;
    int dilation = 1;

    // Fixed-point metadata.
    int frac_in = 0, frac_w = 0, frac_b = 0, frac_out = 0;
    int gap_frac_out = 0;

    // Data properties.
    int bitwidth = 8;
    int is_signed = 1;
    int has_bias = 1;

    // Files relative to the fixture directory.
    std::string input_file = "input.data";
    std::string weights_file = "weights.data";
    std::string bias_file = "bias.data";

    // Optional behavior.
    bool relu_on = false;
    bool gap_enable = false;
};

/** Parse and validate details.txt in a fixture directory. */
static inline DetailsConfig parse_details_file(const std::string& folder) {
    const std::string path = folder + "/details.txt";
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Cannot open details.txt: " + path);

    DetailsConfig cfg;
    bool gap_frac_out_seen = false;
    std::string line;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty() || starts_with(line, "#")) continue;
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        std::string key = trim(line.substr(0, pos));
        std::string val = trim(line.substr(pos + 1));

        if (key == "op_type") cfg.op_type = val;
        else if (key == "activation_layout") cfg.activation_layout = val;
        else if (key == "weights_layout") cfg.weights_layout = val;
        else if (key == "activation_dims") {
            auto v = parse_csv_ints(val);
            if (v.size() == 3) {
                cfg.C_in = v[0];
                cfg.H_in = v[1];
                cfg.W_in = v[2];
            } else if (v.size() == 2) {
                // Linear [spatial, channels] metadata normalizes to CHW [C,1,1].
                if (v[0] != 1) {
                    throw std::runtime_error("Linear activation_dims must be 1,C");
                }
                cfg.C_in = v[1];
                cfg.H_in = 1;
                cfg.W_in = 1;
            }
        } else if (key == "weights_dims") {
            auto v = parse_csv_ints(val);
            if (v.size() == 4) {
                cfg.C_out = v[0];
                cfg.kH = v[2];
                cfg.kW = v[3];
            } else if (v.size() == 2) {
                cfg.C_out = v[0];
                cfg.kH = 1;
                cfg.kW = 1;
                if (cfg.C_in > 0 && v[1] != cfg.C_in) {
                    throw std::runtime_error("Linear weights_dims input channels do not match activation_dims");
                }
            }
        } else if (key == "stride") {
            auto v = parse_csv_ints(val); if (v.size() == 2) { cfg.stride_h = v[0]; cfg.stride_w = v[1]; }
        } else if (key == "padding") {
            auto v = parse_csv_ints(val); if (v.size() == 2) { cfg.pad_h = v[0]; cfg.pad_w = v[1]; }
        } else if (key == "kernel") {
            auto v = parse_csv_ints(val); if (v.size() == 2) { cfg.kH = v[0]; cfg.kW = v[1]; }
        } else if (key == "bitwidth") cfg.bitwidth = std::stoi(val);
        else if (key == "signed") cfg.is_signed = std::stoi(val);
        else if (key == "frac_in") cfg.frac_in = std::stoi(val);
        else if (key == "frac_w") cfg.frac_w = std::stoi(val);
        else if (key == "frac_b") cfg.frac_b = std::stoi(val);
        else if (key == "frac_out") cfg.frac_out = std::stoi(val);
        else if (key == "gap_frac_out") {
            cfg.gap_frac_out = std::stoi(val);
            gap_frac_out_seen = true;
        }
        else if (key == "has_bias") cfg.has_bias = std::stoi(val);
        else if (key == "input_file") cfg.input_file = val;
        else if (key == "weights_file") cfg.weights_file = val;
        else if (key == "bias_file") cfg.bias_file = val;
        else if (key == "gap_enable") cfg.gap_enable = (std::stoi(val) != 0);
        else if (key == "relu") {
            if (val == "1" || val == "true" || val == "True") cfg.relu_on = true; else cfg.relu_on = false;
        }
    }

    if (!gap_frac_out_seen) {
        cfg.gap_frac_out = cfg.frac_out;
    }

    // The kernel accepts symmetric stride, padding, and square kernels.
    if (cfg.activation_layout != "CHW") throw std::runtime_error("Unsupported activation_layout: " + cfg.activation_layout);
    if (cfg.op_type == "Linear") {
        if (cfg.weights_layout != "OI") throw std::runtime_error("Unsupported Linear weights_layout: " + cfg.weights_layout);
    } else if (cfg.weights_layout != "OIHW") {
        throw std::runtime_error("Unsupported weights_layout: " + cfg.weights_layout);
    }
    if (cfg.C_in <= 0 || cfg.H_in <= 0 || cfg.W_in <= 0) throw std::runtime_error("Invalid activation_dims in details.txt");
    if (cfg.C_out <= 0 || cfg.kH <= 0 || cfg.kW <= 0) throw std::runtime_error("Invalid weights_dims/kernel in details.txt");
    if (cfg.stride_h != cfg.stride_w) throw std::runtime_error("Non-symmetric stride not supported by current API");
    if (cfg.pad_h != cfg.pad_w) throw std::runtime_error("Non-symmetric padding not supported by current API");
    if (cfg.kH != cfg.kW) throw std::runtime_error("Non-square kernels not supported by current API");
    if (cfg.bitwidth != 8 || cfg.is_signed != 1) throw std::runtime_error("Only signed int8 is supported currently");

    return cfg;
}

static inline bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary); return f.good();
}

#endif // DETAILS_PARSER_H
