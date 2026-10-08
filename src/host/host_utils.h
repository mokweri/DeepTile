/**
 * @file host_utils.h
 * @brief Legacy fixture I/O, packing, and software-reference helpers.
 * @ingroup deeptile_host
 */

#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <typeinfo>

#include "common_defs.h"

typedef int32_t MACTYPE;

/**
 * @brief Load an exact element count from a binary file.
 *
 * @tparam T The data type of the elements to load.
 * @param filename The path to the binary file to read.
 * @param expected_elements The expected number of elements of type T in the file.
 * @return Loaded elements.
 * @throws std::runtime_error If the file cannot be opened, the size does not match, or reading fails.
 */
template <typename T>
std::vector<T> load_data_from_file(const std::string& filename, size_t expected_elements) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (static_cast<size_t>(size) != expected_elements * sizeof(T)) {
        file.close();
        throw std::runtime_error("File size mismatch for " + filename +
                                 ". Expected " + std::to_string(expected_elements * sizeof(T)) +
                                 " bytes (" + std::to_string(expected_elements) + " elements), got " + std::to_string(size) + " bytes.");
    }

    std::vector<T> buffer(expected_elements);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        file.close();
        throw std::runtime_error("Error reading file: " + filename);
    }
    file.close();
    return buffer;
}

/**
 * @brief Pack int8 elements into kernel words, least-significant element first.
 *
 * @tparam PackedType The type of the packed word (must support .range() assignment).
 * @param in Input vector of int8_t elements to be packed.
 * @param elements_per_word Number of elements to pack into each word.
 * @param bitwidth Number of bits used to represent each element in the packed word.
 * @return Packed kernel words.
 *
 * @note The input size must be a multiple of elements_per_word.
 */
template<typename PackedType>
std::vector<PackedType> pack_data_for_kernel(
    const std::vector<int8_t>& in,
    int elements_per_word,
    int bitwidth
) {
    size_t num_words = in.size() / elements_per_word;
    std::vector<PackedType> out;
    out.reserve(num_words);

    std::cout << "[pack_data_for_kernel] Packing " << in.size() << " elements into "
              << num_words << " words of type '" << typeid(PackedType).name()
              << "' (" << sizeof(PackedType) << " bytes each)" << std::endl;

    for (size_t i = 0; i < num_words; ++i) {
        PackedType word = 0;
        for (int j = 0; j < elements_per_word; ++j) {
            word.range((j + 1) * bitwidth - 1, j * bitwidth) = in[i * elements_per_word + j];
        }
        out.push_back(word);
    }

    std::cout << "[pack_data_for_kernel] Output vector size: " << out.size()
              << " (" << out.size() * sizeof(PackedType) << " bytes)" << std::endl;

    return out;
}

bool read_params_file(const std::string& filename, int& frac_w, int& frac_in, int& frac_b, int& frac_out);

/** Derive output and bias shifts from fixed-point fractional bits. */
inline void compute_shifts(int frac_w, int frac_in, int frac_b, int frac_out,
                           int &shift_out, int &shift_b) {
    shift_out = (frac_in + frac_w) - frac_out;   // Accumulator to output domain.
    shift_b   = (frac_out - frac_b + 1);         // Bias to widened output domain.
}

/**
 * Compute the fixed-point GAP multiplier.
 *
 * The kernel evaluates `(sum * gap_mul) >> GAP_RSHIFT`; GAP_RSHIFT is shared at
 * compile time, and only the signed 16-bit multiplier is passed at runtime.
 */
static constexpr int GAP_RSHIFT   = GAP_SCALE_FRAC_BITS;
static constexpr int GAP_MUL_BITS = 16;
static constexpr int GAP_MUL_MAX  = (1 << (GAP_MUL_BITS - 1)) - 1;

inline int compute_gap_scale(int conv_frac_out, int gap_frac_out,
                             int num_elems) {
    if (num_elems <= 0) {
        throw std::runtime_error(
            "compute_gap_scale: num_elems must be positive, got " +
            std::to_string(num_elems));
    }

    const int frac_delta  = gap_frac_out - conv_frac_out;
    const int total_shift = GAP_RSHIFT + frac_delta;

    if (total_shift < 0) {
        throw std::runtime_error(
            "compute_gap_scale: total_shift is negative (" +
            std::to_string(total_shift) + "); gap_frac_out (" +
            std::to_string(gap_frac_out) + ") is too small relative to "
            "conv_frac_out (" + std::to_string(conv_frac_out) + ")");
    }

    const int64_t numerator = int64_t(1) << total_shift;
    const int gap_mul = static_cast<int>(
        (numerator + num_elems / 2) / num_elems);

    if (gap_mul > GAP_MUL_MAX) {
        throw std::runtime_error(
            "compute_gap_scale: gap_mul (" + std::to_string(gap_mul) +
            ") exceeds " + std::to_string(GAP_MUL_BITS) +
            "-bit signed max (" + std::to_string(GAP_MUL_MAX) +
            "); reduce frac_delta or increase num_elems");
    }

#ifndef NDEBUG
    {
        const int64_t test_sum     = int64_t(num_elems);
        const int64_t test_product = test_sum * gap_mul;
        const int64_t test_result  =
            (test_product + (int64_t(1) << (GAP_RSHIFT - 1))) >> GAP_RSHIFT;
        const int64_t expected     = int64_t(1) << frac_delta;

        assert(std::abs(test_result - expected) <= 1 &&
               "compute_gap_scale: round-trip error exceeds 1 LSB");
    }
#endif

    return gap_mul;
}

/** Normalize a signed output shift for software emulation. */
inline void normalize_shift_out(int shift_out_in, bool &is_right_shift, int &amount) {
    if (shift_out_in > 0) { is_right_shift = true;  amount = shift_out_in - 1; }
    else if (shift_out_in == 0) { is_right_shift = false; amount = 1; }
    else { is_right_shift = false; amount = (-shift_out_in) + 1; }
}

void reorder_ifm(
    const std::vector<int8_t>& ifm_data,
    int8_t* reordered_ifm, 
    int ic, int h, int w
);

std::vector<int8_t> reorder_and_tile_ifm(
    const std::vector<int8_t>& ifm_data,
    int ic,
    int h, int w,
    int tile_size_ic
);

std::vector<int8_t> reorder_weights(
    const std::vector<int8_t>& w_data, 
    int oc, int ic, int kh, int kw
);

std::vector<int8_t> generate_sliding_windows(
    const std::vector<int8_t>& ifm,
    int ifm_h, int ifm_w,
    int k, int s,
    int padding
);

int compute_conv_output_size(
    int ifm_h, int ifm_w, 
    int k, int s, int padding
);

std::vector<acc_t> sw_conv_int(
    const std::vector<int8_t>& ifm_data,
    const std::vector<int8_t>& weights,
    int ifm_h, int ifm_w, int ic, int oc,
    int kernel_size, int stride, int padding
);

/** Reorder OC-major output to the hardware's window-major layout. */
std::vector<acc_t> reorder_ofm_OCxHxW_to_HWxOC(
    const std::vector<acc_t>& ofm_oc_hw,
    int oc, int oh, int ow
);

void Conv2d_FxP(
     const int8_t* input_int8,
    const int8_t* weights_int8,
    const int8_t* bias_int8,
    int8_t* y_int8,
    int N, int C_in, int H_in, int W_in, 
    int C_out, int H_out, int W_out,
    int KERNEL_SIZE, int STRIDE, int PADDING, int DILATION,
    int frac_in, int frac_w, int frac_b, int frac_out,
    bool relu_on
);


void sw_generateWeightStream(
    const std::vector<int8_t> &weights_reordered,
    int  kernel_size, 
    int  oc,
    int  num_windows,
    std::vector<pixel_vector_dt> &out_vec
);

void sw_circularWindowGenerator(
    const std::vector<pixel_vector_dt> &in_vec,
    std::vector<pixel_vector_dt>       &out_vec,
    int  ifm_h,
    int  ifm_w,
    int  kernel_size,
    int  stride
);

void padIFMData(
    const std::vector<int8_t>& ifm_data_reordered,
    std::vector<int8_t>& padded_data,
    int ifm_h, int ifm_w, int ifm_c, int padding
);
