/**
 * @file conv_engine.h
 * @brief Declares the DeepTile dense-convolution engine and packed DSP primitive.
 * @ingroup deeptile_kernel
 */

#ifndef CONV_ENGINE_H_
#define CONV_ENGINE_H_

#include "common_defs.h"

/**
 * @brief Multiplies one signed activation by two packed signed weights.
 *
 * Computes `((w1 << 18) + w0) * a` in one DSP48E2. The caller must correct
 * the high lane once per reduction tree for sign extension from negative low
 * products; summing the packed products directly is not lane-safe.
 *
 * @param a Signed INT8 activation.
 * @param w0 Signed INT8 weight packed into the low product lane.
 * @param w1 Signed INT8 weight packed into the high product lane.
 * @return Exact 35-bit packed product containing both 16-bit products.
 */
static inline ap_int<35> dsp_dual_mul(pixel_t a, weight_t w0, weight_t w1) {
#pragma HLS INLINE
  const ap_int<27> packed_w = (ap_int<27>(w1) << 18) + ap_int<27>(w0);
  ap_int<35> pr = packed_w * a;
#pragma HLS BIND_OP variable = pr op = mul impl = dsp
  return pr;
}

/**
 * @brief Loads weight epochs and executes dense-convolution MAC tiles.
 *
 * Alternates two banked weight buffers so loading an epoch can overlap compute
 * from the previous epoch. Weights are ordered by output-channel bank, layer
 * depth, and input-channel lane. The engine consumes the compute descriptors
 * assigned to each transfer epoch and emits one accumulator group per output
 * spatial group and input-channel group.
 *
 * @param weights Bank-packed weight vectors in device memory.
 * @param desc_xfer Transfer descriptors, exactly `num_xfers` entries.
 * @param pixel_in Input-window streams, one per spatial lane.
 * @param acc_out Output accumulator groups.
 * @param desc_comp Compute descriptors counted by the transfer epochs.
 * @param num_xfers Number of transfer epochs; must be positive.
 * @param kernel_size Convolution kernel width and height.
 */
void conv_engine(const weight_vector_dt *__restrict weights,
                 desc_stream_xfer_t &desc_xfer,
                 ifm_stream_t pixel_in[P_SPATIAL],
                 acc_group_stream_t &acc_out,
                 desc_stream_comp_t &desc_comp,
                 int num_xfers,
                 uint8_t kernel_size);

#endif // CONV_ENGINE_H_
