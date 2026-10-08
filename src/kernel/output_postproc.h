/**
 * @file output_postproc.h
 * @brief Declares accumulation, fused post-processing, and output writeback.
 * @ingroup deeptile_kernel
 */

#ifndef ACC_STORE_H_
#define ACC_STORE_H_

#include "common_defs.h"

/**
 * @brief Accumulates partial sums and emits requantized output tiles.
 *
 * `ACCUM_RESET` replaces the stored partial sum, intermediate descriptors add
 * to it, and `ACCUM_WRITE_BACK` applies bias, round-half-up quantization,
 * saturation, and optional ReLU. The rounding increment is applied only in the
 * second quantization shift.
 *
 * @param acc_in Accumulator groups from the convolution engine.
 * @param bias_cache On-chip packed bias vectors.
 * @param ofm_out Requantized output-vector stream.
 * @param desc_comp Compute descriptors, one per scheduled tile.
 * @param num_tiles Number of descriptors consumed.
 * @param shift_out Layer-wide output requantization shift.
 * @param act_max Upper clamp applied with ReLU: 127 for ReLU, lower for ReLU6.
 */
void acc_quantize(acc_group_stream_t &acc_in,
                  const biases_vector_dt bias_cache[BIAS_BUF_DEPTH],
                  ofm_stream_t &ofm_out,
                  desc_stream_comp_t &desc_comp,
                  int num_tiles,
                  int shift_out,
                  int act_max);

/**
 * @brief Overlaps partial-sum accumulation and output emission.
 *
 * Alternates two partial-sum buffers. Each writeback unit contains the
 * descriptors from reset through writeback and uses the same quantization as
 * `acc_quantize`.
 *
 * @param acc_in Accumulator groups from the convolution engine.
 * @param bias_cache On-chip packed bias vectors.
 * @param ofm_out Requantized output-vector stream.
 * @param desc_comp Compute descriptors through each writeback boundary.
 * @param num_writeback_tiles Number of completed output tiles emitted.
 * @param shift_out Layer-wide output requantization shift.
 * @param act_max Upper clamp applied with ReLU: 127 for ReLU, lower for ReLU6.
 */
void acc_quantize_pp(acc_group_stream_t &acc_in,
                     const biases_vector_dt bias_cache[BIAS_BUF_DEPTH],
                     ofm_stream_t &ofm_out,
                     desc_stream_comp_t &desc_comp,
                     int num_writeback_tiles,
                     int shift_out,
                     int act_max);
/**
 * @brief Applies optional shifted residual addition and post-add ReLU.
 *
 * @param desc_stream_in Input output-tile descriptors.
 * @param ofm_in Main activation stream.
 * @param residual_in Residual activation stream.
 * @param desc_stream_out Forwarded output-tile descriptors.
 * @param ofm_out Result stream.
 * @param num_tiles Number of descriptors consumed and forwarded.
 * @param act_max Upper clamp applied with the post-add ReLU.
 */
void residual_add(desc_stream_ofm_t &desc_stream_in,
                  ofm_stream_t &ofm_in,
                  residual_stream_t &residual_in,
                  desc_stream_ofm_t &desc_stream_out,
                  ofm_stream_t &ofm_out,
                  int num_tiles,
                  int act_max);

/**
 * @brief Applies optional fused 3x3 stride-2 max pooling.
 *
 * Disabled descriptors pass their activation tiles through unchanged.
 *
 * @param desc_stream_in Input output-tile descriptors.
 * @param ofm_in Pre-pool activation stream.
 * @param desc_stream_out Forwarded output-tile descriptors.
 * @param ofm_out Pooled or pass-through activation stream.
 * @param num_tiles Number of descriptors consumed and forwarded.
 */
void maxpool(desc_stream_ofm_t &desc_stream_in,
             ofm_stream_t &ofm_in,
             desc_stream_ofm_t &desc_stream_out,
             ofm_stream_t &ofm_out,
             int num_tiles);

/**
 * @brief Applies optional fused global average pooling.
 *
 * Enabled descriptors reduce the full convolution tile to one vector using a
 * host-computed fixed-point reciprocal. Disabled descriptors pass through.
 *
 * @param desc_stream_in Input output-tile descriptors.
 * @param ofm_in Pre-pool activation stream.
 * @param desc_stream_out Forwarded output-tile descriptors.
 * @param ofm_out Pooled or pass-through activation stream.
 * @param num_tiles Number of descriptors consumed and forwarded.
 * @param gap_mul Fixed-point reciprocal multiplier for the spatial element count.
 */
void gap(desc_stream_ofm_t &desc_stream_in,
         ofm_stream_t &ofm_in,
         desc_stream_ofm_t &desc_stream_out,
         ofm_stream_t &ofm_out,
         int num_tiles,
         int gap_mul);

/**
 * @brief Writes completed output tiles to strided device memory.
 *
 * Descriptors with zero tile height represent accumulation-only work and
 * consume no activation data.
 *
 * @param out Packed output tensor in device memory.
 * @param desc_stream_ofm Output-tile descriptors.
 * @param ofm_stream Completed output-vector stream.
 * @param num_tiles Number of descriptors consumed.
 */
void store_ofm_tile(activations_vector_dt *out,
                    desc_stream_ofm_t &desc_stream_ofm,
                    ofm_stream_t &ofm_stream,
                    int num_tiles);

#endif // ACC_STORE_H_
