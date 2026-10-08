/**
 * @file dw_stage.h
 * @brief Declares the fused streaming depthwise-convolution stage.
 * @ingroup deeptile_kernel
 */

#ifndef DW_STAGE_H_
#define DW_STAGE_H_

#include "common_defs.h"

// The weight-block layout and store sizes (DW_*) are defined in arch_config.h,
// which the host packer shares.

/**
 * @brief Applies optional fused depthwise convolution to an activation stream.
 *
 * Enabled launches apply per-channel 3x3 or 5x5 convolution, requantization,
 * and optional ReLU in raster order. Disabled and accumulation-only descriptors
 * pass through without depthwise computation.
 *
 * @param desc_in Input output-tile descriptors.
 * @param ofm_in Pointwise-output activation stream.
 * @param dw_weights Compact depthwise weights and biases in device memory.
 * @param desc_out Descriptors updated with depthwise output geometry.
 * @param ofm_out Depthwise or pass-through activation stream.
 * @param num_tiles Number of descriptors consumed and forwarded.
 * @param num_cogs Number of compact weight blocks to preload.
 * @param dw_config Enable (bit 0), stride (bits 1-2), ReLU (bit 3), 5x5 kernel
 *        (bit 4, else 3x3) and ReLU clamp (bits 8-15: 127 for ReLU, lower for
 *        ReLU6).
 * @param dw_shift Layer-wide depthwise requantization shift.
 */
void dw_stage(desc_stream_ofm_t &desc_in,
              ofm_stream_t &ofm_in,
              const weight_vector_dt *__restrict dw_weights,
              desc_stream_ofm_t &desc_out,
              ofm_stream_t &ofm_out,
              int num_tiles,
              int num_cogs,
              int dw_config,
              int dw_shift);

#endif // DW_STAGE_H_
