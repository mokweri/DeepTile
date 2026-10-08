/**
 * @file data_orch.h
 * @brief Declares descriptor distribution and input-side data movement.
 * @ingroup deeptile_kernel
 */

#ifndef DATA_ORCH_H_
#define DATA_ORCH_H_

#include "common_defs.h"

// Descriptor FIFO depth shared by the dataflow fan-out streams.
#ifndef DEEPTILE_DESC_STREAM_DEPTH
#define DEEPTILE_DESC_STREAM_DEPTH 16
#endif

/**
 * @brief Distributes external schedule descriptors to kernel dataflow stages.
 *
 * Uses one shared AXI reader with two-epoch transfer lookahead, matching the
 * weight engine's ping-pong order. Compute descriptors for the current epoch
 * are broadcast before fetching the transfer two epochs ahead, so progress
 * does not require the transfer FIFO to hold the entire schedule.
 * Transfer compute counts must sum to num_tiles; num_xfers must be positive.
 *
 * @param xfer_desc Transfer descriptors in device memory.
 * @param desc_xfer Transfer-descriptor output stream.
 * @param num_xfers Number of transfer descriptors.
 * @param tile_desc Compute descriptors in device memory.
 * @param desc_ifm IFM-memory descriptor output stream.
 * @param desc_pad Padding descriptor output stream.
 * @param desc_comp_ifm Compute descriptors for IFM streaming.
 * @param desc_comp_mac Compute descriptors for the MAC engine.
 * @param desc_comp_acc Compute descriptors for accumulation.
 * @param desc_ofm Output and fused-operation descriptors.
 * @param num_tiles Number of compute descriptors.
 * @param kernel_size Dense-convolution kernel width and height.
 * @param use_ifm_cache Enables flat grouping for resident pointwise inputs.
 * @param dw_enable Indicates that the launch contains fused depthwise work.
 */
void schedule_controller(const uram_xfer_desc_t *__restrict xfer_desc,
                         desc_stream_xfer_t &desc_xfer,
                         int num_xfers,
                         const descriptor_dt *__restrict tile_desc,
                         desc_stream_ifm_t &desc_ifm,
                         desc_stream_pad_t &desc_pad,
                         desc_stream_comp_t &desc_comp_ifm,
                         desc_stream_comp_t &desc_comp_mac,
                         desc_stream_comp_t &desc_comp_acc,
                         desc_stream_ofm_t &desc_ofm,
                         int num_tiles,
                         uint8_t kernel_size,
                         ap_uint<1> use_ifm_cache,
                         bool dw_enable);

/**
 * @brief Loads all layer biases into the on-chip bias cache.
 *
 * @param biases Packed bias vectors in device memory.
 * @param bias_cache Destination on-chip cache.
 * @param num_bias_vecs Number of packed vectors to load.
 */
void preload_bias_to_bram(const biases_vector_dt *__restrict biases,
                          biases_vector_dt bias_cache[BIAS_BUF_DEPTH],
                          int num_bias_vecs);

/**
 * @brief Streams residual tiles in output-tile order.
 *
 * Descriptors without residual addition or writeback are forwarded without
 * reading data.
 *
 * @param residual Packed residual tensor in device memory.
 * @param desc_stream_in Input output-tile descriptors.
 * @param desc_stream_out Forwarded output-tile descriptors.
 * @param residual_stream Residual-vector output stream.
 * @param num_tiles Number of descriptors consumed and forwarded.
 */
void residual_loader(const activations_vector_dt *__restrict residual,
                     desc_stream_ofm_t &desc_stream_in,
                     desc_stream_ofm_t &desc_stream_out,
                     residual_stream_t &residual_stream,
                     int num_tiles);

/**
 * @brief Loads input tiles, applies padding, and streams convolution windows.
 *
 * Pointwise layers may retain input-channel groups in the resident cache for
 * reuse across output-channel groups. Other layers alternate two raw-tile
 * buffers to overlap memory loading and pixel streaming.
 *
 * @param ifm Packed input tensor in device memory.
 * @param desc_stream_ifm IFM-memory descriptors.
 * @param desc_stream_pad Padding descriptors.
 * @param desc_comp Compute descriptors for grouping and cache control.
 * @param pixel_stream Output streams, one per spatial lane.
 * @param num_tiles Number of descriptors consumed.
 * @param kernel_size Dense-convolution kernel width and height.
 * @param stride Dense-convolution stride.
 * @param use_ifm_cache Enables resident pointwise input caching.
 * @param spatial_tile_size Cached spatial elements per CIG.
 * @param ifm_cig_stride_vecs Input-plane stride in packed activation vectors.
 */
void load_and_stream_ifm(const ifm_word_dt *__restrict ifm,
                         desc_stream_ifm_t &desc_stream_ifm,
                         desc_stream_pad_t &desc_stream_pad,
                         desc_stream_comp_t &desc_comp,
                         ifm_stream_t pixel_stream[P_SPATIAL],
                         int num_tiles,
                         uint8_t kernel_size,
                         uint8_t stride,
                         ap_uint<1> use_ifm_cache,
                         ap_uint<14> spatial_tile_size,
                         int ifm_cig_stride_vecs);

#endif // DATA_ORCH_H_
