/**
 * @file common_defs.h
 * @brief Defines the shared DeepTile hardware/software ABI and HLS data types.
 * @ingroup deeptile_common
 *
 * Contains packed tensor types, launch descriptors, stream types, buffer
 * geometry, and synthesis-only bounds interpreted by the host and kernel.
 */

#ifndef COMMON_DEFS_H
#define COMMON_DEFS_H

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <iostream>
#include <stdint.h>
#include <stdio.h>

// Architecture knobs remain HLS-independent so native compiler code can share
// them. HLS-derived buffer geometry is defined below.
#include "arch_config.h"
#include "kernel_abi.h"

// Tile and buffer geometry

#define LINE_BUFFER_WIDTH PADDED_TILE_W  // Compatibility alias.

// Full padded-tile element count retained as a compatibility bound.
#define IFM_BUF_DEPTH (PADDED_TILE_H * PADDED_TILE_W)

// Each true-dual-port raw-input copy serves two spatial lanes.
#define IFM_BUF_COPIES ((P_SPATIAL + 1) / 2)

// Raw-input copies use separate even- and odd-column banks.
#define IFM_BANK_DEPTH (PADDED_TILE_H * ((PADDED_TILE_W + 1) / 2))

// Pointwise input storage is indexed by output position without halo padding.
#define IFM_K1_BUF_DEPTH MAX_SPATIAL_TILE_SIZE

// Unified input stores use one fully partitioned bank per spatial lane. The
// same shape supports resident pointwise caching and raw-tile ping-pong.
#define IFM_STORE_ROWS (IFM_CACHE_DEPTH / P_SPATIAL)
#define IFM_STORE_ROWS_LOG2 (IFM_CACHE_DEPTH_LOG2 - P_SPATIAL_LOG2)

#define OFM_BUF_DEPTH MAX_SPATIAL_TILE_SIZE
// Buffer-geometry invariants
#ifdef __cplusplus
static_assert((1 << IFM_STORE_ROWS_LOG2) == IFM_STORE_ROWS,
              "IFM_STORE_ROWS must be a power of two so cache_row_idx_t is exact");
// INVARIANT: The raw padded tile and resident cache share each input-store bank.
static_assert(IFM_BANK_DEPTH <= IFM_STORE_ROWS,
              "IFM store too shallow for the padded K>1 raw tile: raise "
              "IFM_CACHE_DEPTH or lower P_SPATIAL / PADDED_TILE dimensions");
// INVARIANT: Raw-input bank indexing assigns two parity banks to each copy.
static_assert(IFM_BUF_COPIES * 2 == P_SPATIAL,
              "raw-IFM parity banking requires P_SPATIAL even (>= 2); "
              "P_SPATIAL=1 needs a dedicated single-bank loader");
#endif

// Packed arithmetic and tensor types
typedef ap_int<BITWIDTH> DPTYPE;
typedef ap_int<BITWIDTH> pixel_t;
typedef ap_int<BITWIDTH> weight_t;
typedef ap_int<2 * BITWIDTH> product_t;
typedef ap_int<32> acc_t;

#define IFM_VECTOR_WIDTH (IC_TILE_SIZE * BITWIDTH)
typedef ap_uint<IFM_VECTOR_WIDTH> pixel_vector_dt;

/** @brief Two contiguous packed input vectors in one memory word. */
struct ifm_word_dt {
  pixel_vector_dt v[2];
};

#define WEIGHT_VECTOR_WIDTH (IC_TILE_SIZE * BITWIDTH)
typedef ap_uint<WEIGHT_VECTOR_WIDTH> weight_vector_dt;

#define BIAS_VECTOR_WIDTH (OC_TILE_SIZE * 32)
typedef ap_uint<BIAS_VECTOR_WIDTH> biases_vector_dt;

#define OFM_VECTOR_WIDTH (OC_TILE_SIZE * BITWIDTH)
typedef ap_uint<OFM_VECTOR_WIDTH> activations_vector_dt;

typedef struct {
  pixel_vector_dt v[K_MAX * K_MAX];
} window_vector_dt;

/** @brief Weights for all output lanes at one kernel position. */
#define WIDE_WT_WIDTH (P_IC * BITWIDTH)
typedef struct {
  ap_uint<WIDE_WT_WIDTH> v[OC_TILE_SIZE];
} wide_wt_dt;

/** @brief One accumulator per output-channel lane. */
struct acc_vec_dt {
  acc_t v[OC_TILE_SIZE];
};

// Compute-descriptor control flags shared with the compiler.
static const unsigned int ACCUM_RESET = (1U << 0);
static const unsigned int ACCUM_WRITE_BACK = (1U << 1);
static const unsigned int RELU_ON = (1U << 2);
static const unsigned int IS_LAST_TILE = (1U << 3);
static const unsigned int ADD_ENABLE = (1U << 4);
static const unsigned int POST_ADD_RELU = (1U << 5);
static const unsigned int POOL_ENABLE = (1U << 6);
static const unsigned int GAP_ENABLE = (1U << 7);

/**
 * @brief Compiler-to-kernel compute descriptor.
 *
 * This is a stable 32-byte ABI record. Tensor offsets and row strides use
 * packed activation-vector units. Padding packs four 4-bit values in
 * top, bottom, left, right order.
 */
typedef struct alignas(32) descriptor_dt {
  // Input tile, bytes 0-11.
  uint32_t ifm_offset_vecs;      ///< Input base for CIG zero, bytes 0-3.
  uint8_t  ifm_ddr_h;            ///< Boundary-clipped input rows, byte 4.
  uint8_t  ifm_ddr_w;            ///< Boundary-clipped input columns, byte 5.
  uint8_t  ifm_ddr_stride;       ///< Full input-tensor row stride, byte 6.
  uint8_t  ifm_tile_h;           ///< Padded on-chip input height, byte 7.
  uint8_t  ifm_tile_w;           ///< Padded on-chip input width, byte 8.
  uint8_t  reuse_ifm;            ///< Nonzero reads the resident cache, byte 9.
  uint16_t packed_padding;       ///< Input padding nibbles, bytes 10-11.

  // Output tile, bytes 12-19.
  uint32_t ofm_offset_vecs;      ///< Output base for this COG, bytes 12-15.
  uint8_t  ofm_tile_h;           ///< Convolution output height, byte 16.
  uint8_t  ofm_tile_w;           ///< Convolution output width, byte 17.
  uint8_t  ofm_row_stride;       ///< Full output-tensor row stride, byte 18.
  uint8_t  flags;                ///< Compute and fused-operation flags, byte 19.

  // Weight and bias lookup, bytes 20-23.
  uint16_t uram_base_depth;      ///< Active weight-bank base, bytes 20-21.
  uint16_t bias_offset_idx;      ///< Packed bias-cache index, bytes 22-23.

  // Fused post-operations, bytes 24-29.
  int8_t   residual_shift;       ///< Residual-alignment shift, byte 24.
  uint8_t  pooled_ofm_tile_h;    ///< Post-pool output height, byte 25.
  uint8_t  pooled_ofm_tile_w;    ///< Post-pool output width, byte 26.
  int8_t   post_pool_shift;      ///< Post-pool requantization shift, byte 27.
  uint16_t pool_packed_padding;  ///< Pool or depthwise padding, bytes 28-29.

  // Input-channel chaining, bytes 30-31.
  uint8_t  num_cigs;             ///< CIGs consumed by this descriptor, byte 30.
  uint8_t  cache_cig_idx;        ///< First pointwise cache group, byte 31.
} descriptor_dt;

static_assert(sizeof(descriptor_dt) == DEEPTILE_COMPUTE_DESCRIPTOR_BYTES,
              "descriptor_dt size must match the kernel ABI");

/** @brief Input-memory fields forwarded to the IFM loader. */
typedef struct alignas(8) ifm_desc_t {
  uint32_t offset_vecs;   ///< CIG base in packed-vector units.
  uint8_t  ddr_h;         ///< Boundary-clipped row count.
  uint8_t  ddr_w;         ///< Boundary-clipped vectors per row.
  uint8_t  ddr_stride;    ///< Full input-tensor row stride.
} ifm_desc_t;

/** @brief Contiguous packed-weight range. */
typedef struct alignas(4) wgt_desc_t {
  uint32_t offset_vecs;  ///< Start in packed-weight-vector units.
  uint16_t len_vecs;     ///< Number of packed weight vectors.
} wgt_desc_t;

/**
 * @brief One banked weight transfer and its dependent compute range.
 *
 * Weights are ordered by output-channel bank, full layer depth, and
 * input-channel lane. The same per-bank offset and length apply to every bank.
 */
typedef struct alignas(32) uram_xfer_desc_t {
  uint32_t ddr_offset_vecs;    ///< Per-bank vector start, bytes 0-3.
  uint32_t ddr_length_vecs;    ///< Per-bank vector count, bytes 4-7.
  uint32_t total_layer_depth;  ///< Full depth of one bank, bytes 8-11.
  uint16_t depth_count;        ///< Entries loaded per bank, bytes 12-13.
  uint16_t compute_count;      ///< Dependent compute descriptors, bytes 14-15.
  uint8_t  buffer_id;          ///< Ping-pong buffer identifier, byte 16.
  uint8_t  flags;              ///< Transfer flags, byte 17.
  uint8_t  reserved[14];       ///< Reserved ABI bytes 18-31.
} uram_xfer_desc_t;

static_assert(sizeof(uram_xfer_desc_t) ==
                  DEEPTILE_TRANSFER_DESCRIPTOR_BYTES,
              "uram_xfer_desc_t size must match the kernel ABI");

/** @brief Output and fused-operation fields forwarded through post-processing. */
typedef struct alignas(16) ofm_desc_t {
  uint32_t offset_vecs;       ///< COG output base in packed-vector units.
  uint8_t  tile_w;            ///< Final writeback width.
  uint8_t  row_stride;        ///< Full output-tensor row stride.
  uint8_t  tile_h;            ///< Final writeback height; zero disables output.
  uint8_t  flags;             ///< Fused-operation flags.
  int8_t   residual_shift;    ///< Residual-alignment shift.
  uint8_t  conv_tile_w;       ///< Convolution width before fused operations.
  uint8_t  conv_tile_h;       ///< Convolution height before fused operations.
  uint16_t pool_packed_padding;  ///< Pool or depthwise padding nibbles.
  int8_t   post_pool_shift;      ///< Post-pool requantization shift.
  uint8_t  dw_cog_idx;        ///< Compact depthwise weight-block index.
  uint8_t  reserved[2];       ///< Reserved internal fields.
} ofm_desc_t;

/** @brief Compute fields broadcast to the IFM, MAC, and accumulation stages. */
typedef struct alignas(16) comp_desc_t {
  uint16_t uram_base_depth;  ///< Starting depth in the active weight bank.
  uint8_t  ifm_ddr_w;        ///< Boundary-clipped input vectors per row.
  uint8_t  ofm_tile_w;       ///< Convolution output width.
  uint16_t packed_padding;   ///< Input padding nibbles.
  uint8_t  ifm_ddr_h;        ///< Boundary-clipped input rows.
  uint8_t  ofm_tile_h;       ///< Convolution output height.
  uint8_t  ifm_tile_h;       ///< Padded on-chip input height.
  uint8_t  ifm_tile_w;       ///< Padded on-chip input width.
  uint8_t  flags;           ///< Compute and fused-operation flags.
  uint8_t  reuse_ifm;        ///< Zero fills the cache; nonzero reuses it.
  uint8_t  num_cigs;         ///< CIGs consumed by this descriptor.
  // ABI: Bit zero selects row-aligned grouping; bits 1-7 hold the pointwise
  // cache group. Use the accessors below to preserve the 16-byte record.
  uint8_t  mode_bits;
  uint16_t bias_offset_idx;  ///< Packed bias-cache index.
} comp_desc_t;

// Access the two fields packed into `comp_desc_t::mode_bits`.
static inline uint8_t desc_row_aligned(const comp_desc_t &d) {
#pragma HLS INLINE
  return (uint8_t)(d.mode_bits & 0x1U);
}

static inline uint8_t desc_cache_cig_idx(const comp_desc_t &d) {
#pragma HLS INLINE
  return (uint8_t)(d.mode_bits >> 1);
}

static inline uint8_t desc_pack_mode_bits(uint8_t row_aligned,
                                          uint8_t cache_cig_idx) {
#pragma HLS INLINE
  return (uint8_t)((row_aligned & 0x1U) | (uint8_t)(cache_cig_idx << 1));
}

/** @brief Padding and input geometry forwarded to the input streamer. */
typedef struct alignas(8) pad_desc_t {
  uint16_t packed_padding;  ///< Four nibbles: top, bottom, left, right.
  uint8_t  ddr_h;           ///< Boundary-clipped input rows.
  uint8_t  ddr_w;           ///< Boundary-clipped vectors per row.
  uint8_t  tile_h;          ///< Padded on-chip input height.
  uint8_t  tile_w;          ///< Padded on-chip input width.
} pad_desc_t;

/** @brief One packed input vector per spatial lane. */
struct pixel_group_dt {
  pixel_vector_dt v[P_SPATIAL];
};

/** @brief One output-channel accumulator vector per spatial lane. */
struct acc_group_dt {
  acc_vec_dt v[P_SPATIAL];
};

// HLS: Independent per-lane input streams let placement keep each FIFO close
// to its consuming DSP group and avoid a single wide, high-fanout stream.
typedef hls::stream<pixel_vector_dt> ifm_stream_t;
typedef hls::stream<pixel_group_dt> ifm_group_stream_t;  // Compatibility alias.
typedef hls::stream<weight_vector_dt> weight_stream_t;
typedef hls::stream<activations_vector_dt> ofm_stream_t;
typedef hls::stream<activations_vector_dt> residual_stream_t;
typedef hls::stream<window_vector_dt> window_stream_t;

typedef hls::stream<ifm_desc_t> desc_stream_ifm_t;
typedef hls::stream<wgt_desc_t> desc_stream_wgt_t;
typedef hls::stream<uram_xfer_desc_t> desc_stream_uram_wgt_t;
typedef hls::stream<uram_xfer_desc_t> desc_stream_xfer_t;
typedef hls::stream<ofm_desc_t> desc_stream_ofm_t;
typedef hls::stream<comp_desc_t> desc_stream_comp_t;
typedef hls::stream<pad_desc_t> desc_stream_pad_t;

typedef hls::stream<wide_wt_dt> wide_wt_stream_t;
typedef hls::stream<acc_vec_dt> acc_stream_t;
typedef hls::stream<acc_group_dt> acc_group_stream_t;

// AXI co-simulation depths
//
// `scripts/compute_tripcounts.py` supplies case-specific values. Regenerate
// them for co-simulation because undersized depths model incomplete memories.
#ifndef AXI_DEPTH_IFM
#define AXI_DEPTH_IFM      12544
#endif
// `AXI_DEPTH_IFM` counts packed pixel vectors; the memory port transfers two
// vectors per `ifm_word_dt`.
#define AXI_DEPTH_IFM_WORDS ((AXI_DEPTH_IFM + 1) / 2)
#ifndef AXI_DEPTH_WEIGHTS
#define AXI_DEPTH_WEIGHTS  131072
#endif
#ifndef AXI_DEPTH_BIASES
#define AXI_DEPTH_BIASES   128
#endif
#ifndef AXI_DEPTH_OUT
#define AXI_DEPTH_OUT      6272
#endif
#ifndef AXI_DEPTH_RESIDUAL
#define AXI_DEPTH_RESIDUAL 6272
#endif
#ifndef AXI_DEPTH_XDESC
#define AXI_DEPTH_XDESC    2
#endif
// Compact fused-depthwise weights, in memory words: the larger of 128 3x3
// blocks (2,048) and 81 5x5 blocks (2,592).
#ifndef AXI_DEPTH_DW
#define AXI_DEPTH_DW       2592
#endif
#ifndef AXI_DEPTH_CDESC
#define AXI_DEPTH_CDESC    8192
#endif

// HLS synthesis trip-count estimates
//
// Case-specific values come from `scripts/compute_tripcounts.py`. Defaults keep
// synthesis valid when no generated overrides are supplied.
#ifndef TC_K2
#define TC_K2            49
#endif
#ifndef TC_CIG
#define TC_CIG           128
#endif
#ifndef TC_COG
#define TC_COG           128
#endif
#ifndef TC_T_OH
#define TC_T_OH          32
#endif
#ifndef TC_T_OW
#define TC_T_OW          64
#endif
#ifndef TC_NUM_SPATIAL
#define TC_NUM_SPATIAL   4
#endif
#ifndef TC_NUM_XFERS
#define TC_NUM_XFERS     2
#endif
#ifndef TC_NUM_TILES
#define TC_NUM_TILES     8192
#endif
#ifndef TC_NUM_GROUPS
#define TC_NUM_GROUPS    512
#endif
#ifndef TC_SPATIAL
#define TC_SPATIAL       2048
#endif
#ifndef TC_DEPTH_COUNT
#define TC_DEPTH_COUNT   288
#endif
#ifndef TC_COMPUTE_COUNT
#define TC_COMPUTE_COUNT 4096
#endif
#ifndef TC_PADDED_H
#define TC_PADDED_H      37
#endif
#ifndef TC_PADDED_W
#define TC_PADDED_W      69
#endif
#ifndef TC_ROWLEN
#define TC_ROWLEN        127
#endif
#ifndef TC_OFM_WRITEBACK
#define TC_OFM_WRITEBACK 128
#endif
#ifndef TC_NUM_BIAS_VECS
#define TC_NUM_BIAS_VECS 128
#endif
#ifndef TC_PP_LOOP
#define TC_PP_LOOP       1
#endif
#ifndef TC_MAC_FLAT
#define TC_MAC_FLAT      25088
#endif
#ifndef TC_IFM_PP
#define TC_IFM_PP        8191
#endif
#ifndef TC_OFM_PP
#define TC_OFM_PP        127
#endif

#endif // COMMON_DEFS_H
