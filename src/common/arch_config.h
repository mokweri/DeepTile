/**
 * @file arch_config.h
 * @brief Defines the integer-only DeepTile architecture configuration.
 * @ingroup deeptile_common
 *
 * This header intentionally has no Xilinx dependencies so the compiler and
 * other native tools can consume the same configuration as the HLS kernel.
 */

#ifndef DEEPTILE_ARCH_CONFIG_H
#define DEEPTILE_ARCH_CONFIG_H

// Architecture parallelism. Packed dense MACs use two output lanes per DSP.
#define BITWIDTH 8
#define OC_TILE_SIZE 16
#define IC_TILE_SIZE 16
#define P_IC 16
#define P_SPATIAL 4

// HLS: Spatial-group indexing uses shifts and masks in II=1 loops, so
// `P_SPATIAL` must be a supported power of two.
#define P_SPATIAL_LOG2                                                         \
  ((P_SPATIAL) == 1    ? 0                                                     \
   : (P_SPATIAL) == 2  ? 1                                                     \
   : (P_SPATIAL) == 4  ? 2                                                     \
   : (P_SPATIAL) == 8  ? 3                                                     \
   : (P_SPATIAL) == 16 ? 4                                                     \
   : (P_SPATIAL) == 32 ? 5                                                     \
                       : -1)
#define P_SPATIAL_MASK (P_SPATIAL - 1)

// Maximum output-tile geometry and supported kernel edge.
#define MAX_TILE_W 64
#define MAX_TILE_H 32
#define K_MAX 7

// Padded input-tile bounds include the maximum convolution halo.
#define PADDED_TILE_H (MAX_TILE_H + K_MAX - 1)
#define PADDED_TILE_W (MAX_TILE_W + K_MAX - 1)

// INVARIANT: Every output tile must fit this spatial-element budget.
#define MAX_SPATIAL_TILE_SIZE (MAX_TILE_H * MAX_TILE_W)

// Fractional width of the host-computed global-average-pooling reciprocal.
#define GAP_SCALE_FRAC_BITS 16

// Buffer capacities shared with the compiler's hardware specification.

// Resident pointwise-input cache capacity in packed pixel vectors.
#define IFM_CACHE_DEPTH 32768
#define IFM_CACHE_DEPTH_LOG2 15

// Partial-sum entries per spatial and output-channel bank.
#define PSUM_DEPTH (MAX_SPATIAL_TILE_SIZE / P_SPATIAL)

// Weight vectors per output-channel bank and ping-pong buffer.
#define WEIGHT_BUF_DEPTH 512

// Packed output-channel groups retained in the on-chip bias cache, bounding a
// layer to BIAS_BUF_DEPTH * OC_TILE_SIZE output channels.
//
// The cache is one BIAS_VECTOR_WIDTH-bit word per group, so its BRAM cost is
// set by that width, not by this depth: 512 bits forces ceil(512/36) = 15
// BRAM18 primitives in parallel, each 512 words deep. Any depth up to 512 is
// therefore free, and 128 left three quarters of the primitives unused while
// capping layers at 2048 output channels -- exactly ResNet-50's widest layer,
// with no headroom. 512 costs the same and admits 8192.
#define BIAS_BUF_DEPTH 512

// Fused depthwise stage (dw_stage). It computes 3x3 or 5x5 kernels. Each
// compact COG block in device memory holds K*K tap words, then four bias
// words, padded to an even burst stride: 16 words for 3x3 and 32 for 5x5. An
// odd stride once dropped trailing bias words on hardware.
#define DW_KMAX 5
#define DW_TAPS_MAX (DW_KMAX * DW_KMAX)
#define DW_TAPS_3 9
#define DW_TAPS_5 25
#define DW_BIAS_WORDS 4
#define DW_WORDS_PER_COG_3 16
#define DW_WORDS_PER_COG_5 32
#define DW_MAX_COGS 128
// Taps and biases are stored apart, so 5x5 blocks do not double the tap store:
// it holds 128 3x3 blocks (2,048 channels) or 81 5x5 blocks (1,296 channels).
#define DW_TAP_STORE_DEPTH 2048
#define DW_BIAS_STORE_DEPTH (DW_MAX_COGS * DW_BIAS_WORDS)
#define DW_MAX_COGS_5 (DW_TAP_STORE_DEPTH / DW_TAPS_5)

// These integer-only invariants are checked in native and HLS builds.
#ifdef __cplusplus
static_assert(P_SPATIAL_LOG2 >= 0,
              "P_SPATIAL must be a power of two in [1,32]; index math is "
              "shift/mask based and has no divider fallback");
static_assert((1 << P_SPATIAL_LOG2) == P_SPATIAL,
              "P_SPATIAL_LOG2 must be exactly log2(P_SPATIAL)");
static_assert((1 << IFM_CACHE_DEPTH_LOG2) == IFM_CACHE_DEPTH,
              "IFM_CACHE_DEPTH_LOG2 must be exactly log2(IFM_CACHE_DEPTH)");
static_assert(MAX_SPATIAL_TILE_SIZE == MAX_TILE_H * MAX_TILE_W,
              "MAX_SPATIAL_TILE_SIZE must equal MAX_TILE_H * MAX_TILE_W");
static_assert(PADDED_TILE_H == MAX_TILE_H + K_MAX - 1,
              "PADDED_TILE_H must cover the maximum vertical halo");
static_assert(PADDED_TILE_W == MAX_TILE_W + K_MAX - 1,
              "PADDED_TILE_W must cover the maximum horizontal halo");
static_assert(PSUM_DEPTH * P_SPATIAL == MAX_SPATIAL_TILE_SIZE,
              "PSUM_DEPTH must divide exactly: every spatial group is full");
static_assert(DW_MAX_COGS * DW_TAPS_3 <= DW_TAP_STORE_DEPTH,
              "the depthwise tap store must hold every 3x3 block");
static_assert(DW_TAPS_5 + DW_BIAS_WORDS <= DW_WORDS_PER_COG_5 &&
              DW_TAPS_3 + DW_BIAS_WORDS <= DW_WORDS_PER_COG_3,
              "a depthwise COG block must fit its memory stride");
#endif

#endif  // DEEPTILE_ARCH_CONFIG_H
