/**
 * @file descriptors.hpp
 * @brief Packed descriptor ABI shared by the compiler, runtime, and kernel.
 * @ingroup deeptile_compiler
 *
 * These are the byte-for-byte execution contract between the host and the
 * `tile_conv` kernel. They must stay identical to the kernel-side layout in
 *   - src/common/common_defs.h   (descriptor_dt, uram_xfer_desc_t)
 *
 * ABI: Mapping policy does not belong here. ABI tests pin each size and offset.
 */

#pragma once

#include <cstdint>

namespace deeptile {
namespace tile {

// Compute Descriptor (32 bytes)
/**
 * @brief 32-byte compute descriptor sent to the tile_conv kernel.
 *
 * Weights are pre-loaded into URAM ping-pong buffers by transfer epochs; each
 * compute descriptor references its weights via uram_base_depth.
 *
 * Byte layout (alignas(32)):
 *   ── IFM (bytes 0–11) ──────────────────────────────────────────────
 *   0-3:   ifm_offset_vecs    DDR base for CIG 0 (IC-vec units)
 *   4:     ifm_ddr_h          rows to read from DDR (boundary-clipped)
 *   5:     ifm_ddr_w          cols per DDR row (boundary-clipped)
 *   6:     ifm_ddr_stride     DDR row stride = full IFM tensor width W
 *   7:     ifm_tile_h         on-chip tile height = ddr_h + pad_top + pad_bottom
 *   8:     ifm_tile_w         on-chip tile width  = ddr_w + pad_left + pad_right
 *   9:     reuse_ifm          1 = stream from K=1 URAM cache; skip DDR read
 *   10-11: packed_padding     [top(4)|bot(4)|left(4)|right(4)]
 *   ── OFM (bytes 12–19) ─────────────────────────────────────────────
 *   12-15: ofm_offset_vecs    DDR base for this COG tile (OC-vec units)
 *   16:    ofm_tile_h         output tile height T_oh
 *   17:    ofm_tile_w         output tile width  T_ow
 *   18:    ofm_row_stride     DDR row stride = full OFM tensor width
 *   19:    flags              ACCUM_RESET|WRITE_BACK|RELU_ON|ADD_EN|...
 *   ── Weight + Bias (bytes 20–23) ───────────────────────────────────
 *   20-21: uram_base_depth    (local_oc * num_ic_tiles + cig) * K²
 *   22-23: bias_offset_idx    BRAM index = absolute COG tile index
 *   ── Post-ops (bytes 24–29) ────────────────────────────────────────
 *   24:    residual_shift     signed shift to align residual before add
 *   25:    pooled_ofm_tile_h  store tile height after pool
 *   26:    pooled_ofm_tile_w  store tile width  after pool
 *   27:    post_pool_shift    signed shift applied after max-pool
 *   28-29: pool_packed_padding pool zero-pad [top(4)|bot(4)|left(4)|right(4)]
 *   ── Chaining (bytes 30–31) ────────────────────────────────────────
 *   30:    num_cigs           CIGs to chain: 1 = single, N = chained sweep
 *   31:    cache_cig_idx      First K=1 cache row group for this descriptor.
 *                            Chaining advances from this group internally;
 *                            a partial chain need not start at group zero.
 *                            Ignored when the K=1 cache is off. The starting
 *                            group must be in 0..127 (7 internal bits), while
 *                            the chain may advance beyond that range.
 */
struct alignas(32) TileDescriptor {
  // IFM fields.
  std::uint32_t ifm_offset_vecs;     ///< Bytes 0–3.
  std::uint8_t  ifm_ddr_h;           ///< Byte 4.
  std::uint8_t  ifm_ddr_w;           ///< Byte 5.
  std::uint8_t  ifm_ddr_stride;      ///< Byte 6.
  std::uint8_t  ifm_tile_h;          ///< Byte 7.
  std::uint8_t  ifm_tile_w;          ///< Byte 8.
  std::uint8_t  reuse_ifm;           ///< Byte 9.
  std::uint16_t packed_padding;      ///< Bytes 10–11.

  // OFM fields.
  std::uint32_t ofm_offset_vecs;     ///< Bytes 12–15.
  std::uint8_t  ofm_tile_h;          ///< Byte 16.
  std::uint8_t  ofm_tile_w;          ///< Byte 17.
  std::uint8_t  ofm_row_stride;      ///< Byte 18.
  std::uint8_t  flags;               ///< Byte 19.

  // Weight and bias fields.
  std::uint16_t uram_base_depth;     ///< Bytes 20–21.
  std::uint16_t bias_offset_idx;     ///< Bytes 22–23.

  // Post-operation fields.
  std::int8_t   residual_shift;      ///< Byte 24.
  std::uint8_t  pooled_ofm_tile_h;   ///< Byte 25.
  std::uint8_t  pooled_ofm_tile_w;   ///< Byte 26.
  std::int8_t   post_pool_shift;     ///< Byte 27.
  std::uint16_t pool_packed_padding; ///< Bytes 28–29.

  // Chaining fields.
  std::uint8_t  num_cigs;            ///< Byte 30.
  std::uint8_t  cache_cig_idx;       ///< Byte 31.
};

/// Maximum CIG count for an unchained cached schedule.
/// ABI: cache_cig_idx occupies seven bits in the kernel mini-descriptor.
constexpr int kMaxUnchainedCacheCigs = 128;
/// Eight-bit count and seven-bit starting group permit two partial chains.
constexpr int kMaxCigsPerDescriptor = 255;
constexpr int kMaxChainedCacheCigs =
    (kMaxUnchainedCacheCigs - 1) + kMaxCigsPerDescriptor;

static_assert(sizeof(TileDescriptor) == 32,
              "TileDescriptor must be exactly 32 bytes");
static_assert(alignof(TileDescriptor) == 32,
              "TileDescriptor must be 32-byte aligned");

// Weight Transfer Descriptor (32 bytes)
/**
 * @brief Flat 32-byte weight transfer descriptor written to DDR for kernel consumption.
 *
 * Byte layout (must match uram_xfer_desc_t in common_defs.h):
 *   0-3:   ddr_offset_vecs   per-bank start offset (IC-vec units)
 *   4-7:   ddr_length_vecs   per-bank burst length (== depth_count)
 *   8-11:  total_layer_depth full bank depth, layer constant
 *   12-13: depth_count       depths loaded in this transfer
 *   14-15: compute_count     compute descriptors fed by this transfer
 *   16:    buffer_id         0=A, 1=B (ping-pong)
 *   17:    flags             bit0 = last transfer
 *   18-31: reserved[14]
 *
 * ABI: This descriptor is intentionally not alignas(32).
 */
struct WeightTransferDescriptor {
  std::uint32_t ddr_offset_vecs   = 0;  ///< Bytes 0–3.
  std::uint32_t ddr_length_vecs   = 0;  ///< Bytes 4–7.
  std::uint32_t total_layer_depth = 0;  ///< Bytes 8–11.
  std::uint16_t depth_count       = 0;  ///< Bytes 12–13.
  std::uint16_t compute_count     = 0;  ///< Bytes 14–15.
  std::uint8_t  buffer_id         = 0;  ///< Byte 16.
  std::uint8_t  flags             = 0;  ///< Byte 17.
  std::uint8_t  reserved[14]      = {}; ///< Bytes 18–31.
};

static_assert(sizeof(WeightTransferDescriptor) == 32,
              "WeightTransferDescriptor must be exactly 32 bytes");

// Control Flags (TileDescriptor::flags)
enum TileFlags : std::uint8_t {
  ACCUM_RESET      = 1U << 0, ///< Reset accumulator (first Cin group)
  ACCUM_WRITE_BACK = 1U << 1, ///< Write to DDR (last Cin group)
  RELU_ON          = 1U << 2, ///< Apply ReLU activation
  IS_LAST_TILE     = 1U << 3, ///< End of layer marker
  ADD_ENABLE       = 1U << 4, ///< Residual-add post-op enabled
  POST_ADD_RELU    = 1U << 5, ///< Apply ReLU after residual add
  POOL_ENABLE      = 1U << 6, ///< Apply fused maxpool post-op
  GAP_ENABLE       = 1U << 7  ///< Apply fused global average pool post-op
};

/// Transfer descriptor flag bits.
enum TransferFlags : std::uint8_t {
  XFER_LAST = 1U << 0 ///< Final weight transfer of the layer
};

// Packed-Padding Helpers

/// Encode 4 padding values (each 0–15) into a packed_padding field.
inline std::uint16_t pack_padding(std::uint8_t top, std::uint8_t bottom,
                                  std::uint8_t left, std::uint8_t right) {
  return static_cast<std::uint16_t>(
      ((top & 0xFU) << 12) | ((bottom & 0xFU) << 8) |
      ((left & 0xFU) << 4) | (right & 0xFU));
}

/// Decode a packed_padding field into individual values.
inline void unpack_padding(std::uint16_t packed, std::uint8_t &top,
                           std::uint8_t &bottom, std::uint8_t &left,
                           std::uint8_t &right) {
  top    = static_cast<std::uint8_t>((packed >> 12) & 0xFU);
  bottom = static_cast<std::uint8_t>((packed >> 8) & 0xFU);
  left   = static_cast<std::uint8_t>((packed >> 4) & 0xFU);
  right  = static_cast<std::uint8_t>(packed & 0xFU);
}

} // namespace tile
} // namespace deeptile
