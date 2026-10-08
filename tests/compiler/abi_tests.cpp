/**
 * @file abi_tests.cpp
 * @brief ABI lock for the 32-byte tile_conv descriptors.
 *
 * The packed descriptors in descriptors.hpp are the byte-for-byte execution
 * contract with the `tile_conv` kernel. This test pins every field to its
 * absolute byte offset, so any accidental reordering or padding change fails to
 * COMPILE — the strongest possible gate. The expected offsets are the documented
 * layout in descriptors.hpp; they must also match `descriptor_dt` /
 * `uram_xfer_desc_t` in src/common/common_defs.h.
 *
 * Self-contained: depends only on the deeptile::tile headers.
 */

#include "deeptile/tile/descriptors.hpp"
#include "deeptile/tile/hardware_spec.hpp"
#include "deeptile/tile/layer.hpp"
#include "kernel_abi.h"

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace nt = deeptile::tile;

// ── Compile-time layout lock: every field at its contracted byte offset ──────
#define AT_OFFSET(STRUCT, field, off)                                           \
  static_assert(offsetof(STRUCT, field) == (off),                              \
                "ABI offset drift: " #STRUCT "::" #field)

// TileDescriptor (32 bytes) — see descriptors.hpp byte map.
static_assert(sizeof(nt::TileDescriptor) == DEEPTILE_COMPUTE_DESCRIPTOR_BYTES,
              "TileDescriptor must match the kernel ABI");
static_assert(alignof(nt::TileDescriptor) == 32, "TileDescriptor must be 32B aligned");
AT_OFFSET(nt::TileDescriptor, ifm_offset_vecs,      0);
AT_OFFSET(nt::TileDescriptor, ifm_ddr_h,            4);
AT_OFFSET(nt::TileDescriptor, ifm_ddr_w,            5);
AT_OFFSET(nt::TileDescriptor, ifm_ddr_stride,       6);
AT_OFFSET(nt::TileDescriptor, ifm_tile_h,           7);
AT_OFFSET(nt::TileDescriptor, ifm_tile_w,           8);
AT_OFFSET(nt::TileDescriptor, reuse_ifm,            9);
AT_OFFSET(nt::TileDescriptor, packed_padding,      10);
AT_OFFSET(nt::TileDescriptor, ofm_offset_vecs,     12);
AT_OFFSET(nt::TileDescriptor, ofm_tile_h,          16);
AT_OFFSET(nt::TileDescriptor, ofm_tile_w,          17);
AT_OFFSET(nt::TileDescriptor, ofm_row_stride,      18);
AT_OFFSET(nt::TileDescriptor, flags,               19);
AT_OFFSET(nt::TileDescriptor, uram_base_depth,     20);
AT_OFFSET(nt::TileDescriptor, bias_offset_idx,     22);
AT_OFFSET(nt::TileDescriptor, residual_shift,      24);
AT_OFFSET(nt::TileDescriptor, pooled_ofm_tile_h,   25);
AT_OFFSET(nt::TileDescriptor, pooled_ofm_tile_w,   26);
AT_OFFSET(nt::TileDescriptor, post_pool_shift,     27);
AT_OFFSET(nt::TileDescriptor, pool_packed_padding, 28);
AT_OFFSET(nt::TileDescriptor, num_cigs,            30);
AT_OFFSET(nt::TileDescriptor, cache_cig_idx,       31);

// WeightTransferDescriptor (32 bytes).
static_assert(sizeof(nt::WeightTransferDescriptor) ==
                  DEEPTILE_TRANSFER_DESCRIPTOR_BYTES,
              "WeightTransferDescriptor must match the kernel ABI");
static_assert(DEEPTILE_KERNEL_ARGUMENT_COUNT == 23,
              "tile_conv argument count must match the kernel ABI");
AT_OFFSET(nt::WeightTransferDescriptor, ddr_offset_vecs,    0);
AT_OFFSET(nt::WeightTransferDescriptor, ddr_length_vecs,    4);
AT_OFFSET(nt::WeightTransferDescriptor, total_layer_depth,  8);
AT_OFFSET(nt::WeightTransferDescriptor, depth_count,       12);
AT_OFFSET(nt::WeightTransferDescriptor, compute_count,     14);
AT_OFFSET(nt::WeightTransferDescriptor, buffer_id,         16);
AT_OFFSET(nt::WeightTransferDescriptor, flags,             17);
AT_OFFSET(nt::WeightTransferDescriptor, reserved,          18);

// Control-flag bit values are part of the ABI.
static_assert(nt::ACCUM_RESET      == 0x01, "ACCUM_RESET bit");
static_assert(nt::ACCUM_WRITE_BACK == 0x02, "ACCUM_WRITE_BACK bit");
static_assert(nt::RELU_ON          == 0x04, "RELU_ON bit");
static_assert(nt::IS_LAST_TILE     == 0x08, "IS_LAST_TILE bit");
static_assert(nt::ADD_ENABLE       == 0x10, "ADD_ENABLE bit");
static_assert(nt::POST_ADD_RELU    == 0x20, "POST_ADD_RELU bit");
static_assert(nt::POOL_ENABLE      == 0x40, "POOL_ENABLE bit");
static_assert(nt::GAP_ENABLE       == 0x80, "GAP_ENABLE bit");
static_assert(nt::XFER_LAST        == 0x01, "XFER_LAST bit");

// ── Runtime checks ───────────────────────────────────────────────────────────
namespace {

int g_failures = 0;

void check(bool cond, const char *what) {
  if (!cond) {
    std::cerr << "  [FAIL] " << what << "\n";
    ++g_failures;
  }
}

void test_packed_padding() {
  const std::uint16_t packed = nt::pack_padding(1, 2, 3, 4);
  std::uint8_t t, b, l, r;
  nt::unpack_padding(packed, t, b, l, r);
  check(t == 1 && b == 2 && l == 3 && r == 4, "pack/unpack_padding round-trip");
  // Nibble packing [top|bot|left|right]: 0x1234.
  check(packed == 0x1234, "pack_padding nibble encoding");
}

void test_hardware_spec() {
  const nt::HardwareSpec current;
  check(current.p_spatial == P_SPATIAL, "default spatial width follows arch_config");
  check(current.psum_depth == MAX_SPATIAL_TILE_SIZE / P_SPATIAL,
        "default psum depth follows arch_config");
  const auto pp2 = current.with_p_spatial(2);
  check(pp2.macs_per_cycle() == 512, "PP=2 macs_per_cycle == 512");
  check(pp2.psum_depth == 1024, "PP=2 psum_depth re-derived == 1024");
  const auto hw = current.with_p_spatial(4);
  check(hw.macs_per_cycle() == 1024, "PP=4 macs_per_cycle == 1024");
  check(hw.psum_depth == 512, "PP=4 psum_depth == 512");

  const nt::HardwareSpec pp8 = hw.with_p_spatial(8);
  check(pp8.macs_per_cycle() == 2048, "PP=8 macs_per_cycle == 2048");
  check(pp8.psum_depth == 256, "PP=8 psum_depth re-derived == 256");

  // 200 MHz structural peaks.
  check(static_cast<int>(hw.peak_gops(200.0) + 0.5) == 410,
        "PP=4 peak ~409.6 GOPS @200MHz");
  check(static_cast<int>(pp8.peak_gops(200.0) + 0.5) == 819,
        "PP=8 peak ~819.2 GOPS @200MHz");
}

} // namespace

int main() {
  std::cout << "DeepTile compiler descriptor ABI lock\n";
  test_packed_padding();
  test_hardware_spec();

  if (g_failures == 0) {
    std::cout << "OK: all ABI + HardwareSpec checks passed "
                 "(32-byte layout pinned at compile time)\n";
    return 0;
  }
  std::cerr << g_failures << " check(s) failed\n";
  return 1;
}
