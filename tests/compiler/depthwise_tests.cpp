/**
 * @file depthwise_tests.cpp
 * @brief P5.0 unit test: verify the diagonal-dense depthwise lowering.
 *
 * Depthwise convolution is lowered as "diagonal dense":
 * one descriptor per (spatial tile, channel group g), pairing CIG g with COG g,
 * num_cigs = 1, RESET|WRITE_BACK on every descriptor, IFM/OFM at plane g, and
 * uram_base_depth = local_g · K² (no IC-tile dimension). This test drives
 * build_schedule() + emit_descriptors() on a depthwise layer and asserts each of
 * those descriptor fields, plus the weight-epoch depth geometry. It never runs the
 * kernel — it locks the scheduler contract the dense datapath relies on.
 */

#include "deeptile/tile/descriptor_emitter.hpp"
#include "deeptile/tile/schedule_builder.hpp"

#include <iostream>
#include <string>

using namespace deeptile::tile;

static int g_failures = 0;
static int g_tests    = 0;

static void check(bool cond, const std::string &what) {
    ++g_tests;
    if (!cond) {
        std::cerr << "  [FAIL] " << what << "\n";
        ++g_failures;
    }
}

static ConvLayerParams make_dw(int C, int H, int W, int K, int S, int pad) {
    ConvLayerParams p{};
    p.H = H; p.W = W; p.Cin = C; p.Cout = C;
    p.Kh = K; p.Kw = K; p.Sh = S; p.Sw = S;
    p.pad_top = p.pad_bottom = p.pad_left = p.pad_right = pad;
    p.Cpi = 16; p.Cpo = 16; p.elem_bytes = 1;
    p.frac_in = 4; p.frac_w = 6; p.frac_b = 4; p.frac_out = 4;
    p.use_relu = true;
    p.depthwise = true;
    return p;
}

static TileConfig make_cfg(int T_oh, int T_ow) {
    TileConfig cfg{};
    cfg.T_oh = T_oh; cfg.T_ow = T_ow; cfg.T_cog = 1; cfg.T_cig = 1;
    cfg.Cpi = 16; cfg.Cpo = 16;
    return cfg;
}

// A depthwise layer that fits one spatial tile: C=48 (3 channel groups),
// 16×16, K=3 s1 p1. Expect 3 groups × 1 spatial tile = 3 descriptors.
static void test_single_tile_dw() {
    std::cout << "[TEST] depthwise single-tile lowering (C=48, 16x16, K3 s1 p1)\n";
    const ConvLayerParams p = make_dw(48, 16, 16, 3, 1, 1);
    const TileConfig cfg = make_cfg(16, 16);

    // Force a single epoch so uram_base_depth is layer-absolute (local_g == g);
    // the diagonal weight-depth assertion below is then unambiguous. (Multi-epoch
    // correctness — that uram_base_depth is epoch-local — is exercised implicitly
    // by the Pε path in the other cases.)
    ScheduleOptions opts;
    opts.optimize_epoch_depth = false;
    opts.forced_epochs = 1;
    SchedulePlan plan = build_schedule(p, cfg, HardwareSpec{}, opts);
    check(plan.valid, "plan is valid");
    if (!plan.valid) return;
    check(plan.weight_epochs.size() == 1, "single forced epoch");

    // depth_per_oc must be K² (not num_cig·K²) for depthwise.
    for (const auto &ep : plan.weight_epochs) {
        const int oc_tiles_in_ep = ep.oc_tile_count;
        check(ep.depth_count == oc_tiles_in_ep * 9,
              "epoch depth_count == oc_tile_count · K²");
        check(ep.total_layer_depth == 3 * 9,
              "epoch total_layer_depth == num_cog · K² (= 27)");
    }

    DescriptorBundle bundle = emit_descriptors(plan);
    check(bundle.valid, "descriptor bundle is valid");
    check(bundle.compute_descs.size() == 3,
          "3 descriptors (3 groups × 1 spatial tile)");

    const uint32_t oh = static_cast<uint32_t>(p.output_height());   // 16
    const uint32_t ow = static_cast<uint32_t>(p.output_width());    // 16
    for (std::size_t i = 0; i < bundle.compute_descs.size(); ++i) {
        const auto &d = bundle.compute_descs[i];
        const uint32_t g = static_cast<uint32_t>(i);  // one tile, group order
        check(d.num_cigs == 1, "num_cigs == 1");
        check((d.flags & ACCUM_RESET) != 0, "ACCUM_RESET set");
        check((d.flags & ACCUM_WRITE_BACK) != 0, "ACCUM_WRITE_BACK set");
        check((d.flags & RELU_ON) != 0, "RELU_ON set (use_relu)");
        // IFM plane g: ifm_offset(0,0,cg=g) = g·H·W (H=W=16, full tile from origin).
        check(d.ifm_offset_vecs == g * 16u * 16u, "ifm_offset == plane g");
        // OFM plane g: ofm_offset(0,0,cog=g) = g·oh·ow.
        check(d.ofm_offset_vecs == g * oh * ow, "ofm_offset == plane g");
        // Diagonal weight base: local_g · K² (single epoch here → local_g == g).
        check(d.uram_base_depth == static_cast<uint16_t>(g * 9u),
              "uram_base_depth == local_g · K²");
        check(d.bias_offset_idx == static_cast<uint16_t>(g),
              "bias_offset_idx == g");
        check(d.ifm_ddr_stride == 16, "ifm_ddr_stride == W");
    }
    // IS_LAST_TILE on the final descriptor.
    check((bundle.compute_descs.back().flags & IS_LAST_TILE) != 0,
          "IS_LAST_TILE on last descriptor");
}

// A depthwise layer needing spatial tiling: C=16 (one group), 40×40, K3 s1 p1.
// 40×40 > one 32×64 tile in height → multiple spatial tiles, one group each.
static void test_tiled_dw() {
    std::cout << "[TEST] depthwise multi-tile lowering (C=16, 40x40, K3 s1 p1)\n";
    const ConvLayerParams p = make_dw(16, 40, 40, 3, 1, 1);
    const TileConfig cfg = make_cfg(32, 40);

    SchedulePlan plan = build_schedule(p, cfg, HardwareSpec{}, ScheduleOptions{});
    check(plan.valid, "plan is valid");
    if (!plan.valid) return;

    DescriptorBundle bundle = emit_descriptors(plan);
    check(bundle.valid, "descriptor bundle is valid");
    // One channel group → #descriptors == #spatial tiles, each RESET|WRITE_BACK.
    check(bundle.compute_descs.size() == plan.spatial_tiles.size(),
          "descriptors == spatial tiles (single group)");
    for (const auto &d : bundle.compute_descs) {
        check(d.num_cigs == 1, "num_cigs == 1");
        check((d.flags & ACCUM_RESET) != 0, "ACCUM_RESET set");
        check((d.flags & ACCUM_WRITE_BACK) != 0, "ACCUM_WRITE_BACK set");
        check(d.uram_base_depth == 0, "single group → uram_base_depth == 0");
    }
}

// Ragged channel count (not a multiple of 16): C=24 → 2 OC tiles, second half-full.
static void test_ragged_channels_dw() {
    std::cout << "[TEST] depthwise ragged channels (C=24, 8x8, K3 s1 p1)\n";
    const ConvLayerParams p = make_dw(24, 8, 8, 3, 1, 1);
    check(p.num_cog() == 2, "C=24 → 2 OC tiles");
    check(p.num_cig() == 2, "C=24 → 2 IC tiles (== OC for depthwise)");

    SchedulePlan plan = build_schedule(p, make_cfg(8, 8), HardwareSpec{},
                                       ScheduleOptions{});
    check(plan.valid, "plan is valid");
    if (!plan.valid) return;
    DescriptorBundle bundle = emit_descriptors(plan);
    check(bundle.valid, "descriptor bundle is valid");
    check(bundle.compute_descs.size() == 2, "2 descriptors (2 groups × 1 tile)");
}

int main() {
    std::cout << "DeepTile depthwise (P5.0) lowering tests\n";
    for (int i = 0; i < 62; ++i) std::cout << '-';
    std::cout << "\n";

    test_single_tile_dw();
    test_tiled_dw();
    test_ragged_channels_dw();

    for (int i = 0; i < 62; ++i) std::cout << '-';
    std::cout << "\n";
    if (g_failures == 0) {
        std::cout << "OK: all " << g_tests << " depthwise checks passed\n";
        return 0;
    }
    std::cerr << "FAIL: " << g_failures << " / " << g_tests
              << " depthwise checks failed\n";
    return 1;
}
