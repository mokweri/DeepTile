/**
 * @file cost_model_tests.cpp
 * @brief P0 evidence gate: cost model vs measured board cycles for the 7 P0 cases.
 *
 * The analytical cost model (src/cost_model.cpp) is validated against the XRT
 * board measurements retained with the paper artifacts and reproduced by
 * scripts/tools/p0_metrics.py.
 *
 * IMPORTANT: these are the *true P0 testcase dimensions*, which differ from the
 * same-named layers in tools/schedule_inspect_main.cpp and tests/golden_tests.cpp
 * (e.g. golden `best_case_3x3_s1` is 56×56 64→64; P0 TC1 is 28×28 128→128). Do not
 * confuse them.
 *
 * Two assertions per case:
 *   (1) bound_by matches the measured class (hard gate — robust/board-grounded).
 *   (2) predicted_cy within ±20% of board cycles (calibration tolerance; the
 *       COG-0 fill-stall is the one noisy term).
 */

#include "deeptile/tile/cost_model.hpp"
#include "deeptile/tile/schedule_builder.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace deeptile::tile;

// =============================================================================
// Minimal test framework (mirrors golden_tests.cpp)
// =============================================================================
static int g_failures = 0;
static int g_tests    = 0;

static void check(bool cond, const std::string& what) {
    ++g_tests;
    if (!cond) {
        std::cerr << "  [FAIL] " << what << "\n";
        ++g_failures;
    }
}

// =============================================================================
// P0 testcase table — true dims + measured board cycles (analysis §4)
// =============================================================================
struct P0Case {
    const char* name;
    int H, W, Cin, Cout, K, S, pad;
    int T_oh, T_ow;
    bool gap;                      // tail_gap FC
    long board_cy;                 // measured @ 200 MHz
    ScheduleCost::Bound expect;    // measured bound-by class
};

static SchedulePlan build_p0(const P0Case& tc) {
    ConvLayerParams p{};
    p.H = tc.H; p.W = tc.W; p.Cin = tc.Cin; p.Cout = tc.Cout;
    p.Kh = tc.K; p.Kw = tc.K; p.Sh = tc.S; p.Sw = tc.S;
    p.pad_top = tc.pad; p.pad_bottom = tc.pad;
    p.pad_left = tc.pad; p.pad_right = tc.pad;
    p.Cpi = 16; p.Cpo = 16; p.elem_bytes = 1;
    p.frac_in = 5; p.frac_w = 8; p.frac_b = 5; p.frac_out = 3;
    p.use_relu = !tc.gap;

    TileConfig cfg{};
    cfg.T_oh = tc.T_oh; cfg.T_ow = tc.T_ow;
    cfg.T_cog = 1; cfg.T_cig = 1;
    cfg.Cpi = 16; cfg.Cpo = 16;

    PostOpConfig post{};
    post.gap_enable = tc.gap;

    // Pin the spatial width and weight-buffer depth to the measured hardware.
    // The 8 board measurements used P4 and the original 4096-deep URAM weight
    // buffer with the packed (Phase2Default) policy; the shipped default is now
    // WEIGHT_BUF_DEPTH=512 (BRAM ping-pong), which legitimately changes the
    // packed epoch split. This test checks MODEL vs SILICON, so it reproduces the
    // silicon config, not the current default.
    HardwareSpec hw = HardwareSpec{}.with_p_spatial(4);
    hw.weight_buf_depth = 4096;

    return build_schedule(p, cfg, hw,
                          options_for_policy(SchedulePolicy::Phase2Default),
                          0, 0, 0, post);
}

int main() {
    using B = ScheduleCost::Bound;

    // Dimensions and board cycles are reproduced by p0_metrics.py.
    const std::vector<P0Case> cases = {
        {"TC1 best_case_3x3_s1",   28, 28,  128,  128, 3, 1, 1, 28, 28, false, 126475, B::MAC},
        {"TC2 deep_k1_s1",         14, 14, 1024,  256, 1, 1, 0, 14, 14, false,  80302, B::MAC},
        {"TC3 late_k1_s1",          7,  7, 2048,  512, 1, 1, 0,  7,  7, false, 115648, B::Weight},
        {"TC4 proj_bottleneck",    56, 56,  256,  512, 1, 2, 0, 28, 28, false, 156215, B::MAC},
        {"TC5 small_proj",         14, 14, 1024, 2048, 1, 2, 0,  7,  7, false, 214784, B::Weight},
        {"TC6 stride2_3x3_s2",     56, 56,  128,  128, 3, 2, 1, 16, 16, false, 227600, B::IFM},
        {"TC7c tail_gap conv3",     7,  7,  512, 2048, 1, 1, 0,  7,  7, false, 109684, B::Weight},
        {"TC7f tail_gap FC",        1,  1, 2048, 1000, 1, 1, 0,  1,  1, true,  146041, B::Weight},
    };

    std::cout << "P0 cost-model validation vs measured board cycles\n";
    std::cout << std::string(78, '-') << "\n";
    std::cout << std::left << std::setw(24) << "  Case"
              << std::setw(9) << "bound"
              << std::setw(11) << "pred_cy"
              << std::setw(11) << "board_cy"
              << std::setw(9) << "err%" << "verdict\n";
    std::cout << std::string(78, '-') << "\n";

    for (const auto& tc : cases) {
        SchedulePlan plan = build_p0(tc);
        check(plan.valid, std::string(tc.name) + ": schedule valid");
        if (!plan.valid) continue;

        const ScheduleCost cost = estimate_cost(plan);
        const double err = 100.0 * (static_cast<double>(cost.predicted_cy) - static_cast<double>(tc.board_cy))
                         / static_cast<double>(tc.board_cy);
        const bool bound_ok = (cost.bound_by == tc.expect);
        const bool err_ok   = std::fabs(err) <= 20.0;

        std::cout << std::left << std::setw(24) << ("  " + std::string(tc.name))
                  << std::setw(9) << bound_name(cost.bound_by)
                  << std::setw(11) << cost.predicted_cy
                  << std::setw(11) << tc.board_cy
                  << std::setw(9) << (std::to_string(static_cast<int>(std::lround(err))) + "%")
                  << (bound_ok && err_ok ? "OK" : "FAIL")
                  << (bound_ok ? "" : " [bound]")
                  << (err_ok ? "" : " [err>20%]")
                  << "\n";

        check(bound_ok, std::string(tc.name) + ": bound_by == "
              + bound_name(tc.expect) + " (got " + bound_name(cost.bound_by) + ")");
        check(err_ok, std::string(tc.name) + ": predicted within ±20% of board");
    }

    std::cout << std::string(78, '-') << "\n";
    if (g_failures == 0) {
        std::cout << "OK: all " << g_tests << " checks passed\n";
        return 0;
    }
    std::cerr << g_failures << "/" << g_tests << " checks FAILED\n";
    return 1;
}
