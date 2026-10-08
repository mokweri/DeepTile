/**
 * @file cost_model.cpp
 * @brief Diagnostic analytical cost-model implementation.
 *
 * Key unit note: WeightEpoch.weight_vectors holds the per-OC-bank depth
 * (= oc_tile_count * num_cig * K²), not the total vector count. The output-channel
 * banks share one memory port, so load cycles include the bank count.
 */

#include "deeptile/tile/cost_model.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace deeptile {
namespace tile {

namespace {

inline int ceil_div(int a, int b) { return b > 0 ? (a + b - 1) / b : 0; }
inline long lround_pos(double x) { return static_cast<long>(std::llround(x)); }

} // namespace

const char* bound_name(ScheduleCost::Bound b) {
    switch (b) {
        case ScheduleCost::Bound::MAC:    return "MAC";
        case ScheduleCost::Bound::Weight: return "Weight";
        case ScheduleCost::Bound::IFM:    return "IFM";
    }
    return "?";
}

ScheduleCost estimate_cost(const SchedulePlan& plan, const CostModelConstants& k) {
    ScheduleCost c;
    if (!plan.valid) return c;

    const ConvLayerParams& L = plan.layer;
    const int p_spatial = plan.hw.p_spatial;
    const int oc_tile   = plan.hw.oc_tile_size;
    const int K2        = L.Kh * L.Kw;
    const int num_cig   = L.num_cig();
    const bool k1       = (L.Kh == 1 && L.Kw == 1);
    const bool dw       = L.depthwise;

    const int E = static_cast<int>(plan.weight_epochs.size());
    std::vector<long> mac_epoch(static_cast<size_t>(E > 0 ? E : 1), 0);

    // Accumulate per-task compute, post-processing, and IFM demand.
    for (const ComputeTask& t : plan.compute_tasks) {
        const SpatialTile& st = plan.spatial_tiles[static_cast<size_t>(t.spatial_idx)];
        const int num_windows = st.conv_tile_oh * st.conv_tile_ow;
        // The kernel groups K>1 data per row and K=1 data across the full tile.
        //   row-aligned: oh * ceil(ow / P)  (>= flat by the per-row padded lanes)
        //   flat:        ceil(oh*ow / P)
        const int num_groups  = k1
            ? ceil_div(num_windows, p_spatial)
            : st.conv_tile_oh * ceil_div(st.conv_tile_ow, p_spatial);

        const long compute = static_cast<long>(t.cig_count) * num_groups * K2;
        const long gap     = lround_pos(static_cast<double>(t.cig_count) * k.cig_gap);
        // EMIT_LOOP fires once per COG (writeback), i.e. on the task covering the
        // final CIG — not once per CIG. (Chained: every task; unchained: last only.)
        // Depthwise: every task is a self-contained single-CIG reduction that
        // writes back (cig_begin == cog, not a CIG sweep), so it always emits.
        const bool writeback = dw || (t.cig_begin + t.cig_count >= num_cig);
        const long emit = writeback
            ? lround_pos(static_cast<double>(num_groups) * k.emit_ii) : 0;

        c.mac_floor_cy += compute;
        c.acc_emit_cy  += emit;
        c.cig_gap_cy   += gap;
        if (t.epoch_idx >= 0 && t.epoch_idx < E)
            mac_epoch[static_cast<size_t>(t.epoch_idx)] += compute + emit + gap;

        // IFM (BW-bound, one 128-bit vec/cy).
        const InputWindow& w = st.ifm_window;
        if (k1) {
            // K=1 cache: only the fill task (reuse_ifm == false) reads DDR; the
            // reuse tasks replay from URAM (hidden). Fill reads one input vec per
            // output position (K=1 maps output→input 1:1, incl. stride), so the
            // DDR read is num_cig·num_windows — NOT the read_rows×read_cols
            // bounding box, which over-counts strided rows.
            if (!t.reuse_ifm) {
                const long fill_vecs = static_cast<long>(t.cig_count) * num_windows;
                c.ifm_read_cy += fill_vecs;
                const long fill_cy = lround_pos(static_cast<double>(fill_vecs) * k.fill_scale);
                c.cog0_fill_stall_cy += std::max(0L, fill_cy - compute);
            }
        } else {
            // Without caching, each task reloads its padded input tile.
            const long load_vecs =
                static_cast<long>(t.cig_count) * w.total_height * w.total_width;
            c.ifm_read_cy += load_vecs;
        }
    }

    // Output-channel banks share the weight port.
    for (const WeightEpoch& ep : plan.weight_epochs)
        c.weight_read_cy += static_cast<long>(oc_tile) * ep.weight_vectors;
    c.weight_prime_cy = E > 0
        ? static_cast<long>(oc_tile) * plan.weight_epochs[0].weight_vectors
        : 0;
    // Epoch e loads during epoch e-1 compute; excess load time stalls the swap.
    for (int e = 1; e < E; ++e) {
        const long wt_e =
            static_cast<long>(oc_tile) * plan.weight_epochs[static_cast<size_t>(e)].weight_vectors;
        c.inter_epoch_stall_cy += std::max(0L, wt_e - mac_epoch[static_cast<size_t>(e - 1)]);
    }
    // Each ping-pong boundary exposes a fixed fill, swap, and drain overhead.
    c.epoch_overhead_cy = (E > 1)
        ? static_cast<long>(E - 1) * lround_pos(k.epoch_overhead) : 0;

    // Compose serialized and overlapped terms.
    const long mac_work = c.mac_floor_cy + c.acc_emit_cy + c.cig_gap_cy;
    const long drain = static_cast<long>(plan.stats.final_oh) * plan.stats.final_ow
                     + lround_pos(k.drain_const);
    long body;
    if (k1) {
        // Weight/MAC-bound: serial prime + MAC work + the two stalls.
        body = mac_work + c.cog0_fill_stall_cy + c.inter_epoch_stall_cy;
    } else {
        // K>1 input loading and MAC execution overlap; the larger term dominates.
        c.exposed_ifm_cy = std::max(0L, c.ifm_read_cy - mac_work);
        body = std::max(c.ifm_read_cy, mac_work);
    }
    c.predicted_cy = c.weight_prime_cy + body + drain + c.epoch_overhead_cy;
    c.predicted_ms = static_cast<double>(c.predicted_cy) / (k.clock_mhz * 1.0e3);

    // Classify the dominant resource demand.
    if (c.weight_read_cy >= c.ifm_read_cy && c.weight_read_cy >= c.mac_floor_cy)
        c.bound_by = ScheduleCost::Bound::Weight;
    else if (c.ifm_read_cy >= c.weight_read_cy && c.ifm_read_cy >= c.mac_floor_cy)
        c.bound_by = ScheduleCost::Bound::IFM;
    else
        c.bound_by = ScheduleCost::Bound::MAC;

    // Derive roofline metrics from useful work and DDR traffic.
    // Depthwise reduces over one input channel per output channel, so useful
    // MACs are Cout·K²·oh·ow (not Cout·Cin·K²), and the DDR weight bytes are the
    // 16×-inflated diagonal packing = OC_TILE_SIZE·oc_tiles·K²·IC_TILE_SIZE.
    const long useful = dw
        ? static_cast<long>(L.Cout) * K2 * L.output_height() * L.output_width()
        : static_cast<long>(L.Cout) * L.Cin * K2
              * L.output_height() * L.output_width();
    const long wt_bytes  = dw
        ? static_cast<long>(oc_tile) * L.num_cog() * K2 * oc_tile * L.elem_bytes
        : static_cast<long>(L.Cout) * L.Cin * K2 * L.elem_bytes;
    const long ifm_bytes = lround_pos(static_cast<double>(c.ifm_read_cy) * k.bytes_per_cycle);
    const long ofm_bytes = static_cast<long>(L.Cout)
                         * plan.stats.final_oh * plan.stats.final_ow * L.elem_bytes;
    const long useful_ifm = static_cast<long>(L.Cin) * L.H * L.W * L.elem_bytes;
    c.ddr_bytes = wt_bytes + ifm_bytes + ofm_bytes
                + (plan.post_ops.residual_add_enable ? ofm_bytes : 0);
    c.ifm_overfetch = useful_ifm > 0
        ? static_cast<double>(ifm_bytes) / static_cast<double>(useful_ifm) : 0.0;
    c.op_intensity = c.ddr_bytes > 0
        ? static_cast<double>(useful) / static_cast<double>(c.ddr_bytes) : 0.0;
    c.array_eff = c.predicted_cy > 0
        ? static_cast<double>(c.mac_floor_cy) / static_cast<double>(c.predicted_cy) : 0.0;
    c.predicted_gops = c.predicted_cy > 0
        ? 2.0 * static_cast<double>(useful) / static_cast<double>(c.predicted_cy)
            * k.clock_mhz / 1.0e3
        : 0.0;

    return c;
}

void print_cost(const SchedulePlan& plan, const ScheduleCost& cost) {
    std::cout << "\nCost model (predicted)\n";
    for (int i = 0; i < 60; ++i) std::cout << '-';
    std::cout << '\n';

    if (!plan.valid) { std::cout << "  [plan invalid]\n"; return; }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  bound_by:        " << bound_name(cost.bound_by)
              << "    predicted: " << cost.predicted_cy << " cy ("
              << cost.predicted_ms << " ms)   "
              << std::setprecision(1) << cost.predicted_gops << " GOPS\n"
              << std::setprecision(3);

    std::cout << "  resource totals: mac_floor=" << cost.mac_floor_cy
              << "  weight_read=" << cost.weight_read_cy
              << "  ifm_read=" << cost.ifm_read_cy << "\n";
    std::cout << "  weight:          prime=" << cost.weight_prime_cy
              << "  inter_epoch_stall=" << cost.inter_epoch_stall_cy << "\n";
    std::cout << "  ifm:             over-fetch=" << std::setprecision(1)
              << cost.ifm_overfetch << "x"
              << "  cog0_fill_stall=" << cost.cog0_fill_stall_cy
              << "  exposed_ifm=" << cost.exposed_ifm_cy << "\n";
    std::cout << "  post-proc:       acc_emit=" << cost.acc_emit_cy
              << "  cig_gap=" << cost.cig_gap_cy << "\n";
    std::cout << "  roofline:        intensity=" << std::setprecision(1)
              << cost.op_intensity << " MAC/B"
              << "  array_eff=" << std::setprecision(0) << (cost.array_eff * 100.0) << "%"
              << "  ddr=" << cost.ddr_bytes << " B\n";
    std::cout << std::setprecision(6);
}

} // namespace tile
} // namespace deeptile
