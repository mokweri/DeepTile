/**
 * @file cost_model.hpp
 * @brief Diagnostic analytical cycle-cost model for a schedule plan.
 * @ingroup deeptile_compiler
 */

#pragma once

#include "deeptile/tile/schedule.hpp"

namespace deeptile {
namespace tile {

/**
 * @brief Tunable analytical-model constants.
 */
struct CostModelConstants {
    double clock_mhz       = 200.0;  ///< Model clock frequency.
    double bytes_per_cycle = 16.0;   ///< Memory throughput in bytes per cycle.
    double emit_ii         = 4.0;    ///< Quantization emit initiation interval.
    double cig_gap         = 3.0;    ///< Control gap between input-channel groups.
    double fill_scale      = 1.6;    ///< K=1 cache cold-start scale.
    double drain_const     = 500.0;  ///< Fixed pipeline drain in cycles.
    double epoch_overhead  = 256.0;  ///< Ping-pong boundary overhead per extra epoch.
};

/**
 * @brief Cycle and roofline estimates for one layer.
 */
struct ScheduleCost {
    enum class Bound { MAC, Weight, IFM };

    // Resource totals used to classify the bound.
    long mac_floor_cy   = 0;  ///< Σ_tasks compute (= ideal MAC-feed cycles).
    long weight_read_cy = 0;  ///< OC_TILE_SIZE · Σ epoch.weight_vectors (BW-bound, §3.1).
    long ifm_read_cy    = 0;  ///< BW-bound DDR IFM reads (K=1 fill once, or K>1 over-fetch).

    // Exposed and serialized terms.
    long weight_prime_cy      = 0;  ///< First epoch, exposed before any MAC.
    long inter_epoch_stall_cy = 0;  ///< Σ_{e≥1} max(0, wt_epoch[e] − mac_epoch[e-1]) (§1.2b).
    long epoch_overhead_cy    = 0;  ///< Fixed boundary bubbles for extra epochs.
    long cog0_fill_stall_cy   = 0;  ///< K=1 cache cold-start drip-feed (§1.2a, calibrated).
    long acc_emit_cy          = 0;  ///< Σ num_groups·emit_ii (EMIT_LOOP).
    long cig_gap_cy           = 0;  ///< Σ num_cig·cig_gap.
    long exposed_ifm_cy       = 0;  ///< K>1: max(0, ifm_read − mac_work).

    // Composed prediction.
    long   predicted_cy = 0;
    double predicted_ms = 0.0;

    // Verdict and roofline metrics.
    Bound  bound_by      = Bound::MAC;
    double ifm_overfetch = 1.0;  ///< ifm DDR bytes / single-load ifm bytes (H1: ~num_cog).
    long   ddr_bytes     = 0;    ///< weight + ifm(actual) + ofm (+ residual).
    double op_intensity  = 0.0;  ///< useful MAC / DDR byte.
    double array_eff     = 0.0;  ///< mac_floor / predicted (MAC-feed fraction).
    double predicted_gops = 0.0; ///< 2·useful_MAC / predicted_cy · clock.
};

/// Human-readable name for a Bound enum value.
const char* bound_name(ScheduleCost::Bound b);

/// Estimate the cycle cost of a built SchedulePlan (analysis §1–§2).
ScheduleCost estimate_cost(const SchedulePlan& plan,
                           const CostModelConstants& k = CostModelConstants{});

/// Print a compact cost report to stdout (mirrors print_descriptor_table style).
void print_cost(const SchedulePlan& plan, const ScheduleCost& cost);

} // namespace tile
} // namespace deeptile
