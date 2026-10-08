/**
 * @file cost_model_v2_1.hpp
 * @brief Synthesis-grounded schedule cost model for the current kernel.
 * @ingroup deeptile_compiler
 *
 * Loop timing comes from a strict synthesis manifest, while AXI parameters are
 * modeled separately for each execution environment.
 */

#pragma once

#include "deeptile/tile/cost_model.hpp"  // Shared schedule and bound types.

#include <array>
#include <string>

namespace deeptile {
namespace tile {

/**
 * @brief Timing profile for one synthesized pipelined loop.
 *
 * Completion is `first + (N - 1) * ii + tail`. First-token availability and
 * completion remain separate because different dependencies consume them.
 */
struct LoopProfile {
    double ii    = 1.0;  ///< initiation interval
    int    first = 0;    ///< first_result_latency (first token available)
    int    tail  = 0;    ///< tail_latency (last token → completion)

    /// Function completion time for N iterations.
    long done(long n) const {
        if (n <= 0) return 0;
        return static_cast<long>(first + ii * static_cast<double>(n - 1) + tail);
    }
    /// Time the first output token is available (0 iterations → 0).
    long first_token(long n) const { return n <= 0 ? 0 : first; }
    /// Steady output period (one token per II).
    double period() const { return ii; }
};

/**
 * @brief Physical memory-port parameters for an execution environment.
 */
struct AxiPortProfile {
    std::string name = "cosim";
    int    kernel_word_bits   = 256;  ///< widest kernel AXI word (256-bit IFM)
    int    physical_port_bits = 128;  ///< HP port width
    int    max_burst_beats    = 256;
    int    max_outstanding    = 16;
    double address_issue_ii   = 0.0;  ///< cycles between successive burst requests
    double first_beat_latency = 225.0;///< request → first beat (cosim ~225, board ~9)
    double data_beat_ii       = 1.0;  ///< cycles per physical beat once streaming

    /// Physical beats per kernel word (256-bit word over 128-bit port → 2).
    int beats_per_word() const {
        return (kernel_word_bits + physical_port_bits - 1) / physical_port_bits;
    }
};

/// xsim/cosim AXI environment (large fixed per-burst latency, full-width beats).
AxiPortProfile axi_profile_cosim();
/// ZCU104 board environment (small effective per-burst latency, 128-bit beats).
AxiPortProfile axi_profile_board();

/** Synthesized loop profiles and stream capacities used by the cost model. */
struct KernelManifest {
    // Provenance; architecture mismatches are rejected.
    std::string source_commit    = "";        ///< Kernel commit used for profiling.
    std::string hls_version      = "2024.2";
    double      target_clock_ns  = 5.0;        ///< 200 MHz
    std::string architecture_hash = "";        ///< P_SPATIAL/OC_TILE/etc. fingerprint
    std::string provenance       = "csynth-seed"; ///< "csynth-seed" | "extracted"

    // Controller and preload stages.
    LoopProfile schedule_xfer;      ///< FETCH_XFER_LOOP (per weight-transfer desc)
    LoopProfile schedule_tile;      ///< FETCH_LOOP (per compute descriptor)
    LoopProfile bias_preload;       ///< preload_bias_to_bram (per bias vector)

    // Convolution engine.
    LoopProfile weight_depth;       ///< weight_load depth loop (per 128b vector/bank)
    bool        weight_depth_flat = false; ///< BANK_LOOP×DEPTH flattened? (else per-bank restart)
    LoopProfile mac_dense;          ///< MAC_FLAT_LOOP — dense AND standalone-DW lowering
    int         mac_task_entry = 8; ///< per-descriptor control in mac_tile
    /// Restart bubble between input-channel groups in the continuous MAC pipeline.
    int         cig_restart = 4;

    // IFM load and stream stages.
    LoopProfile ifm_load_raw;       ///< K>1 raw-tile load (per 256b word)
    LoopProfile ifm_stream;         ///< window generator (per num_groups·K² element)
    LoopProfile k1_fill;            ///< K=1 cache fill+forward (per 256b word)
    LoopProfile k1_reuse;           ///< K=1 replay (per 4-lane group)

    // Partial-sum accumulation and quantization.
    LoopProfile acc_group;          ///< Accumulate pass per group and CIG restart.
    LoopProfile emit;               ///< EMIT (per output window)

    // Output pipeline.
    LoopProfile residual;           ///< residual_add pass/add (per conv word)
    LoopProfile pool_row;           ///< maxpool padded row/col traversal
    LoopProfile gap_acc;            ///< GAP accumulate (per conv word)
    LoopProfile dw_preload;         ///< fused-DW compact-weight preload (per word)
    LoopProfile dw_local_weight;    ///< get_cog_weights (per writeback tile)
    LoopProfile dw_tile_pix;        ///< DW_TILE_PIX padded raster (per padded pixel)
    LoopProfile store_row;          ///< store_ofm_tile (per OFM word, per-row burst)

    // Stream capacities in tokens.
    int pixel_fifo_depth          = 32;   ///< Per-lane pixel stream.
    int acc_fifo_depth            = 32;   ///< acc_stream
    int intermediate_ofm_fifo_depth = 32; ///< ofm_stream / post/pool/gap streams
    int final_store_fifo_depth    = 512;  ///< dw_stage → store decoupling FIFO
};

/// Return the built-in synthesis seed used when no extracted manifest exists.
KernelManifest default_manifest_csynth();

/// Compile-time architecture fingerprint built from common_defs.h constants; must
/// byte-match a manifest's `arch_hash` for that manifest to be accepted.
std::string architecture_hash_compiled();

/// Load an extracted synthesis manifest and validate required fields.
/// @throws std::runtime_error in strict mode on missing data or hash mismatch.
KernelManifest load_kernel_manifest(const std::string& path, bool strict = true);

/// Return the cached embedded manifest, optionally overridden by DEEPTILE_MANIFEST.
/// A missing override uses the seed; an invalid override raises an error.
const KernelManifest& default_manifest();

enum class ResourceBound { MAC, Weight, IFM, PostOp, Store };
enum class CriticalCause {
    Controller, BiasPreload, WeightPrime, InterEpoch, IfmReady, MacService,
    AccPsum, Emit, PostOp, FusedDw, Store, Drain, Launch, NumCauses
};

struct CostV2_1 {
    long   predicted_cy = 0;
    double predicted_ms = 0.0;

    ResourceBound resource_bound = ResourceBound::MAC;   ///< Total-work roofline bound.
    CriticalCause critical_bound = CriticalCause::MacService; ///< Finish-time dependency.

    // Actor busy totals, including overlapped work.
    long mac_busy_cy    = 0;
    long weight_busy_cy = 0;   ///< DDR weight-port busy (incl. burst α)
    long ifm_busy_cy    = 0;   ///< DDR IFM-port busy
    long emit_busy_cy   = 0;
    long dw_busy_cy     = 0;
    long store_busy_cy  = 0;

    // Critical-path contributions; their sum approximates predicted_cy.
    std::array<long, static_cast<size_t>(CriticalCause::NumCauses)> critical_path_cy{};
    long mac_floor_cy = 0;  ///< ideal MAC-feed cycles (II·trips)

    long   useful_macs = 0;
    double predicted_gops = 0.0;
    double mac_array_active_fraction = 0.0;

    // Diagnostic-model-compatible verdict.
    ScheduleCost::Bound bound_by = ScheduleCost::Bound::MAC;
};

/// Evaluate the schedule for the current kernel. `axi` selects the environment;
/// `post` tells the evaluator which epilogue stages (residual/pool/gap/fused-DW)
/// are instantiated for this launch (the SchedulePlan does not carry post-ops).
CostV2_1 estimate_cost_v2_1(const SchedulePlan& plan,
                            const PostOpConfig& post = PostOpConfig{},
                            const KernelManifest& mf = default_manifest(),
                            const AxiPortProfile& axi = AxiPortProfile{},
                            double clock_mhz = 200.0);

/// Print the V2.1 report (prediction, resource + critical-path decompositions).
void print_cost_v2_1(const SchedulePlan& plan, const CostV2_1& c,
                     const AxiPortProfile& axi);

const char* resource_bound_name(ResourceBound b);
const char* critical_cause_name(CriticalCause c);

} // namespace tile
} // namespace deeptile
