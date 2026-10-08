/**
 * @file schedule_builder.cpp
 * @brief Schedule intermediate-representation builder.
 */

#include "deeptile/tile/schedule_builder.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <string>

#include "deeptile/tile/cost_model.hpp"  // Cost-guided epoch selection.
#include "deeptile/tile/descriptors.hpp" // Descriptor ABI limits.

namespace deeptile {
namespace tile {

// §A  Address math helpers (internal only)

static int div_ceil(int num, int den) {
    if (den <= 0) return 0;
    return (num + den - 1) / den;
}

static int maxpool3x3s2p1_out_dim(int in_dim) {
    return div_ceil(in_dim, 2);
}

// §B  Input window (halo) derivation

static InputWindow derive_input_window(const ConvLayerParams& p,
                                       int oh_start, int ow_start,
                                       int conv_tile_oh, int conv_tile_ow) {
    const int ih_start_padded = oh_start * p.Sh - p.pad_top;
    const int iw_start_padded = ow_start * p.Sw - p.pad_left;
    const int ih_end_padded   = ih_start_padded + (conv_tile_oh - 1) * p.Sh + p.Kh;
    const int iw_end_padded   = iw_start_padded + (conv_tile_ow - 1) * p.Sw + p.Kw;

    const int ih_start_actual = std::max(0, ih_start_padded);
    const int iw_start_actual = std::max(0, iw_start_padded);
    const int ih_end_actual   = std::min(p.H, ih_end_padded);
    const int iw_end_actual   = std::min(p.W, iw_end_padded);

    InputWindow w;
    w.ih_start_actual = ih_start_actual;
    w.iw_start_actual = iw_start_actual;
    w.read_rows    = ih_end_actual - ih_start_actual;
    w.read_cols    = iw_end_actual - iw_start_actual;
    w.pad_top      = static_cast<uint8_t>(std::max(0, -ih_start_padded));
    w.pad_bottom   = static_cast<uint8_t>(std::max(0, ih_end_padded - p.H));
    w.pad_left     = static_cast<uint8_t>(std::max(0, -iw_start_padded));
    w.pad_right    = static_cast<uint8_t>(std::max(0, iw_end_padded - p.W));
    w.total_height = w.pad_top + w.read_rows + w.pad_bottom;
    w.total_width  = w.pad_left + w.read_cols + w.pad_right;
    return w;
}

// §C  Weight epoch planning

struct InternalEpoch {
    int epoch_index;
    int epoch_count;
    int oc_tile_start;
    int oc_tile_count;
    int depth_count;
    int ddr_offset_vecs;
    int ddr_length_vecs;
    int total_layer_depth;
    bool buffer_a;
};

static bool plan_epochs(const ConvLayerParams& params,
                        const TileConfig& cfg,
                        const HardwareSpec& hw,
                        std::vector<InternalEpoch>& out_epochs,
                        int& out_num_ic_tiles,
                        int& out_depth_per_oc,
                        SchedulePlan& plan,
                        int forced_n_epochs = 0) {  // Zero selects automatic packing.
    const int num_oc_tiles = params.num_cog();
    const int num_ic_tiles = params.num_cig();
    const int k_sq = params.Kh * params.Kw;
    // Depthwise pairs COG g with CIG g, so each bank stores K² vectors per OC tile.
    const int depth_per_oc = params.depthwise ? k_sq : num_ic_tiles * k_sq;

    out_num_ic_tiles = num_ic_tiles;
    out_depth_per_oc = depth_per_oc;

    if (depth_per_oc <= 0) {
        plan.valid = false;
        plan.issues.push_back("depth_per_oc_group must be positive");
        return false;
    }

    const int max_oc_per_xfer = hw.weight_buf_depth / depth_per_oc;
    if (max_oc_per_xfer <= 0) {
        plan.valid = false;
        plan.issues.push_back(
            "One OC tile group (all IC tiles) does not fit in one URAM buffer");
        return false;
    }

    const int natural = div_ceil(num_oc_tiles, max_oc_per_xfer);
    // Clamp an override to feasible bounds; automatic mode preserves two-way overlap.
    const int n_epochs = (forced_n_epochs > 0)
        ? std::min(std::max(forced_n_epochs, natural), num_oc_tiles)
        : ((natural == 1 && num_oc_tiles > 1) ? 2 : natural);

    const int total_layer_depth = num_oc_tiles * depth_per_oc;

    out_epochs.reserve(static_cast<size_t>(n_epochs));
    int cursor = 0;
    for (int i = 0; i < n_epochs; ++i) {
        const int base_count  = num_oc_tiles / n_epochs;
        const int extra_count = num_oc_tiles % n_epochs;
        const int count = base_count + (i < extra_count ? 1 : 0);

        InternalEpoch ep;
        ep.epoch_index     = i;
        ep.epoch_count     = n_epochs;
        ep.oc_tile_start   = cursor;
        ep.oc_tile_count   = count;
        ep.depth_count     = count * depth_per_oc;
        ep.ddr_offset_vecs = cursor * depth_per_oc;
        ep.ddr_length_vecs = count  * depth_per_oc;
        ep.total_layer_depth = total_layer_depth;
        ep.buffer_a        = (i % 2 == 0);

        out_epochs.push_back(ep);
        cursor += count;
    }
    (void)cfg;
    return true;
}

// §D  Spatial tile derivation

static bool build_spatial_tiles(
        const ConvLayerParams& params,
        const TileConfig& cfg,
        const PostOpConfig& post_ops,
        int conv_Oh, int conv_Ow,
        std::vector<SpatialTile>& out_tiles,
        int& out_final_Oh, int& out_final_Ow,
        int& out_tile_step_oh, int& out_tile_step_ow,
        SchedulePlan& plan) {

    const bool do_pool = post_ops.maxpool_enable;
    const bool do_gap  = post_ops.gap_enable;
    const bool do_dw   = post_ops.dw_enable;
    const int  dw_s    = post_ops.dw_stride;   // 1 or 2
    const int  dw_k    = post_ops.dw_kernel;   // 3 or 5, same padding
    const int  dw_pad  = dw_k / 2;

    if (do_pool && do_gap) {
        plan.valid = false;
        plan.issues.push_back("GAP and maxpool cannot be fused in the same layer");
        return false;
    }
    if (do_dw && (do_pool || do_gap)) {
        plan.valid = false;
        plan.issues.push_back("fused DW cannot co-occur with maxpool/gap");
        return false;
    }
    if (do_dw && dw_k != 3 && dw_k != 5) {
        plan.valid = false;
        plan.issues.push_back("fused DW supports only 3x3 and 5x5 kernels");
        return false;
    }

    // A same-padded depthwise stage preserves size at stride 1 and halves it
    // at 2; with padding K/2 the output size is the same for 3x3 and 5x5.
    const int dw_out_h = (dw_s == 2) ? maxpool3x3s2p1_out_dim(conv_Oh) : conv_Oh;
    const int dw_out_w = (dw_s == 2) ? maxpool3x3s2p1_out_dim(conv_Ow) : conv_Ow;

    const int final_Oh = do_gap ? 1
                       : (do_pool ? maxpool3x3s2p1_out_dim(conv_Oh)
                       : (do_dw   ? dw_out_h : conv_Oh));
    const int final_Ow = do_gap ? 1
                       : (do_pool ? maxpool3x3s2p1_out_dim(conv_Ow)
                       : (do_dw   ? dw_out_w : conv_Ow));

    if (do_gap && (conv_Oh > cfg.T_oh || conv_Ow > cfg.T_ow)) {
        plan.valid = false;
        plan.issues.push_back(
            "Fused GAP requires the full conv output to fit in one spatial tile");
        return false;
    }

    // DW step: a tile of `step` output rows reads (step-1)*s + K conv rows,
    // which must fit T_oh: s1 gives step = T_oh-(K-1), s2 gives
    // step = (T_oh-K)/2 + 1. Columns follow the same rule when the conv row
    // is wider than one tile; otherwise a single column tile covers the row.
    auto dw_step = [&](int t) {
        return (dw_s == 2) ? std::max(1, (t - dw_k) / 2 + 1)
                           : std::max(1, t - (dw_k - 1));
    };
    const int dw_step_oh = dw_step(cfg.T_oh);
    const int dw_step_ow = conv_Ow <= cfg.T_ow ? dw_out_w : dw_step(cfg.T_ow);
    const int tile_step_oh = do_gap ? final_Oh
                           : (do_pool ? std::max(1, (cfg.T_oh - 1) / 2)
                           : (do_dw   ? dw_step_oh : cfg.T_oh));
    const int tile_step_ow = do_gap ? final_Ow
                           : (do_pool ? std::max(1, (cfg.T_ow - 1) / 2)
                           : (do_dw   ? dw_step_ow : cfg.T_ow));

    out_final_Oh    = final_Oh;
    out_final_Ow    = final_Ow;
    out_tile_step_oh = tile_step_oh;
    out_tile_step_ow = tile_step_ow;

    for (int oh_tile = 0; oh_tile < final_Oh; oh_tile += tile_step_oh) {
        for (int ow_tile = 0; ow_tile < final_Ow; ow_tile += tile_step_ow) {
            SpatialTile st;
            st.oh = oh_tile;
            st.ow = ow_tile;

            // Compute clipped output tile size (post-op output space)
            const int pooled_tile_oh = std::min(tile_step_oh, final_Oh - oh_tile);
            const int pooled_tile_ow = std::min(tile_step_ow, final_Ow - ow_tile);
            st.pooled_tile_oh = pooled_tile_oh;
            st.pooled_tile_ow = pooled_tile_ow;
            st.pool_packed_padding = 0;

            // Derive conv-space window
            if (do_gap) {
                st.conv_oh_start = 0;
                st.conv_ow_start = 0;
                st.conv_tile_oh  = conv_Oh;
                st.conv_tile_ow  = conv_Ow;
            } else if (do_dw) {
                // Reuse the packed-padding field for the depthwise halo; pool is disabled.
                // Output rows [out, out+n) read conv rows
                // [out*s - pad, (out+n-1)*s - pad + K).
                auto dw_axis = [&](int out_tile, int pooled, int conv_dim,
                                   int& cstart, int& ctile, int& pad_lo, int& pad_hi) {
                    const int start_raw = out_tile * dw_s - dw_pad;
                    const int end_raw   = (out_tile + pooled - 1) * dw_s - dw_pad + dw_k;
                    cstart = std::max(0, start_raw);
                    ctile  = std::min(conv_dim, end_raw) - cstart;
                    pad_lo = std::max(0, -start_raw);
                    pad_hi = std::max(0, end_raw - conv_dim);
                };
                int cst_oh, ct_oh, pad_t, pad_b;
                int cst_ow, ct_ow, pad_l, pad_r;
                dw_axis(oh_tile, pooled_tile_oh, conv_Oh, cst_oh, ct_oh, pad_t, pad_b);
                dw_axis(ow_tile, pooled_tile_ow, conv_Ow, cst_ow, ct_ow, pad_l, pad_r);
                st.conv_oh_start = cst_oh;
                st.conv_ow_start = cst_ow;
                st.conv_tile_oh  = ct_oh;
                st.conv_tile_ow  = ct_ow;
                st.pool_packed_padding = static_cast<uint16_t>(
                    ((static_cast<uint8_t>(pad_t) & 0xFU) << 12) |
                    ((static_cast<uint8_t>(pad_b) & 0xFU) <<  8) |
                    ((static_cast<uint8_t>(pad_l) & 0xFU) <<  4) |
                     (static_cast<uint8_t>(pad_r) & 0xFU));
            } else if (do_pool) {
                const int conv_oh_start_raw = oh_tile * 2 - 1;
                const int conv_ow_start_raw = ow_tile * 2 - 1;
                const int conv_oh_end_raw   = oh_tile * 2 + pooled_tile_oh * 2;
                const int conv_ow_end_raw   = ow_tile * 2 + pooled_tile_ow * 2;

                st.conv_oh_start = std::max(0, conv_oh_start_raw);
                st.conv_ow_start = std::max(0, conv_ow_start_raw);
                const int conv_oh_end = std::min(conv_Oh, conv_oh_end_raw);
                const int conv_ow_end = std::min(conv_Ow, conv_ow_end_raw);
                st.conv_tile_oh = conv_oh_end - st.conv_oh_start;
                st.conv_tile_ow = conv_ow_end - st.conv_ow_start;

                const uint8_t pool_pad_top    = static_cast<uint8_t>(std::max(0, -conv_oh_start_raw));
                const uint8_t pool_pad_bottom = static_cast<uint8_t>(std::max(0, conv_oh_end_raw - conv_Oh));
                const uint8_t pool_pad_left   = static_cast<uint8_t>(std::max(0, -conv_ow_start_raw));
                const uint8_t pool_pad_right  = static_cast<uint8_t>(std::max(0, conv_ow_end_raw - conv_Ow));
                // Packed as [top(4)|bot(4)|left(4)|right(4)]
                st.pool_packed_padding = static_cast<uint16_t>(
                    ((pool_pad_top    & 0xFU) << 12) |
                    ((pool_pad_bottom & 0xFU) <<  8) |
                    ((pool_pad_left   & 0xFU) <<  4) |
                     (pool_pad_right  & 0xFU));
            } else {
                st.conv_oh_start = oh_tile;
                st.conv_ow_start = ow_tile;
                st.conv_tile_oh  = pooled_tile_oh;
                st.conv_tile_ow  = pooled_tile_ow;
            }

            // IFM window from conv-space output tile
            st.ifm_window = derive_input_window(
                params, st.conv_oh_start, st.conv_ow_start,
                st.conv_tile_oh, st.conv_tile_ow);

            out_tiles.push_back(st);
        }
    }
    return true;
}

// §E  Compute task building
//     Two emission paths mirror the two loops in generate_tile_schedule().

static void emit_tasks_for_epoch_spatial(
        int epoch_idx,
        int spatial_idx,
        const InternalEpoch& ep,
        bool use_cache,
        bool chain_cigs,
        bool cache_fill_is_global,
        int num_ic_tiles,
        bool depthwise,
        std::vector<ComputeTask>& tasks) {

    for (int cog = ep.oc_tile_start; cog < ep.oc_tile_start + ep.oc_tile_count; ++cog) {
        // Each depthwise task pairs COG g with CIG g and completes one reduction.
        if (depthwise) {
            ComputeTask t;
            t.epoch_idx     = epoch_idx;
            t.spatial_idx   = spatial_idx;
            t.cog           = cog;
            t.cig_begin     = cog;   // IFM plane == COG (channel group g)
            t.cig_count     = 1;
            t.reuse_ifm     = false;
            t.is_cache_fill = false;
            tasks.push_back(t);
            continue;
        }

        const bool fill_cache = use_cache && (
            cache_fill_is_global
                ? (ep.epoch_index == 0 && cog == ep.oc_tile_start)
                : (cog == ep.oc_tile_start));

        ComputeTask t;
        t.epoch_idx   = epoch_idx;
        t.spatial_idx = spatial_idx;
        t.cog         = cog;
        t.cig_begin   = 0;
        // For a reduction wider than the count field, leave a full final
        // chain. Its start is num_ic_tiles - 255, which must fit seven bits.
        const int first_chain = num_ic_tiles > kMaxCigsPerDescriptor
            ? num_ic_tiles - kMaxCigsPerDescriptor : num_ic_tiles;
        t.cig_count   = chain_cigs ? first_chain : 1;
        t.reuse_ifm   = use_cache && !fill_cache;
        t.is_cache_fill = fill_cache;
        tasks.push_back(t);

        if (chain_cigs && first_chain < num_ic_tiles) {
            ComputeTask tail = t;
            tail.cig_begin = first_chain;
            tail.cig_count = num_ic_tiles - first_chain;
            tasks.push_back(tail);
        } else if (!chain_cigs) {
            // One task per CIG — the first task above covers cig=0.
            // Emit tasks for cig=1..num_ic_tiles-1.
            for (int cig = 1; cig < num_ic_tiles; ++cig) {
                ComputeTask tc = t;
                tc.cig_begin  = cig;
                tc.reuse_ifm  = (use_cache && !fill_cache);
                tc.is_cache_fill = false;
                tasks.push_back(tc);
            }
        }
    }
}

// §F  Top-level build_schedule

SchedulePlan build_schedule(
        const ConvLayerParams& params,
        const TileConfig&      tile_cfg,
        const HardwareSpec&    hw,
        const ScheduleOptions& opts,
        uint64_t               base_ifm,
        uint64_t               base_ofm,
        uint64_t               base_bias,
        const PostOpConfig&    post_ops) {

    (void)base_ifm;
    (void)base_ofm;
    (void)base_bias;

    SchedulePlan plan;
    plan.layer    = params;
    plan.tile_cfg = tile_cfg;
    plan.hw       = hw;
    plan.opts     = opts;
    plan.post_ops = post_ops;

    // Validate URAM-mode constraints.
    if (tile_cfg.T_cog != 1 || tile_cfg.T_cig != 1) {
        plan.valid = false;
        plan.issues.push_back("URAM mode requires T_cog=1 and T_cig=1");
        return plan;
    }
    if (params.Kh != params.Kw) {
        plan.valid = false;
        plan.issues.push_back("URAM mode requires square kernels (Kh == Kw)");
        return plan;
    }
    if (params.Sh != params.Sw) {
        plan.valid = false;
        plan.issues.push_back("URAM mode requires equal strides (Sh == Sw)");
        return plan;
    }

    // Build epoch-independent spatial tiles.
    const int conv_Oh = params.output_height();
    const int conv_Ow = params.output_width();
    int final_Oh = 0, final_Ow = 0;
    int tile_step_oh = 0, tile_step_ow = 0;

    if (!build_spatial_tiles(params, tile_cfg, post_ops,
                             conv_Oh, conv_Ow,
                             plan.spatial_tiles,
                             final_Oh, final_Ow,
                             tile_step_oh, tile_step_ow, plan)) {
        return plan;
    }
    // Drain term — estimate_cost() reads these during the epoch-depth sweep below.
    plan.stats.final_oh = final_Oh;
    plan.stats.final_ow = final_Ow;

    const int num_spatial = static_cast<int>(plan.spatial_tiles.size());

    // Single-tile cached schedules fill once and reuse the IFM across all epochs.
    const bool use_cache        = opts.enable_k1_cache && (params.Kh == 1);
    const bool chain_cigs       = use_cache && opts.enable_cig_chaining;
    const bool single_tile_path = use_cache && (num_spatial == 1);

    // Establish epoch-count-independent weight feasibility.
    const int num_oc_tiles = params.num_cog();
    const int num_ic_tiles = params.num_cig();

    if (chain_cigs && num_ic_tiles > kMaxChainedCacheCigs) {
        plan.valid = false;
        plan.issues.push_back(
            "chained K=1 cache exceeds the descriptor start/count range: " +
            std::to_string(kMaxChainedCacheCigs) + " CIGs");
        return plan;
    }

    // ABI: cache_cig_idx has seven bits; reject schedules that would truncate it.
    if (use_cache && !chain_cigs && num_ic_tiles > kMaxUnchainedCacheCigs) {
        plan.valid = false;
        plan.issues.push_back(
            "unchained K=1 cache needs num_cig <= " +
            std::to_string(kMaxUnchainedCacheCigs) +
            " (cache_cig_idx is 7 bits); enable CIG chaining for this layer");
        return plan;
    }
    // Depthwise stores K² depths per OC tile (diagonal), dense stores all IC tiles.
    const int depth_per_oc = params.depthwise
        ? (params.Kh * params.Kw)
        : num_ic_tiles * (params.Kh * params.Kw);
    if (depth_per_oc <= 0) {
        plan.valid = false;
        plan.issues.push_back("depth_per_oc_group must be positive");
        return plan;
    }
    const int max_oc_per_xfer = hw.weight_buf_depth / depth_per_oc;
    if (max_oc_per_xfer <= 0) {
        plan.valid = false;
        plan.issues.push_back(
            "One OC tile group (all IC tiles) does not fit in one URAM buffer");
        return plan;
    }
    const int natural = div_ceil(num_oc_tiles, max_oc_per_xfer);

    // Epoch count changes weight partitioning and ping-pong order, not task semantics.
    auto populate = [&](int n_epochs) -> bool {
        plan.weight_epochs.clear();
        plan.compute_tasks.clear();
        std::vector<InternalEpoch> int_epochs;
        int nic = 0, dpo = 0;
        if (!plan_epochs(params, tile_cfg, hw, int_epochs, nic, dpo, plan, n_epochs))
            return false;
        for (const auto& ep : int_epochs) {
            WeightEpoch we;
            we.epoch_index       = ep.epoch_index;
            we.epoch_count       = ep.epoch_count;
            we.oc_tile_begin     = ep.oc_tile_start;
            we.oc_tile_count     = ep.oc_tile_count;
            we.ic_tiles          = num_ic_tiles;
            we.cog_batch_size    = 1;  // future: opts.cog_batch_size
            we.uram_base_start   = ep.ddr_offset_vecs;
            we.depth_count       = ep.depth_count;
            we.total_layer_depth = ep.total_layer_depth;
            we.weight_vectors    = ep.ddr_length_vecs;
            we.buffer_a          = ep.buffer_a;
            plan.weight_epochs.push_back(we);
        }
        const int n_ep = static_cast<int>(int_epochs.size());
        if (single_tile_path) {
            // Path A: spatial-major (one spatial tile, fill = global).
            for (int si = 0; si < num_spatial; ++si)
                for (int ei = 0; ei < n_ep; ++ei)
                    emit_tasks_for_epoch_spatial(
                        ei, si, int_epochs[static_cast<size_t>(ei)],
                        /*use_cache=*/true, chain_cigs,
                        /*cache_fill_is_global=*/true, num_ic_tiles,
                        params.depthwise, plan.compute_tasks);
        } else {
            // Path B: epoch-major (multi-spatial, all K=3 layers, and depthwise).
            for (int ei = 0; ei < n_ep; ++ei)
                for (int si = 0; si < num_spatial; ++si)
                    emit_tasks_for_epoch_spatial(
                        ei, si, int_epochs[static_cast<size_t>(ei)],
                        use_cache, chain_cigs,
                        /*cache_fill_is_global=*/false, num_ic_tiles,
                        params.depthwise, plan.compute_tasks);
        }
        return true;
    };

    // Select the epoch count.
    if (opts.forced_epochs > 0) {
        // plan_epochs clamps an explicit override to feasible bounds.
        if (!populate(opts.forced_epochs)) return plan;
    } else if (opts.optimize_epoch_depth && num_oc_tiles > 1) {
        // Sweep feasible epoch counts and keep the minimum predicted cycle count.
        const int E_lo = std::max(1, natural);
        const int E_hi = std::min(num_oc_tiles, 32);
        long best_cy = std::numeric_limits<long>::max();
        int  best_E  = E_lo;
        for (int E = E_lo; E <= E_hi; ++E) {
            if (!populate(E)) return plan;
            const long cy = estimate_cost(plan).predicted_cy;
            if (cy < best_cy) { best_cy = cy; best_E = E; }
        }
        if (!populate(best_E)) return plan;
    } else if (!populate(0)) {  // Automatic packing preserves ping-pong overlap.
        return plan;
    }

    // Derive summary statistics.
    plan.stats.num_epochs        = static_cast<int>(plan.weight_epochs.size());
    plan.stats.num_spatial_tiles = num_spatial;
    plan.stats.num_compute_tasks = static_cast<int>(plan.compute_tasks.size());
    plan.stats.num_descriptors   = plan.stats.num_compute_tasks;  // 1:1 when chaining
    plan.stats.uses_cig_chaining = chain_cigs;
    plan.stats.uses_k1_cache     = use_cache;
    plan.stats.uses_cog_batching = opts.enable_cog_batching;
    plan.stats.single_spatial_tile_cache_path = single_tile_path;
    plan.stats.final_oh          = final_Oh;
    plan.stats.final_ow          = final_Ow;

    return plan;
}

} // namespace tile
} // namespace deeptile
