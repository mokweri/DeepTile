/**
 * @file descriptor_emitter.cpp
 * @brief Schedule-plan to descriptor-ABI lowering.
 */

#include "deeptile/tile/descriptor_emitter.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace deeptile {
namespace tile {

// §A  Address math helpers shared with the scheduler contract

// IFM offset in vector units for layout [Cg][H][W].
static uint32_t ifm_offset(int h, int w, int cg, const ConvLayerParams& p) {
    return static_cast<uint32_t>(
        static_cast<int64_t>(cg) * p.H * p.W +
        static_cast<int64_t>(h)  * p.W + w);
}

// OFM offset in vector units for layout [Cg][Oh][Ow].
static uint32_t ofm_offset(int oh, int ow, int cog, int final_Oh, int final_Ow) {
    return static_cast<uint32_t>(
        static_cast<int64_t>(cog) * final_Oh * final_Ow +
        static_cast<int64_t>(oh)  * final_Ow + ow);
}

static uint8_t clip4(int v) {
    return static_cast<uint8_t>(std::min(15, std::max(0, v)));
}

static uint8_t clip8(int v) {
    return static_cast<uint8_t>(std::min(255, std::max(0, v)));
}

// §B  Single TileDescriptor emission

static TileDescriptor make_tile_descriptor(
        const ComputeTask&   task,
        const SpatialTile&   st,
        const WeightEpoch&   ep,
        const ConvLayerParams& p,
        const PostOpConfig&  post,
        int final_Oh, int final_Ow,
        int num_ic_tiles) {

    TileDescriptor d = {};

    const bool do_gap   = post.gap_enable;
    const bool dw       = p.depthwise;
    const int  K        = p.Kh;
    const auto& win     = st.ifm_window;

    // Chaining advances from this task's first IFM plane in hardware.
    // Depthwise: cig_begin == cog, so the IFM plane is the paired channel group.
    const int ifm_cig    = task.cig_begin;
    d.ifm_offset_vecs    = ifm_offset(win.ih_start_actual, win.iw_start_actual,
                                      ifm_cig, p);
    d.ifm_ddr_h          = clip8(win.read_rows);
    d.ifm_ddr_w          = static_cast<uint8_t>(win.read_cols);
    d.ifm_ddr_stride     = static_cast<uint8_t>(p.W);
    d.ifm_tile_h         = static_cast<uint8_t>(win.total_height);
    d.ifm_tile_w         = static_cast<uint8_t>(win.total_width);
    d.reuse_ifm          = task.reuse_ifm ? 1U : 0U;
    d.packed_padding     = pack_padding(clip4(win.pad_top),    clip4(win.pad_bottom),
                                        clip4(win.pad_left),   clip4(win.pad_right));

    // Output activation.
    d.ofm_offset_vecs = do_gap
        ? static_cast<uint32_t>(task.cog)
        : ofm_offset(st.oh, st.ow, task.cog, final_Oh, final_Ow);
    d.ofm_tile_h      = clip8(st.conv_tile_oh);
    d.ofm_tile_w      = static_cast<uint8_t>(st.conv_tile_ow);
    d.ofm_row_stride  = static_cast<uint8_t>(do_gap ? 1 : final_Ow);

    // Weight and bias addressing.
    const int local_oc = task.cog - ep.oc_tile_begin;
    if (dw) {
        // Depthwise uses one diagonal CIG plane per output-channel tile.
        d.uram_base_depth = static_cast<uint16_t>(local_oc * K * K);
        d.num_cigs        = 1U;
    } else {
        d.uram_base_depth = static_cast<uint16_t>(
            (local_oc * num_ic_tiles + task.cig_begin) * K * K);
        d.num_cigs        = static_cast<uint8_t>(task.cig_count);
    }
    d.bias_offset_idx = static_cast<uint16_t>(task.cog);

    // Post-operation arguments.
    const int res_shift  = std::max(-128, std::min(127, post.residual_shift));
    const int pool_shift = std::max(-128, std::min(127, post.post_pool_shift));
    d.residual_shift      = static_cast<int8_t>(res_shift);
    d.pooled_ofm_tile_h   = do_gap ? 1U : clip8(st.pooled_tile_oh);
    d.pooled_ofm_tile_w   = do_gap ? 1U : static_cast<uint8_t>(st.pooled_tile_ow);
    d.post_pool_shift     = static_cast<int8_t>(pool_shift);
    d.pool_packed_padding = st.pool_packed_padding;
    // Both single-CIG tasks and partial chains start at this cache group.
    d.cache_cig_idx = dw ? 0U : static_cast<uint8_t>(task.cig_begin);

    // Only reduction boundaries reset or write back; an interior chain retains
    // partial sums and must not run post-operations or apply bias early.
    const bool is_first_cig = dw || (task.cig_begin == 0);
    const bool is_last_cig  = dw || (task.cig_begin + task.cig_count >= num_ic_tiles);

    d.flags = 0;
    if (is_first_cig)
        d.flags |= static_cast<uint8_t>(ACCUM_RESET);
    if (is_last_cig) {
        d.flags |= static_cast<uint8_t>(ACCUM_WRITE_BACK);
        if (post.maxpool_enable)
            d.flags |= static_cast<uint8_t>(POOL_ENABLE);
        if (post.gap_enable)
            d.flags |= static_cast<uint8_t>(GAP_ENABLE);
        if (post.residual_add_enable) {
            d.flags |= static_cast<uint8_t>(ADD_ENABLE);
            if (post.post_add_relu)
                d.flags |= static_cast<uint8_t>(POST_ADD_RELU);
        } else if (p.use_relu) {
            d.flags |= static_cast<uint8_t>(RELU_ON);
        }
    }

    return d;
}

// §C  emit_descriptors

DescriptorBundle emit_descriptors(const SchedulePlan& plan) {
    DescriptorBundle bundle;

    if (!plan.valid) {
        bundle.valid = false;
        bundle.issues = plan.issues;
        return bundle;
    }
    if (plan.compute_tasks.empty()) {
        bundle.valid = false;
        bundle.issues.push_back("SchedulePlan has no compute tasks");
        return bundle;
    }

    const auto& p          = plan.layer;
    const auto& post       = plan.post_ops;
    const int   final_Oh   = plan.stats.final_oh;
    const int   final_Ow   = plan.stats.final_ow;
    const int   num_ic     = p.num_cig();

    // Compute transfer counts before emitting transfer descriptors.
    const int n_epochs = static_cast<int>(plan.weight_epochs.size());
    std::vector<int> compute_count(static_cast<size_t>(n_epochs), 0);
    for (const auto& task : plan.compute_tasks)
        compute_count[static_cast<size_t>(task.epoch_idx)] += 1;

    // Transfer descriptors follow epoch order.
    bundle.transfers.reserve(plan.weight_epochs.size());
    for (const auto& ep : plan.weight_epochs) {
        WeightTransferDescriptor d = {};
        d.ddr_offset_vecs   = static_cast<uint32_t>(ep.uram_base_start);
        d.ddr_length_vecs   = static_cast<uint32_t>(ep.weight_vectors);
        d.total_layer_depth = static_cast<uint32_t>(ep.total_layer_depth);
        d.depth_count       = static_cast<uint16_t>(ep.depth_count);
        d.compute_count     = static_cast<uint16_t>(
            compute_count[static_cast<size_t>(ep.epoch_index)]);
        d.buffer_id         = ep.buffer_a ? 0U : 1U;
        // Mark the transfer that terminates the layer.
        d.flags             = static_cast<uint8_t>(
            (ep.epoch_index + 1 == ep.epoch_count) ? 0x1 : 0x0);
        bundle.transfers.push_back(d);
    }

    // Compute descriptors follow task order.
    bundle.compute_descs.reserve(plan.compute_tasks.size());
    for (const auto& task : plan.compute_tasks) {
        const SpatialTile& st = plan.spatial_tiles[static_cast<size_t>(task.spatial_idx)];
        const WeightEpoch& ep = plan.weight_epochs[static_cast<size_t>(task.epoch_idx)];

        bundle.compute_descs.push_back(
            make_tile_descriptor(task, st, ep, p, post, final_Oh, final_Ow, num_ic));
    }

    // Terminate the layer on the final compute descriptor.
    if (!bundle.compute_descs.empty())
        bundle.compute_descs.back().flags |= static_cast<uint8_t>(IS_LAST_TILE);

    // INVARIANT: Transfer compute counts cover every compute descriptor.
    int total = 0;
    for (const auto& xd : bundle.transfers)
        total += static_cast<int>(xd.compute_count);
    if (total != static_cast<int>(bundle.compute_descs.size())) {
        bundle.valid = false;
        bundle.issues.push_back("compute_count sum != descriptor count");
    }

    return bundle;
}

} // namespace tile
} // namespace deeptile
