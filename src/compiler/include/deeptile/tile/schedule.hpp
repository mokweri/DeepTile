/**
 * @file schedule.hpp
 * @brief Schedule intermediate representation between layer inputs and ABI descriptors.
 * @ingroup deeptile_compiler
 */

#pragma once

#include "layer.hpp"
#include "hardware_spec.hpp"
#include "schedule_options.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace deeptile {
namespace tile {

/**
 * @brief Receptive-field window into the IFM tensor for one output tile.
 *
 * Captures both the virtual (padded) and physical (actual tensor) coordinates,
 * plus per-edge padding counts. Derived from output tile position + layer
 * geometry; reused across all COGs that share the same spatial tile.
 */
struct InputWindow {
    int ih_start_actual;  ///< First actual IFM row to read
    int iw_start_actual;  ///< First actual IFM col to read
    int read_rows;        ///< Rows to read from DDR
    int read_cols;        ///< Cols per DDR row
    int total_height;     ///< On-chip tile height = pad_top + read_rows + pad_bottom
    int total_width;      ///< On-chip tile width  = pad_left + read_cols + pad_right
    uint8_t pad_top;
    uint8_t pad_bottom;
    uint8_t pad_left;
    uint8_t pad_right;
};

/**
 * @brief One weight transfer epoch: a contiguous range of OC tiles loaded
 *        together into one URAM ping-pong buffer.
 *
 * Maps 1:1 to a WeightTransferDescriptor when emitted. The descriptor emitter
 * derives `compute_count` by counting how many ComputeTasks reference this
 * epoch index.
 */
struct WeightEpoch {
    int epoch_index;          ///< 0-based index in the transfer sequence
    int epoch_count;          ///< Total epochs for this layer
    int oc_tile_begin;        ///< First OC tile index in this epoch
    int oc_tile_count;        ///< Number of OC tiles covered
    int ic_tiles;             ///< = params.num_cig() (all IC tiles per epoch)
    int cog_batch_size;       ///< COGs batched per IFM residency.
    int uram_base_start;      ///< Per-bank DDR offset = oc_tile_begin * depth_per_oc
    int depth_count;          ///< Weight depths in this epoch (bank-relative)
    int total_layer_depth;    ///< Full-bank depth — layer constant across epochs
    int weight_vectors;       ///< Vectors per epoch (for cost model: weight_load cycles)
    bool buffer_a;            ///< true = ping-pong buffer A, false = buffer B
};

/**
 * @brief Geometry of one spatial output tile and its associated IFM window.
 *
 * The post-op geometry fields are layer-uniform, but clipped tile sizes vary at
 * tensor boundaries, so they live here.
 */
struct SpatialTile {
    // Output-space position.
    int oh;             ///< Output tile top row (in output tensor coords)
    int ow;             ///< Output tile left col

    // Convolution output tile before post-operations.
    int conv_oh_start;  ///< Conv output row this tile starts from (= oh unless pooled)
    int conv_ow_start;  ///< Conv output col this tile starts from
    int conv_tile_oh;   ///< Conv output tile height (rows computed by this tile)
    int conv_tile_ow;   ///< Conv output tile width

    // Stored output tile after post-operations.
    int pooled_tile_oh; ///< OFM tile height after maxpool / GAP
    int pooled_tile_ow; ///< OFM tile width  after maxpool / GAP

    // ABI: [top(4)|bottom(4)|left(4)|right(4)], zero without pooling.
    uint16_t pool_packed_padding;

    // Shared IFM window for all output channel groups at this position.
    InputWindow ifm_window;
};

/**
 * @brief One logical compute task: (weight epoch, spatial tile, output channel group).
 *
 * Each task becomes one TileDescriptor with `num_cigs = cig_count`.
 * Chained tasks may cover only part of a reduction; unchained tasks cover
 * one CIG each. Reset and writeback follow reduction boundaries, not tasks.
 *
 * Task order in the SchedulePlan::compute_tasks vector IS the descriptor
 * emission order.
 */
struct ComputeTask {
    int epoch_idx;        ///< Index into SchedulePlan::weight_epochs
    int spatial_idx;      ///< Index into SchedulePlan::spatial_tiles
    int cog;              ///< Absolute OC tile index (0 .. num_cog-1)
    int cig_begin;        ///< First CIG covered by this task.
    int cig_count;        ///< CIG chaining factor: 1 = one-at-a-time, N = chained sweep
    bool reuse_ifm;       ///< true = stream from K=1 URAM cache (skip DDR)
    bool is_cache_fill;   ///< true = this task populates the K=1 URAM cache
};

/** Summary metrics derived from a schedule plan. */
struct ScheduleStats {
    int num_epochs          = 0;
    int num_spatial_tiles   = 0;
    int num_compute_tasks   = 0;
    int num_descriptors     = 0;  ///< After CIG expansion: sum(cig_count or 1 per task)
    bool uses_cig_chaining  = false;
    bool uses_k1_cache      = false;
    bool uses_cog_batching  = false;
    bool single_spatial_tile_cache_path = false;  ///< The special cache ordering path

    // Final tensor dimensions used for descriptor address calculation.
    int final_oh = 0;
    int final_ow = 0;
};

/**
 * @brief The full schedule for one convolution layer, before byte packing.
 *
 * Built by build_schedule() and lowered by emit_descriptors(). Vector order is
 * execution order.
 */
struct SchedulePlan {
    // Inputs captured during planning.
    ConvLayerParams   layer;
    TileConfig        tile_cfg;
    HardwareSpec      hw;
    ScheduleOptions   opts;
    PostOpConfig      post_ops;

    // Ordered IR levels.
    std::vector<WeightEpoch>   weight_epochs;
    std::vector<SpatialTile>   spatial_tiles;
    std::vector<ComputeTask>   compute_tasks;

    // Summary.
    ScheduleStats stats;

    // Validation status.
    bool valid = true;
    std::vector<std::string> issues;
};

} // namespace tile
} // namespace deeptile
