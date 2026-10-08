/**
 * @file schedule_builder.hpp
 * @brief Mapping compiler interface for building schedule plans.
 * @ingroup deeptile_compiler
 */

#pragma once

#include "layer.hpp"
#include "hardware_spec.hpp"
#include "schedule_options.hpp"
#include "schedule.hpp"

#include <cstdint>

namespace deeptile {
namespace tile {

/**
 * @brief Build a SchedulePlan for one convolution layer.
 *
 * URAM mode constraints:
 *   - T_cog == 1 and T_cig == 1
 *   - Square kernel (Kh == Kw) and equal strides (Sh == Sw)
 *   - post_ops.gap_enable and post_ops.maxpool_enable are mutually exclusive
 *   - GAP requires the full output to fit in one spatial tile
 *
 * @param params Convolution geometry and quantization metadata.
 * @param tile_cfg Spatial tile dimensions and channel packing.
 * @param hw Hardware capability model.
 * @param opts Scheduling policy options.
 * @param base_ifm Reserved IFM base address in vector units.
 * @param base_ofm Reserved OFM base address in vector units.
 * @param base_bias Reserved bias base address in vector units.
 * @param post_ops Post-processing configuration.
 *
 * @return A populated plan with validation status and issues.
 */
SchedulePlan build_schedule(
    const ConvLayerParams& params,
    const TileConfig&      tile_cfg,
    const HardwareSpec&    hw        = HardwareSpec{},
    const ScheduleOptions& opts      = ScheduleOptions{},
    uint64_t               base_ifm  = 0,
    uint64_t               base_ofm  = 0,
    uint64_t               base_bias = 0,
    const PostOpConfig&    post_ops  = PostOpConfig{}
);

} // namespace tile
} // namespace deeptile
