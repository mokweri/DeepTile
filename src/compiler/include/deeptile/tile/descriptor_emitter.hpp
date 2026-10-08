/**
 * @file descriptor_emitter.hpp
 * @brief Lowers schedule plans to the packed kernel ABI.
 * @ingroup deeptile_compiler
 */

#pragma once

#include "descriptors.hpp"
#include "schedule.hpp"

#include <string>
#include <vector>

namespace deeptile {
namespace tile {

/**
 * @brief Packed transfer and compute descriptors for one launch.
 */
struct DescriptorBundle {
    std::vector<WeightTransferDescriptor> transfers;    ///< One per WeightEpoch
    std::vector<TileDescriptor>           compute_descs;///< One per ComputeTask
    bool valid = true;
    std::vector<std::string> issues;
};

/**
 * @brief Lower a SchedulePlan to its packed 32-byte descriptor arrays.
 *
 * @param plan Valid schedule produced by build_schedule().
 * @return Packed descriptors and any lowering issues.
 */
DescriptorBundle emit_descriptors(const SchedulePlan& plan);

} // namespace tile
} // namespace deeptile
