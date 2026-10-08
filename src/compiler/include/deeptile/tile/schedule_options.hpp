/**
 * @file schedule_options.hpp
 * @brief Scheduling policy options and named presets.
 * @ingroup deeptile_compiler
 */

#pragma once

namespace deeptile {
namespace tile {

/**
 * @brief Policy switches for the schedule builder.
 */
struct ScheduleOptions {
    bool enable_k1_cache              = true;   ///< Use K=1 IFM URAM cache
    bool enable_cig_chaining          = true;   ///< CIG chaining for K=1 cache layers
    bool enable_cog_batching          = false;  ///< COG batching for K=3 small-spatial (future)
    bool optimize_epoch_depth         = false;  ///< Minimize cost instead of filling URAM.
    int  forced_epochs                = 0;      ///< Positive value overrides epoch selection.
    int  preferred_spatial_tile       = 2048;   ///< Target T_oh × T_ow (ignored when == hw limit)
    int  cog_batch_size               = 1;      ///< COGs per IFM residency epoch (future)
};

/**
 * @brief Reproducible policy presets converted by options_for_policy().
 */
enum class SchedulePolicy {
    Baseline,                       ///< No cache or chaining.
    K1Cache,                        ///< K=1 cache without chaining.
    CigChained,                     ///< K=1 cache with CIG chaining.
    Phase2Default,                  ///< Default ScheduleOptions values.
    CogBatchedExperimental,         ///< K=3 small-spatial COG batching
    DynamicWeightEpochs             ///< Cost-model-selected weight epoch depth.
};

/// Convert a named policy to the corresponding ScheduleOptions.
inline ScheduleOptions options_for_policy(SchedulePolicy policy) {
    ScheduleOptions o;
    switch (policy) {
    case SchedulePolicy::Baseline:
        o.enable_k1_cache     = false;
        o.enable_cig_chaining = false;
        break;
    case SchedulePolicy::K1Cache:
        o.enable_k1_cache     = true;
        o.enable_cig_chaining = false;
        break;
    case SchedulePolicy::CigChained:
        o.enable_k1_cache     = true;
        o.enable_cig_chaining = true;
        break;
    case SchedulePolicy::Phase2Default:
        // Default option values define this preset.
        break;
    case SchedulePolicy::CogBatchedExperimental:
        o.enable_cog_batching = true;
        o.cog_batch_size      = 4;  // Candidate batch size evaluated at runtime.
        break;
    case SchedulePolicy::DynamicWeightEpochs:
        o.optimize_epoch_depth = true;
        break;
    }
    return o;
}

/// Human-readable name for logging and tools.
inline const char* policy_name(SchedulePolicy p) {
    switch (p) {
    case SchedulePolicy::Baseline:                     return "Baseline";
    case SchedulePolicy::K1Cache:                      return "K1Cache";
    case SchedulePolicy::CigChained:                   return "CigChained";
    case SchedulePolicy::Phase2Default:                return "Phase2Default";
    case SchedulePolicy::CogBatchedExperimental:       return "CogBatchedExperimental";
    case SchedulePolicy::DynamicWeightEpochs: return "DynamicWeightEpochs";
    }
    return "Unknown";
}

} // namespace tile
} // namespace deeptile
