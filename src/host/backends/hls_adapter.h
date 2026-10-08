/**
 * @file hls_adapter.h
 * @brief Adapter between runtime plans and the C-simulation kernel interface.
 * @ingroup deeptile_host
 */

#ifndef DEEPTILE_RUNTIME_HLS_ADAPTER_H_
#define DEEPTILE_RUNTIME_HLS_ADAPTER_H_

#include "runtime/runtime.h"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <vector>

namespace deeptile {
namespace runtime {

/** Port word depths required for one launch. */
struct HlsPortDepths {
  std::size_t ifm_words = 0;
  std::size_t weight_words = 0;
  std::size_t bias_words = 0;
  std::size_t output_words = 0;
  std::size_t residual_words = 0;
  std::size_t xfer_desc_words = 0;
  std::size_t comp_desc_words = 0;
};

/** Compute C-simulation port depths for one prepared launch. */
HlsPortDepths get_hls_port_depths(const RuntimeExecutionPlan &plan,
                                  std::size_t launch_index);

HlsPortDepths get_max_hls_port_depths(const RuntimeExecutionPlan &plan);

void print_hls_port_depths(const RuntimeExecutionPlan &plan,
                           std::ostream &log);

/** Execute one prepared launch through the HLS C++ kernel model. */
bool run_hls_launch(RuntimeExecutionPlan &plan, std::size_t launch_index,
                    std::ostream &log);

} // namespace runtime
} // namespace deeptile

#endif // DEEPTILE_RUNTIME_HLS_ADAPTER_H_
