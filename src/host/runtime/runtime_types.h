/**
 * @file runtime_types.h
 * @brief Prepared runtime-plan, buffer, and diagnostic types.
 * @ingroup deeptile_host
 */

#ifndef DEEPTILE_RUNTIME_TYPES_H_
#define DEEPTILE_RUNTIME_TYPES_H_

#include "deeptile/compiler/graph_types.h"
#include "runtime/stem_fold.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace deeptile {
namespace runtime {

enum class RuntimeDiagnosticSeverity { Warning, Error };

struct RuntimeDiagnostic {
  RuntimeDiagnosticSeverity severity = RuntimeDiagnosticSeverity::Error;
  std::string context;
  std::string message;
};

struct RuntimeTensorBuffer {
  std::string id;
  std::string tensor_id;
  std::string kind;
  std::string logical_layout;
  std::string packed_layout;
  std::vector<int> shape;
  std::int64_t element_count = 0;
  std::vector<std::int8_t> raw_data;
  std::vector<std::uint8_t> data_bytes;
  std::vector<std::int32_t> int32_data;

  std::size_t byte_size() const {
    return data_bytes.size() + int32_data.size() * sizeof(std::int32_t);
  }
};

struct RuntimeActivationSlot {
  std::string id;
  std::size_t byte_capacity = 0;
  std::vector<std::uint8_t> data_bytes;
  std::string current_tensor_id;
  std::vector<std::string> assigned_tensors;
};

struct RuntimeTensorStorageBinding {
  std::string tensor_id;
  std::string slot_id;
  std::size_t byte_size = 0;
  std::vector<int> shape;
  int producer_launch = -1;
  int last_consumer_launch = -1;
};

struct RuntimeLaunchBinding {
  std::string id;
  std::vector<std::string> source_nodes;
  std::string ifm_buffer;
  std::string weight_buffer;
  std::string bias_buffer;
  std::optional<std::string> residual_buffer;
  std::string output_buffer;
  int shift_out = 0;
  int shift_bias = 0;
  int residual_shift = 0;
  int gap_mul = 0;
  int kernel_size = 1;  ///< Uniform tile_conv kernel-size argument.
  int stride = 1;       ///< Uniform tile_conv stride argument.
  int act_max = 127;    ///< Activation clamp: 127 for ReLU, lower for ReLU6.
  // ABI: dw_config bit 0 enables depthwise, bits 1–2 encode stride, bit 3
  // enables ReLU, bit 4 selects a 5x5 kernel (else 3x3) and bits 8–15 hold
  // the ReLU clamp (127 for ReLU).
  int dw_config = 0;
  int dw_shift = 0;
  std::string dw_weight_buffer;   ///< Compact depthwise weight/bias buffer ID.
  std::size_t dw_weight_bytes = 0;
  int dw_out_h = 0;
  int dw_out_w = 0;
  bool use_ifm_cache = false;
  int spatial_tile_size = 0;    ///< T_oh * T_ow.
  int num_ic_tiles = 0;         ///< Input-channel group count.
  int ifm_cig_stride_vecs = 0;  ///< IFM plane stride per CIG in vector units.
  int num_xfers = 0;
  int num_tiles = 0;
  int num_writeback_tiles = 0;  ///< Descriptors carrying ACCUM_WRITE_BACK.
  int num_bias_vecs = 0;
  std::size_t ifm_bytes = 0;
  std::size_t weight_bytes = 0;
  std::size_t bias_bytes = 0;
  std::size_t residual_bytes = 0;
  std::size_t output_bytes = 0;
  std::vector<deeptile::tile::WeightTransferDescriptor> transfer_descs;
  std::vector<deeptile::tile::TileDescriptor> compute_descs;
};

struct RuntimeExecutionPlan {
  deeptile::graph::GraphExecutionPlan graph_plan;
  std::vector<RuntimeTensorBuffer> buffers;
  std::map<std::string, std::size_t> buffer_index_by_id;
  std::vector<RuntimeActivationSlot> activation_slots;
  std::map<std::string, std::size_t> activation_binding_by_tensor;
  std::vector<RuntimeTensorStorageBinding> activation_bindings;
  std::vector<RuntimeLaunchBinding> launches;
  std::vector<RuntimeDiagnostic> diagnostics;
  /// Applied space-to-depth stem rewrite; graph inputs set in their original
  /// shape are folded to match.
  StemFold stem_fold;
};

inline bool has_errors(const std::vector<RuntimeDiagnostic> &diagnostics) {
  for (const auto &d : diagnostics) {
    if (d.severity == RuntimeDiagnosticSeverity::Error) {
      return true;
    }
  }
  return false;
}

inline std::string severity_name(RuntimeDiagnosticSeverity severity) {
  return severity == RuntimeDiagnosticSeverity::Error ? "ERROR" : "WARN";
}

} // namespace runtime
} // namespace deeptile

#endif // DEEPTILE_RUNTIME_TYPES_H_
