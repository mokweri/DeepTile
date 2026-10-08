/**
 * @file graph_types.h
 * @brief Graph handoff and compiled execution-plan types.
 * @ingroup deeptile_compiler
 */

#ifndef DEEPTILE_GRAPH_TYPES_H_
#define DEEPTILE_GRAPH_TYPES_H_

#include "deeptile/tile/descriptor_emitter.hpp"
#include "deeptile/tile/schedule_builder.hpp"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace deeptile {
namespace graph {

enum class DiagnosticSeverity { Warning, Error };

struct GraphDiagnostic {
  DiagnosticSeverity severity = DiagnosticSeverity::Error;
  std::string context;
  std::string message;
};

struct TensorSpec {
  std::string id;
  std::string kind;
  std::string dtype;
  std::string layout;
  std::vector<int> shape;
  int frac = 0;
  std::optional<std::string> file;
};

struct NodeSpec {
  std::string id;
  std::string op;
  std::map<std::string, std::string> inputs;
  std::map<std::string, std::string> outputs;
  std::map<std::string, std::vector<int>> attrs_vec;
  std::map<std::string, int> attrs_int;
  std::map<std::string, bool> post_ops_bool;
  std::map<std::string, int> post_ops_int;
};

struct GraphSpec {
  std::string graph_path;
  std::string root_dir;
  std::string schema;
  std::map<std::string, std::string> model;
  std::map<std::string, std::string> target;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::map<std::string, std::string> references;
  std::map<std::string, TensorSpec> tensors;
  std::vector<NodeSpec> nodes;
};

/** Fused depthwise stage appended to a dense or pointwise launch. */
struct DepthwiseFusion {
  bool enable = false;
  int Kh = 3;
  int Kw = 3;
  int Sh = 1;
  int Sw = 1;
  int pad_top = 1;
  int pad_bottom = 1;
  int pad_left = 1;
  int pad_right = 1;
  int frac_in = 0;   ///< Producer output fractional bits.
  int frac_w = 0;
  int frac_b = 0;
  int frac_out = 0;  ///< Final fused output fractional bits.
  int shift_out = 0;   ///< Requantization shift.
  int shift_bias = 0;  ///< Bias-alignment shift.
  bool use_relu = false;
  int act_max = 127;          ///< Activation clamp: 127 for ReLU, lower for ReLU6.
  std::string weight_tensor;  ///< Compact OIHW [C][1][Kh][Kw] int8 weights.
  std::string bias_tensor;    ///< Int8 bias vector with C elements.
  int out_h = 0;
  int out_w = 0;
};

struct LaunchPlan {
  std::string id;
  std::vector<std::string> source_nodes;
  std::string ifm_tensor;
  std::string weight_tensor;
  std::string bias_tensor;
  std::optional<std::string> residual_tensor;
  std::string ofm_tensor;
  deeptile::tile::ConvLayerParams params = {};
  deeptile::tile::TileConfig tile_cfg = {};
  deeptile::tile::PostOpConfig post_ops = {};
  int shift_out = 0;
  int shift_bias = 0;
  int residual_shift = 0;
  int gap_mul = 0;
  bool use_ifm_cache = false;
  DepthwiseFusion dw = {};
  deeptile::tile::DescriptorBundle schedule;
};

struct GraphExecutionPlan {
  GraphSpec graph;
  std::vector<LaunchPlan> launches;
  std::vector<GraphDiagnostic> diagnostics;
};

inline bool has_errors(const std::vector<GraphDiagnostic> &diagnostics) {
  for (const auto &d : diagnostics) {
    if (d.severity == DiagnosticSeverity::Error) {
      return true;
    }
  }
  return false;
}

inline std::string severity_name(DiagnosticSeverity severity) {
  return severity == DiagnosticSeverity::Error ? "ERROR" : "WARN";
}

} // namespace graph
} // namespace deeptile

#endif // DEEPTILE_GRAPH_TYPES_H_
