/**
 * @file graph_compiler.cpp
 * @brief Graph validation, fusion, and launch-plan lowering.
 */

#include "deeptile/compiler/graph_compiler.h"
#include "deeptile/compiler/environment.h"

#include "deeptile/tile/descriptors.hpp"
#include "deeptile/tile/schedule_builder.hpp"
#include "deeptile/tile/schedule_options.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace deeptile {
namespace graph {

namespace hw = deeptile::tile::hw_defaults;

namespace {

void add_diag(std::vector<GraphDiagnostic> &diagnostics,
              DiagnosticSeverity severity, const std::string &context,
              const std::string &message) {
  diagnostics.push_back({severity, context, message});
}

void add_error(std::vector<GraphDiagnostic> &diagnostics,
               const std::string &context, const std::string &message) {
  add_diag(diagnostics, DiagnosticSeverity::Error, context, message);
}

const TensorSpec *find_tensor(const GraphSpec &graph,
                              const std::string &id) {
  auto it = graph.tensors.find(id);
  if (it == graph.tensors.end()) {
    return nullptr;
  }
  return &it->second;
}

std::string input_of(const NodeSpec &node, const std::string &port) {
  auto it = node.inputs.find(port);
  return it == node.inputs.end() ? std::string() : it->second;
}

std::string output_of(const NodeSpec &node, const std::string &port) {
  auto it = node.outputs.find(port);
  return it == node.outputs.end() ? std::string() : it->second;
}

std::vector<int> attr_vec(const NodeSpec &node, const std::string &key) {
  auto it = node.attrs_vec.find(key);
  return it == node.attrs_vec.end() ? std::vector<int>() : it->second;
}

int attr_int(const NodeSpec &node, const std::string &key, int fallback) {
  auto it = node.attrs_int.find(key);
  return it == node.attrs_int.end() ? fallback : it->second;
}

bool post_bool(const NodeSpec &node, const std::string &key) {
  auto it = node.post_ops_bool.find(key);
  return it != node.post_ops_bool.end() && it->second;
}

int post_int(const NodeSpec &node, const std::string &key, int fallback) {
  auto it = node.post_ops_int.find(key);
  return it == node.post_ops_int.end() ? fallback : it->second;
}

// FixQuant's reference clamps a ReLU6 output after saturation at this value.
int relu6_max(int frac_out) {
  return std::min(127, static_cast<int>(std::lround(std::ldexp(6.0, frac_out))));
}

// Every post-op the lowering understands. An unknown one is an error: silently
// ignoring it compiles a graph whose results then differ from the reference.
bool validate_post_ops(const NodeSpec &node,
                       std::vector<GraphDiagnostic> &diagnostics) {
  static const std::set<std::string> kBoolOps = {
      "relu", "relu6", "residual_add", "post_add_relu", "post_add_relu6"};
  static const std::set<std::string> kIntOps = {"residual_shift"};
  bool ok = true;
  for (const auto &entry : node.post_ops_bool) {
    if (kBoolOps.count(entry.first) == 0) {
      add_error(diagnostics, "node " + node.id,
                "unsupported post_op: " + entry.first);
      ok = false;
    }
  }
  for (const auto &entry : node.post_ops_int) {
    if (kIntOps.count(entry.first) == 0) {
      add_error(diagnostics, "node " + node.id,
                "unsupported post_op: " + entry.first);
      ok = false;
    }
  }
  if (post_bool(node, "relu") && post_bool(node, "relu6")) {
    add_error(diagnostics, "node " + node.id, "relu and relu6 are exclusive");
    ok = false;
  }
  if (post_bool(node, "post_add_relu") && post_bool(node, "post_add_relu6")) {
    add_error(diagnostics, "node " + node.id,
              "post_add_relu and post_add_relu6 are exclusive");
    ok = false;
  }
  // The kernel applies one clamp per launch, before and after a residual add.
  const bool pre = post_bool(node, "relu") || post_bool(node, "relu6");
  const bool post = post_bool(node, "post_add_relu") ||
                    post_bool(node, "post_add_relu6");
  if (pre && post &&
      post_bool(node, "relu6") != post_bool(node, "post_add_relu6")) {
    add_error(diagnostics, "node " + node.id,
              "activations before and after a residual add must use the same clamp");
    ok = false;
  }
  return ok;
}

std::map<std::string, int> make_consumer_counts(const GraphSpec &graph) {
  std::map<std::string, int> counts;
  for (const auto &node : graph.nodes) {
    for (const auto &[port, tensor] : node.inputs) {
      (void)port;
      counts[tensor] += 1;
    }
  }
  return counts;
}

int compute_gap_scale_local(int conv_frac_out, int gap_frac_out,
                            int num_elems) {
  if (num_elems <= 0) {
    throw std::runtime_error("GAP element count must be positive");
  }
  const int frac_delta = gap_frac_out - conv_frac_out;
  const int total_shift = hw::kGapScaleFracBits + frac_delta;
  if (total_shift < 0) {
    throw std::runtime_error("GAP fixed-point shift is negative");
  }
  const std::int64_t numerator = std::int64_t(1) << total_shift;
  const int gap_mul =
      static_cast<int>((numerator + num_elems / 2) / num_elems);
  constexpr int kGapMulMax = (1 << 15) - 1;
  if (gap_mul > kGapMulMax) {
    throw std::runtime_error("GAP multiplier exceeds 16-bit signed range");
  }
  return gap_mul;
}

std::vector<int> conv_output_shape(const TensorSpec &ifm,
                                   const TensorSpec &weight,
                                   const NodeSpec &node) {
  const auto padding = attr_vec(node, "padding");
  const auto stride = attr_vec(node, "stride");
  const auto dilation = attr_vec(node, "dilation");
  const int cout = weight.shape[0];
  const int kh = weight.shape[2];
  const int kw = weight.shape[3];
  const int h = ifm.shape[1];
  const int w = ifm.shape[2];
  const int pt = padding[0];
  const int pb = padding[1];
  const int pl = padding[2];
  const int pr = padding[3];
  const int sh = stride[0];
  const int sw = stride[1];
  const int dh = dilation[0];
  const int dw = dilation[1];
  const int oh = (h + pt + pb - dh * (kh - 1) - 1) / sh + 1;
  const int ow = (w + pl + pr - dw * (kw - 1) - 1) / sw + 1;
  return {cout, oh, ow};
}

bool validate_conv_node(const GraphSpec &graph, const NodeSpec &node,
                        std::vector<GraphDiagnostic> &diagnostics) {
  const std::string ctx = "node " + node.id;
  const auto kernel = attr_vec(node, "kernel");
  const auto stride = attr_vec(node, "stride");
  const auto padding = attr_vec(node, "padding");
  const auto dilation = attr_vec(node, "dilation");
  const int groups = attr_int(node, "groups", -1);
  if (kernel.size() != 2 || stride.size() != 2 || padding.size() != 4 ||
      dilation.size() != 2 || groups < 0) {
    add_error(diagnostics, ctx, "conv2d requires kernel/stride/padding/dilation/groups attrs");
    return false;
  }
  // Supported group modes are dense and channel-wise depthwise convolution.
  const TensorSpec *ifm_ts = find_tensor(graph, input_of(node, "ifm"));
  const TensorSpec *weight_ts = find_tensor(graph, input_of(node, "weight"));
  const bool is_depthwise =
      ifm_ts != nullptr && weight_ts != nullptr &&
      ifm_ts->shape.size() == 3 && weight_ts->shape.size() == 4 &&
      groups == ifm_ts->shape[0] && groups == weight_ts->shape[0] &&
      weight_ts->shape[1] == 1;
  if (groups != 1 && !is_depthwise) {
    add_error(diagnostics, ctx,
              "groups must be 1 (dense) or Cin==Cout (depthwise)");
  }
  if (dilation != std::vector<int>({1, 1})) {
    add_error(diagnostics, ctx, "dilation must be [1,1]");
  }
  if (kernel[0] != kernel[1]) {
    add_error(diagnostics, ctx, "kernel must be square");
  }
  if (stride[0] != stride[1]) {
    add_error(diagnostics, ctx, "stride must be equal in both dimensions");
  }
  // The kernel's input loader applies the stride as a shift.
  if (stride[0] != 1 && stride[0] != 2) {
    add_error(diagnostics, ctx, "stride must be 1 or 2");
  }
  if (padding[0] != padding[1] || padding[2] != padding[3]) {
    add_error(diagnostics, ctx, "padding must be symmetric for v1");
  }

  const TensorSpec *ifm = find_tensor(graph, input_of(node, "ifm"));
  const TensorSpec *weight = find_tensor(graph, input_of(node, "weight"));
  const TensorSpec *bias = find_tensor(graph, input_of(node, "bias"));
  const TensorSpec *ofm = find_tensor(graph, output_of(node, "ofm"));
  if (!ifm || !weight || !bias || !ofm) {
    add_error(diagnostics, ctx, "missing ifm/weight/bias/ofm tensor");
    return false;
  }
  if (ifm->layout != "CHW" || ifm->shape.size() != 3) {
    add_error(diagnostics, ctx, "IFM tensor must be CHW with shape [C,H,W]");
  }
  if (weight->layout != "OIHW" || weight->shape.size() != 4) {
    add_error(diagnostics, ctx, "weight tensor must be OIHW");
    return false;
  }
  if (bias->layout != "O" || bias->shape.size() != 1) {
    add_error(diagnostics, ctx, "bias tensor must use layout O");
  }
  if (ofm->layout != "CHW" || ofm->shape.size() != 3) {
    add_error(diagnostics, ctx, "OFM tensor must be CHW with shape [C,H,W]");
  }
  if (!is_depthwise && ifm->shape[0] != weight->shape[1]) {
    add_error(diagnostics, ctx, "IFM channel count does not match weight Cin");
  }
  if (bias->shape.size() == 1 && bias->shape[0] != weight->shape[0]) {
    add_error(diagnostics, ctx, "bias length does not match weight Cout");
  }
  if (kernel != std::vector<int>({weight->shape[2], weight->shape[3]})) {
    add_error(diagnostics, ctx, "kernel attr does not match weight shape");
  }
  if (ofm->shape != conv_output_shape(*ifm, *weight, node)) {
    add_error(diagnostics, ctx, "OFM shape does not match convolution attrs");
  }

  const std::string residual = input_of(node, "residual");
  if (!residual.empty()) {
    const TensorSpec *res = find_tensor(graph, residual);
    if (!res) {
      add_error(diagnostics, ctx, "residual tensor not found");
    } else if (res->shape != ofm->shape) {
      add_error(diagnostics, ctx, "residual shape must match OFM shape");
    }
  }

  return !has_errors(diagnostics);
}

bool validate_maxpool_node(const GraphSpec &graph, const NodeSpec &node,
                           std::vector<GraphDiagnostic> &diagnostics) {
  const std::string ctx = "node " + node.id;
  if (attr_vec(node, "kernel") != std::vector<int>({3, 3}) ||
      attr_vec(node, "stride") != std::vector<int>({2, 2}) ||
      attr_vec(node, "padding") != std::vector<int>({1, 1, 1, 1})) {
    add_error(diagnostics, ctx, "maxpool2d must be 3x3 stride-2 pad-1");
    return false;
  }
  const TensorSpec *ifm = find_tensor(graph, input_of(node, "ifm"));
  const TensorSpec *ofm = find_tensor(graph, output_of(node, "ofm"));
  if (!ifm || !ofm || ifm->shape.size() != 3 || ofm->shape.size() != 3) {
    add_error(diagnostics, ctx, "maxpool requires CHW input/output tensors");
    return false;
  }
  const int oh = (ifm->shape[1] + 2 - 3) / 2 + 1;
  const int ow = (ifm->shape[2] + 2 - 3) / 2 + 1;
  if (ofm->shape != std::vector<int>({ifm->shape[0], oh, ow})) {
    add_error(diagnostics, ctx, "maxpool output shape is incorrect");
  }
  return !has_errors(diagnostics);
}

bool validate_gap_node(const GraphSpec &graph, const NodeSpec &node,
                       std::vector<GraphDiagnostic> &diagnostics) {
  const std::string ctx = "node " + node.id;
  const TensorSpec *ifm = find_tensor(graph, input_of(node, "ifm"));
  const TensorSpec *ofm = find_tensor(graph, output_of(node, "ofm"));
  if (!ifm || !ofm || ifm->shape.size() != 3 || ofm->shape.size() != 3) {
    add_error(diagnostics, ctx, "gap2d requires CHW input/output tensors");
    return false;
  }
  if (ofm->shape != std::vector<int>({ifm->shape[0], 1, 1})) {
    add_error(diagnostics, ctx, "gap2d output shape must be [C,1,1]");
  }
  return !has_errors(diagnostics);
}

bool validate_linear_node(const GraphSpec &graph, const NodeSpec &node,
                          std::vector<GraphDiagnostic> &diagnostics) {
  const std::string ctx = "node " + node.id;
  const TensorSpec *ifm = find_tensor(graph, input_of(node, "ifm"));
  const TensorSpec *weight = find_tensor(graph, input_of(node, "weight"));
  const TensorSpec *bias = find_tensor(graph, input_of(node, "bias"));
  const TensorSpec *ofm = find_tensor(graph, output_of(node, "ofm"));
  if (!ifm || !weight || !bias || !ofm) {
    add_error(diagnostics, ctx, "linear missing ifm/weight/bias/ofm tensor");
    return false;
  }
  if (ifm->shape.size() != 3 || ifm->shape[1] != 1 || ifm->shape[2] != 1) {
    add_error(diagnostics, ctx, "linear input shape must be [Cin,1,1]");
  }
  if (weight->layout != "OI" || weight->shape.size() != 2) {
    add_error(diagnostics, ctx, "linear weight must use OI layout");
    return false;
  }
  if (bias->shape != std::vector<int>({weight->shape[0]})) {
    add_error(diagnostics, ctx, "linear bias length does not match Cout");
  }
  if (ifm->shape[0] != weight->shape[1]) {
    add_error(diagnostics, ctx, "linear input channels do not match weight Cin");
  }
  if (ofm->shape != std::vector<int>({weight->shape[0], 1, 1})) {
    add_error(diagnostics, ctx, "linear output shape must be [Cout,1,1]");
  }
  return !has_errors(diagnostics);
}

// Share hardware limits with the scheduler instead of duplicating constants.
constexpr int kIfmCacheDepth = hw::kK1CacheDepth;
constexpr int kCigStrideAlign = hw::kPSpatial;

int align_up(int value, int alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

bool has_zero_padding(const deeptile::tile::ConvLayerParams &p) {
  return p.pad_top == 0 && p.pad_bottom == 0 && p.pad_left == 0 &&
         p.pad_right == 0;
}

bool is_k1_cache_candidate(const deeptile::tile::ConvLayerParams &p) {
  // Only a descriptor's starting cache group is seven-bit; chained sweeps
  // advance their full-width row address internally. Larger reductions use
  // two legal partial chains rather than truncating the eight-bit CIG count.
  const int max_cigs = deeptile::env_flag("DISABLE_CIG_CHAIN")
      ? deeptile::tile::kMaxUnchainedCacheCigs
      : deeptile::tile::kMaxChainedCacheCigs;
  return p.Kh == 1 && p.Kw == 1 && has_zero_padding(p) &&
         p.num_cog() >= 2 && p.num_cig() <= max_cigs;
}

bool tile_fits_output_storage(int th, int tw) {
  return th > 0 && tw > 0 && th * tw <= hw::kMaxSpatialTile;
}

bool tile_fits_padded_ifm(const deeptile::tile::ConvLayerParams &p, int th,
                          int tw) {
  if (!tile_fits_output_storage(th, tw)) {
    return false;
  }
  const int padded_h = (th - 1) * p.Sh + p.Kh;
  const int padded_w = (tw - 1) * p.Sw + p.Kw;
  return padded_h <= hw::kPaddedTileH && padded_w <= hw::kPaddedTileW;
}

bool tile_fits_ifm_cache(const deeptile::tile::ConvLayerParams &p, int th,
                         int tw) {
  if (!tile_fits_output_storage(th, tw)) {
    return false;
  }
  const int spatial_tile_size = th * tw;
  const int cig_stride = align_up(spatial_tile_size, kCigStrideAlign);
  return p.num_cig() * cig_stride <= kIfmCacheDepth;
}

deeptile::tile::TileConfig choose_tile_config(const deeptile::tile::ConvLayerParams &p,
                                          const deeptile::tile::PostOpConfig &post_ops,
                                          std::vector<GraphDiagnostic> &diagnostics,
                                          const std::string &ctx) {
  // GAP requires the complete convolution output in one launch tile.
  deeptile::tile::TileConfig cfg = {};
  cfg.Cpi = hw::kIcTileSize;
  cfg.Cpo = hw::kOcTileSize;
  cfg.T_cig = 1;
  cfg.T_cog = 1;

  const int oh = p.output_height();
  const int ow = p.output_width();

  if (post_ops.gap_enable) {
    cfg.T_oh = oh;
    cfg.T_ow = ow;
    if (!tile_fits_padded_ifm(p, cfg.T_oh, cfg.T_ow)) {
      add_error(diagnostics, ctx,
                "fused GAP requires full conv output to fit one tile");
    }
    return cfg;
  }

  if (post_ops.dw_enable) {
    // Reserve producer rows for the depthwise halo. A row that fits the line
    // buffer stays whole; a wider one is tiled by column, each conv tile at
    // most one line-buffer row including its halo columns.
    const int k = post_ops.dw_kernel;
    // The producer's padded input tile must also fit on chip, which bounds a
    // strided dense producer such as a stem: take the largest tile whose
    // padded input fits, then honour the output-storage budget.
    const int max_h = (hw::kPaddedTileH - p.Kh) / p.Sh + 1;
    const int max_w = (hw::kPaddedTileW - p.Kw) / p.Sw + 1;
    cfg.T_ow = std::min({hw::kMaxTileW, ow, max_w});
    cfg.T_oh = std::min({hw::kMaxTileH, oh + k - 1, max_h});
    while (cfg.T_oh > 1 && cfg.T_oh * cfg.T_ow > hw::kMaxSpatialTile) {
      --cfg.T_oh;
    }
    // Each side must hold one depthwise window plus one step.
    const int min_side = k + post_ops.dw_stride - 1;
    if (cfg.T_oh < std::min(min_side, oh + k - 1) ||
        cfg.T_ow < std::min(min_side, ow) ||
        !tile_fits_padded_ifm(p, cfg.T_oh, cfg.T_ow)) {
      add_error(diagnostics, ctx, "fused depthwise producer has no legal tile");
    }
    return cfg;
  }

  // INVARIANT: Only the K=1 cache path may exceed padded raw-IFM storage. When
  // caching is disabled, tile selection must honor the raw buffer dimensions.
  const bool k1_cache_usable =
      is_k1_cache_candidate(p) && !deeptile::env_flag("DISABLE_K1_CACHE");
  if (k1_cache_usable) {
    const int full_oh = std::min(hw::kMaxTileH, oh);
    const int full_ow = std::min(hw::kMaxTileW, ow);
    if (full_oh == oh && full_ow == ow &&
        tile_fits_ifm_cache(p, full_oh, full_ow)) {
      cfg.T_oh = full_oh;
      cfg.T_ow = full_ow;
      return cfg;
    }
  }

  cfg.T_oh = std::min(hw::kMaxTileH, oh);
  cfg.T_ow = std::min(hw::kMaxTileW, ow);
  while (cfg.T_oh > 1 && !tile_fits_padded_ifm(p, cfg.T_oh, cfg.T_ow)) {
    if ((cfg.T_oh >= cfg.T_ow && cfg.T_oh > 1) ||
        ((cfg.T_oh - 1) * p.Sh + p.Kh > hw::kPaddedTileH)) {
      --cfg.T_oh;
    } else {
      --cfg.T_ow;
    }
  }
  while (cfg.T_ow > 1 && !tile_fits_padded_ifm(p, cfg.T_oh, cfg.T_ow)) {
    --cfg.T_ow;
  }
  if (!tile_fits_padded_ifm(p, cfg.T_oh, cfg.T_ow)) {
    add_error(diagnostics, ctx, "could not find legal tile config");
  }
  return cfg;
}

deeptile::tile::ConvLayerParams make_conv_params(const GraphSpec &graph,
                                             const NodeSpec &conv_node,
                                             const std::string &ofm_tensor,
                                             std::vector<GraphDiagnostic> &diagnostics) {
  (void)ofm_tensor;
  const TensorSpec *ifm = find_tensor(graph, input_of(conv_node, "ifm"));
  const TensorSpec *weight = find_tensor(graph, input_of(conv_node, "weight"));
  const TensorSpec *bias = find_tensor(graph, input_of(conv_node, "bias"));
  const TensorSpec *ofm = find_tensor(graph, output_of(conv_node, "ofm"));
  deeptile::tile::ConvLayerParams p = {};
  if (!ifm || !weight || !bias || !ofm) {
    add_error(diagnostics, "node " + conv_node.id,
              "cannot build ConvLayerParams due to missing tensors");
    return p;
  }
  const auto stride = attr_vec(conv_node, "stride");
  const auto padding = attr_vec(conv_node, "padding");
  const int groups = attr_int(conv_node, "groups", 1);
  // Depthwise Cin comes from the IFM because weight shape is [C][1][K][K].
  p.depthwise = (groups == ifm->shape[0]) && (groups == weight->shape[0]) &&
                (weight->shape[1] == 1);
  p.Cin = ifm->shape[0];
  p.H = ifm->shape[1];
  p.W = ifm->shape[2];
  p.Cout = weight->shape[0];
  p.Kh = weight->shape[2];
  p.Kw = weight->shape[3];
  p.Sh = stride[0];
  p.Sw = stride[1];
  p.pad_top = padding[0];
  p.pad_bottom = padding[1];
  p.pad_left = padding[2];
  p.pad_right = padding[3];
  p.Cpi = hw::kIcTileSize;
  p.Cpo = hw::kOcTileSize;
  p.elem_bytes = 1;
  p.frac_in = ifm->frac;
  p.frac_w = weight->frac;
  p.frac_b = bias->frac;
  p.frac_out = ofm->frac;
  p.use_relu = post_bool(conv_node, "relu") || post_bool(conv_node, "relu6");
  if (post_bool(conv_node, "relu6") || post_bool(conv_node, "post_add_relu6")) {
    p.act_max = relu6_max(p.frac_out);
  }
  return p;
}

deeptile::tile::ConvLayerParams make_linear_params(const GraphSpec &graph,
                                               const NodeSpec &node,
                                               std::vector<GraphDiagnostic> &diagnostics) {
  const TensorSpec *ifm = find_tensor(graph, input_of(node, "ifm"));
  const TensorSpec *weight = find_tensor(graph, input_of(node, "weight"));
  const TensorSpec *bias = find_tensor(graph, input_of(node, "bias"));
  const TensorSpec *ofm = find_tensor(graph, output_of(node, "ofm"));
  deeptile::tile::ConvLayerParams p = {};
  if (!ifm || !weight || !bias || !ofm) {
    add_error(diagnostics, "node " + node.id,
              "cannot build linear ConvLayerParams due to missing tensors");
    return p;
  }
  p.Cin = ifm->shape[0];
  p.H = 1;
  p.W = 1;
  p.Cout = weight->shape[0];
  p.Kh = 1;
  p.Kw = 1;
  p.Sh = 1;
  p.Sw = 1;
  p.pad_top = p.pad_bottom = p.pad_left = p.pad_right = 0;
  p.Cpi = hw::kIcTileSize;
  p.Cpo = hw::kOcTileSize;
  p.elem_bytes = 1;
  p.frac_in = ifm->frac;
  p.frac_w = weight->frac;
  p.frac_b = bias->frac;
  p.frac_out = ofm->frac;
  // A linear node lowers to a 1x1 convolution, so it carries the same
  // activation post-op block the convolution path reads.
  p.use_relu = post_bool(node, "relu") || post_bool(node, "relu6");
  if (post_bool(node, "relu6")) {
    p.act_max = relu6_max(p.frac_out);
  }
  return p;
}

void fill_shifts(LaunchPlan &launch) {
  // Biases arrive from the exporter as int8.  The runtime widens them to int32
  // after applying the shift computed here.
  launch.shift_out =
      (launch.params.frac_in + launch.params.frac_w) - launch.params.frac_out;
  launch.shift_bias = launch.params.frac_out - launch.params.frac_b + 1;
}

bool finalize_launch(LaunchPlan &launch,
                     std::vector<GraphDiagnostic> &diagnostics) {
  // Finalization is the boundary where a semantic node becomes a concrete
  // tile_conv launch: fixed-point shifts, legal tile shape, and descriptors.
  fill_shifts(launch);
  const std::string ctx = "launch " + launch.id;
  // The kernel preloads one bias vector per output-channel group into a fixed
  // on-chip cache, with no bounds check: overrunning it corrupts memory in
  // simulation and silicon alike, so reject it here instead.
  if (launch.params.num_cog() > hw::kBiasBufDepth) {
    add_error(diagnostics, ctx,
              "output channels exceed the bias cache: " +
                  std::to_string(launch.params.num_cog()) +
                  " output-channel groups against a depth of " +
                  std::to_string(hw::kBiasBufDepth) + " (at most " +
                  std::to_string(hw::kBiasBufDepth * hw::kOcTileSize) +
                  " output channels)");
    return false;
  }
  launch.tile_cfg = choose_tile_config(launch.params, launch.post_ops,
                                       diagnostics, ctx);
  if (has_errors(diagnostics)) {
    return false;
  }
  // Apply policy overrides after tile selection to keep tile geometry stable.
  const bool disable_k1_cache = deeptile::env_flag("DISABLE_K1_CACHE");
  const bool disable_cig_chain = deeptile::env_flag("DISABLE_CIG_CHAIN");

  launch.use_ifm_cache =
      !disable_k1_cache &&
      is_k1_cache_candidate(launch.params) &&
      tile_fits_ifm_cache(launch.params, launch.tile_cfg.T_oh,
                          launch.tile_cfg.T_ow);
  {
    deeptile::tile::ScheduleOptions opts;
    opts.enable_k1_cache = launch.use_ifm_cache;
    opts.enable_cig_chaining = !disable_cig_chain;
    // Environment hooks select packed or explicitly forced epoch schedules.
    opts.optimize_epoch_depth = (deeptile::env_value("PACK_EPOCHS") == nullptr);
    if (const char* fe = deeptile::env_value("FORCE_EPOCHS")) {
        const int n = std::atoi(fe);
        if (n > 0) opts.forced_epochs = n;
    }
    auto plan = deeptile::tile::build_schedule(
        launch.params, launch.tile_cfg,
        deeptile::tile::HardwareSpec{}, opts,
        0, 0, 0, launch.post_ops);
    launch.schedule = deeptile::tile::emit_descriptors(plan);
  }
  if (!launch.schedule.valid) {
    for (const auto &issue : launch.schedule.issues) {
      add_error(diagnostics, ctx, issue);
    }
    return false;
  }
  return true;
}

LaunchPlan make_conv_launch(const GraphSpec &graph, const NodeSpec &conv_node,
                            const std::string &ofm_tensor,
                            std::vector<std::string> source_nodes,
                            std::vector<GraphDiagnostic> &diagnostics) {
  LaunchPlan launch;
  launch.id = conv_node.id;
  launch.source_nodes = std::move(source_nodes);
  launch.ifm_tensor = input_of(conv_node, "ifm");
  launch.weight_tensor = input_of(conv_node, "weight");
  launch.bias_tensor = input_of(conv_node, "bias");
  launch.ofm_tensor = ofm_tensor;
  launch.params = make_conv_params(graph, conv_node, ofm_tensor, diagnostics);

  const std::string residual = input_of(conv_node, "residual");
  if (!residual.empty()) {
    launch.residual_tensor = residual;
    launch.post_ops.residual_add_enable = post_bool(conv_node, "residual_add");
    launch.post_ops.post_add_relu = post_bool(conv_node, "post_add_relu") ||
                                    post_bool(conv_node, "post_add_relu6");
    const TensorSpec *res = find_tensor(graph, residual);
    const TensorSpec *ofm = find_tensor(graph, output_of(conv_node, "ofm"));
    if (res && ofm) {
      launch.residual_shift = ofm->frac - res->frac;
      launch.post_ops.residual_shift =
          post_int(conv_node, "residual_shift", launch.residual_shift);
    }
  }
  return launch;
}

// Fuse an eligible dense/pointwise producer and depthwise consumer into one launch.
// Keep this list beside the switches themselves; a switch that changes
// descriptors without appearing here would be invisible in the run record.
std::string schedule_flag_summary_impl() {
  std::string summary;
  const auto append = [&summary](const std::string &entry) {
    if (!summary.empty()) summary += " ";
    summary += entry;
  };
  if (deeptile::env_value("PACK_EPOCHS") != nullptr) {
    append("PACK_EPOCHS");  // Disables cost-guided epoch selection.
  }
  if (const char *fe = deeptile::env_value("FORCE_EPOCHS")) {
    if (std::atoi(fe) > 0) append(std::string("FORCE_EPOCHS=") + fe);
  }
  if (deeptile::env_flag("DISABLE_K1_CACHE")) {
    append("DISABLE_K1_CACHE");
  }
  if (deeptile::env_flag("DISABLE_CIG_CHAIN")) {
    append("DISABLE_CIG_CHAIN");
  }
  return summary;
}

bool fuse_dw_enabled() {
  const char *env = deeptile::env_value("FUSE_DW");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

// Depthwise nodes use groups == Cin == Cout and [C][1][K][K] weights.
bool node_is_depthwise(const GraphSpec &graph, const NodeSpec &node) {
  if (node.op != "conv2d") {
    return false;
  }
  const int groups = attr_int(node, "groups", -1);
  const TensorSpec *ifm = find_tensor(graph, input_of(node, "ifm"));
  const TensorSpec *weight = find_tensor(graph, input_of(node, "weight"));
  return ifm != nullptr && weight != nullptr && ifm->shape.size() == 3 &&
         weight->shape.size() == 4 && groups == ifm->shape[0] &&
         groups == weight->shape[0] && weight->shape[1] == 1;
}

// Restrict fusion to patterns supported by the streamed depthwise stage.
bool dw_fusion_is_legal(const GraphSpec &graph, const NodeSpec &producer,
                        const NodeSpec &dw) {
  if (node_is_depthwise(graph, producer)) {
    return false;  // The streamed stage accepts a dense producer.
  }
  if (!input_of(producer, "residual").empty()) {
    return false;  // producer must be a plain conv (no residual add)
  }
  if (!input_of(dw, "residual").empty()) {
    return false;  // The depthwise consumer must not carry a residual.
  }
  const auto dw_k = attr_vec(dw, "kernel");
  const auto dw_s = attr_vec(dw, "stride");
  const auto dw_p = attr_vec(dw, "padding");
  if (dw_k.size() != 2 || dw_k[0] != dw_k[1] || (dw_k[0] != 3 && dw_k[0] != 5)) {
    return false;  // The stage computes square 3x3 or 5x5 kernels.
  }
  const int k = dw_k[0];
  if (dw_s.size() != 2 || (dw_s[0] != 1 && dw_s[0] != 2) || dw_s[0] != dw_s[1]) {
    return false;  // stride 1 or 2 only
  }
  if (dw_p != std::vector<int>(4, k / 2)) {
    return false;  // "same" padding only
  }
  // Layers wider than one line-buffer row are tiled by column with a halo;
  // choose_tile_config rejects a producer that leaves no legal tile.
  const TensorSpec *dw_ifm = find_tensor(graph, input_of(dw, "ifm"));
  if (!dw_ifm || dw_ifm->shape.size() != 3) {
    return false;
  }
  // Every channel group's weights must fit the stage's tap store.
  const int cogs = (dw_ifm->shape[0] + hw::kOcTileSize - 1) / hw::kOcTileSize;
  if (cogs > (k == 5 ? hw::kDwMaxCogs5 : hw::kDwMaxCogs3)) {
    return false;
  }
  // The stage decodes its requantization shift into 5-bit amounts.
  const TensorSpec *dw_w = find_tensor(graph, input_of(dw, "weight"));
  const TensorSpec *dw_ofm = find_tensor(graph, output_of(dw, "ofm"));
  if (!dw_w || !dw_ofm) {
    return false;
  }
  const int shift = dw_ifm->frac + dw_w->frac - dw_ofm->frac;
  return shift >= -30 && shift <= 32;
}

// Attach the depthwise consumer to an initialized producer launch.
void attach_depthwise(LaunchPlan &launch, const GraphSpec &graph,
                      const NodeSpec &dw_node) {
  const TensorSpec *dw_ifm = find_tensor(graph, input_of(dw_node, "ifm"));
  const TensorSpec *dw_w = find_tensor(graph, input_of(dw_node, "weight"));
  const TensorSpec *dw_b = find_tensor(graph, input_of(dw_node, "bias"));
  const TensorSpec *dw_ofm = find_tensor(graph, output_of(dw_node, "ofm"));
  const auto stride = attr_vec(dw_node, "stride");
  const auto padding = attr_vec(dw_node, "padding");

  const auto kernel = attr_vec(dw_node, "kernel");

  DepthwiseFusion &dw = launch.dw;
  dw.enable = true;
  dw.Kh = kernel[0];
  dw.Kw = kernel[1];
  dw.Sh = stride[0];
  dw.Sw = stride[1];
  // Select halo-aware tiling and depthwise output geometry.
  launch.post_ops.dw_enable = true;
  launch.post_ops.dw_stride = stride[0];
  launch.post_ops.dw_kernel = kernel[0];
  dw.pad_top = padding[0];
  dw.pad_bottom = padding[1];
  dw.pad_left = padding[2];
  dw.pad_right = padding[3];
  dw.frac_in = dw_ifm ? dw_ifm->frac : 0;   // = producer (PW) output frac
  dw.frac_w = dw_w ? dw_w->frac : 0;
  dw.frac_b = dw_b ? dw_b->frac : 0;
  dw.frac_out = dw_ofm ? dw_ofm->frac : 0;
  dw.shift_out = (dw.frac_in + dw.frac_w) - dw.frac_out;
  dw.shift_bias = dw.frac_out - dw.frac_b + 1;
  dw.use_relu = post_bool(dw_node, "relu") || post_bool(dw_node, "relu6");
  if (post_bool(dw_node, "relu6")) {
    dw.act_max = relu6_max(dw.frac_out);
  }
  dw.weight_tensor = input_of(dw_node, "weight");
  dw.bias_tensor = input_of(dw_node, "bias");
  if (dw_ofm && dw_ofm->shape.size() == 3) {
    dw.out_h = dw_ofm->shape[1];
    dw.out_w = dw_ofm->shape[2];
  }
}

} // namespace

std::string schedule_flag_summary() { return schedule_flag_summary_impl(); }

GraphExecutionPlan compile_graph(const GraphSpec &graph) {
  GraphExecutionPlan plan;
  plan.graph = graph;

  const auto consumers = make_consumer_counts(graph);
  std::set<std::size_t> consumed_nodes;

  // Validate semantics before fusion or lowering.
  for (const auto &node : graph.nodes) {
    validate_post_ops(node, plan.diagnostics);
    if (node.op == "conv2d") {
      validate_conv_node(graph, node, plan.diagnostics);
    } else if (node.op == "maxpool2d") {
      validate_maxpool_node(graph, node, plan.diagnostics);
    } else if (node.op == "gap2d") {
      validate_gap_node(graph, node, plan.diagnostics);
    } else if (node.op == "linear") {
      validate_linear_node(graph, node, plan.diagnostics);
    } else {
      add_error(plan.diagnostics, "node " + node.id,
                "unsupported op: " + node.op);
    }
  }
  if (has_errors(plan.diagnostics)) {
    return plan;
  }

  // Lower nodes and fuse only patterns supported by kernel post-operations.
  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    if (consumed_nodes.count(i) != 0) {
      continue;
    }
    const NodeSpec &node = graph.nodes[i];
    if (node.op == "conv2d") {
      LaunchPlan launch;
      bool fused = false;
      const std::string conv_out = output_of(node, "ofm");
      const auto consumer_it = consumers.find(conv_out);
      if (i + 1 < graph.nodes.size() && consumer_it != consumers.end() &&
          consumer_it->second == 1) {
        const NodeSpec &next = graph.nodes[i + 1];
        if (fuse_dw_enabled() && next.op == "conv2d" &&
            input_of(next, "ifm") == conv_out &&
            node_is_depthwise(graph, next) &&
            dw_fusion_is_legal(graph, node, next)) {
          // Append the streamed depthwise stage and expose its output.
          launch = make_conv_launch(graph, node, output_of(next, "ofm"),
                                    {node.id, next.id}, plan.diagnostics);
          launch.id = node.id + "_fusedw_" + next.id;
          attach_depthwise(launch, graph, next);
          consumed_nodes.insert(i + 1);
          fused = true;
        } else if (next.op == "maxpool2d" && input_of(next, "ifm") == conv_out) {
          // conv2d -> maxpool2d maps to one tile_conv launch with POOL_ENABLE.
          launch = make_conv_launch(graph, node, output_of(next, "ofm"),
                                    {node.id, next.id}, plan.diagnostics);
          launch.id = node.id + "_plus_" + next.id;
          launch.post_ops.maxpool_enable = true;
          const TensorSpec *conv_tensor = find_tensor(graph, conv_out);
          const TensorSpec *pool_tensor =
              find_tensor(graph, output_of(next, "ofm"));
          if (conv_tensor && pool_tensor) {
            launch.post_ops.post_pool_shift =
                pool_tensor->frac - conv_tensor->frac;
          }
          consumed_nodes.insert(i + 1);
          fused = true;
        } else if (next.op == "gap2d" && input_of(next, "ifm") == conv_out) {
          // conv2d -> gap2d maps to one tile_conv launch with GAP_ENABLE.
          launch = make_conv_launch(graph, node, output_of(next, "ofm"),
                                    {node.id, next.id}, plan.diagnostics);
          launch.id = node.id + "_plus_" + next.id;
          launch.post_ops.gap_enable = true;
          const TensorSpec *conv_tensor = find_tensor(graph, conv_out);
          const TensorSpec *gap_tensor =
              find_tensor(graph, output_of(next, "ofm"));
          if (conv_tensor && gap_tensor) {
            launch.gap_mul = compute_gap_scale_local(
                conv_tensor->frac, gap_tensor->frac,
                conv_tensor->shape[1] * conv_tensor->shape[2]);
          }
          consumed_nodes.insert(i + 1);
          fused = true;
        }
      }

      if (!fused) {
        launch = make_conv_launch(graph, node, conv_out, {node.id},
                                  plan.diagnostics);
      }
      if (finalize_launch(launch, plan.diagnostics)) {
        plan.launches.push_back(std::move(launch));
      }
    } else if (node.op == "linear") {
      // Linear is represented as a 1x1 convolution so the backend can reuse
      // the same weight packing, descriptor, and kernel launch path.
      LaunchPlan launch;
      launch.id = node.id;
      launch.source_nodes = {node.id};
      launch.ifm_tensor = input_of(node, "ifm");
      launch.weight_tensor = input_of(node, "weight");
      launch.bias_tensor = input_of(node, "bias");
      launch.ofm_tensor = output_of(node, "ofm");
      launch.params = make_linear_params(graph, node, plan.diagnostics);
      if (finalize_launch(launch, plan.diagnostics)) {
        plan.launches.push_back(std::move(launch));
      }
    } else if (node.op == "maxpool2d" || node.op == "gap2d") {
      add_error(plan.diagnostics, "node " + node.id,
                "standalone " + node.op + " is not supported in v1");
    }
  }

  return plan;
}

} // namespace graph
} // namespace deeptile
