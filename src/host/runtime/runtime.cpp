/**
 * @file runtime.cpp
 * @brief Runtime-plan preparation, tensor packing, and activation-slot planning.
 */

#include "runtime/runtime.h"

#include "deeptile/compiler/graph_compiler.h"
#include "deeptile/compiler/graph_loader.h"
#include "deeptile/tile/hardware_spec.hpp"

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <utility>

namespace deeptile {
namespace runtime {

std::uint64_t realtime_now_ns() {
  // clock_gettime is used rather than std::chrono::system_clock so the clock
  // is named explicitly: it must be the same CLOCK_REALTIME the IIO drivers
  // timestamp power samples with, or the two cannot be aligned.
  timespec now = {};
  clock_gettime(CLOCK_REALTIME, &now);
  return static_cast<std::uint64_t>(now.tv_sec) * 1000000000ULL +
         static_cast<std::uint64_t>(now.tv_nsec);
}

namespace hw = deeptile::tile::hw_defaults;

namespace {

void add_diag(std::vector<RuntimeDiagnostic> &diagnostics,
              RuntimeDiagnosticSeverity severity, const std::string &context,
              const std::string &message) {
  diagnostics.push_back({severity, context, message});
}

void add_error(std::vector<RuntimeDiagnostic> &diagnostics,
               const std::string &context, const std::string &message) {
  add_diag(diagnostics, RuntimeDiagnosticSeverity::Error, context, message);
}

int div_ceil(int a, int b) { return (a + b - 1) / b; }

std::int64_t product(const std::vector<int> &shape) {
  std::int64_t out = 1;
  for (int dim : shape) {
    out *= dim;
  }
  return out;
}

const deeptile::graph::TensorSpec *
find_tensor(const deeptile::graph::GraphSpec &graph, const std::string &id) {
  const auto it = graph.tensors.find(id);
  return it == graph.tensors.end() ? nullptr : &it->second;
}

RuntimeTensorBuffer *find_buffer(RuntimeExecutionPlan &plan,
                                 const std::string &id) {
  const auto it = plan.buffer_index_by_id.find(id);
  if (it == plan.buffer_index_by_id.end()) {
    return nullptr;
  }
  return &plan.buffers[it->second];
}

RuntimeTensorBuffer &upsert_buffer(RuntimeExecutionPlan &plan,
                                   RuntimeTensorBuffer buffer) {
  const auto it = plan.buffer_index_by_id.find(buffer.id);
  if (it != plan.buffer_index_by_id.end()) {
    plan.buffers[it->second] = std::move(buffer);
    return plan.buffers[it->second];
  }
  const std::size_t index = plan.buffers.size();
  plan.buffers.push_back(std::move(buffer));
  plan.buffer_index_by_id[plan.buffers.back().id] = index;
  return plan.buffers.back();
}

std::vector<std::int8_t>
read_i8_file(const std::filesystem::path &path, std::int64_t expected_bytes,
             std::vector<RuntimeDiagnostic> &diagnostics,
             const std::string &context) {
  std::vector<std::int8_t> data;
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    add_error(diagnostics, context, "cannot open file: " + path.string());
    return data;
  }
  const std::streamsize size = input.tellg();
  input.seekg(0, std::ios::beg);
  if (static_cast<std::int64_t>(size) != expected_bytes) {
    std::ostringstream oss;
    oss << "file size mismatch for " << path.string() << " expected "
        << expected_bytes << " bytes got " << size;
    add_error(diagnostics, context, oss.str());
    return data;
  }
  data.resize(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char *>(data.data()), size);
  if (!input) {
    add_error(diagnostics, context, "failed reading file: " + path.string());
    data.clear();
  }
  return data;
}

std::vector<std::uint8_t> pack_chw_to_cghw16(const std::vector<std::int8_t> &src,
                                             const std::vector<int> &shape,
                                             int cpack) {
  // Pack exported CHW activations into channel-grouped pixel vectors.
  const int channels = shape[0];
  const int height = shape[1];
  const int width = shape[2];
  const int groups = div_ceil(channels, cpack);
  std::vector<std::uint8_t> packed(
      static_cast<std::size_t>(groups) * height * width * cpack, 0);

  for (int cg = 0; cg < groups; ++cg) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int lane = 0; lane < cpack; ++lane) {
          const int c = cg * cpack + lane;
          if (c >= channels) {
            continue;
          }
          const std::size_t src_idx =
              static_cast<std::size_t>(c) * height * width +
              static_cast<std::size_t>(h) * width + w;
          const std::size_t dst_idx =
              static_cast<std::size_t>(cg) * height * width * cpack +
              static_cast<std::size_t>(h) * width * cpack +
              static_cast<std::size_t>(w) * cpack + lane;
          packed[dst_idx] = static_cast<std::uint8_t>(src[src_idx]);
        }
      }
    }
  }
  return packed;
}

std::vector<std::uint8_t>
pack_oihw_to_uram_banks(const std::vector<std::int8_t> &src,
                        const deeptile::tile::ConvLayerParams &params, int cpo,
                        int cpi, bool linear_weight) {
  // Each launch owns banked weights, keeping descriptor offsets layer-relative.
  const int out_channels = params.Cout;
  const int in_channels = params.Cin;
  const int kh_dim = linear_weight ? 1 : params.Kh;
  const int kw_dim = linear_weight ? 1 : params.Kw;
  const int oc_tiles = div_ceil(out_channels, cpo);
  const int ic_tiles = div_ceil(in_channels, cpi);
  const int total_layer_depth = oc_tiles * ic_tiles * kh_dim * kw_dim;
  std::vector<std::uint8_t> packed(
      static_cast<std::size_t>(cpo) * total_layer_depth * cpi, 0);

  for (int oc_lane = 0; oc_lane < cpo; ++oc_lane) {
    const std::size_t bank_base =
        static_cast<std::size_t>(oc_lane) * total_layer_depth * cpi;
    for (int oc_tile = 0; oc_tile < oc_tiles; ++oc_tile) {
      const int global_oc = oc_tile * cpo + oc_lane;
      for (int ic_tile = 0; ic_tile < ic_tiles; ++ic_tile) {
        for (int kh = 0; kh < kh_dim; ++kh) {
          for (int kw = 0; kw < kw_dim; ++kw) {
            const int depth_pos =
                (oc_tile * ic_tiles + ic_tile) * kh_dim * kw_dim + kh * kw_dim +
                kw;
            for (int ic_lane = 0; ic_lane < cpi; ++ic_lane) {
              const int global_ic = ic_tile * cpi + ic_lane;
              if (global_oc >= out_channels || global_ic >= in_channels) {
                continue;
              }
              const std::size_t src_idx =
                  linear_weight
                      ? static_cast<std::size_t>(global_oc) * in_channels +
                            global_ic
                      : static_cast<std::size_t>(global_oc) * in_channels *
                                kh_dim * kw_dim +
                            static_cast<std::size_t>(global_ic) * kh_dim *
                                kw_dim +
                            static_cast<std::size_t>(kh) * kw_dim + kw;
              const std::size_t dst_idx =
                  bank_base + static_cast<std::size_t>(depth_pos) * cpi +
                  ic_lane;
              packed[dst_idx] = static_cast<std::uint8_t>(src[src_idx]);
            }
          }
        }
      }
    }
  }
  return packed;
}

// Pack compact [C][1][Kh][Kw] weights diagonally into the dense URAM bank ABI.
// Pairing CIG g with COG g makes one IC lane useful without increasing bank depth.
std::vector<std::uint8_t>
pack_dw_to_uram_banks(const std::vector<std::int8_t> &src,
                      const deeptile::tile::ConvLayerParams &params, int cpo,
                      int cpi) {
  const int channels = params.Cout;  // Depthwise requires Cin == Cout.
  const int kh_dim = params.Kh;
  const int kw_dim = params.Kw;
  const int oc_tiles = div_ceil(channels, cpo);
  const int total_layer_depth = oc_tiles * kh_dim * kw_dim;
  std::vector<std::uint8_t> packed(
      static_cast<std::size_t>(cpo) * total_layer_depth * cpi, 0);

  for (int oc_lane = 0; oc_lane < cpo; ++oc_lane) {
    const std::size_t bank_base =
        static_cast<std::size_t>(oc_lane) * total_layer_depth * cpi;
    for (int oc_tile = 0; oc_tile < oc_tiles; ++oc_tile) {
      const int channel = oc_tile * cpo + oc_lane;
      if (channel >= channels) {
          continue;  // Ragged final-tile lanes remain zero.
      }
      for (int kh = 0; kh < kh_dim; ++kh) {
        for (int kw = 0; kw < kw_dim; ++kw) {
          const int depth_pos = oc_tile * kh_dim * kw_dim + kh * kw_dim + kw;
          // The channel occupies the matching IC lane in its CIG plane.
          const std::size_t dst_idx =
              bank_base + static_cast<std::size_t>(depth_pos) * cpi + oc_lane;
          const std::size_t src_idx =
              static_cast<std::size_t>(channel) * kh_dim * kw_dim +
              static_cast<std::size_t>(kh) * kw_dim + kw;
          packed[dst_idx] = static_cast<std::uint8_t>(src[src_idx]);
        }
      }
    }
  }
  return packed;
}

// Compact fused-depthwise weight and bias ABI, per COG of cpo channels:
//   words 0..Kh*Kw-1 : weight word k, byte lane c = w[cog*cpo+c][tap k]
//   next 4 words     : biases, 4×int32 (little-endian) per word, lane-ordered
// Each COG is padded to the kernel's stride: 16 words for 3x3, 32 for 5x5.
std::vector<std::uint8_t>
pack_dw_compact(const std::vector<std::int8_t> &weight_src,
                const std::vector<std::int8_t> &bias_src, int channels, int kh,
                int kw, int shift_bias, int cpo) {
  const int taps = kh * kw;
  const int oc_tiles = div_ceil(channels, cpo);
  const int word_bytes = 16;
  // ABI: The stride must match DW_WORDS_PER_COG_3/5. An odd 13-word AXI run
  // dropped trailing bias words on hardware despite passing C-sim and co-sim.
  const int words_per_cog = taps > DW_TAPS_3 ? DW_WORDS_PER_COG_5 : DW_WORDS_PER_COG_3;
  std::vector<std::uint8_t> blob(
      static_cast<std::size_t>(oc_tiles) * words_per_cog * word_bytes, 0);
      
  for (int cog = 0; cog < oc_tiles; ++cog) {
    const std::size_t base =
        static_cast<std::size_t>(cog) * words_per_cog * word_bytes;
    for (int k = 0; k < taps; ++k) {
      for (int lane = 0; lane < cpo; ++lane) {
        const int ch = cog * cpo + lane;
        if (ch >= channels) {
          continue;  // Ragged final-tile lanes remain zero.
        }
        blob[base + static_cast<std::size_t>(k) * word_bytes + lane] =
            static_cast<std::uint8_t>(
                weight_src[static_cast<std::size_t>(ch) * taps + k]);
      }
    }
    for (int lane = 0; lane < cpo; ++lane) {
      const int ch = cog * cpo + lane;
      std::int32_t val = 0;
      if (ch < channels) {
        val = bias_src[static_cast<std::size_t>(ch)];
        val = (shift_bias >= 0) ? (val << shift_bias) : (val >> -shift_bias);
      }
      const std::size_t off = base +
                              static_cast<std::size_t>(taps + lane / 4) * word_bytes +
                              static_cast<std::size_t>(lane % 4) * 4;
      blob[off + 0] = static_cast<std::uint8_t>(val & 0xff);
      blob[off + 1] = static_cast<std::uint8_t>((val >> 8) & 0xff);
      blob[off + 2] = static_cast<std::uint8_t>((val >> 16) & 0xff);
      blob[off + 3] = static_cast<std::uint8_t>((val >> 24) & 0xff);
    }
  }
  return blob;
}

std::vector<std::int32_t>
shift_and_pad_bias(const std::vector<std::int8_t> &src, int channels, int cpo,
                   int shift_bias) {
  // Widen and align exported int8 biases for the kernel's int32 bias bus.
  std::vector<std::int32_t> out(shifted_bias_count(channels, cpo), 0);
  for (int i = 0; i < channels; ++i) {
    std::int32_t value = src[static_cast<std::size_t>(i)];
    if (shift_bias >= 0) {
      value <<= shift_bias;
    } else {
      value >>= -shift_bias;
    }
    out[static_cast<std::size_t>(i)] = value;
  }
  return out;
}

std::vector<int> final_output_shape(const deeptile::graph::LaunchPlan &launch) {
  const int cout = launch.params.Cout;
  const int conv_oh = launch.params.output_height();
  const int conv_ow = launch.params.output_width();
  if (launch.dw.enable) {
    // A fused launch exposes the depthwise stage output.
    return {cout, launch.dw.out_h, launch.dw.out_w};
  }
  if (launch.post_ops.gap_enable) {
    return {cout, 1, 1};
  }
  if (launch.post_ops.maxpool_enable) {
    return {cout, div_ceil(conv_oh, 2), div_ceil(conv_ow, 2)};
  }
  return {cout, conv_oh, conv_ow};
}

void convert_graph_diags(const std::vector<deeptile::graph::GraphDiagnostic> &in,
                         std::vector<RuntimeDiagnostic> &out,
                         const std::string &prefix) {
  for (const auto &d : in) {
    out.push_back({d.severity == deeptile::graph::DiagnosticSeverity::Error
                       ? RuntimeDiagnosticSeverity::Error
                       : RuntimeDiagnosticSeverity::Warning,
                   prefix + d.context, d.message});
  }
}

bool prepare_input_buffer(RuntimeExecutionPlan &runtime,
                          const deeptile::graph::TensorSpec &tensor) {
  if (tensor.layout != "CHW" || tensor.shape.size() != 3) {
    add_error(runtime.diagnostics, "tensor " + tensor.id,
              "input activation must use CHW [C,H,W]");
    return false;
  }
  RuntimeTensorBuffer *buffer = find_buffer(runtime, tensor.id);
  if (!buffer || buffer->raw_data.empty()) {
    add_error(runtime.diagnostics, "tensor " + tensor.id,
              "input raw data was not loaded");
    return false;
  }
  buffer->packed_layout = "CgHW16";
  buffer->data_bytes =
      pack_chw_to_cghw16(buffer->raw_data, tensor.shape, hw::kIcTileSize);
  return true;
}

bool ensure_weight_buffer(RuntimeExecutionPlan &runtime,
                          const deeptile::graph::LaunchPlan &launch) {
  if (find_buffer(runtime, launch.weight_tensor) != nullptr &&
      !find_buffer(runtime, launch.weight_tensor)->data_bytes.empty()) {
    return true;
  }
  const auto &graph = runtime.graph_plan.graph;
  const auto *tensor = find_tensor(graph, launch.weight_tensor);
  if (!tensor) {
    add_error(runtime.diagnostics, "launch " + launch.id,
              "weight tensor not found: " + launch.weight_tensor);
    return false;
  }
  RuntimeTensorBuffer *raw = find_buffer(runtime, launch.weight_tensor);
  if (!raw || raw->raw_data.empty()) {
    add_error(runtime.diagnostics, "launch " + launch.id,
              "weight raw data was not loaded: " + launch.weight_tensor);
    return false;
  }
  const bool linear_weight = tensor->layout == "OI";
  if (!(tensor->layout == "OIHW" || linear_weight)) {
    add_error(runtime.diagnostics, "tensor " + tensor->id,
              "weight layout must be OIHW or OI");
    return false;
  }
  RuntimeTensorBuffer buffer = *raw;
  if (launch.params.depthwise) {
    // Pack compact depthwise weights into the dense bank interface.
    buffer.packed_layout = "URAM_BANKS_DW_DIAG";
    buffer.data_bytes = pack_dw_to_uram_banks(
        raw->raw_data, launch.params, hw::kOcTileSize, hw::kIcTileSize);
  } else {
    buffer.packed_layout = linear_weight ? "URAM_BANKS_OI_1x1" : "URAM_BANKS_OIHW";
    buffer.data_bytes = pack_oihw_to_uram_banks(
        raw->raw_data, launch.params, hw::kOcTileSize, hw::kIcTileSize,
        linear_weight);
  }
  upsert_buffer(runtime, std::move(buffer));
  return true;
}

bool ensure_bias_buffer(RuntimeExecutionPlan &runtime,
                        const deeptile::graph::LaunchPlan &launch,
                        std::string &bias_buffer_id) {
  const auto &graph = runtime.graph_plan.graph;
  const auto *tensor = find_tensor(graph, launch.bias_tensor);
  if (!tensor) {
    add_error(runtime.diagnostics, "launch " + launch.id,
              "bias tensor not found: " + launch.bias_tensor);
    return false;
  }
  RuntimeTensorBuffer *raw = find_buffer(runtime, launch.bias_tensor);
  if (!raw || raw->raw_data.empty()) {
    add_error(runtime.diagnostics, "launch " + launch.id,
              "bias raw data was not loaded: " + launch.bias_tensor);
    return false;
  }
  if (tensor->layout != "O" || tensor->shape.size() != 1 ||
      tensor->shape[0] != launch.params.Cout) {
    add_error(runtime.diagnostics, "tensor " + tensor->id,
              "bias tensor must use layout O and match Cout");
    return false;
  }

  bias_buffer_id = launch.id + ":" + launch.bias_tensor + ":bias_i32";
  if (find_buffer(runtime, bias_buffer_id) != nullptr) {
    return true;
  }

  RuntimeTensorBuffer buffer;
  buffer.id = bias_buffer_id;
  buffer.tensor_id = launch.bias_tensor;
  buffer.kind = "bias_i32";
  buffer.logical_layout = "O";
  buffer.packed_layout = "O16_INT32_SHIFTED";
  buffer.shape = {launch.params.Cout};
  buffer.element_count = shifted_bias_count(launch.params.Cout, hw::kOcTileSize);
  buffer.raw_data = raw->raw_data;
  buffer.int32_data = shift_and_pad_bias(raw->raw_data, launch.params.Cout,
                                         hw::kOcTileSize, launch.shift_bias);
  upsert_buffer(runtime, std::move(buffer));
  return true;
}

bool ensure_dw_compact_buffer(RuntimeExecutionPlan &runtime,
                              const deeptile::graph::LaunchPlan &launch,
                              std::string &dw_buffer_id) {
  const auto &dw = launch.dw;
  RuntimeTensorBuffer *w_raw = find_buffer(runtime, dw.weight_tensor);
  RuntimeTensorBuffer *b_raw = find_buffer(runtime, dw.bias_tensor);
  if (!w_raw || w_raw->raw_data.empty()) {
    add_error(runtime.diagnostics, "launch " + launch.id,
              "DW weight raw data not loaded: " + dw.weight_tensor);
    return false;
  }
  if (!b_raw || b_raw->raw_data.empty()) {
    add_error(runtime.diagnostics, "launch " + launch.id,
              "DW bias raw data not loaded: " + dw.bias_tensor);
    return false;
  }
  dw_buffer_id = launch.id + ":dw_compact";
  if (find_buffer(runtime, dw_buffer_id) != nullptr) {
    return true;
  }
  const int channels = launch.params.Cout;  // DW C == producer Cout
  RuntimeTensorBuffer buffer;
  buffer.id = dw_buffer_id;
  buffer.tensor_id = dw.weight_tensor;
  buffer.kind = "dw_weight_compact";
  buffer.logical_layout = "OIHW";
  buffer.packed_layout = dw.Kh == 5 ? "DW_COMPACT_C25_BIAS4" : "DW_COMPACT_C9_BIAS4";
  buffer.shape = {channels, 1, dw.Kh, dw.Kw};
  buffer.data_bytes = pack_dw_compact(w_raw->raw_data, b_raw->raw_data, channels,
                                      dw.Kh, dw.Kw, dw.shift_bias,
                                      hw::kOcTileSize);
  upsert_buffer(runtime, std::move(buffer));
  return true;
}

bool allocate_output_buffer(RuntimeExecutionPlan &runtime,
                            const deeptile::graph::LaunchPlan &launch) {
  const auto *tensor = find_tensor(runtime.graph_plan.graph, launch.ofm_tensor);
  if (!tensor) {
    add_error(runtime.diagnostics, "launch " + launch.id,
              "output tensor not found: " + launch.ofm_tensor);
    return false;
  }
  const std::vector<int> expected = final_output_shape(launch);
  if (tensor->layout != "CHW" || tensor->shape != expected) {
    add_error(runtime.diagnostics, "launch " + launch.id,
              "output tensor shape/layout does not match launch output");
    return false;
  }

  RuntimeTensorBuffer buffer;
  buffer.id = launch.ofm_tensor;
  buffer.tensor_id = launch.ofm_tensor;
  buffer.kind = "activation";
  buffer.logical_layout = tensor->layout;
  buffer.packed_layout = "CgHW16";
  buffer.shape = tensor->shape;
  buffer.element_count = product(tensor->shape);
  buffer.data_bytes.assign(packed_activation_bytes(tensor->shape, hw::kOcTileSize),
                           0);
  upsert_buffer(runtime, std::move(buffer));
  return true;
}

struct ActivationLifetime {
  std::string tensor_id;
  std::size_t byte_size = 0;
  std::vector<int> shape;
  int producer_launch = -1;
  int last_consumer_launch = -1;
};

int lifetime_start(const ActivationLifetime &lifetime) {
  return lifetime.producer_launch < 0 ? 0 : lifetime.producer_launch;
}

void upsert_lifetime(std::map<std::string, ActivationLifetime> &lifetimes,
                     const std::string &tensor_id, std::size_t byte_size,
                     const std::vector<int> &shape, int producer_launch) {
  auto &lifetime = lifetimes[tensor_id];
  lifetime.tensor_id = tensor_id;
  lifetime.byte_size = byte_size;
  lifetime.shape = shape;
  lifetime.producer_launch = producer_launch;
  lifetime.last_consumer_launch =
      std::max(lifetime.last_consumer_launch, producer_launch);
}

void extend_lifetime(std::map<std::string, ActivationLifetime> &lifetimes,
                     const std::string &tensor_id, int consumer_launch) {
  auto it = lifetimes.find(tensor_id);
  if (it != lifetimes.end()) {
    it->second.last_consumer_launch =
        std::max(it->second.last_consumer_launch, consumer_launch);
  }
}

bool prepare_activation_slots(RuntimeExecutionPlan &runtime) {
  // Greedily share physical slots between non-overlapping activation lifetimes.
  std::map<std::string, ActivationLifetime> lifetimes;
  const auto &graph = runtime.graph_plan.graph;
  const int launch_count = static_cast<int>(runtime.launches.size());

  for (const auto &input_id : graph.inputs) {
    const RuntimeTensorBuffer *buffer = find_buffer(runtime, input_id);
    if (!buffer || buffer->data_bytes.empty()) {
      add_error(runtime.diagnostics, "tensor " + input_id,
                "input activation buffer is unavailable for slot planning");
      continue;
    }
    upsert_lifetime(lifetimes, input_id, buffer->data_bytes.size(),
                    buffer->shape, -1);
  }

  for (std::size_t i = 0; i < runtime.launches.size(); ++i) {
    const auto &binding = runtime.launches[i];
    const int launch_index = static_cast<int>(i);
    extend_lifetime(lifetimes, binding.ifm_buffer, launch_index);
    if (binding.residual_buffer.has_value()) {
      extend_lifetime(lifetimes, *binding.residual_buffer, launch_index);
    }

    const RuntimeTensorBuffer *output =
        find_buffer(runtime, binding.output_buffer);
    if (!output || output->data_bytes.empty()) {
      add_error(runtime.diagnostics, "tensor " + binding.output_buffer,
                "output activation buffer is unavailable for slot planning");
      continue;
    }
    upsert_lifetime(lifetimes, binding.output_buffer,
                    output->data_bytes.size(), output->shape, launch_index);
  }

  for (const auto &output_id : graph.outputs) {
    extend_lifetime(lifetimes, output_id, launch_count);
  }

  if (has_errors(runtime.diagnostics)) {
    return false;
  }

  std::vector<ActivationLifetime> ordered;
  ordered.reserve(lifetimes.size());
  for (const auto &entry : lifetimes) {
    ordered.push_back(entry.second);
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const ActivationLifetime &a, const ActivationLifetime &b) {
              const int start_a = lifetime_start(a);
              const int start_b = lifetime_start(b);
              if (start_a != start_b) {
                return start_a < start_b;
              }
              if (a.last_consumer_launch != b.last_consumer_launch) {
                return a.last_consumer_launch > b.last_consumer_launch;
              }
              return a.tensor_id < b.tensor_id;
            });

  std::vector<int> slot_live_until;
  for (const auto &lifetime : ordered) {
    const int start = lifetime_start(lifetime);
    int chosen = -1;
    std::size_t chosen_capacity = 0;
    for (std::size_t i = 0; i < runtime.activation_slots.size(); ++i) {
      const auto &slot = runtime.activation_slots[i];
      // A slot can be reused only after its previous tensor's last consumer.
      if (slot_live_until[i] >= start ||
          slot.byte_capacity < lifetime.byte_size) {
        continue;
      }
      if (chosen < 0 || slot.byte_capacity < chosen_capacity) {
        chosen = static_cast<int>(i);
        chosen_capacity = slot.byte_capacity;
      }
    }

    if (chosen < 0) {
      RuntimeActivationSlot slot;
      slot.id =
          "act_slot_" + std::to_string(runtime.activation_slots.size());
      slot.byte_capacity = lifetime.byte_size;
      slot.data_bytes.assign(slot.byte_capacity, 0);
      runtime.activation_slots.push_back(std::move(slot));
      slot_live_until.push_back(-1);
      chosen = static_cast<int>(runtime.activation_slots.size() - 1);
    }

    RuntimeActivationSlot &slot =
        runtime.activation_slots[static_cast<std::size_t>(chosen)];
    slot.current_tensor_id = lifetime.tensor_id;
    slot.assigned_tensors.push_back(lifetime.tensor_id);
    slot_live_until[static_cast<std::size_t>(chosen)] =
        lifetime.last_consumer_launch;

    const RuntimeTensorBuffer *logical =
        find_buffer(runtime, lifetime.tensor_id);
    if (logical && !logical->data_bytes.empty() && lifetime.producer_launch < 0) {
      std::copy(logical->data_bytes.begin(), logical->data_bytes.end(),
                slot.data_bytes.begin());
    }

    RuntimeTensorStorageBinding storage;
    storage.tensor_id = lifetime.tensor_id;
    storage.slot_id = slot.id;
    storage.byte_size = lifetime.byte_size;
    storage.shape = lifetime.shape;
    storage.producer_launch = lifetime.producer_launch;
    storage.last_consumer_launch = lifetime.last_consumer_launch;
    runtime.activation_binding_by_tensor[storage.tensor_id] =
        runtime.activation_bindings.size();
    runtime.activation_bindings.push_back(std::move(storage));
  }

  return true;
}

} // namespace

std::vector<std::int8_t>
unpack_packed_activation_to_chw(const RuntimeTensorBuffer &buffer) {
  std::vector<std::int8_t> out;
  if (buffer.shape.size() != 3) {
    return out;
  }
  const int channels = buffer.shape[0];
  const int height = buffer.shape[1];
  const int width = buffer.shape[2];
  const int cpack = hw::kOcTileSize;
  out.assign(static_cast<std::size_t>(channels) * height * width, 0);

  for (int c = 0; c < channels; ++c) {
    const int group = c / cpack;
    const int lane = c % cpack;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        const std::size_t packed_idx =
            static_cast<std::size_t>(group) * height * width * cpack +
            static_cast<std::size_t>(h) * width * cpack +
            static_cast<std::size_t>(w) * cpack + lane;
        const std::size_t chw_idx =
            static_cast<std::size_t>(c) * height * width +
            static_cast<std::size_t>(h) * width + w;
        if (packed_idx < buffer.data_bytes.size()) {
          out[chw_idx] =
              static_cast<std::int8_t>(buffer.data_bytes[packed_idx]);
        }
      }
    }
  }
  return out;
}

std::vector<std::uint8_t> pack_activation_chw(const std::vector<std::int8_t> &chw,
                                              const std::vector<int> &shape) {
  return pack_chw_to_cghw16(chw, shape, hw::kIcTileSize);
}

namespace {

/** Shared validation for the two graph-input setters. */
RuntimeTensorBuffer *
resolve_graph_input(RuntimeExecutionPlan &plan, const std::string &tensor_id,
                    std::vector<RuntimeDiagnostic> &diagnostics) {
  const auto &inputs = plan.graph_plan.graph.inputs;
  if (std::find(inputs.begin(), inputs.end(), tensor_id) == inputs.end()) {
    add_error(diagnostics, "tensor " + tensor_id, "not a graph input");
    return nullptr;
  }
  RuntimeTensorBuffer *buffer = find_buffer(plan, tensor_id);
  if (!buffer) {
    add_error(diagnostics, "tensor " + tensor_id, "input buffer not prepared");
    return nullptr;
  }
  if (buffer->shape.size() != 3) {
    add_error(diagnostics, "tensor " + tensor_id,
              "input activation must use CHW [C,H,W]");
    return nullptr;
  }
  return buffer;
}

} // namespace

bool set_graph_input_packed(RuntimeExecutionPlan &plan,
                            const std::string &tensor_id,
                            std::vector<std::uint8_t> packed,
                            std::vector<RuntimeDiagnostic> &diagnostics) {
  RuntimeTensorBuffer *buffer =
      resolve_graph_input(plan, tensor_id, diagnostics);
  if (!buffer) {
    return false;
  }
  const std::size_t expected =
      packed_activation_bytes(buffer->shape, hw::kIcTileSize);
  if (packed.size() != expected) {
    add_error(diagnostics, "tensor " + tensor_id,
              "packed input size " + std::to_string(packed.size()) +
                  " does not match " + std::to_string(expected));
    return false;
  }
  buffer->packed_layout = "CgHW16";
  buffer->data_bytes = std::move(packed);
  return true;
}

bool set_graph_input_chw(RuntimeExecutionPlan &plan,
                         const std::string &tensor_id,
                         const std::vector<std::int8_t> &chw,
                         std::vector<RuntimeDiagnostic> &diagnostics) {
  RuntimeTensorBuffer *buffer =
      resolve_graph_input(plan, tensor_id, diagnostics);
  if (!buffer) {
    return false;
  }
  const StemFold &fold = plan.stem_fold;
  if (fold.active && tensor_id == fold.input_id &&
      chw.size() == static_cast<std::size_t>(fold.channels) * fold.height *
                         fold.width) {
    // The caller holds the input in its original shape; fold it to match.
    return set_graph_input_chw(plan, tensor_id, fold_input_chw(chw, fold),
                               diagnostics);
  }
  const std::size_t expected = static_cast<std::size_t>(buffer->shape[0]) *
                               static_cast<std::size_t>(buffer->shape[1]) *
                               static_cast<std::size_t>(buffer->shape[2]);
  if (chw.size() != expected) {
    add_error(diagnostics, "tensor " + tensor_id,
              "input size " + std::to_string(chw.size()) + " does not match " +
                  std::to_string(expected));
    return false;
  }

  buffer->raw_data = chw;
  buffer->packed_layout = "CgHW16";
  buffer->data_bytes =
      pack_chw_to_cghw16(buffer->raw_data, buffer->shape, hw::kIcTileSize);
  return true;
}

std::size_t packed_activation_bytes(const std::vector<int> &shape, int cpack) {
  if (shape.size() != 3) {
    return 0;
  }
  return static_cast<std::size_t>(div_ceil(shape[0], cpack)) * shape[1] *
         shape[2] * cpack;
}

std::size_t packed_weight_bytes(const deeptile::tile::ConvLayerParams &params,
                                int cpo, int cpi) {
  // Diagonal depthwise packing has no input-channel-tile depth dimension.
  if (params.depthwise) {
    return static_cast<std::size_t>(cpo) * div_ceil(params.Cout, cpo) *
           params.Kh * params.Kw * cpi;
  }
  return static_cast<std::size_t>(cpo) * div_ceil(params.Cout, cpo) *
         div_ceil(params.Cin, cpi) * params.Kh * params.Kw * cpi;
}

std::size_t shifted_bias_count(int channels, int cpo) {
  return static_cast<std::size_t>(div_ceil(channels, cpo)) * cpo;
}

const RuntimeTensorStorageBinding *
find_activation_binding(const RuntimeExecutionPlan &plan,
                        const std::string &tensor_id) {
  const auto it = plan.activation_binding_by_tensor.find(tensor_id);
  if (it == plan.activation_binding_by_tensor.end()) {
    return nullptr;
  }
  return &plan.activation_bindings[it->second];
}

const RuntimeActivationSlot *
find_activation_slot(const RuntimeExecutionPlan &plan,
                     const std::string &tensor_id) {
  const RuntimeTensorStorageBinding *binding =
      find_activation_binding(plan, tensor_id);
  if (!binding) {
    return nullptr;
  }
  for (const auto &slot : plan.activation_slots) {
    if (slot.id == binding->slot_id) {
      return &slot;
    }
  }
  return nullptr;
}

RuntimeActivationSlot *find_activation_slot(RuntimeExecutionPlan &plan,
                                            const std::string &tensor_id) {
  const RuntimeTensorStorageBinding *binding =
      find_activation_binding(plan, tensor_id);
  if (!binding) {
    return nullptr;
  }
  for (auto &slot : plan.activation_slots) {
    if (slot.id == binding->slot_id) {
      return &slot;
    }
  }
  return nullptr;
}

std::size_t activation_tensor_bytes(const RuntimeExecutionPlan &plan,
                                    const std::string &tensor_id) {
  const RuntimeTensorStorageBinding *binding =
      find_activation_binding(plan, tensor_id);
  return binding ? binding->byte_size : 0;
}

void print_activation_memory_plan(const RuntimeExecutionPlan &plan,
                                  std::ostream &out) {
  std::size_t logical_bytes = 0;
  for (const auto &binding : plan.activation_bindings) {
    logical_bytes += binding.byte_size;
  }
  std::size_t slot_bytes = 0;
  for (const auto &slot : plan.activation_slots) {
    slot_bytes += slot.byte_capacity;
  }
  const std::size_t savings =
      logical_bytes > slot_bytes ? logical_bytes - slot_bytes : 0;

  out << "[ACT_MEM] logical_activation_bytes=" << logical_bytes
      << " slot_bytes=" << slot_bytes
      << " slots=" << plan.activation_slots.size()
      << " saved_bytes=" << savings << "\n";
  for (const auto &slot : plan.activation_slots) {
    out << "[ACT_MEM] " << slot.id << " capacity=" << slot.byte_capacity
        << " tensors:";
    for (const auto &tensor_id : slot.assigned_tensors) {
      const RuntimeTensorStorageBinding *binding =
          find_activation_binding(plan, tensor_id);
      out << " " << tensor_id;
      if (binding) {
        out << "(" << binding->producer_launch << "->"
            << binding->last_consumer_launch << ")";
      }
    }
    out << "\n";
  }
}

RuntimeExecutionPlan
prepare_runtime(const deeptile::graph::GraphExecutionPlan &plan) {
  return prepare_runtime(plan, TensorOverrides{});
}

RuntimeExecutionPlan
prepare_runtime(const deeptile::graph::GraphExecutionPlan &plan,
                const TensorOverrides &overrides) {
  // Preparation remains backend-independent.
  RuntimeExecutionPlan runtime;
  runtime.graph_plan = plan;
  convert_graph_diags(plan.diagnostics, runtime.diagnostics, "compiler ");
  if (has_errors(runtime.diagnostics)) {
    return runtime;
  }

  const auto &graph = plan.graph;
  const std::filesystem::path root(graph.root_dir);

  // Load file-backed tensors as raw exported int8 bytes.
  for (const auto &[id, tensor] : graph.tensors) {
    if (!tensor.file.has_value()) {
      continue;
    }
    RuntimeTensorBuffer buffer;
    buffer.id = id;
    buffer.tensor_id = id;
    buffer.kind = tensor.kind;
    buffer.logical_layout = tensor.layout;
    buffer.packed_layout = "RAW";
    buffer.shape = tensor.shape;
    buffer.element_count = product(tensor.shape);
    const auto override_it = overrides.find(id);
    if (override_it != overrides.end()) {
      if (static_cast<std::int64_t>(override_it->second.size()) !=
          buffer.element_count) {
        add_error(runtime.diagnostics, "tensor " + id,
                  "rewritten data does not match its shape");
      }
      buffer.raw_data = override_it->second;
    } else {
      buffer.raw_data = read_i8_file(root / *tensor.file, buffer.element_count,
                                     runtime.diagnostics, "tensor " + id);
    }
    upsert_buffer(runtime, std::move(buffer));
  }
  if (has_errors(runtime.diagnostics)) {
    return runtime;
  }

  for (const auto &input_id : graph.inputs) {
    const auto *tensor = find_tensor(graph, input_id);
    if (!tensor) {
      add_error(runtime.diagnostics, "graph",
                "input tensor not found: " + input_id);
      continue;
    }
    prepare_input_buffer(runtime, *tensor);
  }
  if (has_errors(runtime.diagnostics)) {
    return runtime;
  }

  for (const auto &launch : plan.launches) {
    // Backends resolve logical tensor IDs to their preferred physical storage.
    RuntimeLaunchBinding binding;
    binding.id = launch.id;
    binding.source_nodes = launch.source_nodes;
    binding.ifm_buffer = launch.ifm_tensor;
    binding.weight_buffer = launch.weight_tensor;
    binding.output_buffer = launch.ofm_tensor;
    binding.shift_out = launch.shift_out;
    binding.shift_bias = launch.shift_bias;
    binding.residual_shift = launch.residual_shift;
    binding.gap_mul = launch.gap_mul;
    binding.kernel_size = launch.params.Kh;  // Square kernels only.
    binding.stride = launch.params.Sh;       // Symmetric stride.
    binding.act_max = launch.params.act_max;
    binding.use_ifm_cache = launch.use_ifm_cache;
    binding.spatial_tile_size = launch.tile_cfg.T_oh * launch.tile_cfg.T_ow;
    binding.num_ic_tiles = launch.params.num_cig();
    binding.ifm_cig_stride_vecs = launch.params.H * launch.params.W;
    binding.transfer_descs = launch.schedule.transfers;
    binding.compute_descs = launch.schedule.compute_descs;
    binding.num_xfers = static_cast<int>(binding.transfer_descs.size());
    binding.num_tiles = static_cast<int>(binding.compute_descs.size());
    // Partial-sum ping-pong iterates once per writeback descriptor.
    binding.num_writeback_tiles = 0;
    for (const auto &cd : binding.compute_descs) {
      if (cd.flags & static_cast<std::uint8_t>(deeptile::tile::ACCUM_WRITE_BACK)) {
        ++binding.num_writeback_tiles;
      }
    }
    binding.num_bias_vecs = div_ceil(launch.params.Cout, hw::kOcTileSize);

    const RuntimeTensorBuffer *ifm = find_buffer(runtime, binding.ifm_buffer);
    if (!ifm || ifm->data_bytes.empty()) {
      add_error(runtime.diagnostics, "launch " + launch.id,
                "IFM buffer is not prepared: " + binding.ifm_buffer);
    } else {
      binding.ifm_bytes = ifm->data_bytes.size();
    }

    if (!ensure_weight_buffer(runtime, launch)) {
      continue;
    }
    const RuntimeTensorBuffer *weight =
        find_buffer(runtime, binding.weight_buffer);
    if (weight) {
      binding.weight_bytes = weight->data_bytes.size();
      const std::size_t expected =
          packed_weight_bytes(launch.params, hw::kOcTileSize, hw::kIcTileSize);
      if (binding.weight_bytes != expected) {
        add_error(runtime.diagnostics, "launch " + launch.id,
                  "packed weight byte count mismatch");
      }
    }

    if (!ensure_bias_buffer(runtime, launch, binding.bias_buffer)) {
      continue;
    }
    const RuntimeTensorBuffer *bias = find_buffer(runtime, binding.bias_buffer);
    if (bias) {
      binding.bias_bytes = bias->int32_data.size() * sizeof(std::int32_t);
      if (bias->int32_data.size() !=
          shifted_bias_count(launch.params.Cout, hw::kOcTileSize)) {
        add_error(runtime.diagnostics, "launch " + launch.id,
                  "shifted bias count mismatch");
      }
    }

    if (launch.dw.enable) {
      // Register the fused depthwise blob and scalar launch arguments.
      if (!ensure_dw_compact_buffer(runtime, launch, binding.dw_weight_buffer)) {
        continue;
      }
      const RuntimeTensorBuffer *dw_blob =
          find_buffer(runtime, binding.dw_weight_buffer);
      if (dw_blob) {
        binding.dw_weight_bytes = dw_blob->data_bytes.size();
      }
      binding.dw_config = 1 | ((launch.dw.Sh & 0x3) << 1) |
                          ((launch.dw.use_relu ? 1 : 0) << 3) |
                          ((launch.dw.Kh == 5 ? 1 : 0) << 4) |
                          ((launch.dw.act_max & 0xFF) << 8);
      binding.dw_shift = launch.dw.shift_out;
      binding.dw_out_h = launch.dw.out_h;
      binding.dw_out_w = launch.dw.out_w;
    }

    if (launch.residual_tensor.has_value()) {
      binding.residual_buffer = *launch.residual_tensor;
      const RuntimeTensorBuffer *residual =
          find_buffer(runtime, *binding.residual_buffer);
      if (!residual || residual->data_bytes.empty()) {
        add_error(runtime.diagnostics, "launch " + launch.id,
                  "residual buffer is not prepared: " +
                      *binding.residual_buffer);
      } else {
        binding.residual_bytes = residual->data_bytes.size();
      }
    }

    if (!allocate_output_buffer(runtime, launch)) {
      continue;
    }
    const RuntimeTensorBuffer *output =
        find_buffer(runtime, binding.output_buffer);
    if (output) {
      binding.output_bytes = output->data_bytes.size();
    }

    if (binding.num_xfers !=
        static_cast<int>(launch.schedule.transfers.size())) {
      add_error(runtime.diagnostics, "launch " + launch.id,
                "transfer descriptor count mismatch");
    }
    if (binding.num_tiles !=
        static_cast<int>(launch.schedule.compute_descs.size())) {
      add_error(runtime.diagnostics, "launch " + launch.id,
                "compute descriptor count mismatch");
    }

    if (!has_errors(runtime.diagnostics)) {
      runtime.launches.push_back(std::move(binding));
    }
  }
  if (!has_errors(runtime.diagnostics)) {
    prepare_activation_slots(runtime);
  }
  return runtime;
}

bool load_compile_prepare_runtime(
    const std::string &graph_json_path, RuntimeExecutionPlan &out,
    std::vector<RuntimeDiagnostic> &diagnostics) {
  deeptile::graph::GraphSpec graph;
  std::vector<deeptile::graph::GraphDiagnostic> graph_diags;
  const bool graph_loaded =
      deeptile::graph::load_graph(graph_json_path, graph, graph_diags);
  if (!graph_diags.empty()) {
    convert_graph_diags(graph_diags, diagnostics, "loader ");
  }
  if (!graph_loaded) {
    return false;
  }

  StemFold fold;
  TensorOverrides overrides;
  if (stem_fold_requested()) {
    std::string error;
    if (!fold_stem(graph, fold, overrides, error)) {
      add_error(diagnostics, "stem-fold", error);
      return false;
    }
  }

  deeptile::graph::GraphExecutionPlan graph_plan =
      deeptile::graph::compile_graph(graph);
  if (!graph_plan.diagnostics.empty()) {
    convert_graph_diags(graph_plan.diagnostics, diagnostics, "compiler ");
  }
  if (deeptile::graph::has_errors(graph_plan.diagnostics)) {
    return false;
  }

  out = prepare_runtime(graph_plan, overrides);
  out.stem_fold = fold;
  diagnostics.insert(diagnostics.end(), out.diagnostics.begin(),
                     out.diagnostics.end());
  return !has_errors(diagnostics);
}

} // namespace runtime
} // namespace deeptile
