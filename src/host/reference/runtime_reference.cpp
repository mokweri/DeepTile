/**
 * @file runtime_reference.cpp
 * @brief Bit-exact software implementation of prepared graph launches.
 */

#include "reference/runtime_reference.h"

#include "deeptile/tile/hardware_spec.hpp"

#include <algorithm>
#include <cstdint>
#include <sstream>

namespace deeptile {
namespace runtime {

namespace hw = deeptile::tile::hw_defaults;

namespace {

void add_error(std::vector<RuntimeDiagnostic> &diagnostics,
               const std::string &context, const std::string &message) {
  diagnostics.push_back(
      {RuntimeDiagnosticSeverity::Error, context, message});
}

int div_ceil(int a, int b) { return (a + b - 1) / b; }

std::int64_t product(const std::vector<int> &shape) {
  std::int64_t out = 1;
  for (int dim : shape) {
    out *= dim;
  }
  return out;
}

std::int8_t clamp_i8(std::int32_t v) {
  if (v > 127) {
    return 127;
  }
  if (v < -128) {
    return -128;
  }
  return static_cast<std::int8_t>(v);
}

std::int32_t apply_signed_shift(std::int32_t val, int shift) {
  if (shift > 0) {
    return val << shift;
  }
  if (shift < 0) {
    const int s = -shift;
    const std::int32_t round_bias =
        (static_cast<std::int32_t>(1) << (s - 1)) + (val >> 31);
    return (val + round_bias) >> s;
  }
  return val;
}

const RuntimeTensorBuffer *find_buffer(const RuntimeExecutionPlan &plan,
                                       const std::string &id) {
  const auto it = plan.buffer_index_by_id.find(id);
  if (it == plan.buffer_index_by_id.end()) {
    return nullptr;
  }
  return &plan.buffers[it->second];
}

std::size_t chw_index(int c, int h, int w, int height, int width) {
  return static_cast<std::size_t>(c) * height * width +
         static_cast<std::size_t>(h) * width + w;
}

std::size_t weight_oihw_index(int oc, int ic, int kh, int kw,
                              const deeptile::tile::ConvLayerParams &p) {
  return static_cast<std::size_t>(oc) * p.Cin * p.Kh * p.Kw +
         static_cast<std::size_t>(ic) * p.Kh * p.Kw +
         static_cast<std::size_t>(kh) * p.Kw + kw;
}

std::size_t weight_oi_index(int oc, int ic,
                            const deeptile::tile::ConvLayerParams &p) {
  return static_cast<std::size_t>(oc) * p.Cin + ic;
}

// Depthwise weight layout is [Cout][1][Kh][Kw].
std::size_t weight_dw_index(int oc, int kh, int kw,
                            const deeptile::tile::ConvLayerParams &p) {
  return static_cast<std::size_t>(oc) * p.Kh * p.Kw +
         static_cast<std::size_t>(kh) * p.Kw + kw;
}

std::vector<std::int8_t>
reference_conv_quantize(const RuntimeReferenceTensor &ifm,
                        const std::vector<std::int8_t> &weight,
                        const std::vector<std::int32_t> &bias_i32,
                        const deeptile::graph::LaunchPlan &launch,
                        bool linear_weight) {
  const auto &p = launch.params;
  const int oh_dim = p.output_height();
  const int ow_dim = p.output_width();
  std::vector<std::int8_t> out(
      static_cast<std::size_t>(p.Cout) * oh_dim * ow_dim, 0);

  const bool relu_on =
      p.use_relu && !launch.post_ops.residual_add_enable;

  for (int oc = 0; oc < p.Cout; ++oc) {
    const std::int32_t bias =
        oc < static_cast<int>(bias_i32.size()) ? bias_i32[oc] : 0;
    for (int oh = 0; oh < oh_dim; ++oh) {
      for (int ow = 0; ow < ow_dim; ++ow) {
        std::int32_t acc = 0;
        // Depthwise output channel oc reduces only input channel oc.
        const int ic_lo = p.depthwise ? oc : 0;
        const int ic_hi = p.depthwise ? oc + 1 : p.Cin;
        for (int ic = ic_lo; ic < ic_hi; ++ic) {
          for (int kh = 0; kh < p.Kh; ++kh) {
            for (int kw = 0; kw < p.Kw; ++kw) {
              const int ih = oh * p.Sh + kh - p.pad_top;
              const int iw = ow * p.Sw + kw - p.pad_left;
              if (ih < 0 || ih >= p.H || iw < 0 || iw >= p.W) {
                continue;
              }
              const std::int32_t x =
                  ifm.data_chw[chw_index(ic, ih, iw, p.H, p.W)];
              const std::size_t widx =
                  p.depthwise  ? weight_dw_index(oc, kh, kw, p)
                  : linear_weight ? weight_oi_index(oc, ic, p)
                                  : weight_oihw_index(oc, ic, kh, kw, p);
              const std::int32_t w = weight[widx];
              acc += x * w;
            }
          }
        }

        std::int32_t s1;
        if (launch.shift_out > 0) {
          s1 = (acc >> (launch.shift_out - 1));
        } else if (launch.shift_out == 0) {
          s1 = (acc << 1);
        } else {
          s1 = (acc << ((-launch.shift_out) + 1));
        }

        std::int32_t y = (s1 + bias + 1) >> 1;
        y = std::min(std::max(y, -128), 127);
        if (relu_on) {
          y = std::min(std::max(y, 0), p.act_max);
        }
        out[chw_index(oc, oh, ow, oh_dim, ow_dim)] =
            static_cast<std::int8_t>(y);
      }
    }
  }

  return out;
}

// Match the standalone depthwise requantization path for fused execution.
std::vector<std::int8_t>
reference_depthwise_stage(const std::vector<std::int8_t> &ifm_chw, int channels,
                          int in_h, int in_w,
                          const std::vector<std::int8_t> &weight,
                          const std::vector<std::int8_t> &bias_raw,
                          const deeptile::graph::DepthwiseFusion &dw) {
  const int oh_dim = dw.out_h;
  const int ow_dim = dw.out_w;
  std::vector<std::int8_t> out(
      static_cast<std::size_t>(channels) * oh_dim * ow_dim, 0);
  for (int c = 0; c < channels; ++c) {
    std::int32_t bias =
        c < static_cast<int>(bias_raw.size()) ? bias_raw[c] : 0;
    bias = dw.shift_bias >= 0 ? (bias << dw.shift_bias)
                              : (bias >> -dw.shift_bias);
    for (int oh = 0; oh < oh_dim; ++oh) {
      for (int ow = 0; ow < ow_dim; ++ow) {
        std::int32_t acc = 0;
        for (int kh = 0; kh < dw.Kh; ++kh) {
          for (int kw = 0; kw < dw.Kw; ++kw) {
            const int ih = oh * dw.Sh + kh - dw.pad_top;
            const int iw = ow * dw.Sw + kw - dw.pad_left;
            if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) {
              continue;
            }
            const std::int32_t x = ifm_chw[chw_index(c, ih, iw, in_h, in_w)];
            const std::int32_t w =
                weight[static_cast<std::size_t>(c) * dw.Kh * dw.Kw +
                       static_cast<std::size_t>(kh) * dw.Kw + kw];
            acc += x * w;
          }
        }
        std::int32_t s1;
        if (dw.shift_out > 0) {
          s1 = (acc >> (dw.shift_out - 1));
        } else if (dw.shift_out == 0) {
          s1 = (acc << 1);
        } else {
          s1 = (acc << ((-dw.shift_out) + 1));
        }
        std::int32_t y = (s1 + bias + 1) >> 1;
        y = std::min(std::max(y, -128), 127);
        if (dw.use_relu) {
          y = std::min(std::max(y, 0), dw.act_max);
        }
        out[chw_index(c, oh, ow, oh_dim, ow_dim)] =
            static_cast<std::int8_t>(y);
      }
    }
  }
  return out;
}

std::vector<std::int8_t>
reference_residual_add(const std::vector<std::int8_t> &main_chw,
                       const std::vector<std::int8_t> &residual_chw,
                       int residual_shift, bool post_add_relu,
                       int act_max) {
  std::vector<std::int8_t> out(main_chw.size(), 0);
  for (std::size_t i = 0; i < main_chw.size(); ++i) {
    std::int32_t sum = static_cast<std::int32_t>(main_chw[i]) +
                       apply_signed_shift(
                           static_cast<std::int32_t>(residual_chw[i]),
                           residual_shift);
    if (sum > 127) {
      sum = 127;
    } else if (sum < -128) {
      sum = -128;
    }
    if (post_add_relu) {
      sum = std::min(std::max(sum, 0), act_max);
    }
    out[i] = static_cast<std::int8_t>(sum);
  }
  return out;
}

std::vector<std::int8_t>
reference_maxpool3x3s2p1(const std::vector<std::int8_t> &input,
                         int channels, int height, int width,
                         int post_pool_shift) {
  const int out_h = div_ceil(height, 2);
  const int out_w = div_ceil(width, 2);
  std::vector<std::int8_t> out(
      static_cast<std::size_t>(channels) * out_h * out_w, 0);

  for (int c = 0; c < channels; ++c) {
    for (int oh = 0; oh < out_h; ++oh) {
      for (int ow = 0; ow < out_w; ++ow) {
        std::int32_t max_val = -128;
        for (int kh = 0; kh < 3; ++kh) {
          for (int kw = 0; kw < 3; ++kw) {
            const int ih = oh * 2 + kh - 1;
            const int iw = ow * 2 + kw - 1;
            std::int32_t value = -128;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
              value = input[chw_index(c, ih, iw, height, width)];
            }
            max_val = std::max(max_val, value);
          }
        }
        out[chw_index(c, oh, ow, out_h, out_w)] =
            clamp_i8(apply_signed_shift(max_val, post_pool_shift));
      }
    }
  }
  return out;
}

std::vector<std::int8_t>
reference_gap(const std::vector<std::int8_t> &input, int channels, int height,
              int width, int gap_mul) {
  std::vector<std::int8_t> out(static_cast<std::size_t>(channels), 0);
  const std::int64_t round_bias =
      static_cast<std::int64_t>(1) << (hw::kGapScaleFracBits - 1);

  for (int c = 0; c < channels; ++c) {
    std::int32_t sum = 0;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        sum += input[chw_index(c, h, w, height, width)];
      }
    }
    const std::int64_t scaled =
        (static_cast<std::int64_t>(sum) * gap_mul + round_bias) >>
        hw::kGapScaleFracBits;
    out[static_cast<std::size_t>(c)] =
        clamp_i8(static_cast<std::int32_t>(scaled));
  }
  return out;
}

} // namespace

const RuntimeReferenceTensor *
find_reference_tensor(const RuntimeReferenceResult &result,
                      const std::string &tensor_id) {
  const auto it = result.tensors.find(tensor_id);
  return it == result.tensors.end() ? nullptr : &it->second;
}

RuntimeReferenceResult
generate_bit_exact_reference(const RuntimeExecutionPlan &plan) {
  RuntimeReferenceResult result;

  for (const auto &input_id : plan.graph_plan.graph.inputs) {
    const RuntimeTensorBuffer *buffer = find_buffer(plan, input_id);
    if (!buffer || buffer->raw_data.empty()) {
      add_error(result.diagnostics, "tensor " + input_id,
                "input raw data is unavailable");
      continue;
    }
    result.tensors[input_id] =
        RuntimeReferenceTensor{input_id, buffer->shape, buffer->raw_data};
  }
  if (has_errors(result.diagnostics)) {
    return result;
  }

  for (std::size_t i = 0; i < plan.graph_plan.launches.size(); ++i) {
    const auto &launch = plan.graph_plan.launches[i];
    const auto &binding = plan.launches[i];
    const RuntimeReferenceTensor *ifm =
        find_reference_tensor(result, launch.ifm_tensor);
    const RuntimeTensorBuffer *weight = find_buffer(plan, launch.weight_tensor);
    const RuntimeTensorBuffer *bias = find_buffer(plan, binding.bias_buffer);

    if (!ifm || !weight || weight->raw_data.empty() || !bias ||
        bias->int32_data.empty()) {
      add_error(result.diagnostics, "launch " + launch.id,
                "missing IFM, weight, or bias reference input");
      continue;
    }

    const bool linear_weight =
        weight->logical_layout == "OI" || launch.params.Kh == 1;
    std::vector<std::int8_t> stage = reference_conv_quantize(
        *ifm, weight->raw_data, bias->int32_data, launch, linear_weight);

    int stage_h = launch.params.output_height();
    int stage_w = launch.params.output_width();

    if (launch.dw.enable) {
      // Apply the depthwise stage to the producer output.
      const RuntimeTensorBuffer *dw_w =
          find_buffer(plan, launch.dw.weight_tensor);
      const RuntimeTensorBuffer *dw_b =
          find_buffer(plan, launch.dw.bias_tensor);
      if (!dw_w || dw_w->raw_data.empty() || !dw_b ||
          dw_b->raw_data.empty()) {
        add_error(result.diagnostics, "launch " + launch.id,
                  "missing fused depthwise weight/bias reference input");
        continue;
      }
      stage = reference_depthwise_stage(stage, launch.params.Cout, stage_h,
                                        stage_w, dw_w->raw_data,
                                        dw_b->raw_data, launch.dw);
      stage_h = launch.dw.out_h;
      stage_w = launch.dw.out_w;
    }

    if (launch.post_ops.residual_add_enable) {
      const RuntimeReferenceTensor *residual =
          launch.residual_tensor.has_value()
              ? find_reference_tensor(result, *launch.residual_tensor)
              : nullptr;
      if (!residual || residual->data_chw.size() != stage.size()) {
        add_error(result.diagnostics, "launch " + launch.id,
                  "missing or mismatched residual reference tensor");
        continue;
      }
      stage = reference_residual_add(stage, residual->data_chw,
                                     launch.post_ops.residual_shift,
                                     launch.post_ops.post_add_relu,
                                     launch.params.act_max);
    }

    if (launch.post_ops.maxpool_enable) {
      stage = reference_maxpool3x3s2p1(stage, launch.params.Cout, stage_h,
                                       stage_w,
                                       launch.post_ops.post_pool_shift);
      stage_h = div_ceil(stage_h, 2);
      stage_w = div_ceil(stage_w, 2);
    }

    if (launch.post_ops.gap_enable) {
      stage = reference_gap(stage, launch.params.Cout, stage_h, stage_w,
                            launch.gap_mul);
      stage_h = 1;
      stage_w = 1;
    }

    RuntimeReferenceTensor out;
    out.id = launch.ofm_tensor;
    out.shape = {launch.params.Cout, stage_h, stage_w};
    out.data_chw = std::move(stage);
    const std::int64_t expected = product(out.shape);
    if (expected != static_cast<std::int64_t>(out.data_chw.size())) {
      std::ostringstream oss;
      oss << "reference output size mismatch expected " << expected
          << " got " << out.data_chw.size();
      add_error(result.diagnostics, "launch " + launch.id, oss.str());
      continue;
    }
    result.tensors[out.id] = std::move(out);
  }

  return result;
}

} // namespace runtime
} // namespace deeptile
