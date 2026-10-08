/**
 * @file stem_fold.cpp
 * @brief Implements the space-to-depth stem rewrite.
 */

#include "runtime/stem_fold.h"

#include "deeptile/compiler/environment.h"

#include <filesystem>
#include <fstream>

namespace deeptile {
namespace runtime {
namespace {

bool read_i8(const std::filesystem::path &path, std::size_t count,
             std::vector<std::int8_t> &out, std::string &error) {
  std::ifstream in(path, std::ios::binary);
  out.assign(count, 0);
  if (!in || !in.read(reinterpret_cast<char *>(out.data()),
                      static_cast<std::streamsize>(count)) ||
      in.peek() != std::char_traits<char>::eof()) {
    error = "cannot read " + std::to_string(count) + " bytes from " +
            path.string();
    return false;
  }
  return true;
}

const std::vector<int> *vec_attr(const deeptile::graph::NodeSpec &node,
                                 const char *name) {
  const auto it = node.attrs_vec.find(name);
  return it == node.attrs_vec.end() ? nullptr : &it->second;
}

} // namespace

bool stem_fold_requested() {
  // On by default; DEEPTILE_STEM_S2D=0 (or empty) keeps the stem as exported.
  const char *value = deeptile::env_value("STEM_S2D");
  return value == nullptr || (value[0] != '\0' && value[0] != '0');
}

bool fold_stem(deeptile::graph::GraphSpec &graph, StemFold &fold,
               TensorOverrides &overrides, std::string &error) {
  fold = StemFold{};
  if (graph.inputs.size() != 1) {
    return true;
  }
  const std::string input_id = graph.inputs[0];
  auto input_it = graph.tensors.find(input_id);
  if (input_it == graph.tensors.end() || input_it->second.layout != "CHW" ||
      input_it->second.shape.size() != 3) {
    return true;
  }

  // The input must feed exactly one node, a dense stride-2 convolution.
  deeptile::graph::NodeSpec *stem = nullptr;
  for (auto &node : graph.nodes) {
    for (const auto &[slot, tensor] : node.inputs) {
      if (tensor != input_id) {
        continue;
      }
      if (stem != nullptr || node.op != "conv2d" || slot != "ifm") {
        return true;
      }
      stem = &node;
    }
  }
  if (stem == nullptr) {
    return true;
  }
  const auto *kernel = vec_attr(*stem, "kernel");
  const auto *stride = vec_attr(*stem, "stride");
  const auto *padding = vec_attr(*stem, "padding");
  const auto *dilation = vec_attr(*stem, "dilation");
  const auto groups = stem->attrs_int.find("groups");
  if (!kernel || !stride || !padding || kernel->size() != 2 ||
      stride->size() != 2 || padding->size() != 4 ||
      (dilation && *dilation != std::vector<int>({1, 1})) ||
      (groups != stem->attrs_int.end() && groups->second != 1)) {
    return true;
  }
  const int k = (*kernel)[0];
  const int p = (*padding)[0];
  if ((*kernel)[1] != k || k % 2 == 0 || k > 7 || (*stride)[0] != 2 ||
      (*stride)[1] != 2 || (*padding)[1] != p || (*padding)[2] != p ||
      (*padding)[3] != p) {
    return true;
  }

  const int c = input_it->second.shape[0];
  const int h = input_it->second.shape[1];
  const int w = input_it->second.shape[2];
  if (c < 1 || 4 * c > 16) {
    return true;
  }
  // Pad, rounding the padded size up to even so it folds into whole blocks.
  const int hp = h + 2 * p + (h + 2 * p) % 2;
  const int wp = w + 2 * p + (w + 2 * p) % 2;
  const int kf = (k + 1) / 2;
  // The folded convolution must produce the original output size.
  if (hp / 2 - kf + 1 != (h + 2 * p - k) / 2 + 1 ||
      wp / 2 - kf + 1 != (w + 2 * p - k) / 2 + 1) {
    return true;
  }

  const auto weight_slot = stem->inputs.find("weight");
  if (weight_slot == stem->inputs.end()) {
    return true;
  }
  const std::string weight_id = weight_slot->second;
  for (const auto &node : graph.nodes) {
    if (&node == stem) {
      continue;
    }
    for (const auto &[slot, tensor] : node.inputs) {
      (void)slot;
      if (tensor == weight_id) {
        return true;  // Shared weights would change for the other user too.
      }
    }
  }
  auto weight_it = graph.tensors.find(weight_id);
  if (weight_it == graph.tensors.end() || weight_it->second.layout != "OIHW" ||
      weight_it->second.shape.size() != 4 || !weight_it->second.file ||
      weight_it->second.shape[1] != c || weight_it->second.shape[2] != k ||
      weight_it->second.shape[3] != k) {
    return true;
  }

  const std::filesystem::path root(graph.root_dir);
  const int o = weight_it->second.shape[0];
  std::vector<std::int8_t> weights;
  if (!read_i8(root / *weight_it->second.file,
               static_cast<std::size_t>(o) * c * k * k, weights, error)) {
    return false;
  }

  fold.active = true;
  fold.input_id = input_id;
  fold.node_id = stem->id;
  fold.channels = c;
  fold.height = h;
  fold.width = w;
  fold.pad = p;
  fold.folded_height = hp / 2;
  fold.folded_width = wp / 2;

  // Folded weight [O][4C][kf][kf]: tap (a, b) at block offset (dy, dx) is the
  // original tap (2a + dy, 2b + dx), or zero past an odd kernel's edge.
  const int cf = 4 * c;
  std::vector<std::int8_t> folded(static_cast<std::size_t>(o) * cf * kf * kf, 0);
  for (int oc = 0; oc < o; ++oc) {
    for (int dy = 0; dy < 2; ++dy) {
      for (int dx = 0; dx < 2; ++dx) {
        for (int ic = 0; ic < c; ++ic) {
          const int lane = stem_fold_lane(dy, dx, ic, c);
          for (int a = 0; a < kf; ++a) {
            for (int b = 0; b < kf; ++b) {
              const int ky = 2 * a + dy;
              const int kx = 2 * b + dx;
              if (ky >= k || kx >= k) {
                continue;
              }
              folded[((static_cast<std::size_t>(oc) * cf + lane) * kf + a) * kf + b] =
                  weights[((static_cast<std::size_t>(oc) * c + ic) * k + ky) * k + kx];
            }
          }
        }
      }
    }
  }
  overrides[weight_id] = std::move(folded);
  weight_it->second.shape = {o, cf, kf, kf};

  if (input_it->second.file) {
    std::vector<std::int8_t> input;
    if (!read_i8(root / *input_it->second.file,
                 static_cast<std::size_t>(c) * h * w, input, error)) {
      fold = StemFold{};
      return false;
    }
    overrides[input_id] = fold_input_chw(input, fold);
  }
  input_it->second.shape = {cf, fold.folded_height, fold.folded_width};

  stem->attrs_vec["kernel"] = {kf, kf};
  stem->attrs_vec["stride"] = {1, 1};
  stem->attrs_vec["padding"] = {0, 0, 0, 0};
  return true;
}

std::vector<std::int8_t> fold_input_chw(const std::vector<std::int8_t> &chw,
                                        const StemFold &fold) {
  const int c = fold.channels;
  const int hf = fold.folded_height;
  const int wf = fold.folded_width;
  std::vector<std::int8_t> out(static_cast<std::size_t>(4 * c) * hf * wf, 0);
  for (int dy = 0; dy < 2; ++dy) {
    for (int dx = 0; dx < 2; ++dx) {
      for (int ic = 0; ic < c; ++ic) {
        const int lane = stem_fold_lane(dy, dx, ic, c);
        for (int i = 0; i < hf; ++i) {
          const int y = 2 * i + dy - fold.pad;
          if (y < 0 || y >= fold.height) {
            continue;
          }
          for (int j = 0; j < wf; ++j) {
            const int x = 2 * j + dx - fold.pad;
            if (x < 0 || x >= fold.width) {
              continue;
            }
            out[(static_cast<std::size_t>(lane) * hf + i) * wf + j] =
                chw[(static_cast<std::size_t>(ic) * fold.height + y) * fold.width + x];
          }
        }
      }
    }
  }
  return out;
}

} // namespace runtime
} // namespace deeptile
