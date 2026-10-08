/**
 * @file hls_adapter.cpp
 * @brief Runtime-plan adapter for direct HLS C++ kernel execution.
 */

#include "backends/hls_adapter.h"

#include "tile_conv.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>

namespace deeptile {
namespace runtime {
namespace {

RuntimeTensorBuffer *find_buffer(RuntimeExecutionPlan &plan,
                                 const std::string &id) {
  const auto it = plan.buffer_index_by_id.find(id);
  if (it == plan.buffer_index_by_id.end()) {
    return nullptr;
  }
  return &plan.buffers[it->second];
}

const RuntimeTensorBuffer *find_buffer(const RuntimeExecutionPlan &plan,
                                       const std::string &id) {
  const auto it = plan.buffer_index_by_id.find(id);
  if (it == plan.buffer_index_by_id.end()) {
    return nullptr;
  }
  return &plan.buffers[it->second];
}

template <int WIDTH, typename WordT>
std::vector<WordT> pack_bytes_to_words(const std::vector<std::uint8_t> &bytes) {
  const int bytes_per_word = WIDTH / 8;
  const std::size_t words =
      (bytes.size() + bytes_per_word - 1) / bytes_per_word;
  std::vector<WordT> out(words == 0 ? 1 : words, 0);
  for (std::size_t w = 0; w < words; ++w) {
    WordT word = 0;
    for (int lane = 0; lane < bytes_per_word; ++lane) {
      const std::size_t byte_idx = w * bytes_per_word + lane;
      const std::uint8_t value =
          byte_idx < bytes.size() ? bytes[byte_idx] : 0;
      word.range((lane + 1) * 8 - 1, lane * 8) = value;
    }
    out[w] = word;
  }
  return out;
}

template <int WIDTH, typename WordT>
std::vector<std::uint8_t> unpack_words_to_bytes(const std::vector<WordT> &words,
                                                std::size_t byte_count) {
  const int bytes_per_word = WIDTH / 8;
  std::vector<std::uint8_t> out(byte_count, 0);
  for (std::size_t w = 0; w < words.size(); ++w) {
    const WordT word = words[w];
    for (int lane = 0; lane < bytes_per_word; ++lane) {
      const std::size_t byte_idx = w * bytes_per_word + lane;
      if (byte_idx >= out.size()) {
        return out;
      }
      out[byte_idx] =
          static_cast<std::uint8_t>(word.range((lane + 1) * 8 - 1,
                                               lane * 8));
    }
  }
  return out;
}

std::vector<biases_vector_dt>
pack_bias_words(const std::vector<std::int32_t> &bias_i32, int num_bias_vecs) {
  std::vector<biases_vector_dt> out(num_bias_vecs > 0 ? num_bias_vecs : 1, 0);
  for (int group = 0; group < num_bias_vecs; ++group) {
    biases_vector_dt word = 0;
    for (int lane = 0; lane < OC_TILE_SIZE; ++lane) {
      const std::size_t idx =
          static_cast<std::size_t>(group) * OC_TILE_SIZE + lane;
      const std::uint32_t value =
          idx < bias_i32.size() ? static_cast<std::uint32_t>(bias_i32[idx])
                                : 0u;
      word.range((lane + 1) * 32 - 1, lane * 32) = value;
    }
    out[group] = word;
  }
  return out;
}

template <typename DstT, typename SrcT>
std::vector<DstT> memcpy_descriptors(const std::vector<SrcT> &src) {
  static_assert(sizeof(DstT) == sizeof(SrcT),
                "Descriptor source/destination sizes must match");
  std::vector<DstT> out(src.size());
  if (!src.empty()) {
    std::memcpy(out.data(), src.data(), src.size() * sizeof(DstT));
  }
  return out;
}

bool validate_word_count(const std::string &label, std::size_t bytes,
                         int bytes_per_word, std::ostream &log) {
  if (bytes % bytes_per_word != 0) {
    log << "[HLS_ADAPTER][ERROR] " << label << " byte count " << bytes
        << " is not divisible by " << bytes_per_word << "\n";
    return false;
  }
  return true;
}

std::size_t div_round_up(std::size_t value, std::size_t divisor) {
  return (value + divisor - 1) / divisor;
}

void update_max(std::size_t &dst, std::size_t value) {
  dst = std::max(dst, value);
}

} // namespace

HlsPortDepths get_hls_port_depths(const RuntimeExecutionPlan &plan,
                                  std::size_t launch_index) {
  HlsPortDepths depths;
  if (launch_index >= plan.launches.size()) {
    return depths;
  }

  const RuntimeLaunchBinding &binding = plan.launches[launch_index];
  const RuntimeTensorBuffer *ifm = find_buffer(plan, binding.ifm_buffer);
  const RuntimeTensorBuffer *weight = find_buffer(plan, binding.weight_buffer);
  const RuntimeTensorBuffer *bias = find_buffer(plan, binding.bias_buffer);
  const RuntimeTensorBuffer *output = find_buffer(plan, binding.output_buffer);

  const std::size_t activation_word_bytes = OFM_VECTOR_WIDTH / 8;
  const std::size_t weight_word_bytes = WEIGHT_VECTOR_WIDTH / 8;

  if (ifm) {
    depths.ifm_words = div_round_up(ifm->data_bytes.size(),
                                    activation_word_bytes);
  }
  if (weight) {
    depths.weight_words =
        div_round_up(weight->data_bytes.size(), weight_word_bytes);
  }
  if (bias) {
    depths.bias_words = static_cast<std::size_t>(binding.num_bias_vecs);
  }
  if (output) {
    depths.output_words = div_round_up(output->data_bytes.size(),
                                       activation_word_bytes);
    depths.residual_words = depths.output_words;
  }
  if (binding.residual_buffer.has_value()) {
    const RuntimeTensorBuffer *residual =
        find_buffer(plan, *binding.residual_buffer);
    if (residual) {
      depths.residual_words = div_round_up(residual->data_bytes.size(),
                                           activation_word_bytes);
    }
  }

  depths.xfer_desc_words = static_cast<std::size_t>(binding.num_xfers);
  depths.comp_desc_words = static_cast<std::size_t>(binding.num_tiles);
  return depths;
}

HlsPortDepths get_max_hls_port_depths(const RuntimeExecutionPlan &plan) {
  HlsPortDepths max_depths;
  for (std::size_t i = 0; i < plan.launches.size(); ++i) {
    const HlsPortDepths d = get_hls_port_depths(plan, i);
    update_max(max_depths.ifm_words, d.ifm_words);
    update_max(max_depths.weight_words, d.weight_words);
    update_max(max_depths.bias_words, d.bias_words);
    update_max(max_depths.output_words, d.output_words);
    update_max(max_depths.residual_words, d.residual_words);
    update_max(max_depths.xfer_desc_words, d.xfer_desc_words);
    update_max(max_depths.comp_desc_words, d.comp_desc_words);
  }
  return max_depths;
}

void print_hls_port_depths(const RuntimeExecutionPlan &plan,
                           std::ostream &log) {
  log << "\n[HLS_DEPTHS] per-launch top-level port word depths\n";
  log << "[HLS_DEPTHS] units: ifm/out/residual="
      << (OFM_VECTOR_WIDTH / 8)
      << "B activation words, weights=" << (WEIGHT_VECTOR_WIDTH / 8)
      << "B weight words, biases=" << (BIAS_VECTOR_WIDTH / 8)
      << "B bias words\n";

  for (std::size_t i = 0; i < plan.launches.size(); ++i) {
    const RuntimeLaunchBinding &binding = plan.launches[i];
    const HlsPortDepths d = get_hls_port_depths(plan, i);
    log << "[HLS_DEPTHS] launch " << i << " " << binding.id
        << " ifm=" << d.ifm_words
        << " weights=" << d.weight_words
        << " biases=" << d.bias_words
        << " out=" << d.output_words
        << " residual=" << d.residual_words
        << " xfer_desc=" << d.xfer_desc_words
        << " comp_desc=" << d.comp_desc_words << "\n";
  }

  const HlsPortDepths d = get_max_hls_port_depths(plan);
  log << "[HLS_DEPTHS] recommended max depths for this graph:\n";
  log << "#pragma HLS INTERFACE m_axi port=ifm offset=slave bundle=gmem0 depth="
      << d.ifm_words << "\n";
  log << "#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem1 depth="
      << d.weight_words << "\n";
  log << "#pragma HLS INTERFACE m_axi port=biases offset=slave bundle=gmem2 depth="
      << d.bias_words << "\n";
  log << "#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem3 depth="
      << d.output_words << "\n";
  log << "#pragma HLS INTERFACE m_axi port=residual offset=slave bundle=gmem5 depth="
      << d.residual_words << "\n";
  log << "#pragma HLS INTERFACE m_axi port=xfer_desc offset=slave bundle=gmem4 depth="
      << d.xfer_desc_words << "\n";
  log << "#pragma HLS INTERFACE m_axi port=comp_desc offset=slave bundle=gmem4 depth="
      << d.comp_desc_words << "\n\n";
}

bool run_hls_launch(RuntimeExecutionPlan &plan, std::size_t launch_index,
                    std::ostream &log) {
  static_assert(sizeof(deeptile::tile::WeightTransferDescriptor) ==
                    sizeof(uram_xfer_desc_t),
                "UramTransferDescriptor and uram_xfer_desc_t must match");
  static_assert(sizeof(deeptile::tile::TileDescriptor) == sizeof(descriptor_dt),
                "UramTileDescriptor and descriptor_dt must match");

  if (launch_index >= plan.launches.size()) {
    log << "[HLS_ADAPTER][ERROR] launch index out of range: " << launch_index
        << "\n";
    return false;
  }

  const RuntimeLaunchBinding &binding = plan.launches[launch_index];
  const RuntimeTensorBuffer *ifm = find_buffer(plan, binding.ifm_buffer);
  const RuntimeTensorBuffer *weight = find_buffer(plan, binding.weight_buffer);
  const RuntimeTensorBuffer *bias = find_buffer(plan, binding.bias_buffer);
  RuntimeTensorBuffer *output = find_buffer(plan, binding.output_buffer);

  if (!ifm || !weight || !bias || !output) {
    log << "[HLS_ADAPTER][ERROR] launch " << binding.id
        << " has an unprepared required buffer\n";
    return false;
  }
  if (bias->int32_data.empty()) {
    log << "[HLS_ADAPTER][ERROR] launch " << binding.id
        << " bias buffer has no int32 data\n";
    return false;
  }

  const int vector_bytes = BITWIDTH * IC_TILE_SIZE / 8;
  if (!validate_word_count("IFM", ifm->data_bytes.size(), vector_bytes, log) ||
      !validate_word_count("weights", weight->data_bytes.size(), vector_bytes,
                           log)) {
    return false;
  }

  // HLS: For multi-launch graphs AXI_DEPTH_* is the maximum across launches.
  // ENTER_WRAPC initializes its AXI model
  // by reading exactly AXI_DEPTH_* words from the pointer passed to tile_conv().
  // Every call must therefore provide a buffer of at least AXI_DEPTH_* elements,
  // even when this specific launch's tensor is smaller.  Resize each vector to
  // max(actual, AXI_DEPTH_*) before passing it to tile_conv().
  auto ifm_words =
      pack_bytes_to_words<IFM_VECTOR_WIDTH, pixel_vector_dt>(ifm->data_bytes);
  // HLS: The 256-bit IFM port reads two pixel vectors per word. WRAPC
  // reads AXI_DEPTH_IFM_WORDS words = 2*AXI_DEPTH_IFM_WORDS pixel_vecs; the buffer
  // must hold at least that many, and an even count so the reinterpret is exact.
  // Both IFM paths may read one guarded word past the final row; +2*P_SPATIAL vectors
  // keep that access in-bounds for the last CIG's last row.
  std::size_t ifm_min_vecs =
      std::size_t(2) * std::size_t(AXI_DEPTH_IFM_WORDS) + std::size_t(2 * P_SPATIAL);
  std::size_t ifm_vecs     = std::max(ifm_words.size() + std::size_t(2 * P_SPATIAL), ifm_min_vecs);
  if (ifm_vecs & 1) ifm_vecs += 1;
  ifm_words.resize(ifm_vecs, pixel_vector_dt(0));

  auto weight_words = pack_bytes_to_words<WEIGHT_VECTOR_WIDTH, weight_vector_dt>(
      weight->data_bytes);
  weight_words.resize(
      std::max(weight_words.size(), std::size_t(AXI_DEPTH_WEIGHTS)),
      weight_vector_dt(0));

  auto bias_words = pack_bias_words(bias->int32_data, binding.num_bias_vecs);
  bias_words.resize(
      std::max(bias_words.size(), std::size_t(AXI_DEPTH_BIASES)),
      biases_vector_dt(0));

  // Output: ceiling division (Cout may not be a multiple of OC_TILE_SIZE),
  // then pad to AXI_DEPTH_OUT.
  const std::size_t out_vecs = std::max(
      std::max<std::size_t>(1, div_round_up(output->data_bytes.size(),
                                            std::size_t(vector_bytes))),
      std::size_t(AXI_DEPTH_OUT));
  std::vector<activations_vector_dt> output_words(out_vecs,
                                                  activations_vector_dt(0));

  // Residual: start at AXI_DEPTH_RESIDUAL so launches without a residual also
  // provide a valid (zero-filled) buffer of the right size to ENTER_WRAPC.
  std::vector<activations_vector_dt> residual_words(
      std::max<std::size_t>(1, std::size_t(AXI_DEPTH_RESIDUAL)),
      activations_vector_dt(0));
  if (binding.residual_buffer.has_value()) {
    const RuntimeTensorBuffer *residual =
        find_buffer(plan, *binding.residual_buffer);
    if (!residual) {
      log << "[HLS_ADAPTER][ERROR] launch " << binding.id
          << " residual buffer missing: " << *binding.residual_buffer << "\n";
      return false;
    }
    if (!validate_word_count("residual", residual->data_bytes.size(),
                             vector_bytes, log)) {
      return false;
    }
    auto packed_residual =
        pack_bytes_to_words<OFM_VECTOR_WIDTH, activations_vector_dt>(
            residual->data_bytes);
    packed_residual.resize(
        std::max(packed_residual.size(), std::size_t(AXI_DEPTH_RESIDUAL)),
        activations_vector_dt(0));
    residual_words = std::move(packed_residual);
  }

  // Zero-fill the depthwise port so non-depthwise launches satisfy ENTER_WRAPC.
  std::vector<weight_vector_dt> dw_words(
      std::max<std::size_t>(1, std::size_t(AXI_DEPTH_DW)), weight_vector_dt(0));
  if (!binding.dw_weight_buffer.empty()) {
    const RuntimeTensorBuffer *dw_blob =
        find_buffer(plan, binding.dw_weight_buffer);
    if (dw_blob) {
      auto packed = pack_bytes_to_words<WEIGHT_VECTOR_WIDTH, weight_vector_dt>(
          dw_blob->data_bytes);
      packed.resize(std::max(packed.size(), std::size_t(AXI_DEPTH_DW)),
                    weight_vector_dt(0));
      dw_words = std::move(packed);
    }
  }

  auto xfer_descs =
      memcpy_descriptors<uram_xfer_desc_t>(binding.transfer_descs);
  xfer_descs.resize(std::max(xfer_descs.size(), std::size_t(AXI_DEPTH_XDESC)));
  auto comp_descs = memcpy_descriptors<descriptor_dt>(binding.compute_descs);
  comp_descs.resize(
      std::max(comp_descs.size(), std::size_t(AXI_DEPTH_CDESC)));

  log << "[HLS_ADAPTER] Launch " << launch_index << ": " << binding.id
      << " xfers=" << binding.num_xfers << " tiles=" << binding.num_tiles
      << " bias_vecs=" << binding.num_bias_vecs
      << " shift_out=" << binding.shift_out
      << " gap_mul=" << binding.gap_mul
      << " cig_stride=" << binding.ifm_cig_stride_vecs << "\n";

  tile_conv(reinterpret_cast<const ifm_word_dt *>(ifm_words.data()),
            weight_words.data(), bias_words.data(),
            output_words.data(), residual_words.data(), xfer_descs.data(),
            binding.num_xfers, comp_descs.data(), binding.num_tiles,
            binding.num_bias_vecs, binding.shift_out, binding.gap_mul,
            binding.kernel_size, binding.stride,
            binding.use_ifm_cache ? 1 : 0,
            binding.spatial_tile_size, binding.num_ic_tiles,
            binding.ifm_cig_stride_vecs, binding.act_max,
            dw_words.data(), binding.dw_config, binding.dw_shift,
            binding.num_writeback_tiles);

  output->data_bytes =
      unpack_words_to_bytes<OFM_VECTOR_WIDTH, activations_vector_dt>(
          output_words, output->data_bytes.size());
  return true;
}

} // namespace runtime
} // namespace deeptile
