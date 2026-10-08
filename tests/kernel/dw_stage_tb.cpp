// Standalone testbench for the fused depthwise stage (ctest dw_stage_*).
//
// For a pwdw_* graph it derives, from the proven host reference:
//   - pw_out : the producer conv's INT8 output  = the DW stage input
//   - dw_out : the depthwise reference output    = the golden
// then packs pw_out into the OFM stream (per-COG raster, 16 lanes/pixel), runs
// dw_stage() with the compact weight blob, and compares its output to dw_out
// byte-for-byte. Covers 3x3 and 5x5 kernels, stride-1, stride-2, ragged
// channels, and — when the DW output exceeds one tile height — the row-halo
// overlapped tiling.
//
// Usage: dw_stage_tb <path/to/pwdw_case/graph.json>
#include "dw_stage.h"

#include "runtime/runtime.h"
#include "reference/runtime_reference.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using deeptile::runtime::RuntimeDiagnostic;
using deeptile::runtime::RuntimeExecutionPlan;
using deeptile::runtime::RuntimeTensorBuffer;

namespace {

const RuntimeTensorBuffer *find_buffer(const RuntimeExecutionPlan &plan,
                                       const std::string &id) {
  const auto it = plan.buffer_index_by_id.find(id);
  return it == plan.buffer_index_by_id.end() ? nullptr
                                             : &plan.buffers[it->second];
}

std::size_t chw(int c, int h, int w, int H, int W) {
  return (static_cast<std::size_t>(c) * H + h) * W + w;
}

struct DwTile {
  int cog;       // COG index (weights)
  int first_in;  // first DW-input row fed (global, this COG)
  int conv_h;    // DW-input rows in this tile
  int pad_top;
  int pad_bot;
  int out_row0;  // first DW-output row this tile produces (global)
  int out_rows;  // DW-output rows this tile produces
};

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: dw_stage_tb <graph.json>\n";
    return 2;
  }
  const std::string graph_path = argv[1];

  // ---- flag OFF: reference gives pw_out (DW input) and dw_out (golden) -------
  unsetenv("DEEPTILE_FUSE_DW");
  RuntimeExecutionPlan plan_off;
  std::vector<RuntimeDiagnostic> d_off;
  if (!deeptile::runtime::load_compile_prepare_runtime(graph_path, plan_off,
                                                      d_off)) {
    std::cerr << "[dw_tb] compile (unfused) failed\n";
    return 1;
  }
  const auto ref = deeptile::runtime::generate_bit_exact_reference(plan_off);
  if (deeptile::runtime::has_errors(ref.diagnostics)) {
    std::cerr << "[dw_tb] reference generation failed\n";
    return 1;
  }
  int dw_idx = -1;
  for (std::size_t i = 0; i < plan_off.graph_plan.launches.size(); ++i) {
    if (plan_off.graph_plan.launches[i].params.depthwise) {
      dw_idx = static_cast<int>(i);
    }
  }
  if (dw_idx < 0) {
    std::cerr << "[dw_tb] no depthwise launch in graph\n";
    return 1;
  }
  const auto &L = plan_off.graph_plan.launches[dw_idx];
  const auto *pw_t = deeptile::runtime::find_reference_tensor(ref, L.ifm_tensor);
  const auto *dw_t = deeptile::runtime::find_reference_tensor(ref, L.ofm_tensor);
  if (!pw_t || !dw_t) {
    std::cerr << "[dw_tb] missing pw_out/dw_out reference tensors\n";
    return 1;
  }
  const std::vector<std::int8_t> &pw_out = pw_t->data_chw;  // DW input, CHW
  const std::vector<std::int8_t> &dw_out = dw_t->data_chw;  // golden, CHW

  const int C = L.params.Cin;              // DW channels (== producer Cout)
  const int Hin = L.params.H;
  const int Win = L.params.W;
  const int stride = L.params.Sh;
  const int shift = L.shift_out;
  const bool relu = L.params.use_relu;
  const int kdim = L.params.Kh;   // 3 or 5, same padding
  const int pad = kdim / 2;
  const int outH = L.params.output_height();
  const int outW = L.params.output_width();
  const int num_cog = (C + OC_TILE_SIZE - 1) / OC_TILE_SIZE;

  // ---- flag ON: compact DW blob (registered on the fused launch) ------------
  setenv("DEEPTILE_FUSE_DW", "1", 1);
  RuntimeExecutionPlan plan_on;
  std::vector<RuntimeDiagnostic> d_on;
  const bool ok_on = deeptile::runtime::load_compile_prepare_runtime(
      graph_path, plan_on, d_on);
  unsetenv("DEEPTILE_FUSE_DW");
  if (!ok_on) {
    std::cerr << "[dw_tb] compile (fused) failed\n";
    return 1;
  }
  const std::vector<std::uint8_t> *blob = nullptr;
  for (const auto &b : plan_on.launches) {
    if (!b.dw_weight_buffer.empty()) {
      const auto *buf = find_buffer(plan_on, b.dw_weight_buffer);
      if (buf) {
        blob = &buf->data_bytes;
      }
    }
  }
  if (!blob || blob->empty()) {
    std::cerr << "[dw_tb] compact DW blob not found\n";
    return 1;
  }
  // Size the port array to AXI_DEPTH_DW (the m_axi cosim `depth`); only the
  // first blob words are populated, the rest are zero (unread by the burst).
  const std::size_t blob_words = blob->size() / 16;
  std::vector<weight_vector_dt> dwv(AXI_DEPTH_DW, weight_vector_dt(0));
  for (std::size_t w = 0; w < blob_words; ++w) {
    weight_vector_dt word = 0;
    for (int b = 0; b < 16; ++b) {
      word.range(b * 8 + 7, b * 8) = (*blob)[w * 16 + b];
    }
    dwv[w] = word;
  }

  // ---- build the tile list (row-halo split for tall stride-1 outputs) -------
  const int CHUNK = MAX_TILE_H - (kdim - 1);  // keep conv_h <= MAX_TILE_H
  std::vector<DwTile> tiles;
  for (int g = 0; g < num_cog; ++g) {
    if (stride == 1 && outH > CHUNK) {
      for (int r0 = 0; r0 < outH; r0 += CHUNK) {
        const int r1 = std::min(r0 + CHUNK, outH);
        DwTile t;
        t.cog = g;
        t.pad_top = (r0 == 0) ? pad : 0;
        t.pad_bot = (r1 == outH) ? pad : 0;
        t.first_in = (r0 == 0) ? 0 : (r0 - pad);
        const int last_in =
            (r1 == outH) ? (Hin - 1) : (r1 - 1 + pad);  // s1: Hin == outH
        t.conv_h = last_in - t.first_in + 1;
        t.out_row0 = r0;
        t.out_rows = r1 - r0;
        tiles.push_back(t);
      }
    } else {
      DwTile t;
      t.cog = g;
      t.pad_top = pad;
      t.pad_bot = pad;
      t.first_in = 0;
      t.conv_h = Hin;
      t.out_row0 = 0;
      t.out_rows = outH;
      tiles.push_back(t);
    }
  }

  // ---- feed descriptors + OFM input stream ----------------------------------
  desc_stream_ofm_t desc_in, desc_out;
  ofm_stream_t ofm_in, ofm_out;
  for (const auto &t : tiles) {
    ofm_desc_t desc = {};
    desc.tile_h = static_cast<uint8_t>(t.out_rows);
    desc.tile_w = static_cast<uint8_t>(outW);
    desc.conv_tile_h = static_cast<uint8_t>(t.conv_h);
    desc.conv_tile_w = static_cast<uint8_t>(Win);
    desc.dw_cog_idx = static_cast<uint8_t>(t.cog);
    desc.flags = 0;  // DW ReLU rides dw_config bit3 (layer-uniform), not flags
    desc.pool_packed_padding = (uint16_t)((t.pad_top << 12) | (t.pad_bot << 8) |
                                          (pad << 4) | pad);
    desc_in.write(desc);

    for (int r = 0; r < t.conv_h; ++r) {
      const int in_row = t.first_in + r;
      for (int col = 0; col < Win; ++col) {
        activations_vector_dt word = 0;
        for (int lane = 0; lane < OC_TILE_SIZE; ++lane) {
          const int ch = t.cog * OC_TILE_SIZE + lane;
          std::int8_t v = (ch < C) ? pw_out[chw(ch, in_row, col, Hin, Win)] : 0;
          word.range(lane * 8 + 7, lane * 8) = (ap_uint<8>)(ap_int<8>)v;
        }
        ofm_in.write(word);
      }
    }
  }

  // ---- run the stage --------------------------------------------------------
  const int dw_config = 1 | (stride << 1) | ((relu ? 1 : 0) << 3) |
                        ((kdim == 5 ? 1 : 0) << 4) | (127 << 8);
  dw_stage(desc_in, ofm_in, dwv.data(), desc_out, ofm_out,
           static_cast<int>(tiles.size()), num_cog, dw_config, shift);

  // ---- collect output + compare against dw_out ------------------------------
  std::vector<std::int8_t> got(static_cast<std::size_t>(C) * outH * outW, 0);
  for (const auto &t : tiles) {
    (void)desc_out.read();  // forwarded descriptor
    for (int er = 0; er < t.out_rows; ++er) {
      const int out_row = t.out_row0 + er;
      for (int oc = 0; oc < outW; ++oc) {
        activations_vector_dt word = ofm_out.read();
        for (int lane = 0; lane < OC_TILE_SIZE; ++lane) {
          const int ch = t.cog * OC_TILE_SIZE + lane;
          if (ch < C) {
            got[chw(ch, out_row, oc, outH, outW)] =
                (std::int8_t)(ap_int<8>)word.range(lane * 8 + 7, lane * 8);
          }
        }
      }
    }
  }

  int mismatches = 0;
  int first_c = -1, first_h = -1, first_w = -1;
  for (int c = 0; c < C && mismatches < 1; ++c) {
    for (int h = 0; h < outH; ++h) {
      for (int w = 0; w < outW; ++w) {
        const std::size_t i = chw(c, h, w, outH, outW);
        if (got[i] != dw_out[i]) {
          ++mismatches;
          first_c = c; first_h = h; first_w = w;
          break;
        }
      }
      if (mismatches) break;
    }
  }

  std::cout << "[dw_tb] C=" << C << " in=" << Hin << "x" << Win << " out=" << outH
            << "x" << outW << " s" << stride << " shift=" << shift
            << " relu=" << relu << " tiles=" << tiles.size() << "\n";
  if (mismatches) {
    const std::size_t i = chw(first_c, first_h, first_w, outH, outW);
    std::cerr << "[dw_tb][FAIL] mismatch at c=" << first_c << " h=" << first_h
              << " w=" << first_w << " got=" << (int)got[i]
              << " expected=" << (int)dw_out[i] << "\n";
    return 1;
  }
  std::cout << "[dw_tb][PASS] " << (C * outH * outW)
            << " outputs bit-exact vs reference\n";
  return 0;
}
