/**
 * @file output_postproc.cpp
 * @brief Implements accumulation, fused post-processing, and output writeback.
 */

#include "output_postproc.h"

void acc_quantize(acc_group_stream_t &acc_in,
                  const biases_vector_dt bias_cache[BIAS_BUF_DEPTH],
                  ofm_stream_t &ofm_out, desc_stream_comp_t &desc_comp,
                  int num_tiles, int shift_out, int act_max) {
#pragma HLS INLINE off

  // HLS: Independent spatial and output-channel banks permit parallel access
  // from the fully unrolled lane loops.
  static acc_t psum[P_SPATIAL][OC_TILE_SIZE][PSUM_DEPTH];
#pragma HLS ARRAY_PARTITION variable = psum complete dim = 1
#pragma HLS ARRAY_PARTITION variable = psum complete dim = 2
#pragma HLS BIND_STORAGE variable = psum type = RAM_S2P impl = BRAM
#pragma HLS DEPENDENCE variable = psum inter false

  const bool shift_gt0 = (shift_out > 0);
  const bool shift_eq0 = (shift_out == 0);
  const int shift_pos = shift_gt0 ? (shift_out - 1) : 0;
  const int shift_neg = (shift_out < 0) ? ((-shift_out) + 1) : 0;

ACCQ_TILE_LOOP:
  for (int t = 0; t < num_tiles; t++) {
#pragma HLS loop_tripcount min=TC_NUM_TILES max=TC_NUM_TILES
    comp_desc_t desc = desc_comp.read();

    const int ofm_h = desc.ofm_tile_h;
    const int ofm_w = desc.ofm_tile_w;
    const int num_windows = ofm_h * ofm_w;
    // INVARIANT: Partial-sum addressing must use the grouping mode selected by
    // the IFM producer and propagated through the MAC engine.
    const bool do_row_aligned = (desc_row_aligned(desc) != 0);
    const int groups_per_row = (ofm_w + P_SPATIAL - 1) / P_SPATIAL;
    const int num_groups = do_row_aligned
        ? ofm_h * groups_per_row
        : (num_windows + P_SPATIAL - 1) / P_SPATIAL;

    const bool do_reset = (desc.flags & ACCUM_RESET) != 0;
    const bool do_writeback = (desc.flags & ACCUM_WRITE_BACK) != 0;
    const bool do_relu = (desc.flags & RELU_ON) != 0;

    const int num_cigs = (desc.num_cigs > 0) ? static_cast<int>(desc.num_cigs) : 1;

  ACC_CIG_LOOP:
    for (int cig = 0; cig < num_cigs; cig++) {
#pragma HLS loop_tripcount min=1 max=TC_CIG
// HLS: Do not flatten this loop. Adjacent CIGs revisit each partial-sum address;
// a flattened pipeline can read stale BRAM data when `num_groups == 1`.
#pragma HLS LOOP_FLATTEN off
      const bool cig_reset = do_reset && (cig == 0);

    ACCQ_GROUP_LOOP:
      for (int g = 0; g < num_groups; g++) {
#pragma HLS loop_tripcount min=TC_NUM_GROUPS max=TC_NUM_GROUPS
#pragma HLS PIPELINE II = 1

        acc_group_dt agrp = acc_in.read();

        for (int p = 0; p < P_SPATIAL; p++) {
#pragma HLS UNROLL
          for (int oc = 0; oc < OC_TILE_SIZE; oc++) {
#pragma HLS UNROLL
            if (cig_reset) {
              psum[p][oc][g] = agrp.v[p].v[oc];
            } else {
              psum[p][oc][g] = psum[p][oc][g] + agrp.v[p].v[oc];
            }
          }
        }
      }
    }

    if (do_writeback) {
      acc_t bias_vals[OC_TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable = bias_vals complete dim = 1

      biases_vector_dt bias_word = bias_cache[desc.bias_offset_idx];
    UNPACK_BIAS:
      for (int oc = 0; oc < OC_TILE_SIZE; oc++) {
#pragma HLS UNROLL
        bias_vals[oc] =
            (acc_t)(ap_int<32>)bias_word.range((oc + 1) * 32 - 1, oc * 32);
      }

      // HLS: Running counters map raster output order to row-aligned partial-sum
      // groups without division in the II=1 emission loop.
      int ox = 0;
      int row_group_base = 0;
    EMIT_LOOP:
      for (int w = 0; w < num_windows; w++) {
#pragma HLS loop_tripcount min=TC_SPATIAL max=TC_SPATIAL
#pragma HLS PIPELINE II = 1

        int g_idx, p_idx;
        if (do_row_aligned) {
          g_idx = row_group_base + (ox >> P_SPATIAL_LOG2);  // ox / P_SPATIAL
          p_idx = ox & P_SPATIAL_MASK;                      // ox % P_SPATIAL
        } else {
          g_idx = w >> P_SPATIAL_LOG2;                      // w / P_SPATIAL
          p_idx = w & P_SPATIAL_MASK;                       // w % P_SPATIAL
        }

        activations_vector_dt ofm_word = 0;

        for (int oc = 0; oc < OC_TILE_SIZE; oc++) {
#pragma HLS UNROLL

          acc_t new_val = psum[p_idx][oc][g_idx];

          // Truncate to the intermediate scale without rounding.
          acc_t s1;
          if (shift_gt0) {
            s1 = (new_val >> shift_pos);
          } else if (shift_eq0) {
            s1 = (new_val << 1);
          } else {
            s1 = (new_val << shift_neg);
          }

          // Add bias, round once, saturate, and apply the optional clamped
          // ReLU (act_max is 127 for ReLU, lower for ReLU6).
          acc_t tb = s1 + bias_vals[oc] + 1;
          acc_t y = tb >> 1;
          if (y > 127)  y = 127;
          if (y < -128) y = -128;
          if (do_relu && y < 0) y = 0;
          if (do_relu && y > act_max) y = act_max;

          ap_int<8> y8 = (ap_int<8>)y;
          const int lo = oc * BITWIDTH;
          const int hi = lo + BITWIDTH - 1;
          ofm_word.range(hi, lo) = (ap_uint<8>)y8;
        }

        ofm_out.write(ofm_word);

        if (ox == ofm_w - 1) {
          ox = 0;
          row_group_base += groups_per_row;
        } else {
          ox++;
        }
      }
    }
  }
}

// Accumulates descriptors through one writeback boundary.
static comp_desc_t accumulate_tile(acc_group_stream_t &acc_in,
                                   desc_stream_comp_t &desc_comp,
                                   acc_t psum[P_SPATIAL][OC_TILE_SIZE][PSUM_DEPTH]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=psum complete dim=1
#pragma HLS ARRAY_PARTITION variable=psum complete dim=2
#pragma HLS DEPENDENCE variable=psum inter false
  comp_desc_t desc = {};
  bool wb = false;

ACC_DRAIN_LOOP:
  do {
#pragma HLS loop_tripcount min=1 max=TC_CIG
    desc = desc_comp.read();
    const int ofm_h = desc.ofm_tile_h;
    const int ofm_w = desc.ofm_tile_w;
    const int num_windows = ofm_h * ofm_w;
    const bool do_row_aligned = (desc_row_aligned(desc) != 0);
    const int groups_per_row = (ofm_w + P_SPATIAL - 1) / P_SPATIAL;
    const int num_groups = do_row_aligned
        ? ofm_h * groups_per_row
        : (num_windows + P_SPATIAL - 1) / P_SPATIAL;
    const bool do_reset = (desc.flags & ACCUM_RESET) != 0;
    const int num_cigs = (desc.num_cigs > 0) ? static_cast<int>(desc.num_cigs) : 1;
    wb = (desc.flags & ACCUM_WRITE_BACK) != 0;

  ACC_CIG_LOOP:
    for (int cig = 0; cig < num_cigs; cig++) {
#pragma HLS loop_tripcount min=1 max=TC_CIG
#pragma HLS LOOP_FLATTEN off
      const bool cig_reset = do_reset && (cig == 0);
    ACCQ_GROUP_LOOP:
      for (int g = 0; g < num_groups; g++) {
#pragma HLS loop_tripcount min=TC_NUM_GROUPS max=TC_NUM_GROUPS
#pragma HLS PIPELINE II=1
        acc_group_dt agrp = acc_in.read();
        for (int p = 0; p < P_SPATIAL; p++) {
#pragma HLS UNROLL
          for (int oc = 0; oc < OC_TILE_SIZE; oc++) {
#pragma HLS UNROLL
            if (cig_reset) {
              psum[p][oc][g] = agrp.v[p].v[oc];
            } else {
              psum[p][oc][g] = psum[p][oc][g] + agrp.v[p].v[oc];
            }
          }
        }
      }
    }
  } while (!wb);
  return desc;
}

// Emits one completed tile in raster order using the standard requantization.
static void emit_tile(const acc_t psum[P_SPATIAL][OC_TILE_SIZE][PSUM_DEPTH],
                      const biases_vector_dt bias_cache[BIAS_BUF_DEPTH],
                      ofm_stream_t &ofm_out, comp_desc_t desc, int shift_out,
                      int act_max) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=psum complete dim=1
#pragma HLS ARRAY_PARTITION variable=psum complete dim=2

  const bool shift_gt0 = (shift_out > 0);
  const bool shift_eq0 = (shift_out == 0);
  const int shift_pos = shift_gt0 ? (shift_out - 1) : 0;
  const int shift_neg = (shift_out < 0) ? ((-shift_out) + 1) : 0;

  const int ofm_h = desc.ofm_tile_h;
  const int ofm_w = desc.ofm_tile_w;
  const int num_windows = ofm_h * ofm_w;
  const bool do_row_aligned = (desc_row_aligned(desc) != 0);
  const int groups_per_row = (ofm_w + P_SPATIAL - 1) / P_SPATIAL;
  const bool do_relu = (desc.flags & RELU_ON) != 0;

  acc_t bias_vals[OC_TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=bias_vals complete dim=1
  biases_vector_dt bias_word = bias_cache[desc.bias_offset_idx];
UNPACK_BIAS:
  for (int oc = 0; oc < OC_TILE_SIZE; oc++) {
#pragma HLS UNROLL
    bias_vals[oc] = (acc_t)(ap_int<32>)bias_word.range((oc + 1) * 32 - 1, oc * 32);
  }

  int ox = 0;
  int row_group_base = 0;
EMIT_LOOP:
  for (int w = 0; w < num_windows; w++) {
#pragma HLS loop_tripcount min=TC_SPATIAL max=TC_SPATIAL
#pragma HLS PIPELINE II=1
    int g_idx, p_idx;
    if (do_row_aligned) {
      g_idx = row_group_base + (ox >> P_SPATIAL_LOG2);
      p_idx = ox & P_SPATIAL_MASK;
    } else {
      g_idx = w >> P_SPATIAL_LOG2;
      p_idx = w & P_SPATIAL_MASK;
    }
    activations_vector_dt ofm_word = 0;
    for (int oc = 0; oc < OC_TILE_SIZE; oc++) {
#pragma HLS UNROLL
      acc_t new_val = psum[p_idx][oc][g_idx];
      acc_t s1;
      if (shift_gt0) {
        s1 = (new_val >> shift_pos);
      } else if (shift_eq0) {
        s1 = (new_val << 1);
      } else {
        s1 = (new_val << shift_neg);
      }
      acc_t tb = s1 + bias_vals[oc] + 1;
      acc_t y = tb >> 1;
      if (y > 127) y = 127;
      if (y < -128) y = -128;
      if (do_relu && y < 0) y = 0;
      if (do_relu && y > act_max) y = act_max;
      const int lo = oc * BITWIDTH;
      const int hi = lo + BITWIDTH - 1;
      ofm_word.range(hi, lo) = (ap_uint<8>)(ap_int<8>)y;
    }
    ofm_out.write(ofm_word);
    if (ox == ofm_w - 1) {
      ox = 0;
      row_group_base += groups_per_row;
    } else {
      ox++;
    }
  }
}

void acc_quantize_pp(acc_group_stream_t &acc_in,
                     const biases_vector_dt bias_cache[BIAS_BUF_DEPTH],
                     ofm_stream_t &ofm_out, desc_stream_comp_t &desc_comp,
                     int num_writeback_tiles, int shift_out, int act_max) {
#pragma HLS INLINE off
  acc_t psum_A[P_SPATIAL][OC_TILE_SIZE][PSUM_DEPTH];
  acc_t psum_B[P_SPATIAL][OC_TILE_SIZE][PSUM_DEPTH];
#pragma HLS ARRAY_PARTITION variable=psum_A complete dim=1
#pragma HLS ARRAY_PARTITION variable=psum_A complete dim=2
#pragma HLS ARRAY_PARTITION variable=psum_B complete dim=1
#pragma HLS ARRAY_PARTITION variable=psum_B complete dim=2
#pragma HLS BIND_STORAGE variable=psum_A type=RAM_S2P impl=BRAM
#pragma HLS BIND_STORAGE variable=psum_B type=RAM_S2P impl=BRAM

#pragma HLS allocation function instances=accumulate_tile limit=2
#pragma HLS allocation function instances=emit_tile limit=2

  if (num_writeback_tiles <= 0) return;

  // Prime the first partial-sum buffer.
  comp_desc_t desc0 = accumulate_tile(acc_in, desc_comp, psum_A);

PP_LOOP:
  for (int u = 0; u < num_writeback_tiles - 1; u++) {
#pragma HLS loop_tripcount min=1 max=TC_NUM_TILES
    comp_desc_t desc_next;
    if ((u & 1) == 0) {
      desc_next = accumulate_tile(acc_in, desc_comp, psum_B);
      emit_tile(psum_A, bias_cache, ofm_out, desc0, shift_out, act_max);
    } else {
      desc_next = accumulate_tile(acc_in, desc_comp, psum_A);
      emit_tile(psum_B, bias_cache, ofm_out, desc0, shift_out, act_max);
    }
    desc0 = desc_next;
  }

  // Drain the final completed tile.
  if (((num_writeback_tiles - 1) & 1) == 0) {
    emit_tile(psum_A, bias_cache, ofm_out, desc0, shift_out, act_max);
  } else {
    emit_tile(psum_B, bias_cache, ofm_out, desc0, shift_out, act_max);
  }
}

static inline acc_t apply_signed_shift(acc_t val, int shift) {
#pragma HLS INLINE
  if (shift > 0) {
    return val << shift;
  }
  if (shift < 0) {
    // Round half away from zero with a sign-dependent bias before the
    // arithmetic right shift.
    const int s = -shift;
    const acc_t round_bias = ((acc_t)1 << (s - 1)) + (val >> 31);
    return (val + round_bias) >> s;
  }
  return val;
}

static void pass_ofm_tile(ofm_stream_t &ofm_in,
                          ofm_stream_t &ofm_out,
                          int num_rows,
                          int row_len_vecs) {
  const int total_vecs = num_rows * row_len_vecs;

PASS_OFM_TILE:
  for (int i = 0; i < total_vecs; ++i) {
#pragma HLS loop_tripcount min=TC_SPATIAL max=TC_SPATIAL
#pragma HLS PIPELINE II = 1
    ofm_out.write(ofm_in.read());
  }
}

void residual_add(desc_stream_ofm_t &desc_stream_in,
                  ofm_stream_t &ofm_in,
                  residual_stream_t &residual_in,
                  desc_stream_ofm_t &desc_stream_out,
                  ofm_stream_t &ofm_out,
                  int num_tiles,
                  int act_max) {
#pragma HLS INLINE off

RESIDUAL_TILE_LOOP:
  for (int t = 0; t < num_tiles; ++t) {
#pragma HLS loop_tripcount min=TC_NUM_TILES max=TC_NUM_TILES
    ofm_desc_t desc = desc_stream_in.read();
    desc_stream_out.write(desc);

    if (desc.tile_h == 0) {
      continue;
    }

    const bool do_add = (desc.flags & ADD_ENABLE) != 0;
    const bool post_add_relu = (desc.flags & POST_ADD_RELU) != 0;
    const int total_vecs = desc.conv_tile_h * desc.conv_tile_w;

    if (!do_add) {
      pass_ofm_tile(ofm_in, ofm_out, desc.conv_tile_h,
                    desc.conv_tile_w);
      continue;
    }

  RESIDUAL_VEC_LOOP:
    for (int i = 0; i < total_vecs; ++i) {
#pragma HLS loop_tripcount min=TC_SPATIAL max=TC_SPATIAL
#pragma HLS PIPELINE II = 1
      activations_vector_dt main_word = ofm_in.read();
      activations_vector_dt residual_word = residual_in.read();
      activations_vector_dt out_word = 0;

      for (int oc = 0; oc < OC_TILE_SIZE; ++oc) {
#pragma HLS UNROLL
        const int hi = (oc + 1) * BITWIDTH - 1;
        const int lo = oc * BITWIDTH;
        const acc_t main_val =
            static_cast<acc_t>((ap_int<8>)main_word.range(hi, lo));
        const acc_t residual_val =
            static_cast<acc_t>((ap_int<8>)residual_word.range(hi, lo));
        acc_t sum =
            main_val + apply_signed_shift(residual_val, desc.residual_shift);

        if (sum > 127) {
          sum = 127;
        } else if (sum < -128) {
          sum = -128;
        }
        if (post_add_relu && sum < 0) {
          sum = 0;
        }
        if (post_add_relu && sum > act_max) {
          sum = act_max;
        }

        out_word.range(hi, lo) = static_cast<ap_uint<8>>((ap_int<8>)sum);
      }

      ofm_out.write(out_word);
    }
  }
}

struct pool_padding_t {
  int top;
  int bottom;
  int left;
  int right;
};

static inline pool_padding_t unpack_pool_padding(uint16_t packed) {
#pragma HLS INLINE
  pool_padding_t p;
  p.top    = (packed >> 12) & 0xF;
  p.bottom = (packed >> 8)  & 0xF;
  p.left   = (packed >> 4)  & 0xF;
  p.right  =  packed        & 0xF;
  return p;
}

// HLS: An explicit balanced tree bounds the max-reduction depth.
static inline acc_t max9(acc_t v0, acc_t v1, acc_t v2, acc_t v3, acc_t v4,
                         acc_t v5, acc_t v6, acc_t v7, acc_t v8) {
#pragma HLS INLINE
  const acc_t m01   = (v0 > v1) ? v0 : v1;
  const acc_t m23   = (v2 > v3) ? v2 : v3;
  const acc_t m45   = (v4 > v5) ? v4 : v5;
  const acc_t m67   = (v6 > v7) ? v6 : v7;
  const acc_t m0123 = (m01 > m23) ? m01 : m23;
  const acc_t m4567 = (m45 > m67) ? m45 : m67;
  const acc_t m07   = (m0123 > m4567) ? m0123 : m4567;
  return (m07 > v8) ? m07 : v8;
}

static activations_vector_dt make_pool_pad_word() {
#pragma HLS INLINE
  activations_vector_dt word = 0;
  for (int oc = 0; oc < OC_TILE_SIZE; ++oc) {
#pragma HLS UNROLL
    const int lo = oc * BITWIDTH;
    const int hi = lo + BITWIDTH - 1;
    word.range(hi, lo) = static_cast<ap_uint<8>>((ap_int<8>)-128);
  }
  return word;
}

static void load_padded_pool_row(ofm_stream_t &ofm_in,
                                 activations_vector_dt row[MAX_TILE_W + 2],
                                 int padded_w,
                                 int conv_w,
                                 bool is_data_row,
                                 int pad_left,
                                 const activations_vector_dt &pad_word) {
#pragma HLS INLINE off

POOL_LOAD_ROW:
  for (int c = 0; c < padded_w; ++c) {
#pragma HLS loop_tripcount min=3 max=66
#pragma HLS PIPELINE II = 1
    if (!is_data_row || c < pad_left || c >= pad_left + conv_w) {
      row[c] = pad_word;
    } else {
      row[c] = ofm_in.read();
    }
  }
}

// Streams the third row of each pooling window while retaining only two history
// rows. The streamed row replaces the oldest history row after use.
static void fused_emit_pool_row(ofm_stream_t &ofm_in,
                                activations_vector_dt bufA[MAX_TILE_W + 2],
                                activations_vector_dt bufB[MAX_TILE_W + 2],
                                ofm_stream_t &ofm_out,
                                int padded_w,
                                int conv_w,
                                int pooled_w,
                                bool is_data_row,
                                int pad_left,
                                const activations_vector_dt &pad_word,
                                int post_pool_shift) {
#pragma HLS INLINE off
#pragma HLS DEPENDENCE variable=bufA inter false

  activations_vector_dt r0c0 = pad_word, r0c1 = pad_word, r0c2 = pad_word;
  activations_vector_dt r1c0 = pad_word, r1c1 = pad_word, r1c2 = pad_word;
  activations_vector_dt r2c0 = pad_word, r2c1 = pad_word, r2c2 = pad_word;

FUSED_POOL_COL:
  for (int c = 0; c < padded_w; ++c) {
#pragma HLS loop_tripcount min=3 max=66
#pragma HLS PIPELINE II = 1
    const activations_vector_dt a_new = bufA[c];
    const activations_vector_dt b_new = bufB[c];
    activations_vector_dt c_new;
    if (!is_data_row || c < pad_left || c >= pad_left + conv_w) {
      c_new = pad_word;
    } else {
      c_new = ofm_in.read();
    }

    // Shift the three rows through the current 3-column window.
    r0c0 = r0c1; r0c1 = r0c2; r0c2 = a_new;
    r1c0 = r1c1; r1c1 = r1c2; r1c2 = b_new;
    r2c0 = r2c1; r2c1 = r2c2; r2c2 = c_new;

    // Retain the streamed row as the next window's oldest history row.
    bufA[c] = c_new;

    if (c >= 2 && (c & 1) == 0) {
      const int pc = (c - 2) >> 1;
      if (pc < pooled_w) {
        activations_vector_dt out_word = 0;
        for (int oc = 0; oc < OC_TILE_SIZE; ++oc) {
#pragma HLS UNROLL
          const int lo = oc * BITWIDTH;
          const int hi = lo + BITWIDTH - 1;

          const acc_t v0 = static_cast<acc_t>((ap_int<8>)r0c0.range(hi, lo));
          const acc_t v1 = static_cast<acc_t>((ap_int<8>)r0c1.range(hi, lo));
          const acc_t v2 = static_cast<acc_t>((ap_int<8>)r0c2.range(hi, lo));
          const acc_t v3 = static_cast<acc_t>((ap_int<8>)r1c0.range(hi, lo));
          const acc_t v4 = static_cast<acc_t>((ap_int<8>)r1c1.range(hi, lo));
          const acc_t v5 = static_cast<acc_t>((ap_int<8>)r1c2.range(hi, lo));
          const acc_t v6 = static_cast<acc_t>((ap_int<8>)r2c0.range(hi, lo));
          const acc_t v7 = static_cast<acc_t>((ap_int<8>)r2c1.range(hi, lo));
          const acc_t v8 = static_cast<acc_t>((ap_int<8>)r2c2.range(hi, lo));

          const acc_t max_val = max9(v0, v1, v2, v3, v4, v5, v6, v7, v8);

          acc_t shifted = apply_signed_shift(max_val, post_pool_shift);
          if (shifted > 127) {
            shifted = 127;
          } else if (shifted < -128) {
            shifted = -128;
          }
          out_word.range(hi, lo) = static_cast<ap_uint<8>>((ap_int<8>)shifted);
        }
        ofm_out.write(out_word);
      }
    }
  }
}

void maxpool(desc_stream_ofm_t &desc_stream_in,
             ofm_stream_t &ofm_in,
             desc_stream_ofm_t &desc_stream_out,
             ofm_stream_t &ofm_out,
             int num_tiles) {
#pragma HLS INLINE off
#pragma HLS allocation function instances=load_padded_pool_row limit=1
#pragma HLS allocation function instances=fused_emit_pool_row limit=1

  // HLS: Two history rows are sufficient because the third row streams through;
  // buffer A requires one read and one write per cycle during emission.
  activations_vector_dt bufA[MAX_TILE_W + 2];
  activations_vector_dt bufB[MAX_TILE_W + 2];
#pragma HLS BIND_STORAGE variable = bufA type = RAM_S2P impl = BRAM
#pragma HLS BIND_STORAGE variable = bufB type = RAM_S2P impl = BRAM

  const activations_vector_dt pad_word = make_pool_pad_word();

POOL_TILE_LOOP:
  for (int t = 0; t < num_tiles; ++t) {
#pragma HLS loop_tripcount min=TC_NUM_TILES max=TC_NUM_TILES
    ofm_desc_t desc = desc_stream_in.read();
    desc_stream_out.write(desc);

    if (desc.tile_h == 0) {
      continue;
    }

    const bool do_pool = (desc.flags & POOL_ENABLE) != 0;
    if (!do_pool) {
      pass_ofm_tile(ofm_in, ofm_out, desc.conv_tile_h,
                    desc.conv_tile_w);
      continue;
    }

    const pool_padding_t pad = unpack_pool_padding(desc.pool_packed_padding);
    const int conv_h = desc.conv_tile_h;
    const int conv_w = desc.conv_tile_w;
    const int padded_h = conv_h + pad.top + pad.bottom;
    const int padded_w = conv_w + pad.left + pad.right;

    // Prime two rows, then alternate pooled-row emission with replacement of
    // the middle history row.
  POOL_PADDED_ROW_LOOP:
    for (int y = 0; y < padded_h; ++y) {
#pragma HLS loop_tripcount min=3 max=34
      const bool is_data_row = (y >= pad.top) && (y < pad.top + conv_h);

      if (y == 0) {
        load_padded_pool_row(ofm_in, bufA, padded_w, conv_w, is_data_row,
                             pad.left, pad_word);
      } else if ((y & 1) == 1) {
        load_padded_pool_row(ofm_in, bufB, padded_w, conv_w, is_data_row,
                             pad.left, pad_word);
      } else {
        fused_emit_pool_row(ofm_in, bufA, bufB, ofm_out, padded_w, conv_w,
                            desc.tile_w, is_data_row, pad.left,
                            pad_word, desc.post_pool_shift);
      }
    }
  }
}

// INVARIANT: These widths must cover the maximum supported tile sum and the
// host-computed reciprocal without truncating the product.
static constexpr int GAP_SUM_BITS  = 20;
static constexpr int GAP_MUL_BITS  = 16;
static constexpr int GAP_PROD_BITS = GAP_SUM_BITS + GAP_MUL_BITS;
static constexpr int GAP_RSHIFT    = GAP_SCALE_FRAC_BITS;
static constexpr int GAP_ROUND     = 1 << (GAP_RSHIFT - 1);     // Fixed-point 0.5.

void gap(desc_stream_ofm_t &desc_stream_in,
         ofm_stream_t      &ofm_in,
         desc_stream_ofm_t &desc_stream_out,
         ofm_stream_t      &ofm_out,
         int                num_tiles,
         int                gap_mul) {
#pragma HLS INLINE off

GAP_TILE_LOOP:
  for (int t = 0; t < num_tiles; ++t) {
#pragma HLS loop_tripcount min=TC_NUM_TILES max=TC_NUM_TILES

    const ofm_desc_t desc = desc_stream_in.read();
    desc_stream_out.write(desc);

    if (desc.tile_h == 0) continue;

    if (!(desc.flags & GAP_ENABLE)) {
      pass_ofm_tile(ofm_in, ofm_out, desc.tile_h, desc.tile_w);
      continue;
    }

    acc_t gap_sum[OC_TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=gap_sum complete

    for (int oc = 0; oc < OC_TILE_SIZE; ++oc) {
#pragma HLS UNROLL
      gap_sum[oc] = 0;
    }

    const int total_vecs = desc.conv_tile_h * desc.conv_tile_w;

  GAP_ACC:
    for (int i = 0; i < total_vecs; ++i) {
#pragma HLS loop_tripcount min=TC_SPATIAL max=TC_SPATIAL
#pragma HLS PIPELINE II=1
      const activations_vector_dt word = ofm_in.read();
      for (int oc = 0; oc < OC_TILE_SIZE; ++oc) {
#pragma HLS UNROLL
        const int lo = oc * BITWIDTH;
        gap_sum[oc] += static_cast<acc_t>((ap_int<8>)word.range(lo + BITWIDTH - 1, lo));
      }
    }

    activations_vector_dt out_word = 0;

    for (int oc = 0; oc < OC_TILE_SIZE; ++oc) {
#pragma HLS UNROLL
      const int lo = oc * BITWIDTH;

      const ap_int<GAP_PROD_BITS> prod =
          static_cast<ap_int<GAP_SUM_BITS>>(gap_sum[oc]) *
          static_cast<ap_int<GAP_MUL_BITS>>(gap_mul);

      const ap_int<GAP_SUM_BITS> scaled = (prod + GAP_ROUND) >> GAP_RSHIFT;

      ap_int<8> clamped;
      if      (scaled >  127) clamped =  127;
      else if (scaled < -128) clamped = -128;
      else                    clamped = static_cast<ap_int<8>>(scaled);

      out_word.range(lo + BITWIDTH - 1, lo) = static_cast<ap_uint<8>>(clamped);
    }

    ofm_out.write(out_word);
  }
}

void store_ofm_tile(activations_vector_dt *out,
                    desc_stream_ofm_t &desc_stream_ofm,
                    ofm_stream_t &ofm_stream, int num_tiles) {
OFM_TILE_LOOP:
  for (int t = 0; t < num_tiles; t++) {
#pragma HLS loop_tripcount min=TC_NUM_TILES max=TC_NUM_TILES
    const ofm_desc_t desc = desc_stream_ofm.read();
    const int rows         = desc.tile_h;   // Zero denotes accumulation-only work.
    const int vecs_per_row = desc.tile_w;
    const int elem_offset  = desc.offset_vecs;
    const int stride_elems = desc.row_stride;
    if (rows == 0 || vecs_per_row == 0) continue;

  OFM_STORE_ROWS:
    for (int r = 0; r < rows; r++) {
#pragma HLS loop_tripcount min=TC_T_OH max=TC_T_OH
      const int row_base = elem_offset + r * stride_elems;
    OFM_STORE_COLS:
      for (int c = 0; c < vecs_per_row; c++) {
#pragma HLS loop_tripcount min=TC_T_OW max=TC_T_OW
#pragma HLS PIPELINE II = 1
        out[row_base + c] = ofm_stream.read();
      }
    }
  }
}

#if 0  // Disabled alternative whole-tile ping-pong writer.
static void ofm_fill_buf(ofm_stream_t &ofm_stream,
                         activations_vector_dt buf[OFM_BUF_DEPTH],
                         int total_vecs) {
OFM_FILL_LOOP:
  for (int i = 0; i < total_vecs; i++) {
#pragma HLS loop_tripcount min=TC_SPATIAL max=TC_SPATIAL
#pragma HLS PIPELINE II = 1
    buf[i] = ofm_stream.read();
  }
}

static void ofm_store_buf(activations_vector_dt *__restrict out,
                          activations_vector_dt buf[OFM_BUF_DEPTH],
                          const ofm_desc_t &desc) {
  const int elem_offset = desc.offset_vecs;
  const int vecs_per_row = desc.tile_w;
  const int stride_elems = desc.row_stride;

  int buf_idx = 0;
OFM_STORE_ROWS_PP:
  for (int r = 0; r < desc.tile_h; r++) {
    const int row_base = elem_offset + r * stride_elems;
  OFM_STORE_COLS_PP:
    for (int c = 0; c < vecs_per_row; c++) {
#pragma HLS PIPELINE II = 1
      out[row_base + c] = buf[buf_idx++];
    }
  }
}

static void store_ofm_tile_pingpong(activations_vector_dt *out,
                                    desc_stream_ofm_t &desc_stream_ofm,
                                    ofm_stream_t &ofm_stream, int num_tiles) {
  activations_vector_dt buf0[OFM_BUF_DEPTH];
  activations_vector_dt buf1[OFM_BUF_DEPTH];
#pragma HLS BIND_STORAGE variable = buf0 type = RAM_1P
#pragma HLS BIND_STORAGE variable = buf1 type = RAM_1P
#pragma HLS allocation function instances = ofm_fill_buf limit = 2
#pragma HLS allocation function instances = ofm_store_buf limit = 2

  ofm_desc_t desc0 = desc_stream_ofm.read();
  ofm_fill_buf(ofm_stream, buf0, desc0.tile_h * desc0.tile_w);

OFM_PP_LOOP:
  for (int t = 0; t < num_tiles - 1; t++) {
    ofm_desc_t desc_next = desc_stream_ofm.read();
    if ((t & 1) == 0) {
      ofm_fill_buf(ofm_stream, buf1, desc_next.tile_h * desc_next.tile_w);
      ofm_store_buf(out, buf0, desc0);
    } else {
      ofm_fill_buf(ofm_stream, buf0, desc_next.tile_h * desc_next.tile_w);
      ofm_store_buf(out, buf1, desc0);
    }
    desc0 = desc_next;
  }

  if (((num_tiles - 1) & 1) == 0) ofm_store_buf(out, buf0, desc0);
  else                            ofm_store_buf(out, buf1, desc0);
}
#endif  // Disabled alternative writer.
