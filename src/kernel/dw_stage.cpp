/**
 * @file dw_stage.cpp
 * @brief Implements fused streaming depthwise convolution.
 *
 * A flat raster pipeline maintains four history rows. Each padded input column
 * of five rows is multiplied by all five weight columns at once, and a chain of
 * four partial sums adds the column results into one output per position, of
 * which only stride-aligned ones are emitted. A 3x3 kernel uses the newest
 * three rows and weight columns.
 */

#include "dw_stage.h"

struct dw_padding_t {
  int top;
  int bottom;
  int left;
  int right;
};

static inline dw_padding_t unpack_dw_padding(uint16_t packed) {
#pragma HLS INLINE
  dw_padding_t p;
  p.top = (packed >> 12) & 0xF;
  p.bottom = (packed >> 8) & 0xF;
  p.left = (packed >> 4) & 0xF;
  p.right = packed & 0xF;
  return p;
}

static inline activations_vector_dt make_dw_pad_word() {
#pragma HLS INLINE
  return (activations_vector_dt)0;
}

// Uses the same two-stage integer requantization as dense convolution. The
// layer's shift is decoded once per launch into a direction and a 5-bit
// amount: shift > 0 shifts right by shift - 1, otherwise left by 1 - shift.
// The compiler fuses only layers whose shift lies in [-30, 32], where both
// amounts fit.
static inline ap_int<8> dw_requant(acc_t acc, acc_t bias, bool shift_right,
                                   ap_uint<5> rshift, ap_uint<5> lshift,
                                   bool relu, int act_max) {
#pragma HLS INLINE
  const acc_t s1 = shift_right ? acc_t(acc >> rshift) : acc_t(acc << lshift);
  acc_t tb = s1 + bias + 1;
  acc_t y = tb >> 1;
  if (y > 127) {
    y = 127;
  } else if (y < -128) {
    y = -128;
  }
  if (relu && y < 0) {
    y = 0;
  }
  if (relu && y > act_max) {
    y = act_max;
  }
  return (ap_int<8>)y;
}

// A 3x3 block shifts in 13 slots: three rows of three taps, the first two rows
// each followed by two zeros.
#define DW_K3_SLOTS 13

// Shifts one COG's taps into the per-channel weight chain and loads its biases.
//
// The chain holds the 5x5 window's taps in raster order. A 5x5 block shifts in
// all 25 taps. A 3x3 block's 13 slots leave its taps on the window's
// bottom-right 3x3 positions (12-14, 17-19, 22-24), so no tap needs a computed
// position. Positions 0-11 then keep earlier values, which only ever meet
// window positions a 3x3 kernel holds at zero.
static inline void load_cog_weights(
    const weight_vector_dt tap_store[DW_TAP_STORE_DEPTH],
    const weight_vector_dt bias_store[DW_BIAS_STORE_DEPTH], int g, bool k5,
    ap_int<8> w_chain[OC_TILE_SIZE][DW_TAPS_MAX], acc_t bias_reg[OC_TILE_SIZE]) {
#pragma HLS INLINE

  const int slots = k5 ? DW_TAPS_5 : DW_K3_SLOTS;
  int tap_addr = g * (k5 ? DW_TAPS_5 : DW_TAPS_3);
  const int bias_base = g * DW_BIAS_WORDS;
  int col = 0;

  // Biases load during the first four slots, from their own store.
DW_WLOAD:
  for (int s = 0; s < slots; ++s) {
#pragma HLS loop_tripcount min = DW_K3_SLOTS max = DW_TAPS_5
#pragma HLS PIPELINE II = 1
    const bool gap = !k5 && col >= 3;
    const weight_vector_dt ww = gap ? weight_vector_dt(0) : tap_store[tap_addr];
    if (!gap) {
      ++tap_addr;
    }
    col = (col == DW_KMAX - 1) ? 0 : col + 1;
    for (int oc = 0; oc < OC_TILE_SIZE; ++oc) {
#pragma HLS UNROLL
      for (int i = 0; i < DW_TAPS_MAX - 1; ++i) {
#pragma HLS UNROLL
        w_chain[oc][i] = w_chain[oc][i + 1];
      }
      w_chain[oc][DW_TAPS_MAX - 1] = (ap_int<8>)ww.range(oc * 8 + 7, oc * 8);
    }
    if (s < DW_BIAS_WORDS) {
      const weight_vector_dt bw = bias_store[bias_base + s];
      for (int l = 0; l < 4; ++l) {
#pragma HLS UNROLL
        bias_reg[s * 4 + l] = (acc_t)(ap_int<32>)bw.range(l * 32 + 31, l * 32);
      }
    }
  }
}

// Two products that share an activation, in one DSP: a * (w1 * 2^18 + w0).
static inline ap_int<35> dw_dual_mul(ap_int<8> a, ap_int<8> w0, ap_int<8> w1) {
#pragma HLS INLINE
  const ap_int<27> packed = (ap_int<27>(w1) << 18) + ap_int<27>(w0);
  ap_int<35> pr = packed * a;
#pragma HLS BIND_OP variable = pr op = mul impl = dsp
  return pr;
}

// Column sums of one lane: P_c = sum over rows r of col[r] * w[r][c].
//
// Weight columns 0-1 and 2-3 share each activation in one DSP. The five packed
// products of a pair are summed whole, which is exact: the low field's sum is
// at most 5 * 128 * 128 = 81920 in magnitude, inside the signed 18-bit field,
// so it separates from the high field without loss.
static inline void dw_column_sums(const activations_vector_dt col[DW_KMAX],
                                  const ap_int<8> w[DW_TAPS_MAX], int lo,
                                  acc_t p[DW_KMAX]) {
#pragma HLS INLINE
  ap_int<40> s01 = 0;
  ap_int<40> s23 = 0;
  acc_t s4 = 0;
  for (int r = 0; r < DW_KMAX; ++r) {
#pragma HLS UNROLL
    const ap_int<8> a = (ap_int<8>)col[r].range(lo + BITWIDTH - 1, lo);
    s01 += dw_dual_mul(a, w[r * DW_KMAX + 0], w[r * DW_KMAX + 1]);
    s23 += dw_dual_mul(a, w[r * DW_KMAX + 2], w[r * DW_KMAX + 3]);
    ap_int<16> pr4 = a * w[r * DW_KMAX + 4];
#pragma HLS BIND_OP variable = pr4 op = mul impl = dsp
    s4 += pr4;
  }
  const ap_int<18> lo01 = s01.range(17, 0);
  const ap_int<18> lo23 = s23.range(17, 0);
  p[0] = lo01;
  p[1] = (acc_t)((s01 - lo01) >> 18);
  p[2] = lo23;
  p[3] = (acc_t)((s23 - lo23) >> 18);
  p[4] = s4;
}

static void pass_dw_tile(ofm_stream_t &ofm_in, ofm_stream_t &ofm_out,
                         int num_rows, int row_len_vecs) {
#pragma HLS INLINE

  const int total_vecs = num_rows * row_len_vecs;
DW_PASS_TILE:
  for (int i = 0; i < total_vecs; ++i) {
#pragma HLS loop_tripcount min = 1 max = TC_SPATIAL
#pragma HLS PIPELINE II = 1
    ofm_out.write(ofm_in.read());
  }
}

void dw_stage(desc_stream_ofm_t &desc_in, ofm_stream_t &ofm_in,
              const weight_vector_dt *__restrict dw_weights, desc_stream_ofm_t &desc_out,
              ofm_stream_t &ofm_out, int num_tiles, int num_cogs, int dw_config,
              int dw_shift) {
#pragma HLS INLINE off

  const bool dw_enable = (dw_config & 0x1) != 0;
  const int stride = (dw_config >> 1) & 0x3;
  const bool dw_relu = (dw_config >> 3) & 0x1;
  const bool k5 = (dw_config >> 4) & 0x1;
  // Bits 8-15 hold the ReLU clamp: 127 for ReLU, lower for ReLU6.
  const int dw_act_max = (dw_config >> 8) & 0xFF;
  const int kdim = k5 ? 5 : 3;
  const bool dw_shift_right = dw_shift > 0;
  const ap_uint<5> dw_rshift = dw_shift > 0 ? dw_shift - 1 : 0;
  const ap_uint<5> dw_lshift = dw_shift > 0 ? 0 : 1 - dw_shift;
  const int k_last = kdim - 1;

  const int smask = stride - 1;
  const int sshift = stride >> 1;

  static weight_vector_dt dw_tap_store[DW_TAP_STORE_DEPTH];
#pragma HLS BIND_STORAGE variable = dw_tap_store type = RAM_S2P impl = BRAM
  static weight_vector_dt dw_bias_store[DW_BIAS_STORE_DEPTH];
#pragma HLS BIND_STORAGE variable = dw_bias_store type = RAM_S2P impl = BRAM

  activations_vector_dt lb0[MAX_TILE_W + DW_KMAX - 1];
  activations_vector_dt lb1[MAX_TILE_W + DW_KMAX - 1];
  activations_vector_dt lb2[MAX_TILE_W + DW_KMAX - 1];
  activations_vector_dt lb3[MAX_TILE_W + DW_KMAX - 1];
#pragma HLS BIND_STORAGE variable = lb0 type = RAM_S2P impl = BRAM
#pragma HLS BIND_STORAGE variable = lb1 type = RAM_S2P impl = BRAM
#pragma HLS BIND_STORAGE variable = lb2 type = RAM_S2P impl = BRAM
#pragma HLS BIND_STORAGE variable = lb3 type = RAM_S2P impl = BRAM

  ap_int<8> w_chain[OC_TILE_SIZE][DW_TAPS_MAX];
#pragma HLS ARRAY_PARTITION variable = w_chain complete dim = 0
  acc_t bias_reg[OC_TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable = bias_reg complete dim = 1

  // Defined values from the start: a 3x3 block leaves 12 chain positions as
  // they were, and an undefined weight would stay undefined in RTL simulation
  // even when multiplied by a zero activation.
  for (int oc = 0; oc < OC_TILE_SIZE; ++oc) {
#pragma HLS UNROLL
    for (int i = 0; i < DW_TAPS_MAX; ++i) {
#pragma HLS UNROLL
      w_chain[oc][i] = 0;
    }
  }

  const activations_vector_dt pad_word = make_dw_pad_word();

  if (dw_enable) {
    // Route each COG block's tap words and bias words to their stores.
    const int words_per_cog = k5 ? DW_WORDS_PER_COG_5 : DW_WORDS_PER_COG_3;
    const int taps = k5 ? DW_TAPS_5 : DW_TAPS_3;
    int word = 0;
    int tap_idx = 0;
    int bias_idx = 0;
  DW_WEIGHT_PRELOAD:
    for (int i = 0; i < num_cogs * words_per_cog; ++i) {
#pragma HLS loop_tripcount min = DW_WORDS_PER_COG_3 max = DW_MAX_COGS * DW_WORDS_PER_COG_3
#pragma HLS PIPELINE II = 1
      const weight_vector_dt w = dw_weights[i];
      if (word < taps) {
        if (tap_idx < DW_TAP_STORE_DEPTH) dw_tap_store[tap_idx] = w;
        ++tap_idx;
      } else if (word < taps + DW_BIAS_WORDS) {
        if (bias_idx < DW_BIAS_STORE_DEPTH) dw_bias_store[bias_idx] = w;
        ++bias_idx;
      }
      word = (word == words_per_cog - 1) ? 0 : word + 1;
    }
  }

DW_TILE_LOOP:
  for (int t = 0; t < num_tiles; ++t) {
#pragma HLS loop_tripcount min = TC_NUM_TILES max = TC_NUM_TILES
    ofm_desc_t desc = desc_in.read();

    if (!dw_enable || desc.tile_h == 0) {
      desc_out.write(desc);
      if (desc.tile_h != 0) {
        pass_dw_tile(ofm_in, ofm_out, desc.tile_h, desc.tile_w);
      }
      continue;
    }

    load_cog_weights(dw_tap_store, dw_bias_store, desc.dw_cog_idx, k5, w_chain,
                     bias_reg);

    const dw_padding_t pad = unpack_dw_padding(desc.pool_packed_padding);
    const int conv_h = desc.conv_tile_h;
    const int conv_w = desc.conv_tile_w;
    const bool relu = dw_relu;
    const int padded_h = conv_h + pad.top + pad.bottom;
    const int padded_w = conv_w + pad.left + pad.right;
    // The stride is 1 or 2, so a shift replaces a sequential divider.
    const int out_h = ((padded_h - kdim) >> sshift) + 1;
    const int out_w = ((padded_w - kdim) >> sshift) + 1;
    ofm_desc_t desc_ds = desc;
    desc_ds.tile_h = static_cast<uint8_t>(out_h);
    desc_ds.tile_w = static_cast<uint8_t>(out_w);
    desc_out.write(desc_ds);

    // part[k][oc] holds the partial output that column k+1 of the window adds
    // to; the chain turns column sums into window sums one column at a time.
    acc_t part[DW_KMAX - 1][OC_TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable = part complete dim = 0
    for (int k = 0; k < DW_KMAX - 1; ++k) {
#pragma HLS UNROLL
      for (int oc = 0; oc < OC_TILE_SIZE; ++oc) {
#pragma HLS UNROLL
        part[k][oc] = 0;
      }
    }
    int y = 0;
    int x = 0;

  DW_TILE_PIX:
    for (int i = 0; i < padded_h * padded_w; ++i) {
#pragma HLS loop_tripcount min = 9 max = 2448
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = lb0 inter false
#pragma HLS DEPENDENCE variable = lb1 inter false
#pragma HLS DEPENDENCE variable = lb2 inter false
#pragma HLS DEPENDENCE variable = lb3 inter false

      const bool data_row = (y >= pad.top) && (y < pad.top + conv_h);
      const bool data_col = (x >= pad.left) && (x < pad.left + conv_w);
      const activations_vector_dt p =
          (data_row && data_col) ? ofm_in.read() : pad_word;

      // Promote the four history rows after reading the current column.
      const activations_vector_dt h0 = lb0[x];
      const activations_vector_dt h1 = lb1[x];
      const activations_vector_dt h2 = lb2[x];
      const activations_vector_dt h3 = lb3[x];
      lb0[x] = h1;
      lb1[x] = h2;
      lb2[x] = h3;
      lb3[x] = p;

      // A 3x3 kernel reads only the newest three rows. Holding the rest at
      // zero keeps rows from before this tile out of the sums, where a zero
      // weight alone would not mask an undefined value in RTL.
      const activations_vector_dt col[DW_KMAX] = {
          k5 ? h0 : pad_word, k5 ? h1 : pad_word, h2, h3, p};

      const bool emit = y >= k_last && x >= k_last &&
                        ((y - k_last) & smask) == 0 &&
                        ((x - k_last) & smask) == 0 &&
                        ((y - k_last) >> sshift) < out_h &&
                        ((x - k_last) >> sshift) < out_w;
      activations_vector_dt out_word = 0;
      for (int oc = 0; oc < OC_TILE_SIZE; ++oc) {
#pragma HLS UNROLL
        const int lo = oc * BITWIDTH;
        acc_t pc[DW_KMAX];
#pragma HLS ARRAY_PARTITION variable = pc complete dim = 0
        dw_column_sums(col, w_chain[oc], lo, pc);
        // A 3x3 block leaves weight columns 0-1 of its top row stale.
        if (!k5) {
          pc[0] = 0;
          pc[1] = 0;
        }
        // The window ending at this column: earlier columns' partials plus
        // this column's last-weight-column sum.
        const acc_t acc = part[3][oc] + pc[4];
        part[3][oc] = part[2][oc] + pc[3];
        part[2][oc] = part[1][oc] + pc[2];
        part[1][oc] = part[0][oc] + pc[1];
        part[0][oc] = pc[0];
        out_word.range(lo + BITWIDTH - 1, lo) =
            (ap_uint<8>)dw_requant(acc, bias_reg[oc], dw_shift_right, dw_rshift,
                                  dw_lshift, relu, dw_act_max);
      }
      if (emit) {
        ofm_out.write(out_word);
      }

      if (x == padded_w - 1) {
        x = 0;
        ++y;
      } else {
        ++x;
      }
    }
  }
}
