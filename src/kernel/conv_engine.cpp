/**
 * @file conv_engine.cpp
 * @brief Implements ping-pong weight loading and dense-convolution MACs.
 *
 * Each transfer epoch loads one contiguous region per output-channel bank and
 * then runs the descriptors assigned to that epoch.
 */

#include "conv_engine.h"

// HLS: Replicate activation registers across output-channel groups to bound
// DSP-input fanout. Implementation must preserve equivalent registers or the
// replicas collapse back into one high-fanout net.
static constexpr int ACT_REPL = 4;
static constexpr int OC_PER_REPL = OC_TILE_SIZE / ACT_REPL;
static_assert(OC_TILE_SIZE % ACT_REPL == 0,
              "OC_TILE_SIZE must divide evenly across activation replicas");

// Loads one contiguous transfer region into each output-channel bank.
static void weight_load(const weight_vector_dt *__restrict weights,
                        weight_vector_dt buf[OC_TILE_SIZE][WEIGHT_BUF_DEPTH],
                        int ddr_offset, int depth_count, int total_layer_depth) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = buf complete dim = 1

BANK_LOOP:
  for (int oc = 0; oc < OC_TILE_SIZE; oc++) {
#pragma HLS LOOP_TRIPCOUNT min = 16 max = 16
    const int bank_base = oc * total_layer_depth + ddr_offset;

  DEPTH_LOOP:
    for (int d = 0; d < depth_count; d++) {
#pragma HLS LOOP_TRIPCOUNT min = TC_DEPTH_COUNT max = TC_DEPTH_COUNT
#pragma HLS PIPELINE II = 1
      buf[oc][d] = weights[bank_base + d];
    }
  }
}

// Computes `P_SPATIAL` output positions per cycle with one weight read per
// output-channel bank. Chained descriptors sweep all CIGs before the next tile.
static void mac_tile(weight_vector_dt buf[OC_TILE_SIZE][WEIGHT_BUF_DEPTH],
                     ifm_stream_t pixel_in[P_SPATIAL], acc_group_stream_t &acc_out,
                     desc_stream_comp_t &desc_comp,
                     int compute_count,
                     uint8_t kernel_size) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = pixel_in complete dim = 1

MAC_TILE_LOOP:
  for (int tile = 0; tile < compute_count; tile++) {
#pragma HLS loop_tripcount min = TC_COMPUTE_COUNT max = TC_COMPUTE_COUNT

    comp_desc_t desc = desc_comp.read();

    const int K2          = kernel_size * kernel_size;
    // INVARIANT: Grouping must match the active IFM producer. The MAC body
    // consumes groups in producer order and is otherwise mode-independent.
    const int num_windows = desc.ofm_tile_h * desc.ofm_tile_w;
    const int num_groups  = desc_row_aligned(desc)
        ? desc.ofm_tile_h * ((desc.ofm_tile_w + P_SPATIAL - 1) / P_SPATIAL)
        : (num_windows + P_SPATIAL - 1) / P_SPATIAL;
    const int total_iters = num_groups * K2;
    const int base_depth  = desc.uram_base_depth;
    const int num_cigs    = (desc.num_cigs > 0) ? static_cast<int>(desc.num_cigs) : 1;

CIG_LOOP:
    for (int cig = 0; cig < num_cigs; cig++) {
#pragma HLS loop_tripcount min = 1 max = TC_CIG
      const int cig_base = base_depth + cig * K2;
      int k_cnt = 0;

      // HLS: Per-CIG declaration prevents a false cross-CIG dependence. The
      // post-processing stage performs cross-CIG accumulation.
      acc_t oc_acc[P_SPATIAL][OC_TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable = oc_acc complete dim = 0

    MAC_FLAT_LOOP:
      for (int idx = 0; idx < total_iters; idx++) {
#pragma HLS loop_tripcount min = TC_MAC_FLAT max = TC_MAC_FLAT
#pragma HLS PIPELINE II = 1

        const bool first_k = (k_cnt == 0);
        const bool last_k  = (k_cnt == K2 - 1);
        acc_group_dt agrp;

        // Read and unpack each lane once before activation replication.
        pixel_t act_in[P_SPATIAL][P_IC];
#pragma HLS ARRAY_PARTITION variable = act_in complete dim = 0

        for (int p = 0; p < P_SPATIAL; p++) {
#pragma HLS UNROLL
          const pixel_vector_dt pv = pixel_in[p].read();
          for (int c = 0; c < P_IC; c++) {
#pragma HLS UNROLL
            act_in[p][c] = pv.range((c + 1) * BITWIDTH - 1, c * BITWIDTH);
          }
        }

        // HLS: The latency boundary materializes independent register banks for
        // disjoint output-channel slices without changing the loop II.
        pixel_t act[ACT_REPL][P_SPATIAL][P_IC];
#pragma HLS ARRAY_PARTITION variable = act complete dim = 0
        {
#pragma HLS LATENCY min = 1
          for (int r = 0; r < ACT_REPL; r++) {
#pragma HLS UNROLL
            for (int p = 0; p < P_SPATIAL; p++) {
#pragma HLS UNROLL
              for (int c = 0; c < P_IC; c++) {
#pragma HLS UNROLL
                act[r][p][c] = act_in[p][c];
              }
            }
          }
        }

        // Adjacent output channels share one DSP and one activation replica.
        // Each weight-bank read is reused across all spatial lanes.
      OC_PAIR:
        for (int op = 0; op < OC_TILE_SIZE / 2; op++) {
#pragma HLS UNROLL
          const int oc0 = 2 * op;
          const int oc1 = oc0 + 1;
          const int act_bank = oc0 / OC_PER_REPL;

          weight_vector_dt wt_word0 = buf[oc0][cig_base + k_cnt];
          weight_vector_dt wt_word1 = buf[oc1][cig_base + k_cnt];

          weight_t w_val0[P_IC], w_val1[P_IC];
#pragma HLS ARRAY_PARTITION variable = w_val0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = w_val1 complete dim = 1
          for (int c = 0; c < P_IC; c++) {
#pragma HLS UNROLL
            w_val0[c] = wt_word0.range((c + 1) * BITWIDTH - 1, c * BITWIDTH);
            w_val1[c] = wt_word1.range((c + 1) * BITWIDTH - 1, c * BITWIDTH);
          }

          for (int p = 0; p < P_SPATIAL; p++) {
#pragma HLS UNROLL

            // Extract both products and defer high-lane sign correction until
            // after reduction.
            acc_t prods0[P_IC], prods1[P_IC];
            ap_uint<1> lo_neg[P_IC];
#pragma HLS ARRAY_PARTITION variable = prods0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = prods1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = lo_neg complete dim = 1

            for (int c = 0; c < P_IC; c++) {
#pragma HLS UNROLL
              const ap_int<35> pr =
                  dsp_dual_mul(act[act_bank][p][c], w_val0[c], w_val1[c]);
              prods0[c] = acc_t(ap_int<16>(pr.range(15, 0)));
              prods1[c] = acc_t(ap_int<16>(pr.range(33, 18)));
              lo_neg[c] = pr[15];
            }

            for (int step = P_IC / 2; step > 0; step >>= 1) {
#pragma HLS UNROLL
              for (int i = 0; i < step; i++) {
#pragma HLS UNROLL
                prods0[i] = prods0[i] + prods0[i + step];
                prods1[i] = prods1[i] + prods1[i + step];
              }
            }

            // Correct the high lane once using the negative-low-product count.
            ap_uint<5> nborrow = 0;
            for (int c = 0; c < P_IC; c++) {
#pragma HLS UNROLL
              nborrow += lo_neg[c];
            }

            const acc_t lane0 = prods0[0];
            const acc_t lane1 = prods1[0] + acc_t(nborrow);

            acc_t next_acc0 = lane0;
            acc_t next_acc1 = lane1;
            if (!first_k) {
              next_acc0 = oc_acc[p][oc0] + lane0;
              next_acc1 = oc_acc[p][oc1] + lane1;
            }
            oc_acc[p][oc0] = next_acc0;
            oc_acc[p][oc1] = next_acc1;
            if (last_k) {
              agrp.v[p].v[oc0] = next_acc0;
              agrp.v[p].v[oc1] = next_acc1;
            }
          }
        }

        // HLS: Emit the new accumulator value directly to avoid another read
        // from the register chain.
        if (last_k) {
          acc_out.write(agrp);
        }

        if (last_k) k_cnt = 0;
        else k_cnt++;
      }
    }
  }
}

void conv_engine(const weight_vector_dt *__restrict weights,
                 desc_stream_xfer_t &desc_xfer,
                 ifm_stream_t pixel_in[P_SPATIAL],
                 acc_group_stream_t &acc_out,
                 desc_stream_comp_t &desc_comp,
                 int num_xfers,
                 uint8_t kernel_size) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = pixel_in complete dim = 1

  weight_vector_dt wbuf_A[OC_TILE_SIZE][WEIGHT_BUF_DEPTH];
  weight_vector_dt wbuf_B[OC_TILE_SIZE][WEIGHT_BUF_DEPTH];
#pragma HLS BIND_STORAGE variable = wbuf_A type = RAM_1P impl = bram
#pragma HLS BIND_STORAGE variable = wbuf_B type = RAM_1P impl = bram
#pragma HLS ARRAY_PARTITION variable = wbuf_A complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wbuf_B complete dim = 1

#pragma HLS allocation function instances = weight_load limit = 2
#pragma HLS allocation function instances = mac_tile limit = 2

  // Prime the first weight buffer.
  uram_xfer_desc_t xd0 = desc_xfer.read();
  weight_load(weights, wbuf_A, xd0.ddr_offset_vecs, xd0.depth_count, xd0.total_layer_depth);
  int comp_count0 = xd0.compute_count;

CONV_PP_LOOP:
  for (int ep = 0; ep < num_xfers - 1; ep++) {
#pragma HLS loop_tripcount min = TC_PP_LOOP max = TC_PP_LOOP

    uram_xfer_desc_t xd_next = desc_xfer.read();

    if ((ep & 1) == 0) {
      weight_load(weights, wbuf_B, xd_next.ddr_offset_vecs, xd_next.depth_count, xd_next.total_layer_depth);
      mac_tile(wbuf_A, pixel_in, acc_out, desc_comp, comp_count0, kernel_size);
    } else {
      weight_load(weights, wbuf_A, xd_next.ddr_offset_vecs, xd_next.depth_count, xd_next.total_layer_depth);
      mac_tile(wbuf_B, pixel_in, acc_out, desc_comp, comp_count0, kernel_size);
    }

    comp_count0 = xd_next.compute_count;
  }

  // Drain the final loaded epoch.
  if (((num_xfers - 1) & 1) == 0) {
    mac_tile(wbuf_A, pixel_in, acc_out, desc_comp, comp_count0, kernel_size);
  } else {
    mac_tile(wbuf_B, pixel_in, acc_out, desc_comp, comp_count0, kernel_size);
  }
}
