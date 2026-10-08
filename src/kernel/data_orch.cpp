/**
 * @file data_orch.cpp
 * @brief Implements descriptor distribution and input-side data movement.
 *
 * Provides bias and residual loading, padded input streaming, and resident
 * pointwise input caching.
 */

#include "data_orch.h"

void schedule_controller(const uram_xfer_desc_t *__restrict xfer_desc,
                         desc_stream_xfer_t &desc_xfer,
                         int num_xfers,
                         const descriptor_dt *__restrict tile_desc,
                         desc_stream_ifm_t &desc_ifm,
                         desc_stream_pad_t &desc_pad,
                         desc_stream_comp_t &desc_comp_ifm,
                         desc_stream_comp_t &desc_comp_mac,
                         desc_stream_comp_t &desc_comp_acc,
                         desc_stream_ofm_t &desc_ofm,
                         int num_tiles,
                         uint8_t kernel_size,
                         ap_uint<1> use_ifm_cache,
                         bool dw_enable) {
#pragma HLS INLINE off

  // INVARIANT: Grouping must match the active IFM producer. Cache hits use
  // flat groups; all other paths, including K=1 cache misses, are row-aligned.
  const uint8_t layer_row_aligned =
      (kernel_size == 1 && use_ifm_cache) ? 0 : 1;

  // The weight engine loads the next bank before computing the current one.
  // Prime both epochs, then fetch one more after issuing each epoch's work.
  // Sending every transfer first would block on the bounded transfer FIFO.
  uram_xfer_desc_t current = xfer_desc[0];
  desc_xfer.write(current);
  uint16_t current_count = current.compute_count;
  uint16_t next_count = 0;
  if (num_xfers > 1) {
    uram_xfer_desc_t next = xfer_desc[1];
    desc_xfer.write(next);
    next_count = next.compute_count;
  }
  int tile_index = 0;

FETCH_EPOCH_LOOP:
  for (int epoch = 0; epoch < num_xfers; ++epoch) {
#pragma HLS loop_tripcount min=TC_NUM_XFERS max=TC_NUM_XFERS
  FETCH_LOOP:
    for (int t = 0; t < current_count; t++) {
#pragma HLS loop_tripcount min=TC_COMPUTE_COUNT max=TC_COMPUTE_COUNT
#pragma HLS PIPELINE II = 1
      descriptor_dt d = tile_desc[tile_index + t];

      ifm_desc_t id;
      id.offset_vecs = d.ifm_offset_vecs;
      id.ddr_w       = d.ifm_ddr_w;
      id.ddr_stride  = d.ifm_ddr_stride;
      id.ddr_h       = d.ifm_ddr_h;
      desc_ifm.write(id);

      pad_desc_t pd;
      pd.packed_padding = d.packed_padding;
      pd.ddr_h   = d.ifm_ddr_h;
      pd.ddr_w   = d.ifm_ddr_w;
      pd.tile_h  = d.ifm_tile_h;
      pd.tile_w  = d.ifm_tile_w;
      desc_pad.write(pd);

      ofm_desc_t od = {};
      const bool do_writeback = (d.flags & ACCUM_WRITE_BACK) != 0;
      const bool do_pool = (d.flags & POOL_ENABLE) != 0;
      const bool do_gap = (d.flags & GAP_ENABLE) != 0;
      // INVARIANT: Fused-depthwise launches retain pointwise dimensions until
      // `dw_stage` replaces them with final writeback geometry.
      (void)dw_enable;
      const bool use_pooled_dims = do_pool || do_gap;
      od.offset_vecs = d.ofm_offset_vecs;
      od.tile_w = use_pooled_dims ? d.pooled_ofm_tile_w : d.ofm_tile_w;
      od.row_stride = d.ofm_row_stride;
      od.tile_h =
          do_writeback
              ? (use_pooled_dims ? d.pooled_ofm_tile_h : d.ofm_tile_h)
              : 0;
      od.flags = d.flags;
      od.residual_shift = d.residual_shift;
      od.conv_tile_w = do_writeback ? d.ofm_tile_w : 0;
      od.conv_tile_h = do_writeback ? d.ofm_tile_h : 0;
      od.pool_packed_padding = d.pool_packed_padding;
      od.post_pool_shift = d.post_pool_shift;
      od.dw_cog_idx = static_cast<uint8_t>(d.bias_offset_idx);
      od.reserved[0] = 0;
      od.reserved[1] = 0;
      desc_ofm.write(od);

      comp_desc_t cd;
      cd.uram_base_depth = d.uram_base_depth;
      cd.ifm_ddr_w       = d.ifm_ddr_w;
      cd.ofm_tile_w      = d.ofm_tile_w;
      cd.packed_padding  = d.packed_padding;
      cd.ifm_ddr_h       = d.ifm_ddr_h;
      cd.ofm_tile_h      = d.ofm_tile_h;
      cd.ifm_tile_h      = d.ifm_tile_h;
      cd.ifm_tile_w      = d.ifm_tile_w;
      cd.flags           = d.flags;
      cd.reuse_ifm       = d.reuse_ifm;
      cd.num_cigs        = d.num_cigs;
      // Every descriptor names its first cache group; partial chains advance
      // from that group internally without resetting the reduction.
      cd.mode_bits       = desc_pack_mode_bits(layer_row_aligned, d.cache_cig_idx);
      cd.bias_offset_idx = d.bias_offset_idx;
      desc_comp_ifm.write(cd);
      desc_comp_mac.write(cd);
      desc_comp_acc.write(cd);
    }
    tile_index += current_count;
    current_count = next_count;
    if (epoch + 2 < num_xfers) {
      uram_xfer_desc_t next = xfer_desc[epoch + 2];
      desc_xfer.write(next);
      next_count = next.compute_count;
    }
  }
  // The compiler guarantees sum(transfer.compute_count) == num_tiles.
  (void)num_tiles;
}

void preload_bias_to_bram(const biases_vector_dt *__restrict biases,
                          biases_vector_dt bias_cache[BIAS_BUF_DEPTH],
                          int num_bias_vecs) {
BIAS_PRELOAD:
  for (int i = 0; i < num_bias_vecs; i++) {
#pragma HLS loop_tripcount min=TC_NUM_BIAS_VECS max=TC_NUM_BIAS_VECS
#pragma HLS PIPELINE II = 1
    bias_cache[i] = biases[i];
  }
}

void residual_loader(const activations_vector_dt *__restrict residual,
                     desc_stream_ofm_t &desc_stream_in,
                     desc_stream_ofm_t &desc_stream_out,
                     residual_stream_t &residual_stream,
                     int num_tiles) {
#pragma HLS INLINE off

RESIDUAL_TILE_LOOP:
  for (int t = 0; t < num_tiles; ++t) {
#pragma HLS loop_tripcount min=TC_NUM_TILES max=TC_NUM_TILES
    ofm_desc_t desc = desc_stream_in.read();
    desc_stream_out.write(desc);

    const bool do_add = (desc.flags & ADD_ENABLE) != 0;
    if (!do_add || desc.tile_h == 0) {
      continue;
    }

    // INVARIANT: Global average pooling still consumes the full pre-pool
    // residual tile even though the final writeback geometry is 1x1.
    const bool do_gap = (desc.flags & GAP_ENABLE) != 0;
    const int residual_rows   = desc.conv_tile_h;
    const int residual_cols   = desc.conv_tile_w;
    const int residual_stride = desc.conv_tile_w;
    const int residual_offset =
        do_gap ? desc.offset_vecs * residual_rows * residual_cols
               : desc.offset_vecs;

  RESIDUAL_ROW_LOOP:
    for (int r = 0; r < residual_rows; ++r) {
#pragma HLS loop_tripcount min=TC_T_OH max=TC_T_OH
      const int row_base = residual_offset + r * residual_stride;
    RESIDUAL_COL_LOOP:
      for (int c = 0; c < residual_cols; ++c) {
#pragma HLS loop_tripcount min=TC_T_OW max=TC_T_OW
#pragma HLS PIPELINE II = 1
        residual_stream.write(residual[row_base + c]);
      }
    }
  }
}

template <bool fixed_bound>
static void ifm_load_raw_impl(const ifm_word_dt *__restrict ifm,
                         pixel_vector_dt store[IFM_STORE_ROWS][P_SPATIAL],
                         const ifm_desc_t &ifm_desc) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=store complete dim=2

  const int elem_offset = ifm_desc.offset_vecs;        // Packed-vector units.
  const int ddr_w       = ifm_desc.ddr_w;
  const int stride_vecs = ifm_desc.ddr_stride;
  const int ddr_h       = ifm_desc.ddr_h;
  // One store slot per column pair. This is the layout pitch the reader side
  // assumes, so it depends on the tile width alone, never on its alignment.
  const int half_w      = (ddr_w + 1) >> 1;            // Packed words per row.
  const int tile_words = ((elem_offset & 1) + ddr_w + 1) >> 1;

RAW_TILE:
  for (int y = 0; y < ddr_h; y++) {
#pragma HLS loop_tripcount min=TC_PADDED_H max=TC_PADDED_H
    const int elem0    = elem_offset + y * stride_vecs;
    const int bank_row = y * half_w;
    // DDR is read in column pairs aligned to even elements. A row that starts
    // on an odd element therefore discards the first half-word, and covers one
    // column less than half_w words would suggest; without the extra word the
    // tile's final column is never fetched and the previous tile's data is
    // convolved in its place. Row parity is recomputed per row because an odd
    // row stride flips it. Even-aligned rows keep the previous word count.
    const int row_words = ((elem0 & 0x1) + ddr_w + 1) >> 1;
    const int words = fixed_bound ? tile_words : row_words;
  RAW_ROW:
    for (int j = 0; j < words; j++) {
#pragma HLS loop_tripcount min=TC_ROWLEN max=TC_ROWLEN
#pragma HLS PIPELINE II=1
      // HLS: Each iteration writes distinct banks and rows; suppress the false
      // carried dependence created by the runtime index.
#pragma HLS dependence variable=store type=inter false
      const int         word_idx = (elem0 >> 1) + j;
      const ifm_word_dt w        = ifm[word_idx];
      const int  col0 = (word_idx << 1) - elem0;
      const int  col1 = col0 + 1;
      const bool ok0  = (col0 >= 0) && (col0 < ddr_w);
      const bool ok1  = (col1 >= 0) && (col1 < ddr_w);
      for (int cp = 0; cp < IFM_BUF_COPIES; cp++) {
#pragma HLS UNROLL
        for (int b = 0; b < 2; b++) {
#pragma HLS UNROLL
          // Layout: bank = copy * 2 + column parity; row = y * half_w + col / 2.
          const int bank = cp * 2 + b;
          if      (ok0 && (col0 & 1) == b) store[bank_row + (col0 >> 1)][bank] = w.v[0];
          else if (ok1 && (col1 & 1) == b) store[bank_row + (col1 >> 1)][bank] = w.v[1];
        }
      }
    }
  }
}

static void ifm_load_raw(const ifm_word_dt *__restrict ifm,
                         pixel_vector_dt store[IFM_STORE_ROWS][P_SPATIAL],
                         const ifm_desc_t &ifm_desc) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=store complete dim=2
  // Odd width needs ceil(width/2) words for either start parity. Even stride
  // preserves the start parity across all rows. In both cases the first row's
  // EXACT bound is valid throughout the tile, without a clamp or extra read.
  // The remaining even-width/odd-stride case keeps the original row bound.
  if ((ifm_desc.ddr_w & 1) || !(ifm_desc.ddr_stride & 1)) {
    ifm_load_raw_impl<true>(ifm, store, ifm_desc);
  } else {
    ifm_load_raw_impl<false>(ifm, store, ifm_desc);
  }
}

// Streams row-aligned convolution windows and synthesizes boundary padding.
static void ifm_stream_pixels(pixel_vector_dt store[IFM_STORE_ROWS][P_SPATIAL],
                              ifm_stream_t pixel_stream[P_SPATIAL],
                              const comp_desc_t &desc,
                              uint8_t kernel_size,
                              uint8_t stride) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=store complete dim=2
#pragma HLS ARRAY_PARTITION variable=pixel_stream complete dim=1

  const int raw_w      = desc.ifm_ddr_w;
  const int raw_h      = desc.ifm_ddr_h;
  const int half_w     = (raw_w + 1) >> 1;
  const int pad_top    = (desc.packed_padding >> 12) & 0xF;
  const int pad_left   = (desc.packed_padding >> 4)  & 0xF;
  const int out_h      = desc.ofm_tile_h;
  const int out_w      = desc.ofm_tile_w;
  // INVARIANT: A row-aligned group never crosses an output-row boundary.
  const int groups_per_row = (out_w + P_SPATIAL - 1) / P_SPATIAL;
  const int num_groups     = out_h * groups_per_row;
  const int K2          = kernel_size * kernel_size;
  // INVARIANT: The compiler admits strides of 1 and 2 only.
  const int sshift      = stride >> 1;

  // HLS: Wrapping counters and running row offsets avoid runtime division and
  // multiplication in the II=1 loop body.
  int oy = 0, gx = 0;
  int row0 = -pad_top;               // Raw row of kernel row 0 for this group.
  int bank_row0 = -pad_top * half_w; // Its first word in the store.
GROUP_LOOP:
  for (int g = 0; g < num_groups; g++) {
#pragma HLS loop_tripcount min=TC_NUM_GROUPS max=TC_NUM_GROUPS
    const int gx_col   = gx * P_SPATIAL;
    int ky = 0, kx = 0;
    int raw_row = row0;
    int bank_row = bank_row0;
  K_POS:
    for (int k = 0; k < K2; k++) {
#pragma HLS loop_tripcount min=TC_K2 max=TC_K2
#pragma HLS PIPELINE II=1

      const bool row_ok      = (raw_row >= 0) && (raw_row < raw_h);

      for (int p = 0; p < P_SPATIAL; p++) {
#pragma HLS UNROLL
        const int  ox      = gx_col + p;
        const int  raw_col = ((ox << sshift) + kx) - pad_left;
        const bool col_ok  = (raw_col >= 0) && (raw_col < raw_w);
        pixel_vector_dt pix;
        if (ox < out_w && row_ok && col_ok) {
          // HLS: Compile-time lane copies limit the runtime bank selection to
          // a two-way parity mux.
          const int bank = (p / 2) * 2 + (raw_col & 1);
          pix = store[bank_row + (raw_col >> 1)][bank];
        } else {
          pix = 0;
        }
        pixel_stream[p].write(pix);
      }
      if (kx == kernel_size - 1) {
        kx = 0;
        ky++;
        raw_row++;
        bank_row += half_w;
      } else {
        kx++;
      }
    }
    if (gx == groups_per_row - 1) {
      gx = 0;
      oy++;
      row0 += stride;
      bank_row0 += half_w << sshift;
    } else {
      gx++;
    }
  }
}

// Cache rows are partitioned by spatial lane so the lane index remains static.
typedef ap_uint<IFM_STORE_ROWS_LOG2> cache_row_idx_t;

static void load_cache_and_forward(
    const ifm_word_dt *__restrict ifm,
    pixel_vector_dt k1_cache[IFM_CACHE_DEPTH / P_SPATIAL][P_SPATIAL],
    ifm_stream_t pixel_stream[P_SPATIAL],
    const ifm_desc_t &ifm_desc,
    const comp_desc_t &comp,
    uint8_t stride,
    cache_row_idx_t cig_row_base) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=k1_cache complete dim=2
#pragma HLS ARRAY_PARTITION variable=pixel_stream complete dim=1

  const int out_h       = comp.ofm_tile_h;
  const int out_w       = comp.ofm_tile_w;
  const int base        = ifm_desc.offset_vecs;
  const int rowstr      = ifm_desc.ddr_stride;
  const int rowlen      = ifm_desc.ddr_w;
  const int num_outputs = out_h * out_w;
  // HLS: A loop-invariant word count keeps the row/word nest perfect. Rounding
  // up also covers odd-width row-start straddles.
  const int half_w = (rowlen + 1) >> 1;

  // Each memory word supplies two row-contiguous packed vectors directly to
  // their cache banks and output lanes.
  // INVARIANT: The compiler admits strides of 1 and 2 only.
  const int sshift   = stride >> 1;
  const int smask    = stride - 1;
  const int row_step = rowstr << sshift;

  // HLS: Running row offsets replace per-element multiplication, and the
  // stride test and division are a mask and a shift.
  int row_elem0 = base;  // First input element of this output row.
  int out_row   = 0;     // First output index of this row.
LOAD_FORWARD_OY:
  for (int oy = 0; oy < out_h; oy++) {
#pragma HLS loop_tripcount min=TC_T_OH max=TC_T_OH
  LOAD_FORWARD_W:
    for (int wl = 0; wl < half_w; wl++) {
#pragma HLS loop_tripcount min=TC_ROWLEN max=TC_ROWLEN
#pragma HLS PIPELINE II=1
      // HLS: The two decoded vectors target distinct cache cells.
#pragma HLS dependence variable=k1_cache type=inter false
#pragma HLS dependence variable=k1_cache type=intra false
      const int         w         = (row_elem0 >> 1) + wl;
      const ifm_word_dt wrd       = ifm[w];

      // Decode both packed vectors; boundary and stride checks may suppress either.
      bool v0 = false, v1 = false;
      int  l0 = 0, l1 = 0;
      pixel_vector_dt p0 = 0, p1 = 0;
      {
        const int ix = (w << 1) + 0 - row_elem0;
        if (ix >= 0 && ix < rowlen && (ix & smask) == 0) {
          v0 = true; p0 = wrd.v[0];
          const int o = out_row + (ix >> sshift); l0 = o & P_SPATIAL_MASK;
          k1_cache[static_cast<int>(cig_row_base) + (o >> P_SPATIAL_LOG2)][l0] = p0;
        }
      }
      {
        const int ix = (w << 1) + 1 - row_elem0;
        if (ix >= 0 && ix < rowlen && (ix & smask) == 0) {
          v1 = true; p1 = wrd.v[1];
          const int o = out_row + (ix >> sshift); l1 = o & P_SPATIAL_MASK;
          k1_cache[static_cast<int>(cig_row_base) + (o >> P_SPATIAL_LOG2)][l1] = p1;
        }
      }

      // HLS: Static lane indices permit at most one write per lane each cycle.
    LANE_WR:
      for (int p = 0; p < P_SPATIAL; p++) {
#pragma HLS UNROLL
        if (v0 && l0 == p)      pixel_stream[p].write(p0);
        else if (v1 && l1 == p) pixel_stream[p].write(p1);
      }
    }
    row_elem0 += row_step;
    out_row   += out_w;
  }

  // INVARIANT: Pad the final group so every lane stream has equal length.
  const int rem = num_outputs & P_SPATIAL_MASK;
  if (rem != 0) {
  WRITE_PAD:
    for (int p = 0; p < P_SPATIAL; p++) {
#pragma HLS UNROLL
      if (p >= rem) pixel_stream[p].write(pixel_vector_dt(0));
    }
  }
}

// Streams one cached CIG in flat spatial-group order.
static void stream_from_cache(
    pixel_vector_dt k1_cache[IFM_CACHE_DEPTH / P_SPATIAL][P_SPATIAL],
    ifm_stream_t pixel_stream[P_SPATIAL],
    const comp_desc_t &comp,
    cache_row_idx_t cig_row_base) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=k1_cache complete dim=2
#pragma HLS ARRAY_PARTITION variable=pixel_stream complete dim=1

  const int num_outputs = comp.ofm_tile_h * comp.ofm_tile_w;
  const int num_groups  = (num_outputs + P_SPATIAL - 1) / P_SPATIAL;

CACHE_REUSE_STREAM_LOOP:
  for (int g = 0; g < num_groups; g++) {
#pragma HLS loop_tripcount min=TC_NUM_GROUPS max=TC_NUM_GROUPS
#pragma HLS PIPELINE II=1
    // HLS: Use a native index to avoid widening logic on the cache address.
    const int cache_row = static_cast<int>(cig_row_base) + g;
    for (int p = 0; p < P_SPATIAL; p++) {
#pragma HLS UNROLL
      const int idx = g * P_SPATIAL + p;
      const pixel_vector_dt pix = (idx < num_outputs) ? k1_cache[cache_row][p]
                                                      : pixel_vector_dt(0);
      pixel_stream[p].write(pix);
    }
  }
}

void load_and_stream_ifm(const ifm_word_dt *__restrict ifm,
                         desc_stream_ifm_t &desc_stream_ifm,
                         desc_stream_pad_t &desc_stream_pad,
                         desc_stream_comp_t &desc_comp,
                         ifm_stream_t pixel_stream[P_SPATIAL],
                         int num_tiles,
                         uint8_t kernel_size,
                         uint8_t stride,
                         ap_uint<1> use_ifm_cache,
                         ap_uint<14> spatial_tile_size,
                         int ifm_cig_stride_vecs) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=pixel_stream complete dim=1

  // Pointwise caching uses store A as the resident cache. Other layers use both
  // stores as raw-tile ping-pong buffers.
  pixel_vector_dt ifm_store_a[IFM_STORE_ROWS][P_SPATIAL];
  pixel_vector_dt ifm_store_b[IFM_STORE_ROWS][P_SPATIAL];
#pragma HLS ARRAY_PARTITION variable=ifm_store_a complete dim=2
#pragma HLS ARRAY_PARTITION variable=ifm_store_b complete dim=2
#pragma HLS BIND_STORAGE   variable=ifm_store_a type=RAM_T2P impl=uram
#pragma HLS BIND_STORAGE   variable=ifm_store_b type=RAM_T2P impl=uram

#pragma HLS allocation function instances=ifm_load_raw     limit=2
#pragma HLS allocation function instances=ifm_stream_pixels limit=2
  // HLS: Inline cache helpers to avoid another memory-interface process.

  if (kernel_size == 1 && use_ifm_cache) {
    // Pointwise descriptors cover a CIG range within one COG; each CIG
    // occupies a disjoint cache-row range.
    const cache_row_idx_t cig_row_stride =
        (spatial_tile_size + ap_uint<14>(P_SPATIAL_MASK)) >> P_SPATIAL_LOG2;

  K1_CACHE_LOOP:
    for (int t = 0; t < num_tiles; t++) {
#pragma HLS loop_tripcount min=TC_NUM_TILES max=TC_NUM_TILES
      ifm_desc_t  ifm_d  = desc_stream_ifm.read();
      pad_desc_t  pad_d  = desc_stream_pad.read();  // Drained; unused for K=1.
      comp_desc_t comp_d = desc_comp.read();

      // INVARIANT: The starting group fits seven bits; the full-width row
      // address advances across the chain, which may extend beyond group 127.
      cache_row_idx_t row_base =
          static_cast<cache_row_idx_t>(desc_cache_cig_idx(comp_d)) *
          cig_row_stride;
      uint32_t        ddr_cig_off = 0;
    K1_CIG_LOOP:
      for (int c = 0; c < static_cast<int>(comp_d.num_cigs); c++) {
#pragma HLS loop_tripcount min=1 max=TC_CIG
        if (comp_d.reuse_ifm) {
          stream_from_cache(ifm_store_a, pixel_stream, comp_d, row_base);
        } else {
          ifm_desc_t ifm_cig    = ifm_d;
          ifm_cig.offset_vecs  += ddr_cig_off;
          load_cache_and_forward(ifm, ifm_store_a, pixel_stream, ifm_cig,
                                 comp_d, stride, row_base);
        }
        row_base    += cig_row_stride;
        ddr_cig_off += static_cast<uint32_t>(ifm_cig_stride_vecs);
      }
    }

  } else {
    // The raw-tile path also handles K=1 cache misses. Padding descriptors are
    // drained because boundary padding is reconstructed during streaming.
    ifm_desc_t  ifm_d0  = desc_stream_ifm.read();
    desc_stream_pad.read();
    comp_desc_t comp_d0 = desc_comp.read();
    ifm_load_raw(ifm, ifm_store_a, ifm_d0);

  IFM_PP_LOOP:
    for (int t = 0; t < num_tiles - 1; t++) {
#pragma HLS loop_tripcount min=TC_IFM_PP max=TC_IFM_PP
      ifm_desc_t  ifm_next  = desc_stream_ifm.read();
      desc_stream_pad.read();
      comp_desc_t comp_next = desc_comp.read();

      // HLS: Loading and streaming use separate stores to avoid port contention.
      if ((t & 1) == 0) {
        ifm_load_raw(ifm, ifm_store_b, ifm_next);
        ifm_stream_pixels(ifm_store_a, pixel_stream, comp_d0, kernel_size, stride);
      } else {
        ifm_load_raw(ifm, ifm_store_a, ifm_next);
        ifm_stream_pixels(ifm_store_b, pixel_stream, comp_d0, kernel_size, stride);
      }
      comp_d0 = comp_next;
    }

    // Drain the final buffered tile.
    if (((num_tiles - 1) & 1) == 0) {
      ifm_stream_pixels(ifm_store_a, pixel_stream, comp_d0, kernel_size, stride);
    } else {
      ifm_stream_pixels(ifm_store_b, pixel_stream, comp_d0, kernel_size, stride);
    }
  }
}
