/**
 * @file tile_conv.cpp
 * @brief Connects the DeepTile kernel stages into one HLS dataflow region.
 */

#include <cassert>
#include <cstdio>
#include <hls_stream.h>

#include "output_postproc.h"
#include "conv_engine.h"
#include "data_orch.h"
#include "dw_stage.h"
#include "tile_conv.h"

#ifdef __cplusplus
extern "C" {
#endif

void tile_conv(const ifm_word_dt *__restrict ifm,
               const weight_vector_dt *__restrict weights,
               const biases_vector_dt *__restrict biases,
               activations_vector_dt *__restrict out,
               const activations_vector_dt *__restrict residual,
               const uram_xfer_desc_t *__restrict xfer_desc,
               int num_xfers,
               const descriptor_dt *__restrict comp_desc,
               int num_tiles,
               int num_bias_vecs, int shift_out,
               int gap_mul,
               int kernel_size,
               int stride,
               int use_ifm_cache,
               int spatial_tile_size,
               int num_ic_tiles,
               int ifm_cig_stride_vecs,
               int act_max,
               const weight_vector_dt *__restrict dw_weights,
               int dw_config,
               int dw_shift,
               int num_writeback_tiles) {
// AXI memory interfaces
#pragma HLS INTERFACE m_axi port=ifm       offset=slave bundle=gmem0 depth=AXI_DEPTH_IFM_WORDS
#pragma HLS INTERFACE m_axi port=weights   offset=slave bundle=gmem1 depth=AXI_DEPTH_WEIGHTS
#pragma HLS INTERFACE m_axi port=biases    offset=slave bundle=gmem2 depth=AXI_DEPTH_BIASES
#pragma HLS INTERFACE m_axi port=out       offset=slave bundle=gmem3 depth=AXI_DEPTH_OUT
#pragma HLS INTERFACE m_axi port=residual  offset=slave bundle=gmem5 depth=AXI_DEPTH_RESIDUAL
#pragma HLS INTERFACE m_axi port=xfer_desc offset=slave bundle=gmem4 depth=AXI_DEPTH_XDESC
#pragma HLS INTERFACE m_axi port=comp_desc offset=slave bundle=gmem4 depth=AXI_DEPTH_CDESC
#pragma HLS INTERFACE m_axi port=dw_weights offset=slave bundle=gmem6 depth=AXI_DEPTH_DW

// XRT control interface
#pragma HLS INTERFACE s_axilite port=num_xfers
#pragma HLS INTERFACE s_axilite port=num_tiles
#pragma HLS INTERFACE s_axilite port=num_bias_vecs
#pragma HLS INTERFACE s_axilite port=shift_out
#pragma HLS INTERFACE s_axilite port=gap_mul
#pragma HLS INTERFACE s_axilite port=kernel_size
#pragma HLS INTERFACE s_axilite port=stride
#pragma HLS INTERFACE s_axilite port=use_ifm_cache
#pragma HLS INTERFACE s_axilite port=spatial_tile_size
#pragma HLS INTERFACE s_axilite port=num_ic_tiles
#pragma HLS INTERFACE s_axilite port=ifm_cig_stride_vecs
#pragma HLS INTERFACE s_axilite port=act_max
#pragma HLS INTERFACE s_axilite port=dw_config
#pragma HLS INTERFACE s_axilite port=dw_shift
#pragma HLS INTERFACE s_axilite port=num_writeback_tiles

  biases_vector_dt bias_cache[BIAS_BUF_DEPTH];
#pragma HLS BIND_STORAGE variable=bias_cache type=RAM_1P impl=bram

  const uint8_t layer_kernel_size = static_cast<uint8_t>(kernel_size);
  const uint8_t layer_stride = static_cast<uint8_t>(stride);
  const bool layer_dw_enable = (dw_config & 0x1) != 0;
  const int  layer_num_cogs = num_bias_vecs;
  const ap_uint<1>  layer_use_ifm_cache = use_ifm_cache ? 1 : 0;
  const ap_uint<14> layer_spatial_tile_size =
      static_cast<ap_uint<14>>(spatial_tile_size);
  const int layer_ifm_cig_stride = ifm_cig_stride_vecs;

#pragma HLS DATAFLOW

  // One input FIFO per spatial lane.
  ifm_stream_t pixel_stream[P_SPATIAL];
#pragma HLS ARRAY_PARTITION variable=pixel_stream complete dim=1
#pragma HLS STREAM variable=pixel_stream depth=32
#pragma HLS BIND_STORAGE variable=pixel_stream type=FIFO impl=SRL

  static acc_group_stream_t acc_stream("acc_stream");
#pragma HLS STREAM variable=acc_stream depth=32
#pragma HLS BIND_STORAGE variable=acc_stream type=FIFO impl=SRL


  static ofm_stream_t ofm_stream("ofm_stream");
#pragma HLS STREAM variable=ofm_stream depth=32
#pragma HLS BIND_STORAGE variable=ofm_stream type=FIFO impl=SRL

  static ofm_stream_t ofm_post_stream("ofm_post_stream");
#pragma HLS STREAM variable=ofm_post_stream depth=32
#pragma HLS BIND_STORAGE variable=ofm_post_stream type=FIFO impl=SRL

  static ofm_stream_t ofm_pool_stream("ofm_pool_stream");
#pragma HLS STREAM variable=ofm_pool_stream depth=32
#pragma HLS BIND_STORAGE variable=ofm_pool_stream type=FIFO impl=SRL

  static ofm_stream_t ofm_predw_stream("ofm_predw_stream");
#pragma HLS STREAM variable=ofm_predw_stream depth=32
#pragma HLS BIND_STORAGE variable=ofm_predw_stream type=FIFO impl=SRL

  // HLS: A deep FIFO decouples the strided OFM store from the
  // next tile's MAC+DW compute.
  static ofm_stream_t ofm_gap_stream("ofm_gap_stream");
#pragma HLS STREAM variable=ofm_gap_stream depth=512
#pragma HLS BIND_STORAGE variable=ofm_gap_stream type=FIFO impl=BRAM

  static residual_stream_t residual_stream("residual_stream");
#pragma HLS STREAM variable=residual_stream depth=64
#pragma HLS BIND_STORAGE variable=residual_stream type=FIFO impl=SRL

  static desc_stream_ifm_t desc_stream_ifm("desc_stream_ifm");
#pragma HLS STREAM variable=desc_stream_ifm depth=DEEPTILE_DESC_STREAM_DEPTH

  static desc_stream_xfer_t desc_stream_xfer("desc_stream_xfer");
#pragma HLS STREAM variable=desc_stream_xfer depth=32
#pragma HLS BIND_STORAGE variable=desc_stream_xfer type=FIFO impl=SRL

  static desc_stream_pad_t desc_stream_pad("desc_stream_pad");
#pragma HLS STREAM variable=desc_stream_pad depth=DEEPTILE_DESC_STREAM_DEPTH

  static desc_stream_ofm_t desc_stream_ofm("desc_stream_ofm");
#pragma HLS STREAM variable=desc_stream_ofm depth=32
#pragma HLS BIND_STORAGE variable=desc_stream_ofm type=FIFO impl=SRL

  static desc_stream_ofm_t desc_stream_ofm_add("desc_stream_ofm_add");
#pragma HLS STREAM variable=desc_stream_ofm_add depth=32
#pragma HLS BIND_STORAGE variable=desc_stream_ofm_add type=FIFO impl=SRL

  static desc_stream_ofm_t desc_stream_ofm_post("desc_stream_ofm_post");
#pragma HLS STREAM variable=desc_stream_ofm_post depth=32
#pragma HLS BIND_STORAGE variable=desc_stream_ofm_post type=FIFO impl=SRL

  static desc_stream_ofm_t desc_stream_ofm_pool("desc_stream_ofm_pool");
#pragma HLS STREAM variable=desc_stream_ofm_pool depth=32
#pragma HLS BIND_STORAGE variable=desc_stream_ofm_pool type=FIFO impl=SRL

  static desc_stream_ofm_t desc_stream_ofm_gap("desc_stream_ofm_gap");
#pragma HLS STREAM variable=desc_stream_ofm_gap depth=32
#pragma HLS BIND_STORAGE variable=desc_stream_ofm_gap type=FIFO impl=SRL

  static desc_stream_ofm_t desc_stream_ofm_dw("desc_stream_ofm_dw");
#pragma HLS STREAM variable=desc_stream_ofm_dw depth=32
#pragma HLS BIND_STORAGE variable=desc_stream_ofm_dw type=FIFO impl=SRL

  // The controller broadcasts each compute descriptor to its three consumers.
  static desc_stream_comp_t desc_comp_ifm("desc_comp_ifm");
#pragma HLS STREAM variable=desc_comp_ifm depth=32

  static desc_stream_comp_t desc_comp_mac("desc_comp_mac");
#pragma HLS STREAM variable=desc_comp_mac depth=32

  static desc_stream_comp_t desc_comp_acc("desc_comp_acc");
#pragma HLS STREAM variable=desc_comp_acc depth=32

  schedule_controller(xfer_desc, desc_stream_xfer, num_xfers,
                      comp_desc, desc_stream_ifm,
                      desc_stream_pad,
                      desc_comp_ifm, desc_comp_mac, desc_comp_acc,
                      desc_stream_ofm, num_tiles, layer_kernel_size,
                      layer_use_ifm_cache, layer_dw_enable);

  preload_bias_to_bram(biases, bias_cache, num_bias_vecs);

  load_and_stream_ifm(ifm, desc_stream_ifm, desc_stream_pad,
                      desc_comp_ifm, pixel_stream, num_tiles,
                      layer_kernel_size, layer_stride,
                      layer_use_ifm_cache, layer_spatial_tile_size,
                      layer_ifm_cig_stride);

  conv_engine(weights, desc_stream_xfer, pixel_stream, acc_stream,
              desc_comp_mac, num_xfers, layer_kernel_size);

#ifdef DEEPTILE_NO_PSUM_PP
  // HLS: The single-buffer variant cannot overlap accumulation and emission,
  // so it consumes every compute descriptor.
  acc_quantize(acc_stream, bias_cache, ofm_stream, desc_comp_acc,
               num_tiles, shift_out, act_max);
#else
  acc_quantize_pp(acc_stream, bias_cache, ofm_stream, desc_comp_acc,
                  num_writeback_tiles, shift_out, act_max);
#endif

  residual_loader(residual, desc_stream_ofm, desc_stream_ofm_add,
                  residual_stream, num_tiles);

  residual_add(desc_stream_ofm_add, ofm_stream, residual_stream,
               desc_stream_ofm_post, ofm_post_stream, num_tiles, act_max);

  maxpool(desc_stream_ofm_post, ofm_post_stream,
          desc_stream_ofm_pool, ofm_pool_stream, num_tiles);

  gap(desc_stream_ofm_pool, ofm_pool_stream,
      desc_stream_ofm_gap, ofm_predw_stream, num_tiles, gap_mul);

  dw_stage(desc_stream_ofm_gap, ofm_predw_stream, dw_weights,
           desc_stream_ofm_dw, ofm_gap_stream, num_tiles, layer_num_cogs,
           dw_config, dw_shift);

  store_ofm_tile(out, desc_stream_ofm_dw, ofm_gap_stream, num_tiles);
}

#ifdef __cplusplus
}
#endif
