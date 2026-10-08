// Copyright: Derived from Vitis Vision-style LineBuffer
/**
 * @file linebuffer.hpp
 * @brief Provides a lightweight HLS line buffer without `hls_video.h`.
 */

#ifndef LINEBUFFER_HPP
#define LINEBUFFER_HPP

#include <ap_int.h>
#include <assert.h>

typedef ap_uint<32> XF_SIZE_T;

#define _LB_TPLT_DEC                                                           \
  template <int ROWS, int COLS, typename T, int RESHAPE_FACTOR = 1>
#define _LB_TPLT template <int ROWS, int COLS, typename T, int RESHAPE_FACTOR>
#define _LB_ LineBuffer<ROWS, COLS, T, RESHAPE_FACTOR>

_LB_TPLT_DEC class LineBuffer {
public:
  LineBuffer(){
#pragma HLS inline
#pragma HLS dependence variable = val inter false
#pragma HLS dependence variable = val intra false

#pragma HLS bind_storage variable = val type = RAM_S2P impl = BRAM
#pragma HLS ARRAY_PARTITION variable = val complete dim = 1
  };

  void shift_pixels_up(int col);
  void shift_pixels_down(int col);
  void insert_bottom_row(T value, int col);
  void insert_top_row(T value, int col);
  void get_col(T value[ROWS], int col);
  T &getval(int row, int col);
  const T &getval(int row, int col) const;
  T &operator()(int row, int col);
  const T &operator()(int row, int col) const;

  T val[ROWS][COLS];
};

// Row zero is the top of the image-oriented buffer.
_LB_TPLT void _LB_::shift_pixels_down(int col) {
#pragma HLS inline
  assert(col >= 0 && col < COLS);

  XF_SIZE_T i;
  for (i = ROWS - 1; i > 0; i--) {
#pragma HLS unroll
    val[i][col] = val[i - 1][col];
  }
}

// Row `ROWS - 1` is the bottom of the image-oriented buffer.
_LB_TPLT void _LB_::shift_pixels_up(int col) {
#pragma HLS inline
  assert(col >= 0 && col < COLS);

  XF_SIZE_T i;
  for (i = 0; i < ROWS - 1; i++) {
#pragma HLS unroll

    val[i][col] = val[i + 1][col];
  }
}

_LB_TPLT void _LB_::insert_bottom_row(T value, int col) {
#pragma HLS inline
  assert(col >= 0 && col < COLS);

  val[ROWS - 1][col] = value;
}

_LB_TPLT void _LB_::insert_top_row(T value, int col) {
#pragma HLS inline
  assert(col >= 0 && col < COLS);

  val[0][col] = value;
}

_LB_TPLT void _LB_::get_col(T value[ROWS], int col) {
#pragma HLS inline
  assert(col >= 0 && col < COLS);
  XF_SIZE_T i;
  for (i = 0; i < ROWS; i++) {
#pragma HLS unroll
    value[i] = val[i][col];
  }
}

_LB_TPLT T &_LB_::getval(int row, int col) {
#pragma HLS inline
  assert(row >= 0 && row < ROWS && col >= 0 && col < COLS);
  return val[row][col];
}

_LB_TPLT T &_LB_::operator()(int row, int col) {
#pragma HLS inline
  return getval(row, col);
}

_LB_TPLT const T &_LB_::getval(int row, int col) const {
#pragma HLS inline
  assert(row >= 0 && row < ROWS && col >= 0 && col < COLS);
  return val[row][col];
}

_LB_TPLT const T &_LB_::operator()(int row, int col) const {
#pragma HLS inline
  return getval(row, col);
}

#endif // LINEBUFFER_HPP
