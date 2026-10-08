/**
 * @file layer.hpp
 * @brief Layer geometry, tiling, quantization, and post-operation types.
 * @ingroup deeptile_compiler
 */

#pragma once

namespace deeptile {
namespace tile {

/**
 * @brief Host-side convolution geometry and fixed-point metadata.
 */
struct ConvLayerParams {
  // Tensor dimensions.
  int H;    ///< Input feature map height
  int W;    ///< Input feature map width
  int Cin;  ///< Input channels
  int Cout; ///< Output channels (filters)

  // Kernel and stride.
  int Kh; ///< Kernel height
  int Kw; ///< Kernel width
  int Sh; ///< Stride height
  int Sw; ///< Stride width

  // Static layer padding.
  int pad_top;
  int pad_bottom;
  int pad_left;
  int pad_right;

  // Memory layout.
  int Cpi;        ///< Input channel packing
  int Cpo;        ///< Output channel packing (e.g., 16)
  int elem_bytes; ///< Element size in bytes (1 for int8)

  // Fixed-point metadata used to derive shifts.
  int frac_in;  ///< Input fractional bits
  int frac_w;   ///< Weight fractional bits
  int frac_b;   ///< Bias fractional bits
  int frac_out; ///< Output fractional bits (target)

  // Activation.
  bool use_relu; ///< Apply ReLU activation

  // Convolution mode.
  /// Use channel-wise convolution with compact [C][1][Kh][Kw] weights.
  bool depthwise = false;

  /// Upper clamp applied with the activation: 127 for ReLU, lower for ReLU6.
  int act_max = 127;

  // Derived dimensions.
  int output_height() const { return (H + pad_top + pad_bottom - Kh) / Sh + 1; }
  int output_width() const { return (W + pad_left + pad_right - Kw) / Sw + 1; }
  int num_cig() const { return (Cin + Cpi - 1) / Cpi; }
  int num_cog() const { return (Cout + Cpo - 1) / Cpo; }
};

/**
 * @brief Shape used to partition a convolution into execution tiles.
 */
struct TileConfig {
  int T_oh;  ///< Output tile height (spatial)
  int T_ow;  ///< Output tile width (spatial)
  int T_cog; ///< Output channel groups per tile
  int T_cig; ///< Input channel groups per tile
  int Cpi;
  int Cpo;

  bool is_valid(const ConvLayerParams &p) const {
    if (Cpi != p.Cpi || Cpo != p.Cpo)
      return false;
    if (T_oh <= 0 || T_ow <= 0)
      return false;
    if (T_cog <= 0 || T_cig <= 0)
      return false;
    return true;
  }
};

/**
 * @brief Optional post-processing applied uniformly to a layer.
 */
struct PostOpConfig {
  bool residual_add_enable = false;
  bool post_add_relu       = false;
  int  residual_shift      = 0;
  bool maxpool_enable      = false;
  int  post_pool_shift     = 0;
  bool gap_enable          = false;

  /// Append a streamed depthwise stage; mutually exclusive with pool and GAP.
  bool dw_enable           = false;
  int  dw_stride           = 1;   ///< Depthwise stride, either 1 or 2.
  int  dw_kernel           = 3;   ///< Depthwise kernel edge, 3 or 5, same padding.
};

} // namespace tile
} // namespace deeptile
