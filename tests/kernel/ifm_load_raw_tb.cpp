/**
 * @file ifm_load_raw_tb.cpp
 * @brief Checks raw-loader alignment and exact allocation boundaries under ASan.
 */

// The private helper is exercised directly so allocation overruns are visible
// to a native sanitizer, independently of HLS's AXI simulation wrapper.
#include "../../src/kernel/data_orch.cpp"

#include <iostream>
#include <memory>

static pixel_vector_dt pixel(int index) {
  pixel_vector_dt result = 0;
  for (int lane = 0; lane < IC_TILE_SIZE; ++lane) {
    result.range(8 * lane + 7, 8 * lane) = (index * 17 + lane * 3) & 255;
  }
  return result;
}

int main() {
  static pixel_vector_dt store[IFM_STORE_ROWS][P_SPATIAL];
  int cases = 0;
  for (int width = 1; width <= PADDED_TILE_W; ++width) {
    for (int height : {1, 2, 3, PADDED_TILE_H}) {
      for (int offset : {0, 1}) {
        for (int gap : {0, 1}) {
          const int stride = width + gap;
          const int elements = offset + (height - 1) * stride + width;
          const int words = (elements + 1) / 2;
          // Exactly the allocation promised by the packed-tensor interface.
          auto input = std::make_unique<ifm_word_dt[]>(words);
          for (int i = 0; i < words; ++i) {
            input[i].v[0] = pixel(i * 2);
            input[i].v[1] = pixel(i * 2 + 1);
          }
          ifm_desc_t desc{};
          desc.offset_vecs = offset;
          desc.ddr_w = width;
          desc.ddr_h = height;
          desc.ddr_stride = stride;
          ifm_load_raw(input.get(), store, desc);
          const int pitch = (width + 1) / 2;
          for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
              for (int copy = 0; copy < IFM_BUF_COPIES; ++copy) {
                if (store[y * pitch + x / 2][copy * 2 + x % 2] !=
                    pixel(offset + y * stride + x)) {
                  std::cerr << "FAIL width=" << width << " height=" << height
                            << " offset=" << offset << " stride=" << stride
                            << " at " << y << "," << x << "\n";
                  return 1;
                }
              }
            }
          }
          ++cases;
        }
      }
    }
  }
  std::cout << "PASS: " << cases
            << " exact-allocation alignment cases, all replicated banks\n";
}
