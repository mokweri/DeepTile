/**
 * @file preprocess_anchor_test.cpp
 * @brief Verifies host preprocessing reproduces a model package's input tensor.
 *
 * The exporter's preprocessing is a convention shared by two codebases with
 * nothing enforcing it. This test pins it: decoding the package's source image
 * and preprocessing it here must reproduce the checked-in
 * `inputs/input_0.int8.bin` exactly, using the geometry and fractional bits the
 * package declares.
 *
 * The source image may be a JPEG, which needs a decoder, or a binary PPM, which
 * does not. Supporting both keeps the contract checkable in builds without an
 * image-codec backend.
 */

#include "dataset/image_decode.h"
#include "dataset/preprocess.h"
#include "deeptile/compiler/graph_loader.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using deeptile::dataset::Image8;

std::filesystem::path repo_root() {
#ifdef DEEPTILE_REPO_ROOT
  return DEEPTILE_REPO_ROOT;
#else
  return std::filesystem::current_path();
#endif
}

bool read_ppm(const std::filesystem::path &path, Image8 &out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  std::string magic;
  int width = 0;
  int height = 0;
  int maxval = 0;
  in >> magic >> width >> height >> maxval;
  if (magic != "P6" || maxval != 255 || width <= 0 || height <= 0) {
    return false;
  }
  in.get();
  out.width = width;
  out.height = height;
  out.channels = 3;
  out.pixels.resize(out.expected_size());
  in.read(reinterpret_cast<char *>(out.pixels.data()),
          static_cast<std::streamsize>(out.pixels.size()));
  return in.good() || in.eof();
}

bool load_source_image(const std::filesystem::path &path, Image8 &out) {
  if (path.extension() == ".ppm") {
    return read_ppm(path, out);
  }
  std::string error;
  if (!deeptile::dataset::decode_image_rgb(path.string(), out, error)) {
    std::cerr << "[ERROR] " << error << "\n";
    return false;
  }
  return true;
}

bool read_int8(const std::filesystem::path &path,
               std::vector<std::int8_t> &out) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    return false;
  }
  out.resize(static_cast<std::size_t>(in.tellg()));
  in.seekg(0);
  in.read(reinterpret_cast<char *>(out.data()),
          static_cast<std::streamsize>(out.size()));
  return true;
}

} // namespace

int main(int argc, char **argv) {
  // Defaults describe the current anchor. Once model packages carry their own
  // source image, both arguments come from the package instead.
  const std::filesystem::path image_path =
      argc >= 2 ? std::filesystem::path(argv[1]) : repo_root() / "data" / "new.JPEG";
  const std::filesystem::path package =
      argc >= 3 ? std::filesystem::path(argv[2])
                : repo_root() / "data" / "resnet50_int8_tilecnn";

  std::cout << "[INFO] image:   " << image_path << "\n";
  std::cout << "[INFO] package: " << package << "\n";

  deeptile::graph::GraphSpec graph;
  std::vector<deeptile::graph::GraphDiagnostic> diagnostics;
  if (!deeptile::graph::load_graph((package / "graph.json").string(), graph,
                                  diagnostics)) {
    std::cerr << "[ERROR] cannot load package graph\n";
    return 1;
  }
  if (graph.inputs.empty()) {
    std::cerr << "[ERROR] package declares no graph input\n";
    return 1;
  }

  const auto tensor_it = graph.tensors.find(graph.inputs.front());
  if (tensor_it == graph.tensors.end() || !tensor_it->second.file.has_value()) {
    std::cerr << "[ERROR] input tensor has no exported payload\n";
    return 1;
  }
  const auto &tensor = tensor_it->second;
  if (tensor.shape.size() != 3 || tensor.shape[1] != tensor.shape[2]) {
    std::cerr << "[ERROR] expected a square CHW input tensor\n";
    return 1;
  }

  Image8 source;
  if (!load_source_image(image_path, source)) {
    std::cerr << "[ERROR] cannot load source image\n";
    return 1;
  }
  std::cout << "[INFO] decoded " << source.width << "x" << source.height
            << " RGB\n";

  // The checked-in fixture was produced by the exporter, which squashes rather
  // than short-side resizing.
  deeptile::dataset::PreprocessConfig config;
  config.convention = deeptile::dataset::ResizeConvention::ExportSquash;
  config.target_side = tensor.shape[1];
  config.frac = tensor.frac;
  std::cout << "[INFO] convention="
            << deeptile::dataset::resize_convention_name(config.convention)
            << " side=" << config.target_side << " frac=" << config.frac
            << "\n";

  const std::vector<std::int8_t> got =
      deeptile::dataset::preprocess_rgb(source, config);
  if (got.empty()) {
    std::cerr << "[ERROR] preprocessing produced no output\n";
    return 1;
  }

  std::vector<std::int8_t> expected;
  if (!read_int8(package / *tensor.file, expected)) {
    std::cerr << "[ERROR] cannot read " << *tensor.file << "\n";
    return 1;
  }
  if (got.size() != expected.size()) {
    std::cerr << "[FAIL] size " << got.size() << " != " << expected.size()
              << "\n";
    return 1;
  }

  std::size_t mismatches = 0;
  int reported = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (got[i] == expected[i]) {
      continue;
    }
    ++mismatches;
    if (reported < 8) {
      std::cerr << "  diff idx=" << i << " got " << static_cast<int>(got[i])
                << " want " << static_cast<int>(expected[i]) << "\n";
      ++reported;
    }
  }
  if (mismatches != 0) {
    std::cerr << "[FAIL] " << mismatches << " / " << got.size()
              << " elements differ from the exported input tensor\n";
    return 1;
  }

  std::cout << "[PASS] preprocessing reproduces " << *tensor.file
            << " exactly, elements=" << got.size() << "\n";
  return 0;
}
