/**
 * @file reference_dump_main.cpp
 * @brief Write the software reference tensor for every launch in a graph.
 * @ingroup deeptile_host
 *
 * The HLS testbench and the board host compare only the graph's declared
 * outputs against the exported reference, so a kernel that diverges part-way
 * through a network reports one failure at the end and says nothing about
 * where it started. This tool writes the intermediate tensors the software
 * reference computes, so a suspect launch can be cut into a standalone package
 * and simulated on its own.
 *
 * The software reference shares the kernel's requantization idiom, so these
 * tensors are not an independent oracle. `reference_equivalence_tests` is what
 * binds the reference to the exported one; this tool only localizes a
 * divergence the exported comparison has already found.
 */

#include "reference/runtime_reference.h"
#include "runtime/runtime.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_diagnostics(
    const std::vector<deeptile::runtime::RuntimeDiagnostic> &diagnostics) {
  for (const auto &d : diagnostics) {
    std::cerr << "  [" << deeptile::runtime::severity_name(d.severity) << "] "
              << d.message << "\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << argv[0]
              << " <path/to/graph.json> [out_dir]\n"
              << "\nWrites <out_dir>/<tensor_id>.int8.bin for every launch\n"
              << "output the software reference produces. Default out_dir is\n"
              << "'reference_tensors' beside the graph.\n";
    return 2;
  }

  const std::filesystem::path graph_path(argv[1]);
  const std::filesystem::path out_dir =
      argc == 3 ? std::filesystem::path(argv[2])
                : graph_path.parent_path() / "reference_tensors";

  deeptile::runtime::RuntimeExecutionPlan plan;
  std::vector<deeptile::runtime::RuntimeDiagnostic> diagnostics;
  if (!deeptile::runtime::load_compile_prepare_runtime(graph_path.string(), plan,
                                                      diagnostics)) {
    std::cerr << "[ERROR] compile/prepare failed\n";
    print_diagnostics(diagnostics);
    return 1;
  }

  const deeptile::runtime::RuntimeReferenceResult reference =
      deeptile::runtime::generate_bit_exact_reference(plan);
  if (deeptile::runtime::has_errors(reference.diagnostics)) {
    std::cerr << "[ERROR] reference generation failed\n";
    print_diagnostics(reference.diagnostics);
    return 1;
  }

  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);
  if (ec) {
    std::cerr << "[ERROR] cannot create " << out_dir << ": " << ec.message()
              << "\n";
    return 1;
  }

  int written = 0;
  for (const auto &entry : reference.tensors) {
    const auto &tensor = entry.second;
    const std::filesystem::path file = out_dir / (tensor.id + ".int8.bin");
    std::ofstream out(file, std::ios::binary);
    if (!out) {
      std::cerr << "[ERROR] cannot write " << file << "\n";
      return 1;
    }
    out.write(reinterpret_cast<const char *>(tensor.data_chw.data()),
              static_cast<std::streamsize>(tensor.data_chw.size()));
    if (!out) {
      std::cerr << "[ERROR] short write to " << file << "\n";
      return 1;
    }
    std::cout << tensor.id << "  [";
    for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
      std::cout << (i ? "x" : "") << tensor.shape[i];
    }
    std::cout << "]  " << tensor.data_chw.size() << " bytes\n";
    ++written;
  }

  std::cout << "wrote " << written << " tensors to " << out_dir << "\n";
  return 0;
}
