/**
 * @file tile_conv_tb.cpp
 * @brief Graph-runtime HLS C-simulation testbench for DeepTile.
 *
 * Flow:
 *   1. Load tilecnn.graph.v1 JSON.
 *   2. Compile it into a GraphExecutionPlan.
 *   3. Prepare RuntimeExecutionPlan buffers.
 *   4. Run each launch through the HLS adapter and tile_conv().
 *   5. Compare final graph outputs against the exported reference.
 *
 * The exporter-provided tensors under `data/<package>/refs/` are the sole
 * oracle here, matching the board host. The DeepTile software reference shares
 * its requantization idiom with the kernel, so it cannot independently confirm
 * kernel output; `tests/integration/reference_equivalence_tests.cpp` is the
 * native gate that binds that reference to the exported one.
 */

#include "deeptile/compiler/environment.h"
#include "backends/hls_adapter.h"
#include "runtime/runtime.h"

#include <cstdlib>
#include <cerrno>
#include <climits>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

const deeptile::runtime::RuntimeTensorBuffer *
find_buffer(const deeptile::runtime::RuntimeExecutionPlan &plan,
            const std::string &id) {
  const auto it = plan.buffer_index_by_id.find(id);
  if (it == plan.buffer_index_by_id.end()) {
    return nullptr;
  }
  return &plan.buffers[it->second];
}

std::filesystem::path find_repo_root() {
  // Vitis HLS can launch C-sim from generated project directories, so locate
  // the repository by walking upward instead of assuming the current path.
  std::filesystem::path dir = std::filesystem::current_path();
  while (!dir.empty()) {
    if (std::filesystem::exists(dir / "data" / "phase1_testcases") &&
        std::filesystem::exists(dir / "src" / "compiler")) {
      return dir;
    }
    const std::filesystem::path parent = dir.parent_path();
    if (parent == dir) {
      break;
    }
    dir = parent;
  }
  return std::filesystem::current_path();
}

std::filesystem::path default_graph_path() {
  return find_repo_root() / "data" / "phase1_testcases" / "best_case_3x3_s1" /
         "graph.json";
}

void print_diagnostics(
    const std::vector<deeptile::runtime::RuntimeDiagnostic> &diagnostics) {
  for (const auto &d : diagnostics) {
    std::cerr << "[" << deeptile::runtime::severity_name(d.severity) << "] ";
    if (!d.context.empty()) {
      std::cerr << d.context << ": ";
    }
    std::cerr << d.message << "\n";
  }
}

bool compare_chw(const std::vector<std::int8_t> &observed,
                 const std::vector<std::int8_t> &expected,
                 const std::vector<int> &shape,
                 const std::string &tensor_id,
                 int max_reported = 20) {
  if (observed.size() != expected.size()) {
    std::cerr << "[COMPARE][EXPORTED_REF][ERROR] " << tensor_id
              << " size mismatch observed=" << observed.size()
              << " expected=" << expected.size() << "\n";
    return false;
  }
  if (shape.size() != 3) {
    std::cerr << "[COMPARE][EXPORTED_REF][ERROR] " << tensor_id
              << " output shape is not CHW\n";
    return false;
  }

  const int channels = shape[0];
  const int height = shape[1];
  const int width = shape[2];
  int mismatches = 0;
  int reported = 0;
  for (std::size_t i = 0; i < observed.size(); ++i) {
    if (observed[i] == expected[i]) {
      continue;
    }
    ++mismatches;
    if (reported < max_reported) {
      const std::size_t spatial_size =
          static_cast<std::size_t>(height) * width;
      const int c = static_cast<int>(i / spatial_size);
      const std::size_t spatial = i % spatial_size;
      const int h = static_cast<int>(spatial / width);
      const int w = static_cast<int>(spatial % width);
      std::cerr << "[MISMATCH][EXPORTED_REF] " << tensor_id << " idx=" << i
                << " (c=" << c << ", h=" << h << ", w=" << w
                << ") observed=" << static_cast<int>(observed[i])
                << " expected=" << static_cast<int>(expected[i]) << "\n";
      ++reported;
    }
  }

  if (mismatches != 0) {
    std::cerr << "[COMPARE][EXPORTED_REF][FAIL] " << tensor_id
              << " mismatches=" << mismatches << " / " << observed.size()
              << "\n";
    return false;
  }

  std::cout << "[COMPARE][EXPORTED_REF][PASS] " << tensor_id
            << " exact match, elements=" << observed.size() << " shape="
            << channels << "x" << height << "x" << width << "\n";
  return true;
}

bool compare_exported_reference_outputs(
    const deeptile::runtime::RuntimeExecutionPlan &plan) {
  // The exported reference is the sole oracle and every mismatch is fatal.
  // A missing reference is a package defect, not a tolerated case.
  bool ok = true;
  const auto &graph = plan.graph_plan.graph;
  for (const auto &output_id : graph.outputs) {
    const auto ref_it = graph.references.find(output_id);
    if (ref_it == graph.references.end()) {
      std::cerr << "[COMPARE][EXPORTED_REF][ERROR] no exported reference for "
                << output_id << "\n";
      ok = false;
      continue;
    }

    const auto *observed_buffer = find_buffer(plan, output_id);
    const auto *exported_ref = find_buffer(plan, ref_it->second);
    if (!observed_buffer || !exported_ref) {
      std::cerr << "[COMPARE][EXPORTED_REF][ERROR] missing observed/reference "
                   "for " << output_id << "\n";
      ok = false;
      continue;
    }

    const auto observed =
        deeptile::runtime::unpack_packed_activation_to_chw(*observed_buffer);
    if (!compare_chw(observed, exported_ref->raw_data, exported_ref->shape,
                     output_id)) {
      ok = false;
    }
  }
  return ok;
}

int run_graph(const std::string &graph_path, int repeats) {
  std::cout << "[INFO] graph: " << graph_path << "\n";

  deeptile::runtime::RuntimeExecutionPlan plan;
  std::vector<deeptile::runtime::RuntimeDiagnostic> diagnostics;
  // Shared front-end path: load graph.json, lower to launches, and prepare the
  // same packed runtime buffers used by the XRT host.
  if (!deeptile::runtime::load_compile_prepare_runtime(graph_path, plan,
                                                      diagnostics)) {
    print_diagnostics(diagnostics);
    return EXIT_FAILURE;
  }

  // Print the top-level port depths needed when moving from C-sim to cosim.
  // Off by default (development/tuning detail); set DEEPTILE_PRINT_HLS_DEPTHS=1
  // to see it.
  const char *print_depths = deeptile::env_value("PRINT_HLS_DEPTHS");
  if (print_depths && std::string(print_depths) == "1") {
    deeptile::runtime::print_hls_port_depths(plan, std::cout);
  }

  std::cout << "[INFO] launches: " << plan.launches.size()
            << " repeats: " << repeats << "\n";
  // Two or more invocations expose steady-state DATAFLOW behavior in RTL.
  // Compare every repetition so persistent kernel state cannot hide a failure.
  for (int repeat = 0; repeat < repeats; ++repeat) {
    for (std::size_t i = 0; i < plan.launches.size(); ++i) {
      if (!deeptile::runtime::run_hls_launch(plan, i, std::cout)) {
        std::cerr << "[ERROR] HLS launch failed at index " << i << "\n";
        return EXIT_FAILURE;
      }
    }
    if (!compare_exported_reference_outputs(plan)) {
      std::cerr << "[FAIL] exported reference graph output comparison failed\n";
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
  std::cout << "=================================================\n";
  std::cout << "  DeepTile Graph Runtime HLS Testbench\n";
  std::cout << "=================================================\n\n";

  const std::string graph_path =
      argc >= 2 ? std::string(argv[1]) : default_graph_path().string();
  int repeats = 1;
  if (const char *value = deeptile::env_value("HLS_REPEATS")) {
    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < 1 ||
        parsed > INT_MAX) {
      std::cerr << "[ERROR] DEEPTILE_HLS_REPEATS must be a positive integer\n";
      return EXIT_FAILURE;
    }
    repeats = static_cast<int>(parsed);
  }

  // Exercise prime/drain transitions and FIFO boundaries in one RTL run.
  // Synthesis port depths must cover the largest schedule in this sweep.
  if (const char *sweep = deeptile::env_value("HLS_EPOCH_SWEEP")) {
    std::vector<int> epochs;
    const char *value = sweep;
    while (true) {
      char *end = nullptr;
      errno = 0;
      const long parsed = std::strtol(value, &end, 10);
      if (errno != 0 || end == value || parsed < 1 || parsed > INT_MAX ||
          (*end != ',' && *end != '\0')) {
        std::cerr << "[ERROR] DEEPTILE_HLS_EPOCH_SWEEP must list positive integers\n";
        return EXIT_FAILURE;
      }
      epochs.push_back(static_cast<int>(parsed));
      if (*end == '\0') break;
      value = end + 1;
    }
    for (int count : epochs) {
      const std::string setting = std::to_string(count);
      if (setenv("DEEPTILE_FORCE_EPOCHS", setting.c_str(), 1) != 0) {
        std::cerr << "[ERROR] Cannot set forced epoch count\n";
        return EXIT_FAILURE;
      }
      std::cout << "[INFO] requested epochs: " << count << "\n";
      if (run_graph(graph_path, repeats) != EXIT_SUCCESS) return EXIT_FAILURE;
    }
  } else if (run_graph(graph_path, repeats) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  std::cout << "\n=================================================\n";
  std::cout << "  [PASS] DeepTile graph runtime HLS testbench\n";
  std::cout << "=================================================\n";
  return EXIT_SUCCESS;
}
