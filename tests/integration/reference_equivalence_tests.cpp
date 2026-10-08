/**
 * @file reference_equivalence_tests.cpp
 * @brief Native gate binding the DeepTile software reference to exported refs.
 *
 * The exporter-provided tensors under `data/<package>/refs/` are the single
 * oracle for DeepTile correctness. The software reference in
 * `src/host/reference/` shares its requantization idiom with the kernel by
 * construction, so it cannot detect a divergence it inherits from the kernel.
 * This test is the place where the two oracles are reconciled: every
 * checked-in model package must reproduce its exported reference exactly.
 *
 * Once this gate holds, the C-simulation testbench and the board host compare
 * observed hardware output against the exported reference alone.
 *
 * Packages containing a fusible pointwise->depthwise pair are checked with
 * DEEPTILE_FUSE_DW both off and on, so neither lowering may drift from the
 * exporter.
 */

#include "deeptile/compiler/graph_compiler.h"
#include "deeptile/compiler/graph_loader.h"
#include "runtime/runtime.h"
#include "reference/runtime_reference.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

using deeptile::runtime::RuntimeDiagnostic;
using deeptile::runtime::RuntimeExecutionPlan;

std::filesystem::path find_repo_root() {
#ifdef DEEPTILE_REPO_ROOT
  return DEEPTILE_REPO_ROOT;
#else
  return std::filesystem::current_path();
#endif
}

const deeptile::runtime::RuntimeTensorBuffer *
find_buffer(const RuntimeExecutionPlan &plan, const std::string &id) {
  const auto it = plan.buffer_index_by_id.find(id);
  if (it == plan.buffer_index_by_id.end()) {
    return nullptr;
  }
  return &plan.buffers[it->second];
}

/** Every checked-in model package, discovered rather than enumerated. */
std::vector<std::filesystem::path> discover_packages() {
  std::vector<std::filesystem::path> packages;
  const std::filesystem::path data_root = find_repo_root() / "data";
  if (!std::filesystem::exists(data_root)) {
    return packages;
  }
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(data_root)) {
    if (entry.is_regular_file() && entry.path().filename() == "graph.json") {
      packages.push_back(entry.path());
    }
  }
  std::sort(packages.begin(), packages.end());
  return packages;
}

std::string relative_name(const std::filesystem::path &graph_path) {
  return std::filesystem::relative(graph_path.parent_path(),
                                   find_repo_root() / "data")
      .generic_string();
}

void print_diagnostics(const std::vector<RuntimeDiagnostic> &diagnostics) {
  for (const auto &d : diagnostics) {
    std::cerr << "      [" << deeptile::runtime::severity_name(d.severity)
              << "] ";
    if (!d.context.empty()) {
      std::cerr << d.context << ": ";
    }
    std::cerr << d.message << "\n";
  }
}

/** Report at most `max_reported` differing elements as CHW coordinates. */
bool compare_chw(const std::vector<std::int8_t> &generated,
                 const std::vector<std::int8_t> &exported,
                 const std::vector<int> &shape, const std::string &tensor_id,
                 int max_reported = 8) {
  if (generated.size() != exported.size()) {
    std::cerr << "      size mismatch " << tensor_id
              << " generated=" << generated.size()
              << " exported=" << exported.size() << "\n";
    return false;
  }
  if (shape.size() != 3) {
    std::cerr << "      " << tensor_id << " reference shape is not CHW\n";
    return false;
  }

  const std::size_t spatial =
      static_cast<std::size_t>(shape[1]) * static_cast<std::size_t>(shape[2]);
  std::size_t mismatches = 0;
  int reported = 0;
  for (std::size_t i = 0; i < generated.size(); ++i) {
    if (generated[i] == exported[i]) {
      continue;
    }
    ++mismatches;
    if (reported < max_reported) {
      std::cerr << "      diff " << tensor_id << " idx=" << i << " (c="
                << i / spatial << ", h=" << (i % spatial) / shape[2]
                << ", w=" << (i % spatial) % shape[2]
                << ") generated=" << static_cast<int>(generated[i])
                << " exported=" << static_cast<int>(exported[i]) << "\n";
      ++reported;
    }
  }

  if (mismatches != 0) {
    std::cerr << "      " << tensor_id << " mismatches=" << mismatches << " / "
              << generated.size() << "\n";
    return false;
  }
  return true;
}

/** Compile, prepare, and reference one package, then reconcile the oracles. */
bool check_package(const std::filesystem::path &graph_path, bool fuse_dw) {
  if (fuse_dw) {
    setenv("DEEPTILE_FUSE_DW", "1", 1);
  } else {
    unsetenv("DEEPTILE_FUSE_DW");
  }

  RuntimeExecutionPlan plan;
  std::vector<RuntimeDiagnostic> diagnostics;
  const bool prepared = deeptile::runtime::load_compile_prepare_runtime(
      graph_path.string(), plan, diagnostics);
  unsetenv("DEEPTILE_FUSE_DW");

  if (!prepared) {
    std::cerr << "      compile/prepare failed\n";
    print_diagnostics(diagnostics);
    return false;
  }
  if (plan.stem_fold.active) {
    std::cout << "      stem " << plan.stem_fold.node_id
              << " folded by space-to-depth\n";
  }

  const deeptile::runtime::RuntimeReferenceResult reference =
      deeptile::runtime::generate_bit_exact_reference(plan);
  if (deeptile::runtime::has_errors(reference.diagnostics)) {
    std::cerr << "      reference generation failed\n";
    print_diagnostics(reference.diagnostics);
    return false;
  }

  const auto &graph = plan.graph_plan.graph;
  if (graph.outputs.empty()) {
    std::cerr << "      package declares no graph outputs\n";
    return false;
  }

  bool ok = true;
  for (const auto &output_id : graph.outputs) {
    // An exported reference is mandatory: it is the only oracle the kernel and
    // the software reference do not share assumptions with.
    const auto ref_it = graph.references.find(output_id);
    if (ref_it == graph.references.end()) {
      std::cerr << "      no exported reference declared for " << output_id
                << "\n";
      ok = false;
      continue;
    }
    const auto *generated =
        deeptile::runtime::find_reference_tensor(reference, output_id);
    const auto *exported = find_buffer(plan, ref_it->second);
    if (!generated || !exported) {
      std::cerr << "      missing generated/exported tensor for " << output_id
                << "\n";
      ok = false;
      continue;
    }
    if (!compare_chw(generated->data_chw, exported->raw_data, generated->shape,
                     output_id)) {
      ok = false;
    }
  }
  return ok;
}

/**
 * True when the graph contains a grouped conv2d, i.e. a depthwise candidate.
 *
 * This is deliberately looser than the compiler's own fusion legality check:
 * over-selecting only costs an extra pass, while under-selecting would leave a
 * fused lowering unchecked against the exporter.
 */
bool has_depthwise_candidate(const std::filesystem::path &graph_path) {
  deeptile::graph::GraphSpec graph;
  std::vector<deeptile::graph::GraphDiagnostic> diagnostics;
  if (!deeptile::graph::load_graph(graph_path.string(), graph, diagnostics)) {
    return false;
  }
  return std::any_of(graph.nodes.begin(), graph.nodes.end(),
                     [](const deeptile::graph::NodeSpec &n) {
                       if (n.op != "conv2d") {
                         return false;
                       }
                       const auto it = n.attrs_int.find("groups");
                       return it != n.attrs_int.end() && it->second > 1;
                     });
}

} // namespace

int main() {
  const auto packages = discover_packages();
  if (packages.empty()) {
    std::cerr << "[ERROR] no model packages found under data/\n";
    return 1;
  }

  int passed = 0;
  int failed = 0;
  for (const auto &graph_path : packages) {
    const std::string name = relative_name(graph_path);
    const bool check_fused = has_depthwise_candidate(graph_path);

    for (const bool fuse_dw : {false, true}) {
      if (fuse_dw && !check_fused) {
        continue;
      }
      const std::string label = name + (fuse_dw ? " [fused]" : " [unfused]");
      std::cout << "  " << label << " ... " << std::flush;
      if (check_package(graph_path, fuse_dw)) {
        std::cout << "PASS\n";
        ++passed;
      } else {
        std::cout << "FAIL\n";
        ++failed;
      }
    }
  }

  std::cout << "Reference equivalence: " << passed << " passed, " << failed
            << " failed\n";
  return failed == 0 ? 0 : 1;
}
