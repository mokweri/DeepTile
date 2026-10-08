/**
 * @file model_package_tests.cpp
 * @brief Contract tests for ModelPackage artifact integrity.
 */

#include "deeptile/compiler/model_package.h"
#include "deeptile/compiler/target_package.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ng = deeptile::graph;
namespace nt = deeptile::target;

namespace {

int g_failures = 0;

void check(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "  [FAIL] " << message << "\n";
    ++g_failures;
  }
}

void write_text(const fs::path &path, const std::string &text) {
  std::ofstream output(path, std::ios::binary);
  output << text;
}

std::string file_hash(const fs::path &path) {
  std::string digest;
  std::string error;
  check(nt::sha256_file(path.string(), digest, error),
        "test artifact can be hashed: " + path.string());
  return digest;
}

std::string manifest_text(const fs::path &directory, bool complete,
                          bool list_parameter = true) {
  std::ostringstream out;
  out << R"({
  "schema": "tilecnn.model-package.v1",
  "model": {"release_id": "test/model@v1"},
)";
  if (complete) {
    out << R"(  "integrity": {
    "algorithm": "sha256",
    "coverage": "all-graph-artifacts"
  },
)";
  }
  out << R"(  "artifacts": {
    "graph": {"path": "graph.json", "sha256": ")"
      << file_hash(directory / "graph.json") << R"("},
    "parameters": [)";
  if (list_parameter) {
    out << R"({"path": "params/weight.bin", "sha256": ")"
        << file_hash(directory / "params" / "weight.bin") << R"("})";
  }
  out << R"(],
    "inputs": [{"path": "inputs/input.bin", "sha256": ")"
      << file_hash(directory / "inputs" / "input.bin") << R"("}],
    "references": [{"path": "refs/output.bin", "sha256": ")"
      << file_hash(directory / "refs" / "output.bin") << R"("}]
  }
})";
  return out.str();
}

bool has_warning(const std::vector<ng::GraphDiagnostic> &diagnostics) {
  for (const auto &diagnostic : diagnostics) {
    if (diagnostic.severity == ng::DiagnosticSeverity::Warning) {
      return true;
    }
  }
  return false;
}

void prepare_package(const fs::path &directory) {
  fs::create_directories(directory / "params");
  fs::create_directories(directory / "inputs");
  fs::create_directories(directory / "refs");
  write_text(directory / "params" / "weight.bin", "weight");
  write_text(directory / "inputs" / "input.bin", "input");
  write_text(directory / "refs" / "output.bin", "output");
  write_text(directory / "graph.json", R"({
  "schema": "tilecnn.graph.v1",
  "tensors": {
    "weight": {"kind": "param", "file": "params/weight.bin"},
    "input": {"kind": "input", "file": "inputs/input.bin"},
    "output_ref": {"kind": "reference", "file": "refs/output.bin"}
  }
})");
}

void test_complete_package(const fs::path &directory) {
  write_text(directory / "manifest.json", manifest_text(directory, true));
  std::vector<ng::GraphDiagnostic> diagnostics;
  check(ng::validate_model_package((directory / "graph.json").string(),
                                   diagnostics),
        "complete package is accepted");
  check(diagnostics.empty(), "complete package has no diagnostics");

  write_text(directory / "params" / "weight.bin", "tamper");
  diagnostics.clear();
  check(!ng::validate_model_package((directory / "graph.json").string(),
                                    diagnostics),
        "same-size parameter corruption is rejected");
  write_text(directory / "params" / "weight.bin", "weight");
}

void test_incomplete_inventory(const fs::path &directory) {
  write_text(directory / "manifest.json",
             manifest_text(directory, true, false));
  std::vector<ng::GraphDiagnostic> diagnostics;
  check(!ng::validate_model_package((directory / "graph.json").string(),
                                    diagnostics),
        "complete manifest cannot omit a graph parameter");
}

void test_legacy_manifest(const fs::path &directory) {
  write_text(directory / "manifest.json",
             manifest_text(directory, false, false));
  std::vector<ng::GraphDiagnostic> diagnostics;
  check(ng::validate_model_package((directory / "graph.json").string(),
                                   diagnostics),
        "legacy partial manifest remains usable");
  check(has_warning(diagnostics), "legacy partial manifest reports a warning");
}

} // namespace

int main() {
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path directory = fs::temp_directory_path() /
                             ("deeptile-model-package-test-" +
                              std::to_string(nonce));
  prepare_package(directory);

  test_complete_package(directory);
  test_incomplete_inventory(directory);
  test_legacy_manifest(directory);

  fs::remove_all(directory);
  if (g_failures != 0) {
    std::cerr << g_failures << " model-package check(s) failed\n";
    return 1;
  }
  std::cout << "OK: ModelPackage integrity checks passed\n";
  return 0;
}
