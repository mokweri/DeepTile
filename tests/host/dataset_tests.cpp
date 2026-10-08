/**
 * @file dataset_tests.cpp
 * @brief Tests for dataset indexing and label resolution.
 *
 * Indexing is independent of image decoding, so these run in every native
 * build. Image files are not tracked, so the listing tests build their own
 * fixtures; the prepared evaluation root is exercised only when it exists.
 */

#include "dataset/dataset.h"
#include "dataset/evaluation.h"

#include "deeptile/compiler/target_package.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#define TEST_ASSERT(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "    ASSERT FAILED: " << msg << "\n";                       \
      return false;                                                            \
    }                                                                          \
  } while (0)

namespace {

using deeptile::dataset::DatasetIndex;
using deeptile::dataset::EvaluationContract;
using deeptile::runtime::RuntimeDiagnostic;

std::filesystem::path repo_root() {
#ifdef DEEPTILE_REPO_ROOT
  return DEEPTILE_REPO_ROOT;
#else
  return std::filesystem::current_path();
#endif
}

std::string imagenet_root() {
  return (repo_root() / "data" / "imagenet").string();
}

/** Build a dataset root listing `listed` names, of which `present` exist. */
std::filesystem::path make_fixture(const std::string &tag, int listed,
                                   int present) {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / ("deeptile_dataset_" + tag);
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir / "val");
  std::ofstream listing(dir / "val.txt");
  for (int i = 0; i < listed; ++i) {
    const std::string name = "img" + std::to_string(i) + ".JPEG";
    listing << name << " " << i << "\n";
    if (i < present) {
      std::ofstream(dir / "val" / name) << "placeholder";
    }
  }
  return dir;
}

/** A complete listing indexes cleanly and says nothing is missing. */
bool test_index_complete_listing() {
  const auto dir = make_fixture("complete", 4, 4);
  DatasetIndex index;
  std::vector<RuntimeDiagnostic> diagnostics;
  const bool ok =
      deeptile::dataset::load_imagenet_index(dir.string(), 0, index, diagnostics);
  std::filesystem::remove_all(dir);

  TEST_ASSERT(ok, "a complete listing should index");
  TEST_ASSERT(index.entries.size() == 4, "expected every listed image");
  TEST_ASSERT(index.listed == 4 && index.present == 4 && index.missing == 0,
              "accounting must show a complete set");
  // Listing order is the evaluation order; it must not depend on the
  // filesystem's iteration order.
  TEST_ASSERT(index.entries.front().name == "img0.JPEG",
              "unexpected first entry: " << index.entries.front().name);
  TEST_ASSERT(index.entries.back().label == 3, "labels follow the listing");
  for (const auto &d : diagnostics) {
    TEST_ASSERT(d.severity != deeptile::runtime::RuntimeDiagnosticSeverity::Warning,
                "a complete set must not warn: " << d.message);
  }
  return true;
}

/**
 * val.txt defines the evaluation set, so a short set must not quietly become
 * an accuracy number over an unidentified subset.
 */
bool test_incomplete_listing_is_fatal() {
  const auto dir = make_fixture("incomplete", 4, 3);
  DatasetIndex index;
  std::vector<RuntimeDiagnostic> diagnostics;
  const bool ok =
      deeptile::dataset::load_imagenet_index(dir.string(), 0, index, diagnostics);
  std::filesystem::remove_all(dir);

  TEST_ASSERT(!ok, "an incomplete listing must fail by default");
  TEST_ASSERT(deeptile::runtime::has_errors(diagnostics),
              "an incomplete listing must report an error");
  return true;
}

/** The escape hatch still works, and still reports what it skipped. */
bool test_incomplete_listing_allowed() {
  const auto dir = make_fixture("allowed", 4, 3);
  DatasetIndex index;
  std::vector<RuntimeDiagnostic> diagnostics;
  const bool ok = deeptile::dataset::load_imagenet_index(dir.string(), 0, index,
                                                        diagnostics, true);
  std::filesystem::remove_all(dir);

  TEST_ASSERT(ok, "--allow-missing should index what is present");
  TEST_ASSERT(index.entries.size() == 3, "expected the present images");
  TEST_ASSERT(index.listed == 4 && index.present == 3 && index.missing == 1,
              "accounting must reconcile");
  TEST_ASSERT(!deeptile::runtime::has_errors(diagnostics),
              "the escape hatch must not error");
  bool warned = false;
  for (const auto &d : diagnostics) {
    warned |= d.severity == deeptile::runtime::RuntimeDiagnosticSeverity::Warning;
  }
  TEST_ASSERT(warned, "an incomplete set must warn even when allowed");
  return true;
}

bool test_limit() {
  const auto dir = make_fixture("limit", 4, 4);
  DatasetIndex full;
  std::vector<RuntimeDiagnostic> full_diagnostics;
  const bool full_ok = deeptile::dataset::load_imagenet_index(
      dir.string(), 0, full, full_diagnostics);

  DatasetIndex index;
  std::vector<RuntimeDiagnostic> diagnostics;
  const bool ok =
      deeptile::dataset::load_imagenet_index(dir.string(), 2, index, diagnostics);
  std::filesystem::remove_all(dir);

  TEST_ASSERT(full_ok, "full indexing should succeed");
  TEST_ASSERT(ok, "limited indexing should succeed");
  TEST_ASSERT(index.entries.size() == 2,
              "expected 2 entries, got " << index.entries.size());
  // A limit bounds what is evaluated; it must not distort dataset accounting.
  TEST_ASSERT(index.listed == full.listed, "limit changed listed count");
  TEST_ASSERT(index.present == full.present, "limit changed present count");
  TEST_ASSERT(index.missing == full.missing, "limit changed missing count");
  return true;
}

bool test_missing_root() {
  DatasetIndex index;
  std::vector<RuntimeDiagnostic> diagnostics;
  TEST_ASSERT(!deeptile::dataset::load_imagenet_index(
                  (repo_root() / "data" / "no_such_dataset").string(), 0, index,
                  diagnostics),
              "a missing dataset must fail");
  TEST_ASSERT(deeptile::runtime::has_errors(diagnostics),
              "a missing dataset must report an error");
  return true;
}

bool test_malformed_entries() {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "deeptile_dataset_malformed";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir / "val");
  std::ofstream(dir / "val" / "a.JPEG") << "not really an image";
  std::ofstream(dir / "val.txt") << "a.JPEG 7\n"
                                    "b.JPEG 9\n"
                                    "garbage-line\n"
                                    "\n";

  DatasetIndex index;
  std::vector<RuntimeDiagnostic> diagnostics;
  const bool ok = deeptile::dataset::load_imagenet_index(dir.string(), 0, index,
                                                        diagnostics, true);
  std::filesystem::remove_all(dir);

  TEST_ASSERT(ok, "indexing should succeed when at least one image exists");
  TEST_ASSERT(index.entries.size() == 1,
              "only existing images may be indexed, got "
                  << index.entries.size());
  TEST_ASSERT(index.entries.front().label == 7, "wrong label parsed");
  TEST_ASSERT(index.listed == 2, "malformed lines must not count as listed");
  TEST_ASSERT(index.present == 1, "present image must be counted");
  TEST_ASSERT(index.missing == 1, "absent image must be counted as missing");
  return true;
}

/**
 * The prepared evaluation root is not tracked, so this checks it only where
 * it has been built. Where it exists it must be complete.
 */
bool test_prepared_root_when_present() {
  const std::filesystem::path images =
      std::filesystem::path(imagenet_root()) / "ilsvrc2012" /
      "ILSVRC2012_img_val";
  if (!std::filesystem::is_directory(images)) {
    std::cout << "(skipped: run scripts/data/prepare_imagenet_subset.py) ";
    return true;
  }

  DatasetIndex index;
  std::vector<RuntimeDiagnostic> diagnostics;
  TEST_ASSERT(deeptile::dataset::load_imagenet_index(imagenet_root(), 0, index,
                                                    diagnostics),
              "the prepared root must index without --allow-missing");
  TEST_ASSERT(index.missing == 0, "the prepared root must be complete");
  TEST_ASSERT(index.present == index.entries.size(),
              "an unlimited index should select every present image");
  TEST_ASSERT(index.class_names.size() == 1000,
              "expected 1000 class names, got " << index.class_names.size());
  for (const auto &entry : index.entries) {
    TEST_ASSERT(entry.label >= 0 && entry.label < 1000,
                "label out of range for " << entry.name);
    TEST_ASSERT(std::filesystem::is_regular_file(entry.path),
                "indexed a nonexistent file: " << entry.path);
  }

  // A complete run must reproduce the digest the preparation script printed,
  // which is sha256 of val.txt itself. If these ever diverge, the run record
  // would name a dataset nobody can reconstruct.
  std::ifstream listing(std::filesystem::path(imagenet_root()) / "val.txt",
                        std::ios::binary);
  const std::string contents((std::istreambuf_iterator<char>(listing)),
                             std::istreambuf_iterator<char>());
  TEST_ASSERT(deeptile::dataset::listing_digest(index) ==
                  deeptile::target::sha256_text(contents),
              "listing digest must equal sha256 of val.txt");
  return true;
}

/** A limited run covers a different set, so it must digest differently. */
bool test_listing_digest_tracks_selection() {
  const auto dir = make_fixture("digest", 4, 4);
  DatasetIndex full;
  DatasetIndex partial;
  std::vector<RuntimeDiagnostic> diagnostics;
  deeptile::dataset::load_imagenet_index(dir.string(), 0, full, diagnostics);
  deeptile::dataset::load_imagenet_index(dir.string(), 2, partial, diagnostics);
  std::filesystem::remove_all(dir);

  const std::string full_digest = deeptile::dataset::listing_digest(full);
  TEST_ASSERT(full_digest.size() == 64, "expected a sha256 hex digest");
  TEST_ASSERT(full_digest == deeptile::target::sha256_text(
                                 "img0.JPEG 0\nimg1.JPEG 1\n"
                                 "img2.JPEG 2\nimg3.JPEG 3\n"),
              "digest formula must be '<name> <label>' newline-terminated");
  TEST_ASSERT(deeptile::dataset::listing_digest(partial) != full_digest,
              "a limited run must not claim the full listing's digest");
  return true;
}

deeptile::graph::GraphSpec valid_classification_graph(int input_frac = 5) {
  deeptile::graph::GraphSpec graph;
  graph.inputs = {"input"};
  graph.outputs = {"logits"};

  deeptile::graph::TensorSpec input;
  input.id = "input";
  input.kind = "input";
  input.dtype = "int8";
  input.layout = "CHW";
  input.shape = {3, 32, 32};
  input.frac = input_frac;
  graph.tensors.emplace(input.id, input);

  deeptile::graph::TensorSpec output;
  output.id = "logits";
  output.kind = "activation";
  output.dtype = "int8";
  output.layout = "CHW";
  output.shape = {10, 1, 1};
  graph.tensors.emplace(output.id, output);
  return graph;
}

bool test_evaluation_contract_derives_graph_quantization() {
  const auto graph = valid_classification_graph(3);
  EvaluationContract contract;
  std::vector<RuntimeDiagnostic> diagnostics;
  TEST_ASSERT(deeptile::dataset::resolve_evaluation_contract(
                  graph, contract, diagnostics),
              "valid classifier graph should be accepted");
  TEST_ASSERT(contract.input_id == "input", "wrong input id");
  TEST_ASSERT(contract.output_id == "logits", "wrong output id");
  TEST_ASSERT(contract.target_side == 32, "wrong target side");
  TEST_ASSERT(contract.input_frac == 3,
              "fractional bits must come from the graph");
  TEST_ASSERT(contract.class_count == 10, "wrong class count");
  return true;
}

bool contract_is_rejected(const deeptile::graph::GraphSpec &graph) {
  EvaluationContract contract;
  std::vector<RuntimeDiagnostic> diagnostics;
  return !deeptile::dataset::resolve_evaluation_contract(graph, contract,
                                                        diagnostics) &&
         deeptile::runtime::has_errors(diagnostics);
}

bool test_evaluation_contract_rejects_incompatible_graphs() {
  auto graph = valid_classification_graph();
  graph.tensors["input"].shape[0] = 1;
  TEST_ASSERT(contract_is_rejected(graph), "non-RGB input was accepted");

  graph = valid_classification_graph();
  graph.tensors["input"].shape[2] = 24;
  TEST_ASSERT(contract_is_rejected(graph), "non-square input was accepted");

  graph = valid_classification_graph();
  graph.tensors["logits"].shape = {10, 2, 2};
  TEST_ASSERT(contract_is_rejected(graph), "spatial output was accepted");

  graph = valid_classification_graph();
  graph.tensors["logits"].dtype = "float32";
  TEST_ASSERT(contract_is_rejected(graph), "non-int8 output was accepted");
  return true;
}

bool test_evaluation_labels_match_class_count() {
  DatasetIndex index;
  index.entries = {{"zero.JPEG", "zero.JPEG", 0},
                   {"last.JPEG", "last.JPEG", 9},
                   {"unknown.JPEG", "unknown.JPEG", -1}};
  std::vector<RuntimeDiagnostic> diagnostics;
  TEST_ASSERT(deeptile::dataset::validate_evaluation_labels(index, 10,
                                                           diagnostics),
              "valid and unlabelled entries should be accepted");

  index.entries.push_back({"invalid.JPEG", "invalid.JPEG", 10});
  diagnostics.clear();
  TEST_ASSERT(!deeptile::dataset::validate_evaluation_labels(index, 10,
                                                            diagnostics),
              "out-of-range label was accepted");
  TEST_ASSERT(deeptile::runtime::has_errors(diagnostics),
              "invalid label must report an error");
  return true;
}

int passed = 0;
int failed = 0;

void run(const std::string &name, const std::function<bool()> &fn) {
  std::cout << "  " << name << " ... " << std::flush;
  if (fn()) {
    std::cout << "PASS\n";
    ++passed;
  } else {
    std::cout << "FAIL\n";
    ++failed;
  }
}

} // namespace

int main() {
  run("index_complete_listing", test_index_complete_listing);
  run("incomplete_listing_is_fatal", test_incomplete_listing_is_fatal);
  run("incomplete_listing_allowed", test_incomplete_listing_allowed);
  run("limit", test_limit);
  run("missing_root", test_missing_root);
  run("malformed_entries", test_malformed_entries);
  run("prepared_root_when_present", test_prepared_root_when_present);
  run("listing_digest_tracks_selection", test_listing_digest_tracks_selection);
  run("evaluation_contract_derives_graph_quantization",
      test_evaluation_contract_derives_graph_quantization);
  run("evaluation_contract_rejects_incompatible_graphs",
      test_evaluation_contract_rejects_incompatible_graphs);
  run("evaluation_labels_match_class_count",
      test_evaluation_labels_match_class_count);

  std::cout << "Dataset tests: " << passed << " passed, " << failed
            << " failed\n";
  return failed == 0 ? 0 : 1;
}
