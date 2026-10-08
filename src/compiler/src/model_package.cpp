/**
 * @file model_package.cpp
 * @brief Validates ModelPackage manifests and artifact checksums.
 */

#include "deeptile/compiler/model_package.h"
#include "deeptile/compiler/schema_ids.h"

#include "deeptile/compiler/target_package.h"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>

namespace deeptile {
namespace graph {
namespace {

namespace fs = std::filesystem;

void add_diagnostic(std::vector<GraphDiagnostic> &diagnostics,
                    DiagnosticSeverity severity, const std::string &context,
                    const std::string &message) {
  diagnostics.push_back({severity, context, message});
}

void add_error(std::vector<GraphDiagnostic> &diagnostics,
               const std::string &context, const std::string &message) {
  add_diagnostic(diagnostics, DiagnosticSeverity::Error, context, message);
}

void add_warning(std::vector<GraphDiagnostic> &diagnostics,
                 const std::string &context, const std::string &message) {
  add_diagnostic(diagnostics, DiagnosticSeverity::Warning, context, message);
}

bool is_sha256(const std::string &value) {
  return value.size() == 64U &&
         std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

bool has_parent_component(const fs::path &path) {
  return std::any_of(path.begin(), path.end(), [](const auto &component) {
    return component == "..";
  });
}

bool is_within(const fs::path &root, const fs::path &candidate) {
  auto root_it = root.begin();
  auto candidate_it = candidate.begin();
  for (; root_it != root.end(); ++root_it, ++candidate_it) {
    if (candidate_it == candidate.end() || *root_it != *candidate_it) {
      return false;
    }
  }
  return true;
}

bool resolve_artifact(const fs::path &root, const std::string &relative,
                      fs::path &resolved,
                      std::vector<GraphDiagnostic> &diagnostics) {
  const fs::path path(relative);
  if (relative.empty() || path.is_absolute() || has_parent_component(path)) {
    add_error(diagnostics, "model package",
              "artifact path must stay inside the package: " + relative);
    return false;
  }
  std::error_code error;
  const fs::path canonical_root = fs::weakly_canonical(root, error);
  if (error) {
    add_error(diagnostics, "model package",
              "cannot resolve package directory: " + root.string());
    return false;
  }
  resolved = fs::weakly_canonical(root / path, error);
  if (error || !is_within(canonical_root, resolved) ||
      !fs::is_regular_file(resolved)) {
    add_error(diagnostics, "model package",
              "artifact is missing or outside the package: " + relative);
    return false;
  }
  return true;
}

bool read_json(const fs::path &path, rapidjson::Document &doc,
               std::vector<GraphDiagnostic> &diagnostics,
               const std::string &context) {
  std::ifstream input(path);
  if (!input) {
    add_error(diagnostics, context, "cannot open JSON file: " + path.string());
    return false;
  }
  rapidjson::IStreamWrapper stream(input);
  doc.ParseStream(stream);
  if (doc.HasParseError()) {
    std::ostringstream message;
    message << "JSON parse error at offset " << doc.GetErrorOffset() << ": "
            << rapidjson::GetParseError_En(doc.GetParseError());
    add_error(diagnostics, context, message.str());
    return false;
  }
  if (!doc.IsObject()) {
    add_error(diagnostics, context, "top-level JSON must be an object");
    return false;
  }
  return true;
}

bool validate_record(const fs::path &root, const rapidjson::Value &record,
                     std::string &relative,
                     std::vector<GraphDiagnostic> &diagnostics) {
  if (!record.IsObject() || !record.HasMember("path") ||
      !record["path"].IsString() || !record.HasMember("sha256") ||
      !record["sha256"].IsString()) {
    add_error(diagnostics, "model package",
              "artifact records require string path and sha256 fields");
    return false;
  }
  relative = record["path"].GetString();
  const std::string expected = record["sha256"].GetString();
  if (!is_sha256(expected)) {
    add_error(diagnostics, "model package",
              "invalid SHA-256 for artifact: " + relative);
    return false;
  }
  fs::path resolved;
  if (!resolve_artifact(root, relative, resolved, diagnostics)) {
    return false;
  }
  std::string actual;
  std::string hash_error;
  if (!deeptile::target::sha256_file(resolved.string(), actual, hash_error)) {
    add_error(diagnostics, "model package", hash_error);
    return false;
  }
  if (actual != expected) {
    add_error(diagnostics, "model package",
              "artifact checksum mismatch: " + relative);
    return false;
  }
  return true;
}

using ArtifactSets = std::map<std::string, std::set<std::string>>;

ArtifactSets expected_artifacts(const rapidjson::Document &graph,
                                std::vector<GraphDiagnostic> &diagnostics) {
  ArtifactSets expected = {
      {"parameters", {}}, {"inputs", {}}, {"references", {}}};
  if (!graph.HasMember("tensors") || !graph["tensors"].IsObject()) {
    add_error(diagnostics, "model package", "graph tensors object is missing");
    return expected;
  }
  const auto &tensors = graph["tensors"];
  for (auto it = tensors.MemberBegin(); it != tensors.MemberEnd(); ++it) {
    if (!it->value.IsObject() || !it->value.HasMember("file")) {
      continue;
    }
    if (!it->value["file"].IsString() ||
        !it->value.HasMember("kind") || !it->value["kind"].IsString()) {
      add_error(diagnostics, "model package",
                "file-backed tensor requires string kind and file fields");
      continue;
    }
    const std::string kind = it->value["kind"].GetString();
    const std::string path = it->value["file"].GetString();
    const char *category = nullptr;
    if (kind == "param") {
      category = "parameters";
    } else if (kind == "input") {
      category = "inputs";
    } else if (kind == "reference") {
      category = "references";
    } else {
      add_error(diagnostics, "model package",
                "unsupported file-backed tensor kind: " + kind);
      continue;
    }
    expected[category].insert(path);
  }
  return expected;
}

bool complete_integrity(const rapidjson::Document &manifest,
                        std::vector<GraphDiagnostic> &diagnostics) {
  if (!manifest.HasMember("integrity")) {
    add_warning(diagnostics, "model package",
                "legacy manifest has incomplete artifact coverage");
    return false;
  }
  const auto &integrity = manifest["integrity"];
  if (!integrity.IsObject() || !integrity.HasMember("algorithm") ||
      !integrity["algorithm"].IsString() ||
      !integrity.HasMember("coverage") ||
      !integrity["coverage"].IsString() ||
      std::string(integrity["algorithm"].GetString()) != "sha256" ||
      std::string(integrity["coverage"].GetString()) !=
          "all-graph-artifacts") {
    add_error(diagnostics, "model package",
              "integrity must declare sha256 coverage of all graph artifacts");
    return false;
  }
  return true;
}

} // namespace

bool validate_model_package(const std::string &graph_path,
                            std::vector<GraphDiagnostic> &diagnostics) {
  const fs::path graph_file(graph_path);
  const fs::path root = graph_file.parent_path();
  const fs::path manifest_path = root / "manifest.json";
  if (!fs::is_regular_file(manifest_path)) {
    return true;
  }

  rapidjson::Document manifest;
  if (!read_json(manifest_path, manifest, diagnostics, "model package")) {
    return false;
  }
  if (!manifest.HasMember("schema") || !manifest["schema"].IsString() ||
      !schema::matches(manifest["schema"].GetString(), "model-package.v1")) {
    add_error(diagnostics, "model package",
              "schema must be deeptile.model-package.v1 or "
              "tilecnn.model-package.v1");
  }
  if (!manifest.HasMember("artifacts") ||
      !manifest["artifacts"].IsObject()) {
    add_error(diagnostics, "model package", "artifacts object is missing");
    return false;
  }

  const auto &artifacts = manifest["artifacts"];
  if (!artifacts.HasMember("graph")) {
    add_error(diagnostics, "model package", "graph artifact is missing");
    return false;
  }
  std::string recorded_graph;
  if (!validate_record(root, artifacts["graph"], recorded_graph,
                       diagnostics)) {
    return false;
  }
  std::error_code recorded_error;
  std::error_code requested_error;
  const fs::path recorded_path =
      fs::weakly_canonical(root / recorded_graph, recorded_error);
  const fs::path requested_path =
      fs::weakly_canonical(graph_file, requested_error);
  if (recorded_error || requested_error || recorded_path != requested_path) {
    add_error(diagnostics, "model package",
              "manifest graph does not match the requested graph file");
  }

  rapidjson::Document graph;
  if (!read_json(graph_file, graph, diagnostics, "model package graph")) {
    return false;
  }
  const ArtifactSets expected = expected_artifacts(graph, diagnostics);
  const bool complete = complete_integrity(manifest, diagnostics);
  std::set<std::string> seen = {recorded_graph};

  for (const auto &[category, expected_paths] : expected) {
    if (!artifacts.HasMember(category.c_str())) {
      if (complete) {
        add_error(diagnostics, "model package",
                  "artifact list is missing: " + category);
      }
      continue;
    }
    const auto &records = artifacts[category.c_str()];
    if (!records.IsArray()) {
      add_error(diagnostics, "model package",
                "artifact list must be an array: " + category);
      continue;
    }
    std::set<std::string> actual_paths;
    for (const auto &record : records.GetArray()) {
      std::string relative;
      if (!validate_record(root, record, relative, diagnostics)) {
        continue;
      }
      if (!seen.insert(relative).second) {
        add_error(diagnostics, "model package",
                  "artifact is listed more than once: " + relative);
      }
      actual_paths.insert(relative);
    }
    if (complete && actual_paths != expected_paths) {
      add_error(diagnostics, "model package",
                category + " inventory does not match graph.json");
    }
  }
  return !has_errors(diagnostics);
}

} // namespace graph
} // namespace deeptile
