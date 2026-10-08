/**
 * @file dataset.cpp
 * @brief Implements dataset listing and label resolution.
 */

#include "dataset/dataset.h"

#include "deeptile/compiler/target_package.h"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace deeptile {
namespace dataset {
namespace {

void add_error(std::vector<runtime::RuntimeDiagnostic> &diagnostics,
               const std::string &message) {
  diagnostics.push_back({runtime::RuntimeDiagnosticSeverity::Error, "dataset",
                         message});
}

void add_warning(std::vector<runtime::RuntimeDiagnostic> &diagnostics,
                 const std::string &message) {
  diagnostics.push_back({runtime::RuntimeDiagnosticSeverity::Warning, "dataset",
                         message});
}

/** Locate the image directory; the ILSVRC layout nests it one level deep. */
std::filesystem::path find_image_dir(const std::filesystem::path &root) {
  const std::filesystem::path nested =
      root / "ilsvrc2012" / "ILSVRC2012_img_val";
  if (std::filesystem::is_directory(nested)) {
    return nested;
  }
  const std::filesystem::path flat = root / "val";
  if (std::filesystem::is_directory(flat)) {
    return flat;
  }
  return root;
}

void load_class_names(const std::filesystem::path &path,
                      std::vector<std::string> &out) {
  std::ifstream in(path);
  if (!in) {
    return;
  }
  std::string line;
  while (std::getline(in, line)) {
    // Each line is "<synset> <comma-separated names>"; keep the readable part.
    const std::size_t space = line.find(' ');
    out.push_back(space == std::string::npos ? line : line.substr(space + 1));
  }
}

} // namespace

bool load_imagenet_index(const std::string &root, std::size_t limit,
                         DatasetIndex &out,
                         std::vector<runtime::RuntimeDiagnostic> &diagnostics,
                         bool allow_missing) {
  const std::filesystem::path root_path(root);
  const std::filesystem::path labels_path = root_path / "val.txt";

  std::ifstream labels(labels_path);
  if (!labels) {
    add_error(diagnostics, "cannot open label file: " + labels_path.string());
    return false;
  }

  const std::filesystem::path image_dir = find_image_dir(root_path);
  load_class_names(root_path / "synset_words.txt", out.class_names);

  std::string line;
  while (std::getline(labels, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream parser(line);
    std::string name;
    int label = -1;
    if (!(parser >> name >> label)) {
      add_warning(diagnostics, "skipping malformed label line: " + line);
      continue;
    }
    ++out.listed;

    const std::filesystem::path image_path = image_dir / name;
    if (!std::filesystem::is_regular_file(image_path)) {
      ++out.missing;
      continue;
    }
    ++out.present;
    if (limit != 0 && out.entries.size() >= limit) {
      continue;
    }
    out.entries.push_back({name, image_path.string(), label});
  }

  if (out.missing != 0) {
    const std::string summary =
        "dataset is incomplete: " + std::to_string(out.present) + " of " +
        std::to_string(out.listed) + " images named by " +
        labels_path.string() + " are present under " + image_dir.string();
    if (!allow_missing) {
      // val.txt is the evaluation set. Evaluating a fraction of it would
      // report accuracy over a set nothing identifies.
      add_error(diagnostics, summary);
      return false;
    }
    add_warning(diagnostics, summary);
  }
  if (out.entries.empty()) {
    add_error(diagnostics,
              "no dataset images found under " + image_dir.string());
    return false;
  }
  return true;
}

std::string listing_digest(const DatasetIndex &index) {
  std::string listing;
  for (const auto &entry : index.entries) {
    listing += entry.name;
    listing += ' ';
    listing += std::to_string(entry.label);
    listing += '\n';
  }
  return deeptile::target::sha256_text(listing);
}

} // namespace dataset
} // namespace deeptile
