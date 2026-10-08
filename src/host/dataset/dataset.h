/**
 * @file dataset.h
 * @brief Classification-dataset indexing and label resolution.
 * @ingroup deeptile_host
 */

#ifndef DEEPTILE_DATASET_DATASET_H_
#define DEEPTILE_DATASET_DATASET_H_

#include "runtime/runtime_types.h"

#include <cstddef>
#include <string>
#include <vector>

namespace deeptile {
namespace dataset {

/** One evaluable image and its ground-truth class index. */
struct DatasetEntry {
  std::string name;
  std::string path;
  int label = -1;
};

/**
 * A resolved dataset listing.
 *
 * `entries` contains the selected images in listing order. `present` counts all
 * listed images found on disk before a limit is applied, while `listed` and
 * `missing` describe the complete label file.
 */
struct DatasetIndex {
  std::vector<DatasetEntry> entries;
  std::vector<std::string> class_names;
  std::size_t listed = 0;
  std::size_t present = 0;
  std::size_t missing = 0;
};

/**
 * Index an ImageNet-style validation directory.
 *
 * Expects `val.txt` (one "<file> <class-index>" per line), an image directory,
 * and optionally `synset_words.txt` for human-readable class names.
 *
 * `val.txt` defines the evaluation set, so every image it names must exist. An
 * absent image is an error rather than a warning: a partially present dataset
 * that still reports an accuracy number is the same failure mode as a stale
 * bitstream that still reports a latency. Set @p allow_missing to index
 * whatever is present anyway, which is useful for smoke tests but yields
 * accuracy over an unidentified subset.
 *
 * @param root Dataset root directory.
 * @param limit Maximum entries to index, or zero for no limit.
 * @param out Resolved entries, class names, and listing counts.
 * @param diagnostics Receives parsing, missing-file, and validation messages.
 * @param allow_missing Downgrade an incomplete listing to a warning.
 * @return False if the listing could not be read, is incomplete, or produced
 *         no entries.
 */
bool load_imagenet_index(const std::string &root, std::size_t limit,
                         DatasetIndex &out,
                         std::vector<runtime::RuntimeDiagnostic> &diagnostics,
                         bool allow_missing = false);

/**
 * Digest the selected images: SHA-256 over the ordered "<name> <label>\n"
 * lines of `index.entries`.
 *
 * One value answers which images a run covered, without listing them. The
 * formula matches `scripts/data/prepare_imagenet_subset.py`, so a run over a
 * complete listing reproduces the digest that script printed when the dataset
 * was prepared; a limited or incomplete run differs, which is the point.
 */
std::string listing_digest(const DatasetIndex &index);

} // namespace dataset
} // namespace deeptile

#endif // DEEPTILE_DATASET_DATASET_H_
