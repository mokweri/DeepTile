/**
 * @file schema_ids.h
 * @brief Declares record-format identifiers under DeepTile's current and former names.
 * @ingroup deeptile_compiler
 */

#ifndef DEEPTILE_SCHEMA_IDS_H_
#define DEEPTILE_SCHEMA_IDS_H_

#include <string_view>

namespace deeptile {
namespace schema {

// Identifiers written into new records. Records from before the rename carry
// the tilecnn.* form of the same formats, which readers still accept. See
// docs/reference/naming.md.
inline constexpr std::string_view kGraph = "deeptile.graph.v1";
inline constexpr std::string_view kModelPackage = "deeptile.model-package.v1";
inline constexpr std::string_view kTarget = "deeptile.target.v1";
inline constexpr std::string_view kRun = "deeptile.run.v1";

/** Source-revision field written into new records. */
inline constexpr std::string_view kRevisionField = "deeptile_revision";

/** The same field in records written before the rename; read, never written. */
inline constexpr std::string_view kFormerRevisionField = "tilecnn_revision";

/**
 * True when @p actual names @p format ("graph.v1", "target.v1", ...) under
 * either namespace. The two namespaces identify the same formats.
 */
inline bool matches(std::string_view actual, std::string_view format) {
  for (std::string_view prefix : {std::string_view("deeptile."),
                                  std::string_view("tilecnn.")}) {
    if (actual.size() == prefix.size() + format.size() &&
        actual.substr(0, prefix.size()) == prefix &&
        actual.substr(prefix.size()) == format) {
      return true;
    }
  }
  return false;
}

}  // namespace schema
}  // namespace deeptile

#endif  // DEEPTILE_SCHEMA_IDS_H_
