/**
 * @file model_package.h
 * @brief Declares ModelPackage integrity validation.
 * @ingroup deeptile_compiler
 */

#ifndef DEEPTILE_MODEL_PACKAGE_H_
#define DEEPTILE_MODEL_PACKAGE_H_

#include "deeptile/compiler/graph_types.h"

#include <string>
#include <vector>

namespace deeptile {
namespace graph {

/**
 * Validate the manifest beside a graph when one is present.
 *
 * Graph-only regression fixtures remain supported. A legacy manifest without
 * complete integrity coverage is verified as far as possible and produces a
 * warning; a complete manifest must inventory every file-backed graph tensor.
 */
bool validate_model_package(
    const std::string &graph_path,
    std::vector<GraphDiagnostic> &diagnostics);

} // namespace graph
} // namespace deeptile

#endif // DEEPTILE_MODEL_PACKAGE_H_
