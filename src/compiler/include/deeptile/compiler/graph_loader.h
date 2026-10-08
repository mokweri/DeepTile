/**
 * @file graph_loader.h
 * @brief Graph handoff loader interface.
 * @ingroup deeptile_compiler
 */

#ifndef DEEPTILE_GRAPH_LOADER_H_
#define DEEPTILE_GRAPH_LOADER_H_

#include "deeptile/compiler/graph_types.h"

#include <string>
#include <vector>

namespace deeptile {
namespace graph {

/**
 * Load and validate a graph handoff document.
 *
 * @param graph_path Path to the graph JSON file.
 * @param graph Receives the parsed graph when loading succeeds.
 * @param diagnostics Receives validation warnings and errors.
 * @return True when the document was loaded and contains no errors.
 */
bool load_graph(const std::string &graph_path, GraphSpec &graph,
                std::vector<GraphDiagnostic> &diagnostics);

} // namespace graph
} // namespace deeptile

#endif // DEEPTILE_GRAPH_LOADER_H_
