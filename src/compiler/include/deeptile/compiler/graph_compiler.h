/**
 * @file graph_compiler.h
 * @brief Graph-to-launch-plan compiler interface.
 * @ingroup deeptile_compiler
 */

#ifndef DEEPTILE_GRAPH_COMPILER_H_
#define DEEPTILE_GRAPH_COMPILER_H_

#include "deeptile/compiler/graph_types.h"

#include <string>

namespace deeptile {
namespace graph {

/** Compile a validated graph into ordered runtime launch plans. */
GraphExecutionPlan compile_graph(const GraphSpec &graph);

/**
 * Summarize compile-time lowering switches that depart from their defaults.
 *
 * Environment switches change the emitted descriptors, so a run driven by one
 * is not reproducible from the model and target alone. Returns an empty string
 * for a default compilation, which is the ordinary case; anything else belongs
 * in the run record. Fusion is reported separately as a plan property.
 */
std::string schedule_flag_summary();

} // namespace graph
} // namespace deeptile

#endif // DEEPTILE_GRAPH_COMPILER_H_
