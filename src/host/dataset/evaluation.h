/**
 * @file evaluation.h
 * @brief Dataset-evaluation contract derived from a model graph.
 * @ingroup deeptile_host
 */

#ifndef DEEPTILE_DATASET_EVALUATION_H_
#define DEEPTILE_DATASET_EVALUATION_H_

#include "dataset/dataset.h"
#include "runtime/runtime_types.h"
#include "deeptile/compiler/graph_types.h"

#include <string>
#include <vector>

namespace deeptile {
namespace dataset {

/** Model properties required by the RGB classification pipeline. */
struct EvaluationContract {
  std::string input_id;
  std::string output_id;
  std::vector<int> input_shape;
  std::vector<int> output_shape;
  int input_frac = 0;
  int target_side = 0;
  int class_count = 0;
};

/** Validate a graph and derive its preprocessing and classification contract. */
bool resolve_evaluation_contract(
    const graph::GraphSpec &graph, EvaluationContract &out,
    std::vector<runtime::RuntimeDiagnostic> &diagnostics);

/** Ensure every labelled dataset entry fits the model's output index space. */
bool validate_evaluation_labels(
    const DatasetIndex &index, int class_count,
    std::vector<runtime::RuntimeDiagnostic> &diagnostics);

} // namespace dataset
} // namespace deeptile

#endif // DEEPTILE_DATASET_EVALUATION_H_
