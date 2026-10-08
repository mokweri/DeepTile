/**
 * @file evaluation.cpp
 * @brief Implements graph and label validation for dataset evaluation.
 */

#include "dataset/evaluation.h"

namespace deeptile {
namespace dataset {
namespace {

void add_error(std::vector<runtime::RuntimeDiagnostic> &diagnostics,
               const std::string &message) {
  diagnostics.push_back(
      {runtime::RuntimeDiagnosticSeverity::Error, "dataset-eval", message});
}

const graph::TensorSpec *find_tensor(const graph::GraphSpec &graph,
                                     const std::string &id) {
  const auto it = graph.tensors.find(id);
  return it == graph.tensors.end() ? nullptr : &it->second;
}

} // namespace

bool resolve_evaluation_contract(
    const graph::GraphSpec &graph, EvaluationContract &out,
    std::vector<runtime::RuntimeDiagnostic> &diagnostics) {
  out = EvaluationContract{};
  if (graph.inputs.size() != 1) {
    add_error(diagnostics, "dataset evaluation requires a single-input graph");
    return false;
  }
  if (graph.outputs.size() != 1) {
    add_error(diagnostics, "dataset evaluation requires a single-output graph");
    return false;
  }

  out.input_id = graph.inputs.front();
  out.output_id = graph.outputs.front();
  const graph::TensorSpec *input = find_tensor(graph, out.input_id);
  const graph::TensorSpec *output = find_tensor(graph, out.output_id);
  if (!input) {
    add_error(diagnostics, "graph input tensor is missing: " + out.input_id);
    return false;
  }
  if (!output) {
    add_error(diagnostics, "graph output tensor is missing: " + out.output_id);
    return false;
  }

  if (input->dtype != "int8" || input->layout != "CHW" ||
      input->shape.size() != 3 || input->shape[0] != 3 ||
      input->shape[1] <= 0 || input->shape[1] != input->shape[2]) {
    add_error(diagnostics,
              "graph input must be an int8 CHW tensor shaped [3,H,H]");
    return false;
  }
  if (input->frac < 0 || input->frac > 30) {
    add_error(diagnostics, "graph input fractional bits must be in [0,30]");
    return false;
  }

  if (output->dtype != "int8" || output->layout != "CHW" ||
      output->shape.size() != 3 || output->shape[0] <= 0 ||
      output->shape[1] != 1 || output->shape[2] != 1) {
    add_error(diagnostics,
              "graph output must be an int8 CHW classifier tensor shaped "
              "[classes,1,1]");
    return false;
  }

  out.input_shape = input->shape;
  out.output_shape = output->shape;
  out.input_frac = input->frac;
  out.target_side = input->shape[1];
  out.class_count = output->shape[0];
  return true;
}

bool validate_evaluation_labels(
    const DatasetIndex &index, int class_count,
    std::vector<runtime::RuntimeDiagnostic> &diagnostics) {
  if (class_count <= 0) {
    add_error(diagnostics, "model class count must be positive");
    return false;
  }
  for (const auto &entry : index.entries) {
    if (entry.label >= class_count) {
      add_error(diagnostics, "label " + std::to_string(entry.label) + " for " +
                                 entry.name + " is outside model class range [0," +
                                 std::to_string(class_count - 1) + "]");
      return false;
    }
  }
  return true;
}

} // namespace dataset
} // namespace deeptile
