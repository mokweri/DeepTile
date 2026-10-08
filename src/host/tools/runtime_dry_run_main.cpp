/**
 * @file runtime_dry_run_main.cpp
 * @brief CLI for inspecting a prepared runtime plan without device execution.
 */

#include "runtime/runtime.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using deeptile::runtime::RuntimeDiagnostic;
using deeptile::runtime::RuntimeExecutionPlan;

namespace {

void print_diagnostics(const std::vector<RuntimeDiagnostic> &diagnostics) {
  for (const auto &d : diagnostics) {
    std::cerr << "[" << deeptile::runtime::severity_name(d.severity) << "] ";
    if (!d.context.empty()) {
      std::cerr << d.context << ": ";
    }
    std::cerr << d.message << "\n";
  }
}

std::size_t total_raw_bytes(const RuntimeExecutionPlan &plan) {
  std::size_t out = 0;
  for (const auto &buffer : plan.buffers) {
    out += buffer.raw_data.size();
  }
  return out;
}

std::size_t total_packed_bytes(const RuntimeExecutionPlan &plan) {
  std::size_t out = 0;
  for (const auto &buffer : plan.buffers) {
    out += buffer.data_bytes.size();
  }
  return out;
}

std::size_t total_i32_bytes(const RuntimeExecutionPlan &plan) {
  std::size_t out = 0;
  for (const auto &buffer : plan.buffers) {
    out += buffer.int32_data.size() * sizeof(std::int32_t);
  }
  return out;
}

void print_plan(const RuntimeExecutionPlan &plan) {
  const auto &graph = plan.graph_plan.graph;
  std::cout << "DeepTile runtime dry run\n";
  std::cout << "  graph: " << graph.graph_path << "\n";
  std::cout << "  schema: " << graph.schema << "\n";
  std::cout << "  tensors: " << graph.tensors.size()
            << " nodes: " << graph.nodes.size()
            << " launches: " << plan.launches.size() << "\n";
  std::cout << "  buffers: " << plan.buffers.size()
            << " raw_bytes=" << total_raw_bytes(plan)
            << " packed_bytes=" << total_packed_bytes(plan)
            << " int32_param_bytes=" << total_i32_bytes(plan) << "\n";
  std::cout << "  inputs:";
  for (const auto &id : graph.inputs) {
    std::cout << " " << id;
  }
  std::cout << "\n  outputs:";
  for (const auto &id : graph.outputs) {
    std::cout << " " << id;
  }
  std::cout << "\n\n";
  deeptile::runtime::print_activation_memory_plan(plan, std::cout);
  std::cout << "\n";

  for (std::size_t i = 0; i < plan.launches.size(); ++i) {
    const auto &launch = plan.launches[i];
    std::cout << "Launch " << i << ": " << launch.id << "\n";
    std::cout << "  source_nodes:";
    for (const auto &id : launch.source_nodes) {
      std::cout << " " << id;
    }
    std::cout << "\n";
    std::cout << "  buffers: ifm=" << launch.ifm_buffer
              << " weight=" << launch.weight_buffer
              << " bias=" << launch.bias_buffer
              << " output=" << launch.output_buffer;
    if (launch.residual_buffer.has_value()) {
      std::cout << " residual=" << *launch.residual_buffer;
    }
    std::cout << "\n";
    std::cout << "  bytes: ifm=" << launch.ifm_bytes
              << " weight=" << launch.weight_bytes
              << " bias=" << launch.bias_bytes
              << " output=" << launch.output_bytes;
    if (launch.residual_buffer.has_value()) {
      std::cout << " residual=" << launch.residual_bytes;
    }
    std::cout << "\n";
    std::cout << "  scalars: shift_out=" << launch.shift_out
              << " shift_bias=" << launch.shift_bias
              << " residual_shift=" << launch.residual_shift
              << " gap_mul=" << launch.gap_mul
              << " num_bias_vecs=" << launch.num_bias_vecs << "\n";
    std::cout << "  descriptors: transfers=" << launch.num_xfers
              << " compute=" << launch.num_tiles << "\n\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <path/to/graph.json>\n";
    return 2;
  }

  RuntimeExecutionPlan plan;
  std::vector<RuntimeDiagnostic> diagnostics;
  if (!deeptile::runtime::load_compile_prepare_runtime(argv[1], plan,
                                                      diagnostics)) {
    print_diagnostics(diagnostics);
    return 1;
  }
  print_plan(plan);
  return 0;
}
