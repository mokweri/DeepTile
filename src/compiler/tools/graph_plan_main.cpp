#include <algorithm>
/**
 * @file graph_plan_main.cpp
 * @brief CLI for compiling and summarizing a graph execution plan.
 */

#include "deeptile/compiler/graph_compiler.h"
#include "deeptile/compiler/graph_loader.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using deeptile::graph::DiagnosticSeverity;
using deeptile::graph::GraphDiagnostic;
using deeptile::graph::GraphExecutionPlan;
using deeptile::graph::GraphSpec;
using deeptile::graph::LaunchPlan;

namespace {

void print_diagnostics(const std::vector<GraphDiagnostic> &diagnostics) {
  for (const auto &d : diagnostics) {
    std::cerr << "[" << deeptile::graph::severity_name(d.severity) << "] ";
    if (!d.context.empty()) {
      std::cerr << d.context << ": ";
    }
    std::cerr << d.message << "\n";
  }
}

std::string flags_for(const LaunchPlan &launch) {
  std::string flags;
  if (launch.params.use_relu) {
    flags += " relu";
  }
  if (launch.post_ops.residual_add_enable) {
    flags += " residual_add";
  }
  if (launch.post_ops.post_add_relu) {
    flags += " post_add_relu";
  }
  if (launch.post_ops.maxpool_enable) {
    flags += " maxpool";
  }
  if (launch.post_ops.gap_enable) {
    flags += " gap";
  }
  return flags.empty() ? " none" : flags;
}

void print_launch(const LaunchPlan &launch, std::size_t index) {
  const auto &p = launch.params;
  const auto &cfg = launch.tile_cfg;
  std::cout << "Launch " << index << ": " << launch.id << "\n";
  std::cout << "  source_nodes:";
  for (const auto &id : launch.source_nodes) {
    std::cout << " " << id;
  }
  std::cout << "\n";
  std::cout << "  tensors: ifm=" << launch.ifm_tensor
            << " weight=" << launch.weight_tensor
            << " bias=" << launch.bias_tensor
            << " ofm=" << launch.ofm_tensor;
  if (launch.residual_tensor.has_value()) {
    std::cout << " residual=" << *launch.residual_tensor;
  }
  std::cout << "\n";
  std::cout << "  shape: " << p.Cin << "x" << p.H << "x" << p.W << " -> "
            << p.Cout << "x" << p.output_height() << "x"
            << p.output_width() << " k=" << p.Kh << "x" << p.Kw
            << " stride=" << p.Sh << " pad=(" << p.pad_top << ","
            << p.pad_bottom << "," << p.pad_left << "," << p.pad_right
            << ")\n";
  std::cout << "  tile: T_oh=" << cfg.T_oh << " T_ow=" << cfg.T_ow
            << " T_cig=" << cfg.T_cig << " T_cog=" << cfg.T_cog
            << " Cpi=" << cfg.Cpi << " Cpo=" << cfg.Cpo << "\n";
  std::cout << "  quant: frac_in=" << p.frac_in << " frac_w=" << p.frac_w
            << " frac_b=" << p.frac_b << " frac_out=" << p.frac_out
            << " shift_out=" << launch.shift_out
            << " shift_bias=" << launch.shift_bias;
  if (launch.residual_tensor.has_value()) {
    std::cout << " residual_shift=" << launch.post_ops.residual_shift;
  }
  if (launch.post_ops.gap_enable) {
    std::cout << " gap_mul=" << launch.gap_mul;
  }
  std::cout << "\n";
  std::cout << "  post_ops:" << flags_for(launch) << "\n";
  // Schedule policy actually applied to this launch, so an ablation run can be
  // confirmed from the plan dump rather than inferred from descriptor counts.
  const bool cig_chained =
      std::any_of(launch.schedule.compute_descs.begin(),
                  launch.schedule.compute_descs.end(),
                  [](const deeptile::tile::TileDescriptor &d) {
                    return d.num_cigs > 1;
                  });
  std::cout << "  schedule: transfers=" << launch.schedule.transfers.size()
            << " compute_tiles=" << launch.schedule.compute_descs.size()
            << " ifm_cache=" << (launch.use_ifm_cache ? "on" : "off")
            << " cig_chain=" << (cig_chained ? "on" : "off")
            << "\n";
}

void print_plan(const GraphExecutionPlan &plan) {
  std::cout << "DeepTile graph plan\n";
  std::cout << "  graph: " << plan.graph.graph_path << "\n";
  std::cout << "  schema: " << plan.graph.schema << "\n";
  std::cout << "  tensors: " << plan.graph.tensors.size()
            << " nodes: " << plan.graph.nodes.size()
            << " launches: " << plan.launches.size() << "\n";
  std::cout << "  inputs:";
  for (const auto &id : plan.graph.inputs) {
    std::cout << " " << id;
  }
  std::cout << "\n  outputs:";
  for (const auto &id : plan.graph.outputs) {
    std::cout << " " << id;
  }
  std::cout << "\n\n";

  for (std::size_t i = 0; i < plan.launches.size(); ++i) {
    print_launch(plan.launches[i], i);
    std::cout << "\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <path/to/graph.json>\n";
    return 2;
  }

  GraphSpec graph;
  std::vector<GraphDiagnostic> diagnostics;
  const bool loaded = deeptile::graph::load_graph(argv[1], graph, diagnostics);
  if (!diagnostics.empty()) {
    print_diagnostics(diagnostics);
  }
  if (!loaded) {
    return 1;
  }

  GraphExecutionPlan plan = deeptile::graph::compile_graph(graph);
  if (!plan.diagnostics.empty()) {
    print_diagnostics(plan.diagnostics);
  }
  if (deeptile::graph::has_errors(plan.diagnostics)) {
    return 1;
  }

  print_plan(plan);
  return 0;
}
