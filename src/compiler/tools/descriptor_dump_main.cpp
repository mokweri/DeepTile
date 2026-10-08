/**
 * @file descriptor_dump_main.cpp
 * @brief Emits size-tagged descriptor bytes for same-machine compiler checks.
 *
 * Link this inspection tool against each compiler library, then compare its
 * binary stdout. This is an inspection format, not a deployment-plan ABI.
 */

#include "deeptile/compiler/graph_compiler.h"
#include "deeptile/compiler/graph_loader.h"

#include <cstdint>
#include <iostream>
#include <vector>

template <typename T> void dump(const std::vector<T> &items) {
  const std::uint64_t count = items.size();
  std::cout.write(reinterpret_cast<const char *>(&count), sizeof(count));
  if (!items.empty())
    std::cout.write(reinterpret_cast<const char *>(items.data()),
                    items.size() * sizeof(T));
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " graph.json > descriptors.bin\n";
    return 2;
  }
  deeptile::graph::GraphSpec graph;
  std::vector<deeptile::graph::GraphDiagnostic> diagnostics;
  if (!deeptile::graph::load_graph(argv[1], graph, diagnostics)) return 1;
  const auto plan = deeptile::graph::compile_graph(graph);
  if (deeptile::graph::has_errors(plan.diagnostics)) return 1;
  const std::uint64_t count = plan.launches.size();
  std::cout.write(reinterpret_cast<const char *>(&count), sizeof(count));
  for (const auto &launch : plan.launches) {
    dump(launch.schedule.transfers);
    dump(launch.schedule.compute_descs);
  }
  return std::cout ? 0 : 1;
}
