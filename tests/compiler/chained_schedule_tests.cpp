/**
 * @file chained_schedule_tests.cpp
 * @brief Checks partial CIG chains against existing descriptor address limits.
 */

#include "deeptile/compiler/graph_compiler.h"
#include "deeptile/tile/descriptor_emitter.hpp"
#include "deeptile/tile/schedule_builder.hpp"

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <utility>

using namespace deeptile::tile;

namespace {
int failures = 0;
void check(bool valid, const char *message) {
  if (!valid) {
    std::cerr << "[FAIL] " << message << '\n';
    ++failures;
  }
}

void check_schedule(int cigs, bool spatial, bool chained) {
  ConvLayerParams p{};
  p.Cin = cigs * 16 - 3;
  p.Cout = 33;
  p.H = spatial ? 3 : 1;
  p.W = spatial ? 5 : 1;
  p.Kh = p.Kw = p.Sh = p.Sw = 1;
  p.Cpi = p.Cpo = 16;
  p.elem_bytes = 1;
  p.use_relu = true;
  TileConfig cfg{};
  cfg.T_oh = spatial ? 2 : 1;
  cfg.T_ow = spatial ? 3 : 1;
  cfg.T_cog = cfg.T_cig = 1;
  cfg.Cpi = cfg.Cpo = 16;
  ScheduleOptions opts;
  opts.enable_cig_chaining = chained;
  auto plan = build_schedule(p, cfg, HardwareSpec{}, opts);
  const int limit = chained ? kMaxChainedCacheCigs : kMaxUnchainedCacheCigs;
  check(plan.valid == (cigs <= limit), "cache schedule eligibility");
  if (!plan.valid) return;
  auto bundle = emit_descriptors(plan);
  check(bundle.valid, "descriptor emission");
  check(bundle.compute_descs.size() == plan.compute_tasks.size(), "one descriptor per task");
  std::map<std::pair<int, int>, int> covered;
  int writebacks = 0;
  for (std::size_t i = 0; i < plan.compute_tasks.size(); ++i) {
    const auto &task = plan.compute_tasks[i];
    const auto &desc = bundle.compute_descs[i];
    const auto &ep = plan.weight_epochs[task.epoch_idx];
    const auto &st = plan.spatial_tiles[task.spatial_idx];
    auto &next = covered[{task.spatial_idx, task.cog}];
    check(task.cig_begin == next, "CIG coverage is ordered without gaps/overlap");
    next += task.cig_count;
    check(task.cig_count > 0 && task.cig_count <= 255, "count fits eight bits");
    check(task.cig_begin < 128, "starting cache group fits seven bits");
    check(desc.num_cigs == task.cig_count, "count is not truncated");
    check(desc.cache_cig_idx == task.cig_begin, "cache starting group");
    check(desc.ifm_offset_vecs == static_cast<unsigned>(
        task.cig_begin * p.H * p.W + st.ifm_window.ih_start_actual * p.W +
        st.ifm_window.iw_start_actual), "partial-chain IFM starting plane");
    check(desc.uram_base_depth == (task.cog - ep.oc_tile_begin) * cigs +
        task.cig_begin, "partial-chain weight starting depth");
    check(bool(desc.flags & ACCUM_RESET) == (task.cig_begin == 0), "reset only at reduction start");
    check(bool(desc.flags & ACCUM_WRITE_BACK) == (next == cigs), "writeback only at reduction end");
    check(bool(desc.flags & RELU_ON) == (next == cigs), "post-op only at reduction end");
    const bool fill = task.cog == ep.oc_tile_begin && (spatial || task.epoch_idx == 0);
    check(bool(desc.reuse_ifm) == !fill, "both chunks share cache fill/reuse policy");
    writebacks += bool(desc.flags & ACCUM_WRITE_BACK);
  }
  for (const auto &entry : covered) check(entry.second == cigs, "complete CIG reduction");
  check(writebacks == p.num_cog() * static_cast<int>(plan.spatial_tiles.size()), "one output per spatial/OC tile");
  std::size_t count = 0;
  for (const auto &transfer : bundle.transfers) count += transfer.compute_count;
  check(count == bundle.compute_descs.size(), "transfer counts cover all partial chains");
}

void check_graph_policy(int cigs, bool disable_chain, bool disable_cache) {
  using namespace deeptile::graph;
  setenv("DEEPTILE_DISABLE_CIG_CHAIN", disable_chain ? "1" : "0", 1);
  setenv("DEEPTILE_DISABLE_K1_CACHE", disable_cache ? "1" : "0", 1);
  GraphSpec graph;
  graph.inputs = {"input"};
  graph.outputs = {"output"};
  graph.tensors["input"] = {"input", "input", "int8", "CHW", {cigs * 16, 1, 1}, 4, {}};
  graph.tensors["weights"] = {"weights", "param", "int8", "OI", {33, cigs * 16}, 4, {}};
  graph.tensors["bias"] = {"bias", "param", "int8", "O", {33}, 4, {}};
  graph.tensors["output"] = {"output", "activation", "int8", "CHW", {33, 1, 1}, 4, {}};
  NodeSpec node;
  node.id = "fc";
  node.op = "linear";
  node.inputs = {{"ifm", "input"}, {"weight", "weights"}, {"bias", "bias"}};
  node.outputs = {{"ofm", "output"}};
  graph.nodes = {node};
  const auto plan = compile_graph(graph);
  check(!has_errors(plan.diagnostics) && plan.launches.size() == 1, "wide graph compiles or falls back safely");
  if (plan.launches.empty()) return;
  const int limit = disable_chain ? kMaxUnchainedCacheCigs : kMaxChainedCacheCigs;
  check(plan.launches[0].use_ifm_cache == (!disable_cache && cigs <= limit), "graph cache policy honors field limits and overrides");
  for (const auto &d : plan.launches[0].schedule.compute_descs)
    check(d.num_cigs != 0, "no wrapped zero-length chain");
}
} // namespace

int main() {
  unsetenv("DEEPTILE_FORCE_EPOCHS");
  unsetenv("DEEPTILE_PACK_EPOCHS");
  for (int cigs : {1, 2, 127, 128, 129, 254, 255, 256, 257, 381, 382, 383}) {
    for (bool spatial : {false, true}) {
      check_schedule(cigs, spatial, true);
      check_schedule(cigs, spatial, false);
    }
    for (bool disable_chain : {false, true})
      for (bool disable_cache : {false, true})
        check_graph_policy(cigs, disable_chain, disable_cache);
  }
  if (failures) return EXIT_FAILURE;
  std::cout << "Partial-chain limits, addresses, reductions, cache policy and fallbacks pass\n";
  return EXIT_SUCCESS;
}
