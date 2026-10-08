/**
 * @file ifm_traffic_main.cpp
 * @brief Derives scheduled IFM DDR traffic from a compiled graph.
 *
 * This intentionally counts transfer words only. It does not use cost-model
 * service cycles, which include AXI latency and request-overlap effects.
 */

#include "deeptile/compiler/environment.h"
#include "deeptile/compiler/graph_compiler.h"
#include "deeptile/compiler/graph_loader.h"

#include "deeptile/tile/schedule_builder.hpp"
#include "deeptile/tile/schedule_options.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

// Baseline uses the 128-bit loader; final and P0–P3 use the 256-bit loader.
// P0–P3 vary only scheduling policy.
enum class TrafficPolicy { Baseline, Final, P0, P1, P2, P3 };

// Only the baseline policy models 128-bit IFM words.
bool uses_legacy_128bit_loader(TrafficPolicy policy) {
  return policy == TrafficPolicy::Baseline;
}

struct TrafficCount {
  std::uint64_t fresh_descriptors = 0;
  std::uint64_t cached_descriptors = 0;
  std::uint64_t read_bursts = 0;
  std::uint64_t read_words = 0;
  std::uint64_t word_bytes = 0;
};

int ceil_div(int value, int divisor) {
  return divisor > 0 ? (value + divisor - 1) / divisor : 0;
}

void print_diagnostics(
    const std::vector<deeptile::graph::GraphDiagnostic> &diagnostics) {
  for (const auto &diagnostic : diagnostics) {
    std::cerr << "[" << deeptile::graph::severity_name(diagnostic.severity)
              << "] ";
    if (!diagnostic.context.empty()) {
      std::cerr << diagnostic.context << ": ";
    }
    std::cerr << diagnostic.message << "\n";
  }
}

bool parse_policy(const std::string &text, TrafficPolicy &policy) {
  if (text == "baseline") { policy = TrafficPolicy::Baseline; return true; }
  if (text == "final")    { policy = TrafficPolicy::Final;    return true; }
  if (text == "p0")       { policy = TrafficPolicy::P0;       return true; }
  if (text == "p1")       { policy = TrafficPolicy::P1;       return true; }
  if (text == "p2")       { policy = TrafficPolicy::P2;       return true; }
  if (text == "p3")       { policy = TrafficPolicy::P3;       return true; }
  return false;
}

deeptile::tile::ScheduleOptions schedule_options(
    TrafficPolicy policy, bool launch_cache_eligible) {
  using deeptile::tile::SchedulePolicy;
  using deeptile::tile::options_for_policy;

  if (policy == TrafficPolicy::Baseline) {
    return options_for_policy(SchedulePolicy::Baseline);
  }

  // Final and P3 select the same adaptive schedule.
  const bool adaptive_epochs =
      policy == TrafficPolicy::Final || policy == TrafficPolicy::P3;
  const bool want_cache = policy != TrafficPolicy::P0;
  const bool want_chain = policy != TrafficPolicy::P0 &&
                          policy != TrafficPolicy::P1;

  auto options = options_for_policy(adaptive_epochs
                                        ? SchedulePolicy::DynamicWeightEpochs
                                        : SchedulePolicy::Phase2Default);
  options.optimize_epoch_depth = adaptive_epochs;
  // Policy may disable caching but cannot make an ineligible layer cacheable.
  options.enable_k1_cache = want_cache && launch_cache_eligible;
  options.enable_cig_chaining = want_chain && launch_cache_eligible;
  return options;
}

TrafficCount count_traffic(const deeptile::tile::SchedulePlan &plan,
                           TrafficPolicy policy) {
  TrafficCount count;
  const bool baseline = uses_legacy_128bit_loader(policy);
  const bool k1_cache = !baseline && plan.stats.uses_k1_cache &&
                        plan.layer.Kh == 1 && plan.layer.Kw == 1;
  count.word_bytes = baseline ? 16U : 32U;

  for (const auto &task : plan.compute_tasks) {
    const auto &tile =
        plan.spatial_tiles[static_cast<std::size_t>(task.spatial_idx)];
    const auto &window = tile.ifm_window;
    const std::uint64_t cigs =
        static_cast<std::uint64_t>(task.cig_count > 0 ? task.cig_count : 1);

    if (k1_cache && task.reuse_ifm) {
      ++count.cached_descriptors;
      continue;
    }
    ++count.fresh_descriptors;

    if (baseline) {
      // Historical kernel: one 128-bit vector for every boundary-clipped
      // element in every scheduled row. Zero padding was synthesized on-chip.
      count.read_bursts +=
          static_cast<std::uint64_t>(window.read_rows) * cigs;
      count.read_words += static_cast<std::uint64_t>(window.read_rows) *
                          static_cast<std::uint64_t>(window.read_cols) * cigs;
    } else if (k1_cache) {
      // Current K=1 fill loop: one 256-bit word holds two 128-bit channel
      // vectors. It visits conv output rows and advances the IFM row by stride.
      const std::uint64_t rows =
          static_cast<std::uint64_t>(tile.conv_tile_oh);
      const std::uint64_t words_per_row =
          static_cast<std::uint64_t>(ceil_div(window.read_cols, 2));
      count.read_bursts += rows * cigs;
      count.read_words += rows * words_per_row * cigs;
    } else {
      // Current raw loader: boundary-clipped rows, rounded to whole 256-bit
      // words. Padding is reconstructed while streaming from the on-chip tile.
      const std::uint64_t rows =
          static_cast<std::uint64_t>(window.read_rows);
      const std::uint64_t words_per_row =
          static_cast<std::uint64_t>(ceil_div(window.read_cols, 2));
      count.read_bursts += rows * cigs;
      count.read_words += rows * words_per_row * cigs;
    }
  }
  return count;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr
        << "Usage: " << argv[0]
        << " <path/to/graph.json> <baseline|final|p0|p1|p2|p3>\n\n"
        << "  baseline  historical 128-bit IFM loader, unoptimized schedule\n"
        << "  final     current 256-bit loader, shipped schedule\n"
        << "  p0..p3    schedule-policy ablation, all on the current 256-bit\n"
        << "            loader: p0 no reuse/no chain, p1 +reuse, p2 +chain,\n"
        << "            p3 +adaptive prefetch (== final schedule)\n";
    return 2;
  }

  TrafficPolicy policy;
  if (!parse_policy(argv[2], policy)) {
    std::cerr << "Unknown traffic policy: " << argv[2] << "\n";
    return 2;
  }

  // Reject environment overrides because the explicit policy is authoritative.
  for (const char *var : {"DISABLE_K1_CACHE", "DISABLE_CIG_CHAIN"}) {
    if (deeptile::env_flag(var)) {
      std::cerr << "Refusing to run: DEEPTILE_" << var << " is set, but this tool takes "
                   "the schedule policy as an argument. Unset it and pass "
                   "p0/p1/p2/p3 instead.\n";
      return 2;
    }
  }

  deeptile::graph::GraphSpec graph;
  std::vector<deeptile::graph::GraphDiagnostic> diagnostics;
  const bool loaded = deeptile::graph::load_graph(argv[1], graph, diagnostics);
  if (!diagnostics.empty()) {
    print_diagnostics(diagnostics);
  }
  if (!loaded) {
    return 1;
  }

  deeptile::graph::GraphExecutionPlan compiled =
      deeptile::graph::compile_graph(graph);
  if (!compiled.diagnostics.empty()) {
    print_diagnostics(compiled.diagnostics);
  }
  if (deeptile::graph::has_errors(compiled.diagnostics)) {
    return 1;
  }

  std::cout << "launch_index,layer_name,fresh_descriptors,cached_descriptors,"
               "ifm_read_bursts,ifm_read_words,axi_word_bytes,"
               "scheduled_ifm_ddr_read_bytes\n";

  for (std::size_t index = 0; index < compiled.launches.size(); ++index) {
    const auto &launch = compiled.launches[index];
    const auto options = schedule_options(policy, launch.use_ifm_cache);
    const auto plan = deeptile::tile::build_schedule(
        launch.params, launch.tile_cfg, deeptile::tile::HardwareSpec{}, options,
        0, 0, 0, launch.post_ops);
    if (!plan.valid) {
      std::cerr << "Invalid rebuilt schedule for launch " << index << " ("
                << launch.id << ")\n";
      for (const auto &issue : plan.issues) {
        std::cerr << "  " << issue << "\n";
      }
      return 1;
    }

    const TrafficCount count = count_traffic(plan, policy);
    const std::uint64_t bytes = count.read_words * count.word_bytes;
    std::cout << index << ',' << launch.id << ',' << count.fresh_descriptors
              << ',' << count.cached_descriptors << ',' << count.read_bursts
              << ',' << count.read_words << ',' << count.word_bytes << ','
              << bytes << '\n';
  }
  return 0;
}
