/**
 * @file graph_schedule_main.cpp
 * @brief CLI for inspecting graph launch schedules and descriptors.
 */

#include "deeptile/compiler/graph_compiler.h"
#include "deeptile/compiler/graph_loader.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_usage(const char *argv0) {
  std::cerr << "Usage: " << argv0 << " <graph.json>\n";
}

void print_diagnostics(
    const std::vector<deeptile::graph::GraphDiagnostic> &diags) {
  for (const auto &d : diags) {
    std::cerr << "[" << deeptile::graph::severity_name(d.severity) << "] ";
    if (!d.context.empty()) std::cerr << d.context << ": ";
    std::cerr << d.message << "\n";
  }
}

// Decode descriptor flags for tabular output.
std::string decode_flags(uint8_t f) {
  std::string s;
  if (f & static_cast<uint8_t>(deeptile::tile::ACCUM_RESET))      s += "RST|";
  if (f & static_cast<uint8_t>(deeptile::tile::ACCUM_WRITE_BACK)) s += "WB|";
  if (f & static_cast<uint8_t>(deeptile::tile::RELU_ON))          s += "RELU|";
  if (f & static_cast<uint8_t>(deeptile::tile::IS_LAST_TILE))     s += "LAST|";
  if (f & static_cast<uint8_t>(deeptile::tile::ADD_ENABLE))       s += "ADD|";
  if (f & static_cast<uint8_t>(deeptile::tile::POST_ADD_RELU))    s += "PADD_RELU|";
  if (f & static_cast<uint8_t>(deeptile::tile::POOL_ENABLE))      s += "POOL|";
  if (f & static_cast<uint8_t>(deeptile::tile::GAP_ENABLE))       s += "GAP|";
  if (!s.empty()) s.pop_back();
  return s.empty() ? "0" : s;
}

void inspect_launch(const deeptile::graph::LaunchPlan &launch,
                    std::ostream &out) {
  const auto &p  = launch.params;
  const auto &tc = launch.tile_cfg;
  const auto &sc = launch.schedule;

  out << "══════════════════════════════════════════════════════════\n";
  out << "  Launch: " << launch.id << "\n";
  out << "  shape:  " << p.Cin << "×" << p.H << "×" << p.W
      << " → " << p.Cout
      << "×" << p.output_height() << "×" << p.output_width()
      << "  K=" << p.Kh << "  stride=" << p.Sh << "\n";
  out << "  tile:   T_oh=" << tc.T_oh << " T_ow=" << tc.T_ow
      << "  Cpi=" << tc.Cpi << " Cpo=" << tc.Cpo
      << "  ifm_cache=" << (launch.use_ifm_cache ? "yes" : "no") << "\n";

  const int total_descs = static_cast<int>(sc.compute_descs.size());
  const int total_xfers = static_cast<int>(sc.transfers.size());

  // Classify the descriptor chaining mode.
  int n_chained = 0;
  uint8_t chained_num_cigs = 0;
  for (const auto &d : sc.compute_descs) {
    if (d.num_cigs > 1) { ++n_chained; chained_num_cigs = d.num_cigs; }
  }

  out << "  schedule: transfers=" << total_xfers
      << "  descriptors=" << total_descs;
  if (n_chained == total_descs && n_chained > 0) {
    out << "  [CIG-chained  num_cigs=" << static_cast<int>(chained_num_cigs) << "]";
  } else if (n_chained > 0) {
    out << "  [mixed: " << n_chained << " chained  "
        << (total_descs - n_chained) << " legacy]";
  } else {
    out << "  [legacy: 1 CIG per descriptor]";
  }
  out << "\n\n";

  const char *hdr =
      "   idx  R cigs  uram_base  bias   ifm_off   ofm_off  flags";
  const char *sep =
      "  ----- - ---- ---------- ----- ---------- ---------- ---------";

  int desc_cursor = 0;
  int xfer_idx = 0;
  for (const auto &xfer : sc.transfers) {
    const char *buf_id = (xfer.buffer_id == 0) ? "A" : "B";
    out << "  Transfer " << xfer_idx++
        << "  buf=" << buf_id
        << "  depth=" << xfer.depth_count
        << "  compute=" << xfer.compute_count;
    if (xfer.flags & 0x1) out << "  [last-xfer]";
    out << "\n" << hdr << "\n" << sep << "\n";

    for (int i = 0; i < xfer.compute_count; ++i) {
      const auto &d =
          sc.compute_descs[static_cast<std::size_t>(desc_cursor + i)];
      out << "  " << std::setw(5) << (desc_cursor + i)
          << "  " << (d.reuse_ifm ? 'r' : 'F')
          << " " << std::setw(4) << static_cast<int>(d.num_cigs)
          << " " << std::setw(10) << d.uram_base_depth
          << " " << std::setw(5) << d.bias_offset_idx
          << " " << std::setw(10) << d.ifm_offset_vecs
          << " " << std::setw(10) << d.ofm_offset_vecs
          << "  " << decode_flags(d.flags)
          << "\n";
    }
    desc_cursor += xfer.compute_count;
    out << "\n";
  }

  int fresh = 0;
  for (const auto &d : sc.compute_descs) {
    if (!d.reuse_ifm) ++fresh;
  }
  out << "  IFM: " << fresh << " fresh  " << (total_descs - fresh) << " cached\n\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    print_usage(argv[0]);
    return 2;
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

  deeptile::graph::GraphExecutionPlan plan =
      deeptile::graph::compile_graph(graph);
  if (!plan.diagnostics.empty()) print_diagnostics(plan.diagnostics);
  if (deeptile::graph::has_errors(plan.diagnostics)) return 1;

  for (const auto &launch : plan.launches) {
    inspect_launch(launch, std::cout);
  }

  return 0;
}
