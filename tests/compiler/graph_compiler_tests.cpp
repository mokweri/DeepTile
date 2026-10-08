#include "deeptile/compiler/graph_compiler.h"
#include "deeptile/compiler/graph_loader.h"
#include "deeptile/compiler/schema_ids.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using deeptile::graph::GraphDiagnostic;
using deeptile::graph::GraphExecutionPlan;
using deeptile::graph::GraphSpec;

#define TEST_ASSERT(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "    ASSERT FAILED: " << msg << "\n";                      \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define TEST_ASSERT_EQ(a, b, msg)                                              \
  do {                                                                         \
    if ((a) != (b)) {                                                          \
      std::cerr << "    ASSERT FAILED: " << msg << " expected=" << (b)        \
                << " got=" << (a) << "\n";                                   \
      return false;                                                            \
    }                                                                          \
  } while (0)

namespace {

void print_diagnostics(const std::vector<GraphDiagnostic> &diagnostics) {
  for (const auto &d : diagnostics) {
    std::cerr << "    [" << deeptile::graph::severity_name(d.severity) << "] ";
    if (!d.context.empty()) {
      std::cerr << d.context << ": ";
    }
    std::cerr << d.message << "\n";
  }
}

std::filesystem::path find_repo_root() {
#ifdef DEEPTILE_REPO_ROOT
  return DEEPTILE_REPO_ROOT;
#else
  std::filesystem::path dir = std::filesystem::current_path();
  while (!dir.empty()) {
    if (std::filesystem::exists(dir / "data" / "phase1_testcases") &&
        std::filesystem::exists(dir / "src" / "compiler")) {
      return dir;
    }
    const std::filesystem::path parent = dir.parent_path();
    if (parent == dir) {
      break;
    }
    dir = parent;
  }
  return std::filesystem::current_path();
#endif
}

// `rel_dir` is a repo-relative directory under data/, e.g.
// "phase1_testcases/best_case_3x3_s1" or "resnet50_int8_tilecnn".
std::filesystem::path exported_graph_path(const std::string &rel_dir) {
  static const std::filesystem::path repo_root = find_repo_root();
  return repo_root / "data" / rel_dir / "graph.json";
}


bool load_and_compile(const std::string &path, GraphExecutionPlan &plan,
                      std::vector<GraphDiagnostic> &diagnostics) {
  GraphSpec graph;
  if (!deeptile::graph::load_graph(path, graph, diagnostics)) {
    return false;
  }
  plan = deeptile::graph::compile_graph(graph);
  diagnostics.insert(diagnostics.end(), plan.diagnostics.begin(),
                     plan.diagnostics.end());
  return !deeptile::graph::has_errors(diagnostics);
}

bool test_exported_case(const std::string &name, int expected_launches) {
  std::cout << "[TEST] exported " << name << "\n";
  GraphExecutionPlan plan;
  std::vector<GraphDiagnostic> diagnostics;
  const std::filesystem::path path = exported_graph_path(name);
  const bool ok = load_and_compile(path.string(), plan, diagnostics);
  if (!ok) {
    print_diagnostics(diagnostics);
  }
  TEST_ASSERT(ok, "compile should pass");
  TEST_ASSERT_EQ(static_cast<int>(plan.launches.size()), expected_launches,
                 "launch count");
  for (const auto &launch : plan.launches) {
    TEST_ASSERT(launch.schedule.valid, "schedule should be valid");
    TEST_ASSERT(!launch.schedule.compute_descs.empty(),
                "schedule should have compute descriptors");
    TEST_ASSERT(!launch.schedule.transfers.empty(),
                "schedule should have transfers");
    TEST_ASSERT(launch.tile_cfg.T_oh > 0 && launch.tile_cfg.T_ow > 0,
                "tile dims should be positive");
  }
  return true;
}


void write_bytes(const std::filesystem::path &path, int count) {
  std::ofstream out(path, std::ios::binary);
  for (int i = 0; i < count; ++i) {
    char v = 0;
    out.write(&v, 1);
  }
}

std::string valid_single_graph_json(const std::string &op = "conv2d",
                                    const std::string &extra_attrs = "",
                                    const std::string &extra_inputs = "",
                                    const std::string &output_shape = "[16,4,4]") {
  return std::string(R"JSON({
  "schema": "tilecnn.graph.v1",
  "model": {"name": "neg"},
  "target": {
    "bitwidth": 8,
    "signed": true,
    "activation_layout": "CHW",
    "conv_weight_layout": "OIHW",
    "linear_weight_layout": "OI",
    "bias_dtype": "int8"
  },
  "graph": {"inputs": ["input"], "outputs": ["out"]},
  "tensors": {
    "input": {"kind": "input", "dtype": "int8", "layout": "CHW", "shape": [16,4,4], "frac": 5, "file": "inputs/input.bin"},
    "w": {"kind": "param", "dtype": "int8", "layout": "OIHW", "shape": [16,16,3,3], "frac": 8, "file": "params/w.bin"},
    "b": {"kind": "param", "dtype": "int8", "layout": "O", "shape": [16], "frac": 3, "file": "params/b.bin"},
    "out": {"kind": "activation", "dtype": "int8", "layout": "CHW", "shape": )JSON") +
         output_shape + R"JSON(, "frac": 5}
  },
  "nodes": [{
    "id": "n0",
    "op": ")JSON" + op + R"JSON(",
    "inputs": {"ifm": "input", "weight": "w", "bias": "b")JSON" +
         extra_inputs + R"JSON(},
    "outputs": {"ofm": "out"},
    "attrs": {"kernel": [3,3], "stride": [1,1], "padding": [1,1,1,1], "dilation": [1,1], "groups": 1)JSON" +
         extra_attrs + R"JSON(}
  }]
})JSON";
}

std::filesystem::path make_temp_case(const std::string &json,
                                     bool write_all_files = true) {
  auto dir = std::filesystem::temp_directory_path() /
             ("deeptile_graph_test_" + std::to_string(std::rand()));
  std::filesystem::create_directories(dir / "inputs");
  std::filesystem::create_directories(dir / "params");
  std::ofstream(dir / "graph.json") << json;
  if (write_all_files) {
    write_bytes(dir / "inputs/input.bin", 16 * 4 * 4);
    write_bytes(dir / "params/w.bin", 16 * 16 * 3 * 3);
    write_bytes(dir / "params/b.bin", 16);
  }
  return dir;
}

bool expect_failure(const std::string &name, const std::string &json,
                    bool write_all_files = true) {
  std::cout << "[TEST] negative " << name << "\n";
  const auto dir = make_temp_case(json, write_all_files);
  GraphExecutionPlan plan;
  std::vector<GraphDiagnostic> diagnostics;
  const bool ok = load_and_compile((dir / "graph.json").string(), plan,
                                   diagnostics);
  std::filesystem::remove_all(dir);
  TEST_ASSERT(!ok, "case should fail");
  return true;
}

bool test_negative_cases() {
  TEST_ASSERT(expect_failure("missing file", valid_single_graph_json(), false),
              "missing file negative");
  TEST_ASSERT(expect_failure("unsupported op",
                             valid_single_graph_json("batchnorm")),
              "unsupported op negative");
  return true;
}

bool test_bad_residual_shape() {
  std::cout << "[TEST] negative bad_residual_shape\n";
  const std::string json = R"JSON({
  "schema": "tilecnn.graph.v1",
  "model": {"name": "neg"},
  "target": {"bitwidth": 8, "signed": true, "activation_layout": "CHW", "conv_weight_layout": "OIHW", "linear_weight_layout": "OI", "bias_dtype": "int8"},
  "graph": {"inputs": ["input", "res"], "outputs": ["out"]},
  "tensors": {
    "input": {"kind": "input", "dtype": "int8", "layout": "CHW", "shape": [16,4,4], "frac": 5, "file": "inputs/input.bin"},
    "res": {"kind": "input", "dtype": "int8", "layout": "CHW", "shape": [16,2,2], "frac": 5, "file": "inputs/res.bin"},
    "w": {"kind": "param", "dtype": "int8", "layout": "OIHW", "shape": [16,16,3,3], "frac": 8, "file": "params/w.bin"},
    "b": {"kind": "param", "dtype": "int8", "layout": "O", "shape": [16], "frac": 3, "file": "params/b.bin"},
    "out": {"kind": "activation", "dtype": "int8", "layout": "CHW", "shape": [16,4,4], "frac": 5}
  },
  "nodes": [{
    "id": "n0", "op": "conv2d",
    "inputs": {"ifm": "input", "weight": "w", "bias": "b", "residual": "res"},
    "outputs": {"ofm": "out"},
    "attrs": {"kernel": [3,3], "stride": [1,1], "padding": [1,1,1,1], "dilation": [1,1], "groups": 1},
    "post_ops": {"residual_add": true, "post_add_relu": true}
  }]
})JSON";
  const auto dir = make_temp_case(json, false);
  write_bytes(dir / "inputs/input.bin", 16 * 4 * 4);
  write_bytes(dir / "inputs/res.bin", 16 * 2 * 2);
  write_bytes(dir / "params/w.bin", 16 * 16 * 3 * 3);
  write_bytes(dir / "params/b.bin", 16);
  GraphExecutionPlan plan;
  std::vector<GraphDiagnostic> diagnostics;
  const bool ok = load_and_compile((dir / "graph.json").string(), plan,
                                   diagnostics);
  std::filesystem::remove_all(dir);
  TEST_ASSERT(!ok, "bad residual shape should fail");
  return true;
}

bool test_non_topological() {
  const std::string json = R"JSON({
  "schema": "tilecnn.graph.v1",
  "model": {"name": "neg"},
  "target": {"bitwidth": 8, "signed": true, "activation_layout": "CHW", "conv_weight_layout": "OIHW", "linear_weight_layout": "OI", "bias_dtype": "int8"},
  "graph": {"inputs": ["input"], "outputs": ["out2"]},
  "tensors": {
    "input": {"kind": "input", "dtype": "int8", "layout": "CHW", "shape": [16,4,4], "frac": 5, "file": "inputs/input.bin"},
    "mid": {"kind": "activation", "dtype": "int8", "layout": "CHW", "shape": [16,4,4], "frac": 5},
    "out2": {"kind": "activation", "dtype": "int8", "layout": "CHW", "shape": [16,4,4], "frac": 5},
    "w": {"kind": "param", "dtype": "int8", "layout": "OIHW", "shape": [16,16,3,3], "frac": 8, "file": "params/w.bin"},
    "b": {"kind": "param", "dtype": "int8", "layout": "O", "shape": [16], "frac": 3, "file": "params/b.bin"}
  },
  "nodes": [
    {"id": "n1", "op": "conv2d", "inputs": {"ifm": "mid", "weight": "w", "bias": "b"}, "outputs": {"ofm": "out2"}, "attrs": {"kernel": [3,3], "stride": [1,1], "padding": [1,1,1,1], "dilation": [1,1], "groups": 1}},
    {"id": "n0", "op": "conv2d", "inputs": {"ifm": "input", "weight": "w", "bias": "b"}, "outputs": {"ofm": "mid"}, "attrs": {"kernel": [3,3], "stride": [1,1], "padding": [1,1,1,1], "dilation": [1,1], "groups": 1}}
  ]
})JSON";
  return expect_failure("non_topological", json);
}

bool test_unsupported_maxpool() {
  std::cout << "[TEST] negative unsupported maxpool\n";
  const auto dir = std::filesystem::temp_directory_path() /
                   ("deeptile_graph_test_" + std::to_string(std::rand()));
  std::filesystem::create_directories(dir / "inputs");
  std::ofstream(dir / "graph.json") << R"JSON({
  "schema": "tilecnn.graph.v1",
  "model": {"name": "neg"},
  "target": {"bitwidth": 8, "signed": true, "activation_layout": "CHW", "conv_weight_layout": "OIHW", "linear_weight_layout": "OI", "bias_dtype": "int8"},
  "graph": {"inputs": ["input"], "outputs": ["out"]},
  "tensors": {
    "input": {"kind": "input", "dtype": "int8", "layout": "CHW", "shape": [16,4,4], "frac": 5, "file": "inputs/input.bin"},
    "out": {"kind": "activation", "dtype": "int8", "layout": "CHW", "shape": [16,2,2], "frac": 5}
  },
  "nodes": [{"id": "pool", "op": "maxpool2d", "inputs": {"ifm": "input"}, "outputs": {"ofm": "out"}, "attrs": {"kernel": [2,2], "stride": [2,2], "padding": [0,0,0,0]}}]
})JSON";
  write_bytes(dir / "inputs/input.bin", 16 * 4 * 4);
  GraphExecutionPlan plan;
  std::vector<GraphDiagnostic> diagnostics;
  const bool ok = load_and_compile((dir / "graph.json").string(), plan,
                                   diagnostics);
  std::filesystem::remove_all(dir);
  TEST_ASSERT(!ok, "unsupported maxpool should fail");
  return true;
}

bool test_standalone_gap() {
  std::cout << "[TEST] negative standalone_gap\n";
  const auto dir = std::filesystem::temp_directory_path() /
                   ("deeptile_graph_test_" + std::to_string(std::rand()));
  std::filesystem::create_directories(dir / "inputs");
  std::ofstream(dir / "graph.json") << R"JSON({
  "schema": "tilecnn.graph.v1",
  "model": {"name": "neg"},
  "target": {"bitwidth": 8, "signed": true, "activation_layout": "CHW", "conv_weight_layout": "OIHW", "linear_weight_layout": "OI", "bias_dtype": "int8"},
  "graph": {"inputs": ["input"], "outputs": ["out"]},
  "tensors": {
    "input": {"kind": "input", "dtype": "int8", "layout": "CHW", "shape": [16,4,4], "frac": 5, "file": "inputs/input.bin"},
    "out": {"kind": "activation", "dtype": "int8", "layout": "CHW", "shape": [16,1,1], "frac": 5}
  },
  "nodes": [{"id": "gap", "op": "gap2d", "inputs": {"ifm": "input"}, "outputs": {"ofm": "out"}}]
})JSON";
  write_bytes(dir / "inputs/input.bin", 16 * 4 * 4);
  GraphExecutionPlan plan;
  std::vector<GraphDiagnostic> diagnostics;
  const bool ok = load_and_compile((dir / "graph.json").string(), plan,
                                   diagnostics);
  std::filesystem::remove_all(dir);
  TEST_ASSERT(!ok, "standalone gap should fail");
  return true;
}

std::filesystem::path dw_case_path(const std::string &name) {
  static const std::filesystem::path repo_root = find_repo_root();
  return repo_root / "data" / "dw_testcases" / name / "graph.json";
}

bool compile_dw_case(const std::string &name, bool fuse,
                     GraphExecutionPlan &plan) {
  if (fuse) {
    setenv("DEEPTILE_FUSE_DW", "1", 1);
  } else {
    unsetenv("DEEPTILE_FUSE_DW");
  }
  std::vector<GraphDiagnostic> diagnostics;
  const bool ok = load_and_compile(dw_case_path(name).string(), plan,
                                   diagnostics);
  unsetenv("DEEPTILE_FUSE_DW");
  if (!ok) {
    print_diagnostics(diagnostics);
  }
  return ok;
}

// P5 Phase 1: the fusion pass produces the right launch structure and DW
// metadata (flag on), and is completely inert (flag off).
bool test_dw_fusion_structure() {
  std::cout << "[TEST] dw fusion structure\n";
  {  // Flag OFF: no fusion; producer + DW stay two separate launches.
    GraphExecutionPlan plan;
    TEST_ASSERT(compile_dw_case("pwdw_single", false, plan), "compile off");
    TEST_ASSERT_EQ(static_cast<int>(plan.launches.size()), 2,
                   "flag-off launch count");
    for (const auto &l : plan.launches) {
      TEST_ASSERT(!l.dw.enable, "flag off must not fuse");
    }
  }
  {  // Flag ON: single fused launch, DW metadata correct, producer stays dense.
    GraphExecutionPlan plan;
    TEST_ASSERT(compile_dw_case("pwdw_single", true, plan), "compile on");
    TEST_ASSERT_EQ(static_cast<int>(plan.launches.size()), 1,
                   "flag-on launch count");
    const auto &l = plan.launches[0];
    TEST_ASSERT(l.dw.enable, "launch is fused");
    TEST_ASSERT(!l.params.depthwise, "producer weight stays dense");
    TEST_ASSERT_EQ(l.dw.Kh, 3, "dw kernel");
    TEST_ASSERT_EQ(l.dw.Sh, 1, "dw stride");
    TEST_ASSERT_EQ(l.dw.shift_out, 6, "dw shift_out");
    TEST_ASSERT_EQ(l.dw.out_h, 16, "dw out height");
    TEST_ASSERT_EQ(l.dw.out_w, 16, "dw out width");
    TEST_ASSERT(l.ofm_tensor == "dw_out", "fused output is the DW output");
    TEST_ASSERT(l.schedule.valid, "fused schedule valid");
  }
  {  // Stride-2: DW output dims halved.
    GraphExecutionPlan plan;
    TEST_ASSERT(compile_dw_case("pwdw_stride2", true, plan), "compile s2");
    TEST_ASSERT_EQ(static_cast<int>(plan.launches.size()), 1, "s2 launches");
    const auto &l = plan.launches[0];
    TEST_ASSERT(l.dw.enable, "s2 fused");
    TEST_ASSERT_EQ(l.dw.Sh, 2, "s2 stride");
    TEST_ASSERT_EQ(l.dw.out_h, 8, "s2 out height");
    TEST_ASSERT_EQ(l.dw.out_w, 8, "s2 out width");
  }
  {  // V2 block: expand+DW fuse; the linear project stays its own launch.
    GraphExecutionPlan plan;
    TEST_ASSERT(compile_dw_case("pwdw_v2_block", true, plan), "compile v2");
    TEST_ASSERT_EQ(static_cast<int>(plan.launches.size()), 2, "v2 launches");
    TEST_ASSERT(plan.launches[0].dw.enable, "expand+dw fused");
    TEST_ASSERT(!plan.launches[1].dw.enable, "project not fused");
    TEST_ASSERT(plan.launches[1].residual_tensor.has_value(),
                "project keeps its residual");
  }
  return true;
}

// Fused 5x5 depthwise: kernel metadata, the 4-row halo, and seam geometry.
bool test_dw5_fusion_structure() {
  std::cout << "[TEST] dw 5x5 fusion structure\n";
  {
    GraphExecutionPlan plan;
    TEST_ASSERT(compile_dw_case("pwdw_k5_single", true, plan), "compile k5");
    TEST_ASSERT_EQ(static_cast<int>(plan.launches.size()), 1, "k5 launches");
    const auto &l = plan.launches[0];
    TEST_ASSERT(l.dw.enable, "k5 fused");
    TEST_ASSERT_EQ(l.dw.Kh, 5, "k5 kernel");
    TEST_ASSERT_EQ(l.post_ops.dw_kernel, 5, "k5 post-op kernel");
    TEST_ASSERT_EQ(l.dw.pad_top, 2, "k5 padding");
    TEST_ASSERT_EQ(l.tile_cfg.T_oh, 20, "producer rows include the 4-row halo");
    TEST_ASSERT_EQ(l.dw.out_h, 16, "k5 out height");
  }
  {  // Two row tiles of 28 output rows; the seam tile reads 2 rows either side.
    GraphExecutionPlan plan;
    TEST_ASSERT(compile_dw_case("pwdw_k5_rowtiled", true, plan), "compile rowtiled");
    const auto &l = plan.launches[0];
    TEST_ASSERT(l.dw.enable, "rowtiled fused");
    TEST_ASSERT_EQ(l.tile_cfg.T_oh, 32, "rowtiled T_oh");
    // Output rows 0-27 read conv rows 0-29 with 2 padded rows on top; output
    // rows 28-55 read conv rows 26-55 with 2 padded rows below.
    const auto &descs = l.schedule.compute_descs;
    TEST_ASSERT(!descs.empty(), "rowtiled descriptors");
    bool saw_top = false;
    bool saw_bottom = false;
    for (const auto &d : descs) {
      TEST_ASSERT_EQ(static_cast<int>(d.ofm_tile_h), 30, "conv rows per tile");
      TEST_ASSERT_EQ(static_cast<int>(d.pooled_ofm_tile_h), 28, "output rows per tile");
      saw_top = saw_top || d.pool_packed_padding == 0x2022;
      saw_bottom = saw_bottom || d.pool_packed_padding == 0x0222;
    }
    TEST_ASSERT(saw_top && saw_bottom, "halo padding on both seams");
  }
  {  // 72 channel groups still fit the tap store.
    GraphExecutionPlan plan;
    TEST_ASSERT(compile_dw_case("pwdw_k5_deep", true, plan), "compile deep");
    TEST_ASSERT_EQ(static_cast<int>(plan.launches.size()), 1, "deep launches");
    TEST_ASSERT(plan.launches[0].dw.enable, "deep fused");
  }
  return true;
}

// Rows wider than the line buffer tile by column; the producer's input tile
// bounds a strided stem.
bool test_dw_wide_fusion_structure() {
  std::cout << "[TEST] dw wide fusion structure\n";
  {  // 112-wide 1x1 producer, 3x3 stride-2 depthwise: two column tiles.
    GraphExecutionPlan plan;
    TEST_ASSERT(compile_dw_case("pwdw_wide_s2", true, plan), "compile wide_s2");
    TEST_ASSERT_EQ(static_cast<int>(plan.launches.size()), 1, "wide_s2 launches");
    const auto &l = plan.launches[0];
    TEST_ASSERT(l.dw.enable, "wide_s2 fused");
    TEST_ASSERT_EQ(l.tile_cfg.T_ow, 64, "conv tile is one line-buffer row");
    // Output columns 0-30 read conv columns 0-61 with one padded column on
    // the left; columns 31-55 read 61-111 and need no padding, since output
    // 55 is centred on column 110 of an even-width row.
    bool saw_left = false;
    bool saw_right = false;
    for (const auto &d : l.schedule.compute_descs) {
      const int pad_l = (d.pool_packed_padding >> 4) & 0xF;
      const int pad_r = d.pool_packed_padding & 0xF;
      TEST_ASSERT(d.ofm_tile_w <= 64, "conv tile fits the line buffer");
      saw_left = saw_left || (pad_l == 1 && pad_r == 0 && d.pooled_ofm_tile_w == 31 &&
                                     d.ofm_tile_w == 62);
      saw_right = saw_right || (pad_l == 0 && pad_r == 0 && d.pooled_ofm_tile_w == 25 &&
                                       d.ofm_tile_w == 51);
    }
    TEST_ASSERT(saw_left && saw_right, "column seam padding");
  }
  {  // A 3x3 stride-2 stem feeding 112-wide depthwise: bounded by its input.
    GraphExecutionPlan plan;
    TEST_ASSERT(compile_dw_case("pwdw_stem_wide", true, plan), "compile stem_wide");
    TEST_ASSERT_EQ(static_cast<int>(plan.launches.size()), 1, "stem_wide launches");
    const auto &l = plan.launches[0];
    TEST_ASSERT(l.dw.enable, "stem_wide fused");
    TEST_ASSERT_EQ(l.tile_cfg.T_oh, 18, "stem conv rows fit the padded input");
    TEST_ASSERT_EQ(l.tile_cfg.T_ow, 34, "stem conv columns fit the padded input");
  }
  return true;
}

// A pointwise producer feeding a same-padded depthwise layer of C channels.
std::string pwdw_graph(int channels, int kernel) {
  const std::string c = std::to_string(channels);
  const std::string k = std::to_string(kernel);
  const std::string p = std::to_string(kernel / 2);
  return R"JSON({
  "schema": "deeptile.graph.v1",
  "model": {"name": "pwdw_limit"},
  "target": {"bitwidth": 8, "signed": true, "activation_layout": "CHW", "conv_weight_layout": "OIHW", "linear_weight_layout": "OI", "bias_dtype": "int8"},
  "graph": {"inputs": ["input"], "outputs": ["dw_out"]},
  "tensors": {
    "input": {"kind": "input", "dtype": "int8", "layout": "CHW", "shape": [16,4,4], "frac": 5, "file": "inputs/input.bin"},
    "pw_out": {"kind": "activation", "dtype": "int8", "layout": "CHW", "shape": [)JSON" + c + R"JSON(,4,4], "frac": 4},
    "dw_out": {"kind": "activation", "dtype": "int8", "layout": "CHW", "shape": [)JSON" + c + R"JSON(,4,4], "frac": 4},
    "pw_w": {"kind": "param", "dtype": "int8", "layout": "OIHW", "shape": [)JSON" + c + R"JSON(,16,1,1], "frac": 7, "file": "params/pw_w.bin"},
    "pw_b": {"kind": "param", "dtype": "int8", "layout": "O", "shape": [)JSON" + c + R"JSON(], "frac": 5, "file": "params/pw_b.bin"},
    "dw_w": {"kind": "param", "dtype": "int8", "layout": "OIHW", "shape": [)JSON" + c + ",1," + k + "," + k + R"JSON(], "frac": 7, "file": "params/dw_w.bin"},
    "dw_b": {"kind": "param", "dtype": "int8", "layout": "O", "shape": [)JSON" + c + R"JSON(], "frac": 5, "file": "params/dw_b.bin"}
  },
  "nodes": [
    {"id": "pw", "op": "conv2d", "inputs": {"ifm": "input", "weight": "pw_w", "bias": "pw_b"}, "outputs": {"ofm": "pw_out"}, "attrs": {"kernel": [1,1], "stride": [1,1], "padding": [0,0,0,0], "dilation": [1,1], "groups": 1}, "post_ops": {"relu6": true}},
    {"id": "dw", "op": "conv2d", "inputs": {"ifm": "pw_out", "weight": "dw_w", "bias": "dw_b"}, "outputs": {"ofm": "dw_out"}, "attrs": {"kernel": [)JSON" + k + "," + k + R"JSON(], "stride": [1,1], "padding": [)JSON" + p + "," + p + "," + p + "," + p + R"JSON(], "dilation": [1,1], "groups": )JSON" + c + R"JSON(}, "post_ops": {"relu6": true}}
  ]
})JSON";
}

// Returns whether the depthwise layer fused into its producer.
bool pwdw_fuses(int channels, int kernel, bool &fused) {
  const auto dir = make_temp_case(pwdw_graph(channels, kernel), false);
  write_bytes(dir / "inputs/input.bin", 16 * 4 * 4);
  write_bytes(dir / "params/pw_w.bin", static_cast<std::size_t>(channels) * 16);
  write_bytes(dir / "params/pw_b.bin", static_cast<std::size_t>(channels));
  write_bytes(dir / "params/dw_w.bin",
              static_cast<std::size_t>(channels) * kernel * kernel);
  write_bytes(dir / "params/dw_b.bin", static_cast<std::size_t>(channels));
  setenv("DEEPTILE_FUSE_DW", "1", 1);
  GraphExecutionPlan plan;
  std::vector<GraphDiagnostic> diagnostics;
  const bool ok = load_and_compile((dir / "graph.json").string(), plan, diagnostics);
  unsetenv("DEEPTILE_FUSE_DW");
  std::filesystem::remove_all(dir);
  if (!ok) {
    print_diagnostics(diagnostics);
    return false;
  }
  fused = plan.launches.size() == 1 && plan.launches[0].dw.enable;
  return true;
}

// The tap store bounds fused 5x5 layers at 81 channel groups; 7x7 never fuses.
bool test_dw_fusion_limits() {
  std::cout << "[TEST] dw fusion limits\n";
  bool fused = false;
  TEST_ASSERT(pwdw_fuses(81 * 16, 5, fused), "compile 5x5 at the limit");
  TEST_ASSERT(fused, "81 groups of 5x5 fuse");
  TEST_ASSERT(pwdw_fuses(81 * 16 + 1, 5, fused), "compile 5x5 past the limit");
  TEST_ASSERT(!fused, "82 groups of 5x5 run unfused");
  TEST_ASSERT(pwdw_fuses(128 * 16, 3, fused), "compile 3x3 at the limit");
  TEST_ASSERT(fused, "128 groups of 3x3 fuse");
  TEST_ASSERT(pwdw_fuses(128 * 16 + 1, 3, fused), "compile 3x3 past the limit");
  TEST_ASSERT(!fused, "129 groups of 3x3 run unfused");
  TEST_ASSERT(pwdw_fuses(32, 7, fused), "compile 7x7");
  TEST_ASSERT(!fused, "7x7 runs unfused");
  return true;
}

} // namespace

// One 3x3 convolution whose output has frac 4, with the given post_ops block.
std::string single_conv_graph(const std::string &post_ops) {
  return std::string(R"JSON({
  "schema": "deeptile.graph.v1",
  "model": {"name": "post_ops"},
  "target": {"bitwidth": 8, "signed": true, "activation_layout": "CHW", "conv_weight_layout": "OIHW", "linear_weight_layout": "OI", "bias_dtype": "int8"},
  "graph": {"inputs": ["input"], "outputs": ["out"]},
  "tensors": {
    "input": {"kind": "input", "dtype": "int8", "layout": "CHW", "shape": [16,4,4], "frac": 5, "file": "inputs/input.bin"},
    "out": {"kind": "activation", "dtype": "int8", "layout": "CHW", "shape": [16,4,4], "frac": 4},
    "w": {"kind": "param", "dtype": "int8", "layout": "OIHW", "shape": [16,16,3,3], "frac": 8, "file": "params/w.bin"},
    "b": {"kind": "param", "dtype": "int8", "layout": "O", "shape": [16], "frac": 3, "file": "params/b.bin"}
  },
  "nodes": [
    {"id": "n0", "op": "conv2d", "inputs": {"ifm": "input", "weight": "w", "bias": "b"}, "outputs": {"ofm": "out"}, "attrs": {"kernel": [3,3], "stride": [1,1], "padding": [1,1,1,1], "dilation": [1,1], "groups": 1}, "post_ops": )JSON") +
         post_ops + "}\n  ]\n}";
}

// ReLU6 lowers to a clamped ReLU at round(6 * 2^frac_out) = 96 for frac 4.
bool test_relu6_lowers_to_a_clamp() {
  std::cout << "[TEST] relu6 clamp\n";
  const auto dir = make_temp_case(single_conv_graph(R"({"relu6": true})"), true);
  GraphExecutionPlan plan;
  std::vector<GraphDiagnostic> diagnostics;
  const bool ok = load_and_compile((dir / "graph.json").string(), plan, diagnostics);
  std::filesystem::remove_all(dir);
  if (!ok) print_diagnostics(diagnostics);
  TEST_ASSERT(ok, "relu6 graph compiles");
  TEST_ASSERT_EQ(plan.launches.size(), static_cast<std::size_t>(1), "one launch");
  TEST_ASSERT(plan.launches[0].params.use_relu, "relu6 enables the activation");
  TEST_ASSERT_EQ(plan.launches[0].params.act_max, 96, "relu6 clamp");
  return true;
}

// A post-op the lowering does not implement must not compile silently.
bool test_unknown_post_ops_are_rejected() {
  return expect_failure("unknown post_op", single_conv_graph(R"({"leaky_relu": true})")) &&
         expect_failure("non-scalar post_op", single_conv_graph(R"({"relu": "yes"})")) &&
         expect_failure("relu with relu6", single_conv_graph(R"({"relu": true, "relu6": true})"));
}

// Both namespaces name the same formats; only their exact prefixes match.
bool test_schema_namespaces() {
  namespace s = deeptile::schema;
  return s::matches("tilecnn.graph.v1", "graph.v1") &&
         s::matches("deeptile.graph.v1", "graph.v1") &&
         !s::matches("deeptile.graph.v2", "graph.v1") &&
         !s::matches("other.graph.v1", "graph.v1") &&
         !s::matches("deeptile.target.v1", "graph.v1") &&
         !s::matches("xdeeptile.graph.v1", "graph.v1");
}

int main() {
  int passed = 0;
  int failed = 0;
  const auto run = [&](const std::string &name, bool (*fn)()) {
    if (fn()) {
      ++passed;
    } else {
      ++failed;
      std::cerr << "[FAIL] " << name << "\n";
    }
  };

  // Exported-graph coverage. The single-layer cases keep the fast basic checks;
  // the two full networks subsume the retired conv_pool / residual_block /
  // residual_block_downsample / tail_gap_fc fixtures — ResNet-50 carries the
  // fused conv+maxpool stem, residual adds, downsample projections, GAP and FC,
  // and MobileNetV2 carries standalone depthwise lowering.
  run("schema_namespaces", test_schema_namespaces);
  run("relu6_clamp", test_relu6_lowers_to_a_clamp);
  run("unknown_post_ops", test_unknown_post_ops_are_rejected);
  run("single_conv_k3", [] {
    return test_exported_case("phase1_testcases/best_case_3x3_s1", 1);
  });
  run("single_conv_k1", [] {
    return test_exported_case("phase1_testcases/deep_k1_s1", 1);
  });
  run("proj_conv_k1_s2", [] {
    return test_exported_case("phase1_testcases/proj_bottleneck_1x1_s2", 1);
  });
  run("resnet50_full_graph", [] {
    return test_exported_case("resnet50_int8_tilecnn", 54);
  });
  run("mobilenetv2_full_graph", [] {
    return test_exported_case("mobilenet_v2_int8_tilecnn", 53);
  });
  run("basic_negative_cases", test_negative_cases);
  run("bad_residual_shape", test_bad_residual_shape);
  run("non_topological", test_non_topological);
  run("unsupported_maxpool", test_unsupported_maxpool);
  run("standalone_gap", test_standalone_gap);
  run("dw_fusion_structure", test_dw_fusion_structure);
  run("dw5_fusion_structure", test_dw5_fusion_structure);
  run("dw_fusion_limits", test_dw_fusion_limits);
  run("dw_wide_fusion_structure", test_dw_wide_fusion_structure);

  std::cout << "Graph compiler tests: " << passed << " passed, " << failed
            << " failed\n";
  return failed == 0 ? 0 : 1;
}
