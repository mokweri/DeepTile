#include "deeptile/compiler/graph_compiler.h"
#include "deeptile/compiler/graph_loader.h"
#include "deeptile/tile/hardware_spec.hpp"
#include "runtime/runtime.h"
#include "reference/runtime_reference.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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

namespace hw = deeptile::tile::hw_defaults;

using deeptile::graph::GraphDiagnostic;
using deeptile::graph::GraphExecutionPlan;
using deeptile::graph::GraphSpec;
using deeptile::runtime::RuntimeDiagnostic;
using deeptile::runtime::RuntimeExecutionPlan;

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

std::filesystem::path dry_run_tool_path() {
  const std::filesystem::path local =
      std::filesystem::current_path() / "runtime_dry_run";
  if (std::filesystem::exists(local)) {
    return local;
  }
  return find_repo_root() / "build" / "tools" / "runtime_dry_run";
}

void print_runtime_diagnostics(
    const std::vector<RuntimeDiagnostic> &diagnostics) {
  for (const auto &d : diagnostics) {
    std::cerr << "    [" << deeptile::runtime::severity_name(d.severity)
              << "] ";
    if (!d.context.empty()) {
      std::cerr << d.context << ": ";
    }
    std::cerr << d.message << "\n";
  }
}

bool load_and_compile(const std::filesystem::path &path,
                      GraphExecutionPlan &plan) {
  GraphSpec graph;
  std::vector<GraphDiagnostic> diagnostics;
  if (!deeptile::graph::load_graph(path.string(), graph, diagnostics)) {
    return false;
  }
  plan = deeptile::graph::compile_graph(graph);
  return !deeptile::graph::has_errors(plan.diagnostics);
}

const deeptile::runtime::RuntimeTensorBuffer *
find_buffer(const RuntimeExecutionPlan &plan, const std::string &id) {
  const auto it = plan.buffer_index_by_id.find(id);
  if (it == plan.buffer_index_by_id.end()) {
    return nullptr;
  }
  return &plan.buffers[it->second];
}

std::filesystem::path dw_testcase_graph_path(const std::string &name) {
  static const std::filesystem::path repo_root = find_repo_root();
  return repo_root / "data" / "dw_testcases" / name / "graph.json";
}

// Compile+prepare+reference a graph with DEEPTILE_FUSE_DW forced on/off and
// return the first graph output's bytes (plus whether a fused launch appeared).
bool reference_output_for(const std::string &graph_path, bool fuse,
                          std::vector<std::int8_t> &out, int &num_launches,
                          bool &has_fused_launch) {
  if (fuse) {
    setenv("DEEPTILE_FUSE_DW", "1", 1);
  } else {
    unsetenv("DEEPTILE_FUSE_DW");
  }
  RuntimeExecutionPlan plan;
  std::vector<RuntimeDiagnostic> diagnostics;
  const bool ok = deeptile::runtime::load_compile_prepare_runtime(
      graph_path, plan, diagnostics);
  unsetenv("DEEPTILE_FUSE_DW");
  if (!ok) {
    print_runtime_diagnostics(diagnostics);
    return false;
  }
  num_launches = static_cast<int>(plan.launches.size());
  has_fused_launch = false;
  for (const auto &l : plan.graph_plan.launches) {
    if (l.dw.enable) {
      has_fused_launch = true;
    }
  }
  const auto reference = deeptile::runtime::generate_bit_exact_reference(plan);
  if (deeptile::runtime::has_errors(reference.diagnostics)) {
    print_runtime_diagnostics(reference.diagnostics);
    return false;
  }
  const auto &out_id = plan.graph_plan.graph.outputs.front();
  const auto *t = deeptile::runtime::find_reference_tensor(reference, out_id);
  if (!t) {
    return false;
  }
  out = t->data_chw;
  return true;
}

// The master gate: fused (DEEPTILE_FUSE_DW=1) must be byte-identical to unfused
// (the two/three-launch decomposition) on the final graph output.
bool test_dw_fusion_equivalence(const std::string &name) {
  std::cout << "[TEST] dw fusion equivalence " << name << "\n";
  const std::string path = dw_testcase_graph_path(name).string();
  std::vector<std::int8_t> unfused, fused;
  int nl_off = 0, nl_on = 0;
  bool has_off = false, has_on = false;
  TEST_ASSERT(reference_output_for(path, false, unfused, nl_off, has_off),
              "unfused reference");
  TEST_ASSERT(reference_output_for(path, true, fused, nl_on, has_on),
              "fused reference");
  TEST_ASSERT(!has_off, "flag off must not produce a fused launch");
  TEST_ASSERT(has_on, "flag on must produce a fused launch");
  TEST_ASSERT(nl_on < nl_off, "fusion must reduce launch count");
  TEST_ASSERT_EQ(fused.size(), unfused.size(), "fused/unfused output size");
  TEST_ASSERT(fused == unfused, "fused output byte-identical to unfused");
  return true;
}

// Round-trip the compact DW blob: unpack per §4.3 layout and compare to the raw
// depthwise weights and widened biases.
bool test_dw_compact_roundtrip(const std::string &name) {
  std::cout << "[TEST] dw compact packer " << name << "\n";
  setenv("DEEPTILE_FUSE_DW", "1", 1);
  RuntimeExecutionPlan plan;
  std::vector<RuntimeDiagnostic> diagnostics;
  const bool ok = deeptile::runtime::load_compile_prepare_runtime(
      dw_testcase_graph_path(name).string(), plan, diagnostics);
  unsetenv("DEEPTILE_FUSE_DW");
  if (!ok) {
    print_runtime_diagnostics(diagnostics);
  }
  TEST_ASSERT(ok, "prepare with fusion enabled");
  const int cpo = hw::kOcTileSize;
  const int word_bytes = 16;
  int checked = 0;
  for (std::size_t i = 0; i < plan.launches.size(); ++i) {
    const auto &binding = plan.launches[i];
    const auto &launch = plan.graph_plan.launches[i];
    if (!launch.dw.enable) {
      continue;
    }
    ++checked;
    const auto *blob = find_buffer(plan, binding.dw_weight_buffer);
    const auto *w_raw = find_buffer(plan, launch.dw.weight_tensor);
    const auto *b_raw = find_buffer(plan, launch.dw.bias_tensor);
    TEST_ASSERT(blob && !blob->data_bytes.empty(), "compact blob present");
    TEST_ASSERT(w_raw && b_raw, "DW raw weight/bias present");
    const int C = launch.params.Cout;
    const int taps = launch.dw.Kh * launch.dw.Kw;
    // Each COG block is padded to an EVEN, burst-clean stride to match the
    // kernel's DW_WORDS_PER_COG_3/5 (arch_config.h): 16 words for 3x3 and 32
    // for 5x5. Layout: words [0,taps) = tap rows, [taps, taps+4) = bias words,
    // trailing words = zero pad. The earlier odd 13-word stride (taps +
    // bias_words) made the gmem6 preload an odd-length m_axi run that dropped
    // the trailing bias words on silicon.
    const int bias_words = (cpo * 4 + word_bytes - 1) / word_bytes;
    const int words_per_cog = taps == 25 ? 32 : 16;
    TEST_ASSERT(taps == 9 || taps == 25, "fused depthwise is 3x3 or 5x5");
    TEST_ASSERT(taps + bias_words <= words_per_cog,
                "COG payload must fit the stride");
    TEST_ASSERT_EQ(binding.dw_config >> 4 & 1, taps == 25 ? 1 : 0,
                   "dw_config selects the kernel size");
    const int oc_tiles = (C + cpo - 1) / cpo;
    TEST_ASSERT_EQ(blob->data_bytes.size(),
                   static_cast<std::size_t>(oc_tiles) * words_per_cog *
                       word_bytes,
                   "compact blob byte size");
    TEST_ASSERT_EQ(binding.dw_weight_bytes, blob->data_bytes.size(),
                   "binding dw_weight_bytes");
    for (int c = 0; c < C; ++c) {
      const int cog = c / cpo;
      const int lane = c % cpo;
      const std::size_t base =
          static_cast<std::size_t>(cog) * words_per_cog * word_bytes;
      for (int k = 0; k < taps; ++k) {
        const auto got = static_cast<std::int8_t>(
            blob->data_bytes[base + static_cast<std::size_t>(k) * word_bytes +
                             lane]);
        const auto exp =
            w_raw->raw_data[static_cast<std::size_t>(c) * taps + k];
        TEST_ASSERT_EQ(static_cast<int>(got), static_cast<int>(exp),
                       "unpacked DW weight byte");
      }
      std::int32_t exp_b = b_raw->raw_data[c];
      exp_b = launch.dw.shift_bias >= 0 ? (exp_b << launch.dw.shift_bias)
                                        : (exp_b >> -launch.dw.shift_bias);
      const std::size_t off =
          base + static_cast<std::size_t>(taps + lane / 4) * word_bytes +
          static_cast<std::size_t>(lane % 4) * 4;
      const std::int32_t got_b = static_cast<std::int32_t>(
          static_cast<std::uint32_t>(blob->data_bytes[off]) |
          (static_cast<std::uint32_t>(blob->data_bytes[off + 1]) << 8) |
          (static_cast<std::uint32_t>(blob->data_bytes[off + 2]) << 16) |
          (static_cast<std::uint32_t>(blob->data_bytes[off + 3]) << 24));
      TEST_ASSERT_EQ(got_b, exp_b, "unpacked DW bias int32");
    }
  }
  TEST_ASSERT(checked > 0, "at least one fused launch was checked");
  return true;
}

bool test_exported_case(const std::string &name, int expected_launches) {
  std::cout << "[TEST] runtime exported " << name << "\n";
  RuntimeExecutionPlan plan;
  std::vector<RuntimeDiagnostic> diagnostics;
  const bool ok = deeptile::runtime::load_compile_prepare_runtime(
      exported_graph_path(name).string(), plan, diagnostics);
  if (!ok) {
    print_runtime_diagnostics(diagnostics);
  }
  TEST_ASSERT(ok, "runtime prepare should pass");
  TEST_ASSERT_EQ(static_cast<int>(plan.launches.size()), expected_launches,
                 "runtime launch count");
  TEST_ASSERT_EQ(static_cast<int>(plan.graph_plan.launches.size()),
                 expected_launches, "compiler launch count");
  TEST_ASSERT(!plan.activation_slots.empty(), "activation slots planned");
  TEST_ASSERT(!plan.activation_bindings.empty(),
              "activation tensor bindings planned");
  TEST_ASSERT(plan.activation_slots.size() <= plan.activation_bindings.size(),
              "activation slots do not exceed logical activations");

  for (std::size_t i = 0; i < plan.launches.size(); ++i) {
    const auto &binding = plan.launches[i];
    const auto &launch = plan.graph_plan.launches[i];
    const auto *ifm = find_buffer(plan, binding.ifm_buffer);
    const auto *weight = find_buffer(plan, binding.weight_buffer);
    const auto *bias = find_buffer(plan, binding.bias_buffer);
    const auto *output = find_buffer(plan, binding.output_buffer);
    const auto *ifm_storage =
        deeptile::runtime::find_activation_binding(plan, binding.ifm_buffer);
    const auto *output_storage =
        deeptile::runtime::find_activation_binding(plan, binding.output_buffer);

    TEST_ASSERT(ifm && !ifm->data_bytes.empty(), "IFM buffer prepared");
    TEST_ASSERT(weight && !weight->data_bytes.empty(), "weight buffer prepared");
    TEST_ASSERT(bias && !bias->int32_data.empty(), "bias buffer prepared");
    TEST_ASSERT(output && !output->data_bytes.empty(), "output buffer prepared");
    TEST_ASSERT_EQ(binding.ifm_bytes, ifm->data_bytes.size(), "IFM byte count");
    TEST_ASSERT_EQ(binding.weight_bytes, weight->data_bytes.size(),
                   "weight byte count");
    TEST_ASSERT_EQ(binding.bias_bytes,
                   bias->int32_data.size() * sizeof(std::int32_t),
                   "bias byte count");
    TEST_ASSERT_EQ(binding.output_bytes, output->data_bytes.size(),
                   "output byte count");
    TEST_ASSERT(ifm_storage != nullptr, "IFM activation slot binding");
    TEST_ASSERT(output_storage != nullptr, "output activation slot binding");
    TEST_ASSERT_EQ(deeptile::runtime::activation_tensor_bytes(
                       plan, binding.output_buffer),
                   output->data_bytes.size(), "output slot byte count");
    TEST_ASSERT(ifm_storage->slot_id != output_storage->slot_id,
                "launch output does not overwrite same-launch IFM slot");
    TEST_ASSERT_EQ(weight->data_bytes.size(),
                   deeptile::runtime::packed_weight_bytes(
                       launch.params, hw::kOcTileSize,
                       hw::kIcTileSize),
                   "packed weight bytes");
    TEST_ASSERT_EQ(bias->int32_data.size(),
                   deeptile::runtime::shifted_bias_count(
                       launch.params.Cout, hw::kOcTileSize),
                   "shifted bias count");
    TEST_ASSERT_EQ(binding.transfer_descs.size(),
                   launch.schedule.transfers.size(), "transfer desc count");
    TEST_ASSERT_EQ(binding.compute_descs.size(),
                   launch.schedule.compute_descs.size(), "compute desc count");
    if (binding.residual_buffer.has_value()) {
      const auto *residual = find_buffer(plan, *binding.residual_buffer);
      const auto *residual_storage =
          deeptile::runtime::find_activation_binding(
              plan, *binding.residual_buffer);
      TEST_ASSERT(residual && !residual->data_bytes.empty(),
                  "residual buffer prepared");
      TEST_ASSERT(residual_storage != nullptr,
                  "residual activation slot binding");
      TEST_ASSERT(residual_storage->slot_id != output_storage->slot_id,
                  "launch output does not overwrite same-launch residual slot");
    }
  }

  if (name == "residual_block") {
    const auto *input_storage = deeptile::runtime::find_activation_binding(
        plan, plan.graph_plan.graph.inputs[0]);
    TEST_ASSERT(input_storage != nullptr, "residual block input storage");
    TEST_ASSERT_EQ(input_storage->producer_launch, -1,
                   "residual block input producer");
    TEST_ASSERT_EQ(input_storage->last_consumer_launch, 2,
                   "residual block input lives until skip add");
  }

  if (name == "residual_block_downsample") {
    const auto *input_storage = deeptile::runtime::find_activation_binding(
        plan, plan.graph_plan.graph.inputs[0]);
    TEST_ASSERT(input_storage != nullptr, "downsample block input storage");
    TEST_ASSERT_EQ(input_storage->producer_launch, -1,
                   "downsample block input producer");
    TEST_ASSERT_EQ(input_storage->last_consumer_launch, 1,
                   "downsample block input feeds main and projection branches");
    const auto *projection_storage =
        deeptile::runtime::find_activation_binding(
            plan, "layer2_0_downsample_0_out");
    TEST_ASSERT(projection_storage != nullptr,
                "downsample projection output storage");
    TEST_ASSERT_EQ(projection_storage->last_consumer_launch, 3,
                   "projection output lives until residual add");
  }

  const deeptile::runtime::RuntimeReferenceResult reference =
      deeptile::runtime::generate_bit_exact_reference(plan);
  TEST_ASSERT(!deeptile::runtime::has_errors(reference.diagnostics),
              "bit-exact reference generation should pass");
  for (const auto &output_id : plan.graph_plan.graph.outputs) {
    const auto *tensor =
        deeptile::runtime::find_reference_tensor(reference, output_id);
    TEST_ASSERT(tensor != nullptr, "final output reference exists");
    TEST_ASSERT_EQ(tensor->data_chw.size(),
                   static_cast<std::size_t>(tensor->shape[0]) *
                       tensor->shape[1] * tensor->shape[2],
                   "reference output size");
  }

  const std::string command = "\"" + dry_run_tool_path().string() + "\" \"" +
                              exported_graph_path(name).string() +
                              "\" >/dev/null";
  TEST_ASSERT_EQ(std::system(command.c_str()), 0, "dry-run CLI exits 0");
  return true;
}

bool expect_prepare_failure(const std::string &name,
                            const GraphExecutionPlan &graph_plan) {
  std::cout << "[TEST] runtime negative " << name << "\n";
  RuntimeExecutionPlan runtime =
      deeptile::runtime::prepare_runtime(graph_plan);
  TEST_ASSERT(deeptile::runtime::has_errors(runtime.diagnostics),
              "runtime prepare should fail");
  return true;
}

bool test_missing_producer_activation() {
  GraphExecutionPlan plan;
  TEST_ASSERT(load_and_compile(exported_graph_path("phase1_testcases/best_case_3x3_s1"), plan),
              "compile fixture");
  plan.launches[0].ifm_tensor = "__missing_ifm";
  return expect_prepare_failure("missing producer activation", plan);
}

bool test_bad_activation_layout() {
  GraphExecutionPlan plan;
  TEST_ASSERT(load_and_compile(exported_graph_path("phase1_testcases/best_case_3x3_s1"), plan),
              "compile fixture");
  plan.graph.tensors[plan.graph.inputs[0]].layout = "HWC";
  return expect_prepare_failure("bad activation layout", plan);
}

bool test_mismatched_output_shape() {
  GraphExecutionPlan plan;
  TEST_ASSERT(load_and_compile(exported_graph_path("phase1_testcases/best_case_3x3_s1"), plan),
              "compile fixture");
  plan.graph.tensors[plan.launches[0].ofm_tensor].shape = {1, 1, 1};
  return expect_prepare_failure("mismatched output shape", plan);
}

void write_bytes(const std::filesystem::path &path, int count) {
  std::ofstream out(path, std::ios::binary);
  for (int i = 0; i < count; ++i) {
    char v = 0;
    out.write(&v, 1);
  }
}

bool test_missing_parameter_file() {
  std::cout << "[TEST] runtime negative missing parameter file\n";
  const auto dir = std::filesystem::temp_directory_path() /
                   ("deeptile_runtime_test_" + std::to_string(std::rand()));
  std::filesystem::create_directories(dir / "inputs");
  std::filesystem::create_directories(dir / "params");
  std::ofstream(dir / "graph.json") << R"JSON({
  "schema": "tilecnn.graph.v1",
  "model": {"name": "missing_param"},
  "target": {"bitwidth": 8, "signed": true, "activation_layout": "CHW", "conv_weight_layout": "OIHW", "linear_weight_layout": "OI", "bias_dtype": "int8"},
  "graph": {"inputs": ["input"], "outputs": ["out"]},
  "tensors": {
    "input": {"kind": "input", "dtype": "int8", "layout": "CHW", "shape": [16,4,4], "frac": 5, "file": "inputs/input.bin"},
    "w": {"kind": "param", "dtype": "int8", "layout": "OIHW", "shape": [16,16,3,3], "frac": 8, "file": "params/w.bin"},
    "b": {"kind": "param", "dtype": "int8", "layout": "O", "shape": [16], "frac": 3, "file": "params/b.bin"},
    "out": {"kind": "activation", "dtype": "int8", "layout": "CHW", "shape": [16,4,4], "frac": 5}
  },
  "nodes": [{"id": "n0", "op": "conv2d", "inputs": {"ifm": "input", "weight": "w", "bias": "b"}, "outputs": {"ofm": "out"}, "attrs": {"kernel": [3,3], "stride": [1,1], "padding": [1,1,1,1], "dilation": [1,1], "groups": 1}}]
})JSON";
  write_bytes(dir / "inputs/input.bin", 16 * 4 * 4);
  write_bytes(dir / "params/b.bin", 16);

  RuntimeExecutionPlan plan;
  std::vector<RuntimeDiagnostic> diagnostics;
  const bool ok = deeptile::runtime::load_compile_prepare_runtime(
      (dir / "graph.json").string(), plan, diagnostics);
  std::filesystem::remove_all(dir);
  TEST_ASSERT(!ok, "missing parameter file should fail");
  return true;
}

bool test_residual_tensor_not_prepared() {
  GraphExecutionPlan plan;
  TEST_ASSERT(load_and_compile(exported_graph_path("phase1_testcases/best_case_3x3_s1"), plan),
              "compile fixture");
  plan.launches[0].residual_tensor = plan.launches[0].ofm_tensor;
  return expect_prepare_failure("residual tensor not prepared", plan);
}

} // namespace

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
  // P5 Phase 1 — fused PW→DW (DEEPTILE_FUSE_DW): fused ≡ unfused byte-identical.
  run("dw_equiv_pwdw_single",
      [] { return test_dw_fusion_equivalence("pwdw_single"); });
  run("dw_equiv_pwdw_stride2",
      [] { return test_dw_fusion_equivalence("pwdw_stride2"); });
  run("dw_equiv_pwdw_ragged",
      [] { return test_dw_fusion_equivalence("pwdw_ragged"); });
  run("dw_equiv_pwdw_rowtiled",
      [] { return test_dw_fusion_equivalence("pwdw_rowtiled"); });
  run("dw_equiv_pwdw_after_k3",
      [] { return test_dw_fusion_equivalence("pwdw_after_k3"); });
  run("dw_equiv_pwdw_v2_block",
      [] { return test_dw_fusion_equivalence("pwdw_v2_block"); });
  run("dw_packer_pwdw_single",
      [] { return test_dw_compact_roundtrip("pwdw_single"); });
  run("dw_packer_pwdw_ragged",
      [] { return test_dw_compact_roundtrip("pwdw_ragged"); });
  run("dw_packer_pwdw_v2_block",
      [] { return test_dw_compact_roundtrip("pwdw_v2_block"); });
  for (const char *name : {"pwdw_k5_single", "pwdw_k5_stride2", "pwdw_k5_ragged",
                           "pwdw_k5_rowtiled", "pwdw_k5_rowtiled_s2", "pwdw_k5_wide",
                           "pwdw_k5_deep", "pwdw_k5_block"}) {
    const std::string case_name = name;
    if (test_dw_fusion_equivalence(case_name)) {
      ++passed;
    } else {
      ++failed;
      std::cerr << "[FAIL] dw_equiv_" << case_name << "\n";
    }
  }
  run("dw_packer_pwdw_k5_ragged",
      [] { return test_dw_compact_roundtrip("pwdw_k5_ragged"); });
  run("dw_packer_pwdw_k5_deep",
      [] { return test_dw_compact_roundtrip("pwdw_k5_deep"); });

  run("missing_producer_activation", test_missing_producer_activation);
  run("bad_activation_layout", test_bad_activation_layout);
  run("mismatched_output_shape", test_mismatched_output_shape);
  run("missing_parameter_file", test_missing_parameter_file);
  run("residual_tensor_not_prepared", test_residual_tensor_not_prepared);

  std::cout << "Runtime tests: " << passed << " passed, " << failed
            << " failed\n";
  return failed == 0 ? 0 : 1;
}
