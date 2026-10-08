/**
 * @file run_record_tests.cpp
 * @brief Tests for the minimal DeepTile RunRecord writer.
 */

#include "runtime/run_record.h"

#include "deeptile/compiler/target_package.h"

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include <chrono>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#define TEST_ASSERT(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "    ASSERT FAILED: " << msg << "\n";                       \
      return false;                                                            \
    }                                                                          \
  } while (0)

namespace {

namespace fs = std::filesystem;
using deeptile::runtime::DatasetRunSummary;
using deeptile::runtime::RunRecordContext;
using deeptile::runtime::SingleRunSummary;

fs::path test_root() {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  return fs::temp_directory_path() /
         ("deeptile-run-record-tests-" + std::to_string(stamp));
}

bool read_json(const fs::path &path, rapidjson::Document &document) {
  std::ifstream input(path);
  if (!input) {
    return false;
  }
  rapidjson::IStreamWrapper stream(input);
  document.ParseStream(stream);
  return !document.HasParseError() && document.IsObject();
}

RunRecordContext context() {
  RunRecordContext value;
  value.model = "resnet50/imagenet1k/int8-tqt@v1.0.0";
  value.target = "zcu104-p4-pp@v1.0.0";
  value.source_revision = "97a4308";
  value.source_dirty = false;
  value.launches = 54;
  return value;
}

bool test_single_record(const fs::path &root) {
  SingleRunSummary summary;
  summary.passed = true;
  summary.warmup_runs = 1;
  summary.measured_runs = 3;
  summary.comparison_performed = true;
  summary.exact_reference_match = true;
  summary.compared_elements = 1000;
  summary.kernel_ms = 27.784;
  summary.kernel_gops = 294.352;
  summary.launch_profile = "launch-profile.csv";

  std::string error;
  const fs::path path = root / "single" / "run.json";
  TEST_ASSERT(deeptile::runtime::write_single_run_record(
                  path.string(), context(), summary, error),
              error);

  rapidjson::Document record;
  TEST_ASSERT(read_json(path, record), "cannot parse single RunRecord");
  TEST_ASSERT(std::string(record["schema"].GetString()) == "deeptile.run.v1",
              "wrong schema");
  TEST_ASSERT(std::string(record["software"]["deeptile_revision"].GetString()) ==
                  "97a4308",
              "source revision missing");
  TEST_ASSERT(std::string(record["mode"].GetString()) == "single",
              "wrong single mode");
  TEST_ASSERT(record["execution"]["launches"].GetUint64() == 54,
              "wrong launch count");
  TEST_ASSERT(record["results"]["exact_reference_match"].GetBool(),
              "exact comparison missing");
  TEST_ASSERT(record["results"]["compared_elements"].GetUint64() == 1000,
              "wrong comparison size");
  TEST_ASSERT(std::string(
                  record["artifacts"]["launch_profile"].GetString()) ==
                  "launch-profile.csv",
              "wrong profile artifact");
  TEST_ASSERT(!record["results"].HasMember("graph_wall_ms"),
              "legacy window must not be reported as input-to-output latency");
  TEST_ASSERT(std::string(record["results"]["kernel_timing_scope"].GetString()) ==
                  "host_launch_and_wait",
              "host timing must not be labeled device time");
  return true;
}

bool test_latency_record(const fs::path &root) {
  SingleRunSummary summary;
  summary.passed = true;
  summary.measure_latency = true;
  summary.graph_wall_ms = 8.5;
  summary.graph_profile = "graph-profile.csv";
  std::string error;
  const auto path = root / "latency" / "run.json";
  TEST_ASSERT(deeptile::runtime::write_single_run_record(
                  path.string(), context(), summary, error), error);
  rapidjson::Document record;
  TEST_ASSERT(read_json(path, record), "cannot parse latency record");
  TEST_ASSERT(record["results"]["graph_wall_ms"].GetDouble() == 8.5,
              "missing complete graph latency");
  TEST_ASSERT(std::string(record["results"]["graph_timing_scope"].GetString()) ==
                  "input_upload_through_output_download",
              "graph timing boundary must be explicit");
  TEST_ASSERT(std::string(record["artifacts"]["graph_profile"].GetString()) ==
                  "graph-profile.csv", "missing per-inference samples");
  return true;
}

bool test_dataset_record(const fs::path &root) {
  RunRecordContext run_context = context();
  run_context.model = "mobilenet_v2/imagenet1k/int8-tqt-cle@v1.0.0";
  run_context.launches = 38;
  run_context.fused_depthwise = true;
  run_context.launch_mode = "direct";
  run_context.digests.model_sha256 = std::string(64, 'a');
  run_context.digests.xclbin_sha256 = std::string(64, 'b');
  run_context.digests.boot_sha256 = std::string(64, 'c');

  DatasetRunSummary summary;
  summary.passed = true;
  summary.dataset = "data/imagenet";
  summary.limit = 100;
  summary.workers = 2;
  summary.queue_depth = 4;
  summary.top_k = 5;
  summary.preprocessing = "eval";
  summary.short_side = 256;
  summary.images = 100;
  summary.top1_percent = 77.0;
  summary.topk_percent = 91.0;
  summary.kernel_ms_per_image = 8.145;
  summary.kernel_ms_median = 8.130;
  summary.kernel_ms_min = 7.950;
  summary.wall_ms_per_image = 13.181;
  summary.images_per_second = 75.864;
  summary.accelerator_busy_percent = 61.79;
  summary.graph_ops = 601600000.0;
  summary.avg_kernel_gops = 73.86;
  summary.peak_kernel_gops = 75.67;
  summary.end_to_end_gops = 45.64;
  summary.dataset_sha256 = std::string(64, 'd');
  summary.predictions = "predictions.csv";

  std::string error;
  const fs::path path = root / "dataset" / "run.json";
  TEST_ASSERT(deeptile::runtime::write_dataset_run_record(
                  path.string(), run_context, summary, error),
              error);

  rapidjson::Document record;
  TEST_ASSERT(read_json(path, record), "cannot parse dataset RunRecord");
  TEST_ASSERT(std::string(record["mode"].GetString()) == "dataset",
              "wrong dataset mode");
  TEST_ASSERT(record["execution"]["fused_depthwise"].GetBool(),
              "fused-depthwise setting missing");
  TEST_ASSERT(std::string(record["execution"]["launch_mode"].GetString()) ==
                  "direct",
              "launch mode missing");
  TEST_ASSERT(record["execution"]["workers"].GetInt() == 2,
              "wrong worker count");
  TEST_ASSERT(std::string(record["execution"]["tie_policy"].GetString()) ==
                  "lowest-class-index",
              "wrong tie policy");
  TEST_ASSERT(record["results"]["images"].GetUint64() == 100,
              "wrong image count");
  // The record must state which artifacts ran, not merely name them.
  TEST_ASSERT(record.HasMember("identity"), "identity block missing");
  TEST_ASSERT(std::string(record["identity"]["model_sha256"].GetString()) ==
                  std::string(64, 'a'),
              "model digest missing");
  TEST_ASSERT(std::string(record["identity"]["xclbin_sha256"].GetString()) ==
                  std::string(64, 'b'),
              "xclbin digest missing");
  TEST_ASSERT(std::string(record["identity"]["boot_sha256"].GetString()) ==
                  std::string(64, 'c'),
              "boot digest missing");
  TEST_ASSERT(std::string(record["execution"]["dataset_sha256"].GetString()) ==
                  std::string(64, 'd'),
              "dataset digest missing");
  TEST_ASSERT(!record["execution"].HasMember("schedule_flags"),
              "default lowering must not emit schedule_flags");
  TEST_ASSERT(record["results"]["top1_percent"].GetDouble() == 77.0,
              "wrong top-1 result");
  // Both timing scopes must be recorded: a lone throughput number cannot be
  // interpreted without its measurement boundary.
  TEST_ASSERT(record["results"]["avg_kernel_gops"].GetDouble() == 73.86,
              "average kernel throughput missing");
  TEST_ASSERT(record["results"]["peak_kernel_gops"].GetDouble() == 75.67,
              "peak kernel throughput missing");
  TEST_ASSERT(record["results"]["end_to_end_gops"].GetDouble() == 45.64,
              "end-to-end throughput missing");
  TEST_ASSERT(std::string(record["results"]["op_convention"].GetString()) ==
                  "1 MAC = 2 ops",
              "operation convention missing");
  TEST_ASSERT(record["results"]["kernel_ms_min"].GetDouble() == 7.950,
              "kernel-time minimum missing");
  TEST_ASSERT(record["results"]["accelerator_busy_percent"].GetDouble() == 61.79,
              "accelerator busy fraction missing");
  TEST_ASSERT(std::string(record["artifacts"]["predictions"].GetString()) ==
                  "predictions.csv",
              "wrong predictions artifact");
  return true;
}

bool test_context_identity(const fs::path &root) {
  const fs::path package = root / "package";
  fs::create_directories(package);
  std::ofstream(package / "manifest.json")
      << R"({"model":{"release_id":"resnet50/imagenet1k/int8-tqt@v1.0.0"}})";

  deeptile::runtime::RuntimeExecutionPlan plan;
  plan.graph_plan.graph.root_dir = package.string();
  plan.graph_plan.graph.model["name"] = "fallback";
  plan.launches.resize(2);
  plan.launches[1].dw_config = 1;
  const RunRecordContext value =
      deeptile::runtime::make_run_record_context(plan, "target@v1");
  TEST_ASSERT(value.model == "resnet50/imagenet1k/int8-tqt@v1.0.0",
              "release identity not resolved");
  TEST_ASSERT(value.target == "target@v1", "target identity not retained");
  TEST_ASSERT(value.fused_depthwise, "fused launch not detected");
  TEST_ASSERT(!value.source_revision.empty(), "software revision is empty");

  // Direct register control is the default; the variable selects XRT's
  // scheduler, whose per-launch cost the record must be able to show.
  unsetenv("DEEPTILE_XRT_LAUNCH");
  unsetenv("TILECNN_XRT_LAUNCH");
  TEST_ASSERT(deeptile::runtime::make_run_record_context(plan, "t").launch_mode ==
                  "direct",
              "direct control is not the default");
  setenv("DEEPTILE_XRT_LAUNCH", "run", 1);
  TEST_ASSERT(deeptile::runtime::make_run_record_context(plan, "t").launch_mode ==
                  "xrt-run",
              "scheduler mode not recorded");
  unsetenv("DEEPTILE_XRT_LAUNCH");
  return true;
}

/**
 * A run with no ModelPackage and no TargetPackage has no integrity claim to
 * make. It must omit the block rather than record empty digests, so an absent
 * field always means "no such artifact" and never "unchecked".
 */
bool test_source_scope_recorded(const fs::path &root) {
  RunRecordContext run_context = context();
  run_context.source_dirty = false;
  run_context.source_scope = "src scripts";
  SingleRunSummary summary;
  summary.passed = true;
  summary.measured_runs = 1;

  std::string error;
  const fs::path path = root / "scope" / "run.json";
  TEST_ASSERT(deeptile::runtime::write_single_run_record(path.string(),
                                                        run_context, summary,
                                                        error),
              error);
  rapidjson::Document record;
  TEST_ASSERT(read_json(path, record), "cannot parse RunRecord");
  const auto &sw = record["software"];
  // A clean flag is only meaningful next to the scope it was checked over.
  TEST_ASSERT(sw.HasMember("source_scope") &&
                  std::string(sw["source_scope"].GetString()) == "src scripts",
              "source_scope must accompany source_dirty");
  TEST_ASSERT(sw["source_dirty"].IsBool() && !sw["source_dirty"].GetBool(),
              "source_dirty should be false here");
  return true;
}

bool test_source_scope_omitted_when_unknown(const fs::path &root) {
  RunRecordContext run_context = context();
  run_context.source_scope.clear();
  SingleRunSummary summary;
  summary.passed = true;
  summary.measured_runs = 1;

  std::string error;
  const fs::path path = root / "no-scope" / "run.json";
  TEST_ASSERT(deeptile::runtime::write_single_run_record(path.string(),
                                                        run_context, summary,
                                                        error),
              error);
  rapidjson::Document record;
  TEST_ASSERT(read_json(path, record), "cannot parse RunRecord");
  TEST_ASSERT(!record["software"].HasMember("source_scope"),
              "an unknown scope must be absent, not an empty string");
  return true;
}

bool test_window_omitted_when_unset(const fs::path &root) {
  RunRecordContext run_context = context();
  SingleRunSummary summary;
  summary.passed = true;
  summary.measured_runs = 1;

  std::string error;
  const fs::path path = root / "no-window" / "run.json";
  TEST_ASSERT(deeptile::runtime::write_single_run_record(path.string(),
                                                        run_context, summary,
                                                        error),
              error);
  rapidjson::Document record;
  TEST_ASSERT(read_json(path, record), "cannot parse RunRecord");
  // Absence says no power trace can be attributed to this run, which is not
  // the same claim as a window of zero length.
  TEST_ASSERT(!record.HasMember("window"),
              "window must be absent when the run recorded none");
  return true;
}

bool test_window_recorded(const fs::path &root) {
  RunRecordContext run_context = context();
  DatasetRunSummary summary;
  summary.passed = true;
  summary.dataset = "data/imagenet";
  summary.images = 8;
  summary.window.start_ns = 1788270568225818643ULL;
  summary.window.end_ns = 1788270583225818643ULL;

  std::string error;
  const fs::path path = root / "window" / "run.json";
  TEST_ASSERT(deeptile::runtime::write_dataset_run_record(path.string(),
                                                         run_context, summary,
                                                         error),
              error);
  rapidjson::Document record;
  TEST_ASSERT(read_json(path, record), "cannot parse RunRecord");
  TEST_ASSERT(record.HasMember("window") && record["window"].IsObject(),
              "window block missing");
  const auto &window = record["window"];
  TEST_ASSERT(window.HasMember("clock") &&
                  std::string(window["clock"].GetString()) == "CLOCK_REALTIME",
              "window must name the clock it was taken on");
  // Nanosecond timestamps exceed a double's exact integer range, so the writer
  // must emit them as integers or a sampler aligns against a rounded value.
  TEST_ASSERT(window["start_ns"].IsUint64() && window["end_ns"].IsUint64(),
              "window bounds must be integers, not doubles");
  TEST_ASSERT(window["start_ns"].GetUint64() == summary.window.start_ns,
              "start_ns round-tripped inexactly");
  TEST_ASSERT(window["end_ns"].GetUint64() == summary.window.end_ns,
              "end_ns round-tripped inexactly");
  return true;
}

bool test_window_rejects_reversed_bounds(const fs::path &root) {
  RunRecordContext run_context = context();
  SingleRunSummary summary;
  summary.passed = true;
  summary.measured_runs = 1;
  summary.window.start_ns = 2000;
  summary.window.end_ns = 1000;

  std::string error;
  const fs::path path = root / "bad-window" / "run.json";
  TEST_ASSERT(deeptile::runtime::write_single_run_record(path.string(),
                                                        run_context, summary,
                                                        error),
              error);
  rapidjson::Document record;
  TEST_ASSERT(read_json(path, record), "cannot parse RunRecord");
  TEST_ASSERT(!record.HasMember("window"),
              "a window that ends before it starts must not be published");
  return true;
}

bool test_realtime_clock_advances() {
  const std::uint64_t first = deeptile::runtime::realtime_now_ns();
  TEST_ASSERT(first > 1'600'000'000'000'000'000ULL,
              "realtime_now_ns must return nanoseconds since the epoch");
  std::uint64_t second = first;
  for (int i = 0; i < 1000000 && second == first; ++i) {
    second = deeptile::runtime::realtime_now_ns();
  }
  TEST_ASSERT(second >= first, "realtime clock went backwards within a run");
  return true;
}

bool test_identity_omitted_without_artifacts(const fs::path &root) {
  RunRecordContext run_context = context();
  SingleRunSummary summary;
  summary.passed = true;
  summary.measured_runs = 1;

  std::string error;
  const fs::path path = root / "no-identity" / "run.json";
  TEST_ASSERT(deeptile::runtime::write_single_run_record(path.string(),
                                                        run_context, summary,
                                                        error),
              error);
  rapidjson::Document record;
  TEST_ASSERT(read_json(path, record), "cannot parse RunRecord");
  TEST_ASSERT(!record.HasMember("identity"),
              "identity block must be absent when no artifact carries one");
  return true;
}

/** A partially identified run records what it has and omits the rest. */
bool test_identity_is_partial_when_artifacts_are(const fs::path &root) {
  RunRecordContext run_context = context();
  run_context.digests.xclbin_sha256 = std::string(64, 'e');

  SingleRunSummary summary;
  summary.passed = true;
  summary.measured_runs = 1;

  std::string error;
  const fs::path path = root / "partial-identity" / "run.json";
  TEST_ASSERT(deeptile::runtime::write_single_run_record(path.string(),
                                                        run_context, summary,
                                                        error),
              error);
  rapidjson::Document record;
  TEST_ASSERT(read_json(path, record), "cannot parse RunRecord");
  TEST_ASSERT(record.HasMember("identity"), "identity block missing");
  TEST_ASSERT(std::string(record["identity"]["xclbin_sha256"].GetString()) ==
                  std::string(64, 'e'),
              "xclbin digest missing");
  TEST_ASSERT(!record["identity"].HasMember("model_sha256"),
              "absent model digest must be omitted, not empty");
  TEST_ASSERT(!record["identity"].HasMember("boot_sha256"),
              "absent boot digest must be omitted, not empty");
  return true;
}

/** Non-default lowering must be visible; the switch is what changed the run. */
bool test_schedule_flags_recorded(const fs::path &root) {
  setenv("DEEPTILE_FORCE_EPOCHS", "3", 1);
  deeptile::runtime::RuntimeExecutionPlan plan;
  plan.graph_plan.graph.model["name"] = "m";
  plan.launches.resize(1);
  const RunRecordContext run_context =
      deeptile::runtime::make_run_record_context(plan, "target@v1");
  unsetenv("DEEPTILE_FORCE_EPOCHS");
  TEST_ASSERT(run_context.schedule_flags == "FORCE_EPOCHS=3",
              "expected FORCE_EPOCHS=3, got '" << run_context.schedule_flags
                                               << "'");

  SingleRunSummary summary;
  summary.passed = true;
  summary.measured_runs = 1;
  std::string error;
  const fs::path path = root / "flags" / "run.json";
  TEST_ASSERT(deeptile::runtime::write_single_run_record(path.string(),
                                                        run_context, summary,
                                                        error),
              error);
  rapidjson::Document record;
  TEST_ASSERT(read_json(path, record), "cannot parse RunRecord");
  TEST_ASSERT(std::string(record["execution"]["schedule_flags"].GetString()) ==
                  "FORCE_EPOCHS=3",
              "schedule flags not recorded");
  return true;
}

/** A graph-only fixture makes no integrity claim, so it yields no digest. */
bool test_model_digest_from_manifest(const fs::path &root) {
  const fs::path bare = root / "bare-graph";
  fs::create_directories(bare);
  std::ofstream(bare / "graph.json") << "{}";
  TEST_ASSERT(deeptile::runtime::model_package_digest(
                  (bare / "graph.json").string())
                  .empty(),
              "a manifest-less fixture must not produce a digest");

  const fs::path pkg = root / "with-manifest";
  fs::create_directories(pkg);
  std::ofstream(pkg / "graph.json") << "{}";
  std::ofstream(pkg / "manifest.json") << R"({"schema":"tilecnn.model-package.v1"})";
  const std::string digest =
      deeptile::runtime::model_package_digest((pkg / "graph.json").string());
  TEST_ASSERT(digest.size() == 64, "expected a sha256 hex digest");
  // The manifest inventories every artifact, so hashing it covers the package.
  TEST_ASSERT(digest == deeptile::target::sha256_text(
                            R"({"schema":"tilecnn.model-package.v1"})"),
              "model digest must be the manifest's own sha256");
  return true;
}

} // namespace

int main() {
  const fs::path root = test_root();
  fs::create_directories(root);
  int passed = 0;
  int failed = 0;

  const auto run = [&](const char *name, bool (*test)(const fs::path &)) {
    std::cout << "[TEST] " << name << "\n";
    if (test(root)) {
      ++passed;
    } else {
      ++failed;
    }
  };

  run("single RunRecord", test_single_record);
  run("batch-one latency RunRecord", test_latency_record);
  run("dataset RunRecord", test_dataset_record);
  run("RunRecord context", test_context_identity);
  run("identity omitted without artifacts",
      test_identity_omitted_without_artifacts);
  run("identity partial when artifacts are",
      test_identity_is_partial_when_artifacts_are);
  run("schedule flags recorded", test_schedule_flags_recorded);
  run("model digest from manifest", test_model_digest_from_manifest);
  run("source scope recorded", test_source_scope_recorded);
  run("source scope omitted when unknown", test_source_scope_omitted_when_unknown);
  run("window omitted when unset", test_window_omitted_when_unset);
  run("window recorded", test_window_recorded);
  run("window rejects reversed bounds", test_window_rejects_reversed_bounds);
  run("realtime clock advances",
      [](const fs::path &) { return test_realtime_clock_advances(); });

  std::error_code ec;
  fs::remove_all(root, ec);
  std::cout << "RunRecord tests: " << passed << " passed, " << failed
            << " failed\n";
  return failed == 0 ? 0 : 1;
}
