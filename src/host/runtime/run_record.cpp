/**
 * @file run_record.cpp
 * @brief Implements the minimal DeepTile execution-result record.
 */

#include "runtime/run_record.h"

#include "deeptile/compiler/environment.h"
#include "deeptile/compiler/graph_compiler.h"
#include "deeptile/compiler/schema_ids.h"
#include "deeptile/compiler/target_package.h"

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <system_error>

#ifndef DEEPTILE_GIT_REVISION
#define DEEPTILE_GIT_REVISION "unknown"
#endif

#ifndef DEEPTILE_SOURCE_DIRTY
#define DEEPTILE_SOURCE_DIRTY 0
#endif

#ifndef DEEPTILE_SOURCE_SCOPE
#define DEEPTILE_SOURCE_SCOPE ""
#endif

namespace deeptile {
namespace runtime {
namespace {

namespace fs = std::filesystem;
using JsonWriter = rapidjson::PrettyWriter<rapidjson::StringBuffer>;

std::string utc_timestamp() {
  const std::time_t now = std::chrono::system_clock::to_time_t(
      std::chrono::system_clock::now());
  std::tm utc = {};
  gmtime_r(&now, &utc);
  char value[32] = {};
  std::strftime(value, sizeof(value), "%Y-%m-%dT%H:%M:%SZ", &utc);
  return value;
}

std::string model_identity(const graph::GraphSpec &graph) {
  const fs::path manifest_path = fs::path(graph.root_dir) / "manifest.json";
  std::ifstream input(manifest_path);
  if (input) {
    rapidjson::IStreamWrapper stream(input);
    rapidjson::Document manifest;
    manifest.ParseStream(stream);
    if (!manifest.HasParseError() && manifest.IsObject() &&
        manifest.HasMember("model") && manifest["model"].IsObject()) {
      const auto &model = manifest["model"];
      if (model.HasMember("release_id") && model["release_id"].IsString()) {
        return model["release_id"].GetString();
      }
    }
  }

  const auto name = graph.model.find("name");
  return name == graph.model.end() ? "unknown" : name->second;
}

void write_common(JsonWriter &json, const RunRecordContext &context,
                  const char *mode, bool passed) {
  json.Key("schema");
  json.String(deeptile::schema::kRun.data());
  json.Key("created_at");
  json.String(utc_timestamp().c_str());
  json.Key("status");
  json.String(passed ? "pass" : "fail");
  json.Key("mode");
  json.String(mode);
  json.Key("model");
  json.String(context.model.c_str());
  json.Key("target");
  json.String(context.target.c_str());

  json.Key("software");
  json.StartObject();
  json.Key(deeptile::schema::kRevisionField.data());
  json.String(context.source_revision.c_str());
  json.Key("source_dirty");
  json.Bool(context.source_dirty);
  // Without the scope the flag is unreadable: a reader cannot tell whether a
  // clean result means the whole tree or only the build inputs were checked.
  if (!context.source_scope.empty()) {
    json.Key("source_scope");
    json.String(context.source_scope.c_str());
  }
  json.EndObject();

  // Naming an artifact says which one was intended; the digest says which one
  // ran. Absent fields mean the run had no such artifact, not that it went
  // unchecked.
  const auto &digests = context.digests;
  if (!digests.model_sha256.empty() || !digests.xclbin_sha256.empty() ||
      !digests.boot_sha256.empty()) {
    json.Key("identity");
    json.StartObject();
    const auto emit = [&json](const char *key, const std::string &value) {
      if (!value.empty()) {
        json.Key(key);
        json.String(value.c_str());
      }
    };
    emit("model_sha256", digests.model_sha256);
    emit("xclbin_sha256", digests.xclbin_sha256);
    emit("boot_sha256", digests.boot_sha256);
    json.EndObject();
  }
}

/**
 * Emit the measurement window, or nothing when the run did not record one.
 *
 * Absence is meaningful: it says no power trace can be attributed to this run,
 * which is different from a window of zero length.
 */
void write_window(JsonWriter &json, const MeasurementWindow &window) {
  if (!window.valid()) {
    return;
  }
  json.Key("window");
  json.StartObject();
  json.Key("clock");
  json.String("CLOCK_REALTIME");
  json.Key("start_ns");
  json.Uint64(window.start_ns);
  json.Key("end_ns");
  json.Uint64(window.end_ns);
  json.EndObject();
}

void write_execution_common(JsonWriter &json,
                            const RunRecordContext &context) {
  json.Key("launches");
  json.Uint64(context.launches);
  json.Key("fused_depthwise");
  json.Bool(context.fused_depthwise);
  json.Key("stem_space_to_depth");
  json.Bool(context.stem_space_to_depth);
  if (!context.launch_mode.empty()) {
    json.Key("launch_mode");
    json.String(context.launch_mode.c_str());
  }
  // Present only when lowering departed from its defaults, which would
  // otherwise make the run unreproducible from the model and target alone.
  if (!context.schedule_flags.empty()) {
    json.Key("schedule_flags");
    json.String(context.schedule_flags.c_str());
  }
}

bool finish_record(const fs::path &path,
                   const rapidjson::StringBuffer &buffer,
                   std::string &error) {
  std::error_code ec;
  if (!path.parent_path().empty()) {
    fs::create_directories(path.parent_path(), ec);
    if (ec) {
      error = "cannot create run directory: " + ec.message();
      return false;
    }
  }

  const fs::path temporary = path.string() + ".tmp";
  std::ofstream output(temporary);
  if (!output) {
    error = "cannot open RunRecord: " + temporary.string();
    return false;
  }
  output << buffer.GetString() << "\n";
  output.close();
  if (!output) {
    error = "cannot write RunRecord: " + temporary.string();
    fs::remove(temporary, ec);
    return false;
  }

  fs::rename(temporary, path, ec);
  if (ec) {
    error = "cannot publish RunRecord: " + ec.message();
    fs::remove(temporary, ec);
    return false;
  }
  return true;
}

} // namespace

std::string model_package_digest(const std::string &graph_path) {
  const fs::path manifest = fs::path(graph_path).parent_path() / "manifest.json";
  if (!fs::is_regular_file(manifest)) {
    return {};  // A graph-only fixture makes no integrity claim to record.
  }
  std::string digest;
  std::string error;
  if (!deeptile::target::sha256_file(manifest.string(), digest, error)) {
    return {};
  }
  return digest;
}

std::string xrt_launch_mode() {
  const char *mode = deeptile::env_value("XRT_LAUNCH");
  return (mode != nullptr && std::string(mode) == "run") ? "xrt-run" : "direct";
}

RunRecordContext make_run_record_context(
    const RuntimeExecutionPlan &plan, const std::string &target_id,
    const RunArtifactDigests &digests) {
  RunRecordContext context;
  context.model = model_identity(plan.graph_plan.graph);
  context.target = target_id.empty() ? "legacy-xclbin" : target_id;
  context.source_revision = DEEPTILE_GIT_REVISION;
  context.source_dirty = DEEPTILE_SOURCE_DIRTY != 0;
  context.source_scope = DEEPTILE_SOURCE_SCOPE;
  context.launches = plan.launches.size();
  context.fused_depthwise = std::any_of(
      plan.launches.begin(), plan.launches.end(),
      [](const RuntimeLaunchBinding &launch) {
        return (launch.dw_config & 1) != 0;
      });
  context.stem_space_to_depth = plan.stem_fold.active;
  context.launch_mode = xrt_launch_mode();
  context.digests = digests;
  context.schedule_flags = deeptile::graph::schedule_flag_summary();
  return context;
}

bool write_single_run_record(const std::string &path,
                             const RunRecordContext &context,
                             const SingleRunSummary &summary,
                             std::string &error) {
  rapidjson::StringBuffer buffer;
  JsonWriter json(buffer);
  json.SetIndent(' ', 2);
  json.SetMaxDecimalPlaces(6);
  json.StartObject();
  write_common(json, context, "single", summary.passed);
  write_window(json, summary.window);

  json.Key("execution");
  json.StartObject();
  write_execution_common(json, context);
  json.Key("warmup_runs");
  json.Int(summary.warmup_runs);
  json.Key("measured_runs");
  json.Int(summary.measured_runs);
  json.EndObject();

  json.Key("results");
  json.StartObject();
  if (summary.comparison_performed) {
    json.Key("exact_reference_match");
    json.Bool(summary.exact_reference_match);
    json.Key("compared_elements");
    json.Uint64(summary.compared_elements);
  }
  json.Key("kernel_ms");
  json.Double(summary.kernel_ms);
  json.Key("kernel_gops");
  json.Double(summary.kernel_gops);
  json.Key("kernel_timing_scope");
  json.String("host_launch_and_wait");
  if (summary.measure_latency) {
    json.Key("graph_wall_ms");
    json.Double(summary.graph_wall_ms);
    json.Key("graph_timing_scope");
    json.String("input_upload_through_output_download");
  }
  json.EndObject();

  json.Key("artifacts");
  json.StartObject();
  if (!summary.launch_profile.empty()) {
    json.Key("launch_profile");
    json.String(summary.launch_profile.c_str());
  }
  if (!summary.graph_profile.empty()) {
    json.Key("graph_profile");
    json.String(summary.graph_profile.c_str());
  }
  json.EndObject();
  json.EndObject();
  return finish_record(path, buffer, error);
}

bool write_dataset_run_record(const std::string &path,
                              const RunRecordContext &context,
                              const DatasetRunSummary &summary,
                              std::string &error) {
  rapidjson::StringBuffer buffer;
  JsonWriter json(buffer);
  json.SetIndent(' ', 2);
  json.SetMaxDecimalPlaces(6);
  json.StartObject();
  write_common(json, context, "dataset", summary.passed);
  write_window(json, summary.window);

  json.Key("execution");
  json.StartObject();
  write_execution_common(json, context);
  json.Key("dataset");
  json.String(summary.dataset.c_str());
  if (!summary.dataset_sha256.empty()) {
    json.Key("dataset_sha256");
    json.String(summary.dataset_sha256.c_str());
  }
  json.Key("limit");
  json.Uint64(summary.limit);
  json.Key("workers");
  json.Int(summary.workers);
  json.Key("queue_depth");
  json.Uint64(summary.queue_depth);
  json.Key("top_k");
  json.Int(summary.top_k);
  json.Key("tie_policy");
  json.String("lowest-class-index");
  json.Key("preprocessing");
  json.String(summary.preprocessing.c_str());
  if (summary.short_side > 0) {
    json.Key("short_side");
    json.Int(summary.short_side);
  }
  json.EndObject();

  json.Key("results");
  json.StartObject();
  json.Key("images");
  json.Uint64(summary.images);
  json.Key("top1_percent");
  json.Double(summary.top1_percent);
  json.Key("topk_percent");
  json.Double(summary.topk_percent);
  json.Key("kernel_ms_per_image");
  json.Double(summary.kernel_ms_per_image);
  json.Key("kernel_ms_median");
  json.Double(summary.kernel_ms_median);
  json.Key("kernel_ms_min");
  json.Double(summary.kernel_ms_min);
  json.Key("wall_ms_per_image");
  json.Double(summary.wall_ms_per_image);
  json.Key("images_per_second");
  json.Double(summary.images_per_second);
  json.Key("accelerator_busy_percent");
  json.Double(summary.accelerator_busy_percent);
  // Throughput is recorded at both scopes with the operation count beside it,
  // so a reader never has to guess the measurement boundary.
  json.Key("graph_ops");
  json.Double(summary.graph_ops);
  json.Key("avg_kernel_gops");
  json.Double(summary.avg_kernel_gops);
  json.Key("peak_kernel_gops");
  json.Double(summary.peak_kernel_gops);
  json.Key("end_to_end_gops");
  json.Double(summary.end_to_end_gops);
  json.Key("op_convention");
  json.String("1 MAC = 2 ops");
  json.EndObject();

  json.Key("artifacts");
  json.StartObject();
  if (!summary.predictions.empty()) {
    json.Key("predictions");
    json.String(summary.predictions.c_str());
  }
  json.EndObject();
  json.EndObject();
  return finish_record(path, buffer, error);
}

} // namespace runtime
} // namespace deeptile
