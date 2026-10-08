/**
 * @file main.cpp
 * @brief DeepTile graph host command-line entry point.
 */

#include "runtime/runtime.h"
#include "runtime/run_record.h"
#include "runtime/target_artifact.h"
#include "backends/dataset_eval.h"
#include "backends/xrt_backend.h"
#include "deeptile/compiler/target_package.h"

#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct CliOptions {
  deeptile::runtime::XrtRunOptions xrt;
  deeptile::runtime::DatasetEvalOptions dataset;
  std::string target_argument;
  std::string target_id;
  deeptile::runtime::RunArtifactDigests digests;
  std::string output_dir;
  std::string run_record_path;
  bool dataset_mode = false;
  bool kernel_overridden = false;
};

void print_diagnostics(
    const std::vector<deeptile::runtime::RuntimeDiagnostic> &diagnostics);

// Parse options, prepare the graph, execute it through XRT, and compare outputs.
void print_usage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0
      << " <target.json|xclbin> <graph.json> [options]\n"
      << "\nSingle-image mode (default): runs the graph's exported input and\n"
      << "compares against its exported reference.\n"
      << "  --runs N             measured runs, default 1\n"
      << "  --warmup N           warm-up runs, default 1\n"
      << "  --csv FILE           per-launch profile CSV\n"
      << "  --no-compare         skip the exported-reference comparison\n"
      << "  --measure-latency    time input upload through output download per run\n"
      << "\nDataset mode: --dataset selects full-dataset inference.\n"
      << "  --dataset ROOT       ImageNet-style dataset root\n"
      << "  --limit N            evaluate at most N images\n"
      << "  --workers N          preparation threads, default 2\n"
      << "  --queue N            prepared images buffered\n"
      << "  --topk N             top-k reported alongside top-1, default 5\n"
      << "  --convention NAME    'eval' (default) or 'export'\n"
      << "  --short-side N       eval-convention resize target, default 256\n"
      << "  --predictions FILE   per-image prediction CSV\n"
      << "  --allow-missing      evaluate even if val.txt names absent images\n"
      << "\nCommon:\n"
      << "  --output DIR         write run.json and the mode-specific CSV\n"
      << "  --kernel NAME        kernel name, default tile_conv\n"
      << "  --device N           device index, default 0\n"
      << "  --verbose            per-launch or per-image detail\n";
}

bool parse_int_arg(const char *name, const char *value, int &out) {
  try {
    out = std::stoi(value);
  } catch (const std::exception &) {
    std::cerr << "[ERROR] invalid integer for " << name << ": " << value
              << "\n";
    return false;
  }
  return true;
}

bool parse_cli(int argc, char **argv, CliOptions &out) {
  if (argc < 3) {
    print_usage(argv[0]);
    return false;
  }

  out.target_argument = argv[1];
  out.xrt.graph_path = argv[2];

  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto require_value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << "[ERROR] missing value for " << name << "\n";
        return nullptr;
      }
      return argv[++i];
    };

    if (arg == "--runs") {
      const char *value = require_value("--runs");
      if (!value || !parse_int_arg("--runs", value, out.xrt.measured_runs)) {
        return false;
      }
    } else if (arg == "--warmup") {
      const char *value = require_value("--warmup");
      if (!value || !parse_int_arg("--warmup", value, out.xrt.warmup_runs)) {
        return false;
      }
    } else if (arg == "--csv") {
      const char *value = require_value("--csv");
      if (!value) {
        return false;
      }
      out.xrt.csv_path = value;
    } else if (arg == "--measure-latency") {
      out.xrt.measure_latency = true;
    } else if (arg == "--kernel") {
      const char *value = require_value("--kernel");
      if (!value) {
        return false;
      }
      out.xrt.kernel_name = value;
      out.kernel_overridden = true;
    } else if (arg == "--device") {
      const char *value = require_value("--device");
      if (!value || !parse_int_arg("--device", value, out.xrt.device_index)) {
        return false;
      }
    } else if (arg == "--dataset") {
      const char *value = require_value("--dataset");
      if (!value) {
        return false;
      }
      out.dataset.dataset_root = value;
      out.dataset_mode = true;
    } else if (arg == "--limit") {
      const char *value = require_value("--limit");
      int limit = 0;
      if (!value || !parse_int_arg("--limit", value, limit) || limit < 0) {
        return false;
      }
      out.dataset.limit = static_cast<std::size_t>(limit);
    } else if (arg == "--workers") {
      const char *value = require_value("--workers");
      if (!value || !parse_int_arg("--workers", value, out.dataset.workers) ||
          out.dataset.workers < 1) {
        std::cerr << "[ERROR] --workers must be at least 1\n";
        return false;
      }
    } else if (arg == "--queue") {
      const char *value = require_value("--queue");
      int depth = 0;
      if (!value || !parse_int_arg("--queue", value, depth) || depth < 1) {
        return false;
      }
      out.dataset.queue_depth = static_cast<std::size_t>(depth);
    } else if (arg == "--topk") {
      const char *value = require_value("--topk");
      if (!value || !parse_int_arg("--topk", value, out.dataset.top_k) ||
          out.dataset.top_k < 1) {
        return false;
      }
    } else if (arg == "--convention") {
      const char *value = require_value("--convention");
      if (!value ||
          !deeptile::dataset::parse_resize_convention(
              value, out.dataset.preprocess.convention)) {
        std::cerr << "[ERROR] unknown convention; expected 'eval' or 'export'\n";
        return false;
      }
    } else if (arg == "--short-side") {
      const char *value = require_value("--short-side");
      if (!value || !parse_int_arg("--short-side", value,
                                   out.dataset.preprocess.short_side)) {
        return false;
      }
    } else if (arg == "--predictions") {
      const char *value = require_value("--predictions");
      if (!value) {
        return false;
      }
      out.dataset.predictions_csv = value;
    } else if (arg == "--output") {
      const char *value = require_value("--output");
      if (!value) {
        return false;
      }
      out.output_dir = value;
    } else if (arg == "--no-compare") {
      out.xrt.compare_outputs = false;
    } else if (arg == "--allow-missing") {
      out.dataset.allow_missing = true;
    } else if (arg == "--verbose") {
      out.xrt.verbose = true;
      out.dataset.verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return false;
    } else {
      std::cerr << "[ERROR] unknown argument: " << arg << "\n";
      print_usage(argv[0]);
      return false;
    }
  }

  if (out.xrt.measured_runs <= 0) {
    std::cerr << "[ERROR] --runs must be positive\n";
    return false;
  }
  if (out.xrt.warmup_runs < 0) {
    std::cerr << "[ERROR] --warmup cannot be negative\n";
    return false;
  }
  if (out.xrt.device_index < 0) {
    std::cerr << "[ERROR] --device cannot be negative\n";
    return false;
  }
  return true;
}

bool prepare_output(CliOptions &options) {
  if (options.dataset_mode && options.xrt.measure_latency) {
    std::cerr << "[ERROR] --measure-latency applies to single-image mode\n";
    return false;
  }
  if (options.output_dir.empty()) {
    return true;
  }

  const std::filesystem::path directory(options.output_dir);
  std::error_code ec;
  std::filesystem::create_directories(directory, ec);
  if (ec || !std::filesystem::is_directory(directory)) {
    std::cerr << "[ERROR] cannot create output directory: "
              << options.output_dir << "\n";
    return false;
  }

  const std::filesystem::path record = directory / "run.json";
  if (std::filesystem::exists(record)) {
    std::cerr << "[ERROR] output already contains run.json: "
              << record.string() << "\n";
    return false;
  }
  options.run_record_path = record.string();

  if (options.dataset_mode) {
    if (options.dataset.predictions_csv.empty()) {
      options.dataset.predictions_csv =
          (directory / "predictions.csv").string();
    }
  } else if (options.xrt.csv_path.empty()) {
    options.xrt.csv_path = (directory / "launch-profile.csv").string();
  }
  return true;
}

std::string artifact_reference(const CliOptions &options,
                               const std::string &path) {
  if (path.empty() || options.output_dir.empty()) {
    return path;
  }
  std::error_code ec;
  const auto relative = std::filesystem::relative(
      std::filesystem::absolute(path),
      std::filesystem::absolute(options.output_dir), ec);
  return ec ? path : relative.generic_string();
}

bool resolve_target(CliOptions &options) {
  const std::filesystem::path argument(options.target_argument);
  if (argument.extension() != ".json") {
    options.xrt.xclbin_path = options.target_argument;
    return true;
  }

  deeptile::target::TargetPackage target;
  std::vector<deeptile::target::TargetDiagnostic> target_diagnostics;
  if (!deeptile::target::load_target_package(options.target_argument, target,
                                            target_diagnostics)) {
    for (const auto &diagnostic : target_diagnostics) {
      std::cerr << "[" << deeptile::target::severity_name(diagnostic.severity)
                << "] " << diagnostic.context << ": " << diagnostic.message
                << "\n";
    }
    return false;
  }
  for (const auto &diagnostic : target_diagnostics) {
    std::cerr << "[" << deeptile::target::severity_name(diagnostic.severity)
              << "] " << diagnostic.context << ": " << diagnostic.message
              << "\n";
  }
  if (options.kernel_overridden && options.xrt.kernel_name != target.kernel_name) {
    std::cerr << "[ERROR] --kernel conflicts with target manifest kernel "
              << target.kernel_name << "\n";
    return false;
  }

  std::vector<deeptile::runtime::RuntimeDiagnostic> artifact_diagnostics;
  if (!deeptile::runtime::resolve_target_artifact(
          target, options.xrt.xclbin_path, artifact_diagnostics)) {
    print_diagnostics(artifact_diagnostics);
    return false;
  }
  options.xrt.kernel_name = target.kernel_name;
  options.target_id = target.id;
  // resolve_target_artifact has just confirmed the files match these, so the
  // record can state which artifacts ran rather than which were named.
  options.digests.xclbin_sha256 = target.xclbin_sha256;
  options.digests.boot_sha256 = target.boot_sha256;
  return true;
}

void print_diagnostics(
    const std::vector<deeptile::runtime::RuntimeDiagnostic> &diagnostics) {
  for (const auto &d : diagnostics) {
    std::cerr << "[" << deeptile::runtime::severity_name(d.severity) << "] ";
    if (!d.context.empty()) {
      std::cerr << d.context << ": ";
    }
    std::cerr << d.message << "\n";
  }
}

const deeptile::runtime::RuntimeTensorBuffer *
find_buffer(const deeptile::runtime::RuntimeExecutionPlan &plan,
            const std::string &id) {
  const auto it = plan.buffer_index_by_id.find(id);
  if (it == plan.buffer_index_by_id.end()) {
    return nullptr;
  }
  return &plan.buffers[it->second];
}

bool compare_chw(const std::vector<std::int8_t> &observed,
                 const std::vector<std::int8_t> &expected,
                 const std::vector<int> &shape,
                 const std::string &tensor_id,
                 int max_reported = 20) {
  if (observed.size() != expected.size()) {
    std::cerr << "[COMPARE][ERROR] " << tensor_id
              << " size mismatch observed=" << observed.size()
              << " expected=" << expected.size() << "\n";
    return false;
  }
  if (shape.size() != 3) {
    std::cerr << "[COMPARE][ERROR] " << tensor_id
              << " output shape is not CHW\n";
    return false;
  }

  const int height = shape[1];
  const int width = shape[2];
  int mismatches = 0;
  int reported = 0;
  for (std::size_t i = 0; i < observed.size(); ++i) {
    if (observed[i] == expected[i]) {
      continue;
    }
    ++mismatches;
    if (reported < max_reported) {
      const std::size_t spatial_size =
          static_cast<std::size_t>(height) * width;
      const int c = static_cast<int>(i / spatial_size);
      const std::size_t spatial = i % spatial_size;
      const int h = static_cast<int>(spatial / width);
      const int w = static_cast<int>(spatial % width);
      std::cerr << "[MISMATCH][EXPORTED_REF] " << tensor_id << " idx=" << i
                << " (c=" << c << ", h=" << h << ", w=" << w
                << ") observed=" << static_cast<int>(observed[i])
                << " expected=" << static_cast<int>(expected[i]) << "\n";
      ++reported;
    }
  }

  if (mismatches != 0) {
    std::cerr << "[COMPARE][FAIL] " << tensor_id
              << " mismatches=" << mismatches << " / " << observed.size()
              << "\n";
    return false;
  }

  std::cout << "[COMPARE][PASS] " << tensor_id
            << " exact match, elements=" << observed.size() << "\n";
  return true;
}

bool compare_outputs(const deeptile::runtime::RuntimeExecutionPlan &plan,
                     std::size_t &compared_elements) {
  // Board validation compares against exporter-provided reference tensors.
  bool ok = true;
  for (const auto &output_id : plan.graph_plan.graph.outputs) {
    const auto ref_it = plan.graph_plan.graph.references.find(output_id);
    if (ref_it == plan.graph_plan.graph.references.end()) {
      std::cerr << "[COMPARE][ERROR] missing exported reference mapping for "
                << output_id << "\n";
      ok = false;
      continue;
    }

    const auto *observed_buffer = find_buffer(plan, output_id);
    const auto *reference_buffer = find_buffer(plan, ref_it->second);
    if (!observed_buffer || !reference_buffer) {
      std::cerr << "[COMPARE][ERROR] missing observed/exported reference for "
                << output_id << "\n";
      ok = false;
      continue;
    }
    if (observed_buffer->shape != reference_buffer->shape) {
      std::cerr << "[COMPARE][ERROR] " << output_id
                << " shape mismatch observed=[";
      for (std::size_t i = 0; i < observed_buffer->shape.size(); ++i) {
        std::cerr << (i == 0 ? "" : ",") << observed_buffer->shape[i];
      }
      std::cerr << "] reference=[";
      for (std::size_t i = 0; i < reference_buffer->shape.size(); ++i) {
        std::cerr << (i == 0 ? "" : ",") << reference_buffer->shape[i];
      }
      std::cerr << "]\n";
      ok = false;
      continue;
    }

    const auto observed =
        deeptile::runtime::unpack_packed_activation_to_chw(*observed_buffer);
    compared_elements += observed.size();
    if (!compare_chw(observed, reference_buffer->raw_data,
                     reference_buffer->shape, output_id)) {
      ok = false;
    }
  }
  return ok;
}

} // namespace

int main(int argc, char **argv) {
  CliOptions options;
  if (!parse_cli(argc, argv, options)) {
    return EXIT_FAILURE;
  }
  if (!prepare_output(options)) {
    return EXIT_FAILURE;
  }
  if (!resolve_target(options)) {
    return EXIT_FAILURE;
  }

  std::cout << "==========================================================\n";
  std::cout << "  DeepTile Graph XRT Host \n";
  std::cout << "==========================================================\n";
  std::cout << "[XRT] xclbin: " << options.xrt.xclbin_path << "\n";
  std::cout << "[XRT] kernel: " << options.xrt.kernel_name << "\n";
  if (!options.target_id.empty()) {
    std::cout << "[TARGET] id: " << options.target_id << "\n";
  }
  std::cout << "[GRAPH] path: " << options.xrt.graph_path << "\n";
  if (options.dataset_mode) {
    std::cout << "[DATASET] root: " << options.dataset.dataset_root
              << " convention="
              << deeptile::dataset::resize_convention_name(
                     options.dataset.preprocess.convention)
              << " topk=" << options.dataset.top_k << "\n";
  } else {
    std::cout << "[RUNS] warmup=" << options.xrt.warmup_runs
              << " measured=" << options.xrt.measured_runs << "\n";
  }

  deeptile::runtime::RuntimeExecutionPlan plan;
  std::vector<deeptile::runtime::RuntimeDiagnostic> diagnostics;
  // Prepare the complete packed runtime plan from the graph handoff.
  options.digests.model_sha256 =
      deeptile::runtime::model_package_digest(options.xrt.graph_path);
  if (!deeptile::runtime::load_compile_prepare_runtime(options.xrt.graph_path,
                                                      plan, diagnostics)) {
    print_diagnostics(diagnostics);
    return EXIT_FAILURE;
  }
  std::cout << "[GRAPH] launches=" << plan.launches.size()
            << " tensors=" << plan.graph_plan.graph.tensors.size() << "\n";
  if (options.xrt.verbose) {
    deeptile::runtime::print_activation_memory_plan(plan, std::cout);
  }

  if (options.dataset_mode) {
    // Dataset mode owns its own session and reporting; the single-image
    // comparison path does not apply because each image has no exported
    // reference.
    const deeptile::runtime::DatasetEvalResult dataset_result =
        deeptile::runtime::run_dataset_evaluation(plan, options.xrt,
                                                 options.dataset);
    print_diagnostics(dataset_result.diagnostics);
    if (!dataset_result.ok) {
      return EXIT_FAILURE;
    }
    deeptile::runtime::print_dataset_eval_summary(dataset_result, std::cout);
    if (!options.dataset.predictions_csv.empty()) {
      std::cout << "[CSV] wrote " << options.dataset.predictions_csv << "\n";
    }
    if (!options.run_record_path.empty()) {
      const double images = static_cast<double>(dataset_result.evaluated);
      deeptile::runtime::DatasetRunSummary summary;
      summary.passed = true;
      summary.window = dataset_result.window;
      summary.dataset = options.dataset.dataset_root;
      summary.limit = options.dataset.limit;
      summary.workers = dataset_result.workers;
      summary.queue_depth = dataset_result.queue_depth;
      summary.top_k = options.dataset.top_k;
      summary.preprocessing = deeptile::dataset::resize_convention_name(
          options.dataset.preprocess.convention);
      summary.short_side = options.dataset.preprocess.short_side;
      summary.images = dataset_result.evaluated;
      summary.top1_percent =
          100.0 * dataset_result.metrics.top1_accuracy();
      summary.topk_percent =
          100.0 * dataset_result.metrics.topk_accuracy();
      summary.kernel_ms_median = dataset_result.kernel_ms_median;
      summary.kernel_ms_min = dataset_result.kernel_ms_min;
      summary.graph_ops = static_cast<double>(dataset_result.graph_ops);
      if (dataset_result.evaluated != 0) {
        summary.kernel_ms_per_image = dataset_result.kernel_ms / images;
        summary.wall_ms_per_image = dataset_result.wall_ms / images;
        // Same scopes the console summary reports, so a record and a terminal
        // transcript of the same run cannot disagree.
        summary.avg_kernel_gops = deeptile::runtime::compute_rate(
            dataset_result.graph_ops, summary.kernel_ms_per_image);
        summary.peak_kernel_gops = deeptile::runtime::compute_rate(
            dataset_result.graph_ops, dataset_result.kernel_ms_min);
        summary.end_to_end_gops = deeptile::runtime::compute_rate(
            dataset_result.graph_ops, summary.wall_ms_per_image);
      }
      if (dataset_result.wall_ms > 0.0) {
        summary.images_per_second = 1000.0 * images / dataset_result.wall_ms;
        summary.accelerator_busy_percent =
            100.0 * dataset_result.kernel_ms / dataset_result.wall_ms;
      }
      summary.dataset_sha256 = dataset_result.listing_sha256;
      summary.predictions = artifact_reference(
          options, options.dataset.predictions_csv);

      const auto context = deeptile::runtime::make_run_record_context(
          plan, options.target_id, options.digests);
      std::string error;
      if (!deeptile::runtime::write_dataset_run_record(
              options.run_record_path, context, summary, error)) {
        std::cerr << "[ERROR] run-record: " << error << "\n";
        return EXIT_FAILURE;
      }
      std::cout << "[RUN] wrote " << options.run_record_path << "\n";
    }
    return EXIT_SUCCESS;
  }

  const deeptile::runtime::XrtBackend backend(options.xrt);
  // Session preparation is excluded; launch/wait and graph wall differ in scope.
  deeptile::runtime::XrtRunResult result = backend.run(plan);
  if (!result.ok) {
    print_diagnostics(result.diagnostics);
    return EXIT_FAILURE;
  }

  deeptile::runtime::print_xrt_run_summary(result, std::cout, options.xrt.verbose);
  if (!options.xrt.csv_path.empty()) {
    std::vector<deeptile::runtime::RuntimeDiagnostic> csv_diagnostics;
    if (!deeptile::runtime::write_xrt_profile_csv(result, options.xrt.csv_path,
                                                 csv_diagnostics)) {
      print_diagnostics(csv_diagnostics);
      return EXIT_FAILURE;
    }
    std::cout << "[CSV] wrote " << options.xrt.csv_path << "\n";
  }

  std::size_t compared_elements = 0;
  bool comparison_passed = true;
  if (options.xrt.compare_outputs) {
    comparison_passed = compare_outputs(plan, compared_elements);
  }

  if (!options.run_record_path.empty()) {
    std::string graph_profile;
    if (options.xrt.measure_latency) {
      graph_profile =
          (std::filesystem::path(options.output_dir) / "graph-profile.csv").string();
      if (!deeptile::runtime::write_xrt_graph_profile_csv(
              result, graph_profile, result.diagnostics)) {
        print_diagnostics(result.diagnostics);
        return EXIT_FAILURE;
      }
    }
    deeptile::runtime::SingleRunSummary summary;
    summary.passed = comparison_passed;
    summary.window = result.window;
    summary.warmup_runs = options.xrt.warmup_runs;
    summary.measured_runs = options.xrt.measured_runs;
    summary.comparison_performed = options.xrt.compare_outputs;
    summary.exact_reference_match = comparison_passed;
    summary.compared_elements = compared_elements;
    summary.kernel_ms = result.avg_kernel_latency_ms;
    summary.kernel_gops = result.gops;
    summary.measure_latency = options.xrt.measure_latency;
    summary.graph_wall_ms = result.avg_graph_wall_ms;
    summary.graph_profile = artifact_reference(options, graph_profile);
    summary.launch_profile = artifact_reference(options, options.xrt.csv_path);

    const auto context = deeptile::runtime::make_run_record_context(
        plan, options.target_id, options.digests);
    std::string error;
    if (!deeptile::runtime::write_single_run_record(
            options.run_record_path, context, summary, error)) {
      std::cerr << "[ERROR] run-record: " << error << "\n";
      return EXIT_FAILURE;
    }
    std::cout << "[RUN] wrote " << options.run_record_path << "\n";
  }

  if (!comparison_passed) {
    return EXIT_FAILURE;
  }

  std::cout << "==========================================================\n";
  std::cout << "  [PASS] DeepTile graph XRT host\n";
  std::cout << "==========================================================\n";
  return EXIT_SUCCESS;
}
