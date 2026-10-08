/**
 * @file dataset_eval.cpp
 * @brief Implements dataset inference over an XRT session.
 *
 * The session is opened once for the whole dataset. Re-opening it per image
 * would re-upload every weight and descriptor, which dominates the per-image
 * cost by orders of magnitude.
 */

#include "backends/dataset_eval.h"

#include "dataset/dataset.h"
#include "dataset/evaluation.h"
#include "dataset/image_decode.h"
#include "deeptile/tile/hardware_spec.hpp"
#include "dataset/pipeline.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <iostream>
#include <thread>

namespace deeptile {
namespace runtime {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void add_error(std::vector<RuntimeDiagnostic> &diagnostics,
               const std::string &message) {
  diagnostics.push_back(
      {RuntimeDiagnosticSeverity::Error, "dataset-eval", message});
}

int resolve_workers(int requested) {
  return requested > 0 ? requested : dataset::kDefaultWorkers;
}

} // namespace

DatasetEvalResult run_dataset_evaluation(RuntimeExecutionPlan &plan,
                                         const XrtRunOptions &xrt_options,
                                         const DatasetEvalOptions &options) {
  DatasetEvalResult result;
  result.metrics.k = options.top_k;

  if (!dataset::image_decoding_available()) {
    add_error(result.diagnostics,
              "this build has no image decoder; rebuild with OpenCV available");
    return result;
  }
  // The contract is the model's: with a folded stem, judge it against the
  // input as exported, not as folded.
  const StemFold &fold = plan.stem_fold;
  dataset::EvaluationContract contract;
  {
    deeptile::graph::GraphSpec model_view;
    const deeptile::graph::GraphSpec *graph = &plan.graph_plan.graph;
    if (fold.active) {
      model_view = plan.graph_plan.graph;
      model_view.tensors[fold.input_id].shape = {fold.channels, fold.height,
                                                 fold.width};
      graph = &model_view;
    }
    if (!dataset::resolve_evaluation_contract(*graph, contract,
                                              result.diagnostics)) {
      return result;
    }
  }

  const std::string &input_id = contract.input_id;
  const std::string &output_id = contract.output_id;

  dataset::DatasetIndex index;
  if (!dataset::load_imagenet_index(options.dataset_root, options.limit, index,
                                    result.diagnostics,
                                    options.allow_missing)) {
    return result;
  }
  result.listed = index.listed;
  result.present = index.present;
  result.selected = index.entries.size();
  result.listing_sha256 = dataset::listing_digest(index);
  if (!dataset::validate_evaluation_labels(index, contract.class_count,
                                           result.diagnostics)) {
    return result;
  }

  // Graph input geometry and quantization own the preprocessing contract.
  dataset::PreprocessConfig preprocess = options.preprocess;
  preprocess.target_side = contract.target_side;
  preprocess.frac = contract.input_frac;
  preprocess.fold_pad = fold.active ? fold.pad : -1;
  const std::vector<int> device_input_shape =
      fold.active ? std::vector<int>{4 * fold.channels, fold.folded_height,
                                     fold.folded_width}
                  : contract.input_shape;
  const RuntimeTensorBuffer *input_buffer = nullptr;
  for (const auto &buffer : plan.buffers) {
    if (buffer.id == input_id) {
      input_buffer = &buffer;
      break;
    }
  }
  if (!input_buffer || input_buffer->shape != device_input_shape) {
    add_error(result.diagnostics,
              "prepared graph input does not match the evaluation contract");
    return result;
  }

  std::ofstream predictions;
  if (!options.predictions_csv.empty()) {
    predictions.open(options.predictions_csv);
    if (!predictions) {
      add_error(result.diagnostics,
                "cannot open predictions CSV: " + options.predictions_csv);
      return result;
    }
    predictions << "index,image,label,top1,correct,top" << options.top_k << "\n";
    predictions.flush();
    if (!predictions) {
      add_error(result.diagnostics,
                "cannot write predictions CSV: " + options.predictions_csv);
      return result;
    }
  }

  std::unique_ptr<XrtSession> session =
      XrtSession::open(plan, xrt_options, result.diagnostics);
  if (!session) {
    return result;
  }
  result.graph_macs = session->graph_macs();
  result.graph_ops = macs_to_ops(result.graph_macs);

  const int workers = resolve_workers(options.workers);
  result.workers = workers;
  // The thread submitting to the accelerator needs a core of its own. Without
  // one, its wake-up latency is charged to the device and inflates kernel time.
  const unsigned cores = std::thread::hardware_concurrency();
  if (cores != 0 && static_cast<unsigned>(workers) + 1u > cores) {
    result.cpu_oversubscribed = true;
    result.diagnostics.push_back(
        {RuntimeDiagnosticSeverity::Warning, "dataset-eval",
         "workers=" + std::to_string(workers) + " on " +
             std::to_string(cores) +
             " cores leaves none for submission; kernel-only timing will be "
             "inflated and must not be quoted as accelerator throughput"});
  }
  const std::size_t queue_depth =
      options.queue_depth != 0 ? options.queue_depth
                               : static_cast<std::size_t>(workers) * 2;
  result.queue_depth = queue_depth;

  // Pack on the worker threads: at full input size this is real per-image work
  // and doing it between inference calls would idle the accelerator.
  const std::vector<int> input_shape = input_buffer->shape;
  // An input that fits one channel group can be produced directly in packed
  // form, skipping the intermediate CHW tensor and its repack. Anything wider
  // spans multiple groups and takes the general path.
  const int direct_lanes =
      input_shape[0] <= deeptile::tile::hw_defaults::kIcTileSize
          ? deeptile::tile::hw_defaults::kIcTileSize
          : 0;
  dataset::PreprocessPipeline pipeline(
      index, preprocess, workers, queue_depth, dataset::default_image_loader(),
      [input_shape](const std::vector<std::int8_t> &chw) {
        return pack_activation_chw(chw, input_shape);
      },
      direct_lanes, dataset::default_region_loader());

  std::vector<XrtLaunchProfile> profiles;
  profiles.reserve(plan.launches.size());
  std::vector<double> per_image_kernel_ms;
  per_image_kernel_ms.reserve(index.entries.size());
  dataset::PreparedImage image;
  // Both clocks are read together: the steady clock measures the interval, the
  // realtime one names it for a power sampler that cannot see this process.
  const auto wall_start = Clock::now();
  result.window.start_ns = realtime_now_ns();

  while (pipeline.next(image)) {
    if (!image.ok) {
      std::cerr << "[WARN] " << image.error << "\n";
      ++result.prepare_failed;
      continue;
    }

    const auto upload_start = Clock::now();
    if (!set_graph_input_packed(plan, input_id, std::move(image.packed),
                                result.diagnostics)) {
      return result;
    }
    if (!session->upload_inputs(result.diagnostics)) {
      return result;
    }
    result.upload_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - upload_start)
            .count();
    profiles.clear();
    if (!session->execute(&profiles, static_cast<int>(image.index),
                          result.diagnostics)) {
      return result;
    }
    const auto download_start = Clock::now();
    if (!session->download_outputs(result.diagnostics)) {
      return result;
    }
    result.download_ms += std::chrono::duration<double, std::milli>(
                              Clock::now() - download_start)
                              .count();
    double image_kernel_ms = 0.0;
    for (const auto &profile : profiles) {
      image_kernel_ms += profile.kernel_ms;
    }
    result.kernel_ms += image_kernel_ms;
    per_image_kernel_ms.push_back(image_kernel_ms);

    const RuntimeTensorBuffer *output = nullptr;
    for (const auto &buffer : plan.buffers) {
      if (buffer.id == output_id) {
        output = &buffer;
        break;
      }
    }
    if (!output) {
      add_error(result.diagnostics, "missing graph output buffer");
      return result;
    }
    const std::vector<std::int8_t> logits =
        unpack_packed_activation_to_chw(*output);
    result.metrics.accumulate(logits, image.label);
    ++result.evaluated;

    if (predictions.is_open() || options.verbose) {
      const std::vector<int> ranked =
          dataset::top_k(logits, std::max(options.top_k, 1));
      const int top1 = ranked.empty() ? -1 : ranked.front();
      const bool correct = top1 == image.label;
      if (predictions.is_open()) {
        predictions << image.index << "," << image.name << "," << image.label
                    << "," << top1 << "," << (correct ? 1 : 0) << ",";
        for (std::size_t i = 0; i < ranked.size(); ++i) {
          predictions << (i == 0 ? "" : " ") << ranked[i];
        }
        predictions << "\n";
      }
      if (options.verbose) {
        std::cout << "  " << image.name << " label=" << image.label << " ("
                  << dataset::class_label(index.class_names, image.label)
                  << ") top1=" << top1 << " ("
                  << dataset::class_label(index.class_names, top1) << ") "
                  << (correct ? "OK" : "MISS") << "\n";
      }
    }
  }

  result.window.end_ns = realtime_now_ns();
  result.wall_ms = ms_since(wall_start);
  if (!per_image_kernel_ms.empty()) {
    std::sort(per_image_kernel_ms.begin(), per_image_kernel_ms.end());
    result.kernel_ms_min = per_image_kernel_ms.front();
    result.kernel_ms_median =
        per_image_kernel_ms[per_image_kernel_ms.size() / 2];
  }
  const dataset::PipelineStats stats = pipeline.stats();
  result.prepare_cpu_ms = stats.decode_ms + stats.preprocess_ms;
  if (predictions.is_open()) {
    predictions.flush();
    if (!predictions) {
      add_error(result.diagnostics,
                "failed while writing predictions CSV: " +
                    options.predictions_csv);
      return result;
    }
  }
  result.ok = true;
  return result;
}

void print_dataset_eval_summary(const DatasetEvalResult &result,
                                std::ostream &out) {
  out << "==========================================================\n";
  out << "  Dataset evaluation\n";
  out << "==========================================================\n";
  out << "  images listed        : " << result.listed << "\n";
  out << "  images present       : " << result.present << "\n";
  out << "  images selected      : " << result.selected << "\n";
  out << "  images evaluated     : " << result.evaluated << "\n";
  if (result.prepare_failed != 0) {
    out << "  preparation failures : " << result.prepare_failed << "\n";
  }
  if (result.metrics.unlabelled != 0) {
    out << "  unlabelled           : " << result.metrics.unlabelled << "\n";
  }

  const auto flags = out.flags();
  out << std::fixed << std::setprecision(2);
  out << "  top-1 accuracy       : " << 100.0 * result.metrics.top1_accuracy()
      << " %  (" << result.metrics.top1_correct << "/"
      << result.metrics.evaluated << ")\n";
  out << "  top-" << result.metrics.k << " accuracy       : "
      << 100.0 * result.metrics.topk_accuracy() << " %  ("
      << result.metrics.topk_correct << "/" << result.metrics.evaluated
      << ")\n";

  if (result.evaluated != 0) {
    const double n = static_cast<double>(result.evaluated);
    const double kernel_ms_per_image = result.kernel_ms / n;
    const double wall_ms_per_image = result.wall_ms / n;

    out << std::setprecision(3);
    out << "  kernel time          : " << kernel_ms_per_image
        << " ms/img  (median " << result.kernel_ms_median << ", min "
        << result.kernel_ms_min << ")\n";
    out << "  preparation CPU      : " << result.prepare_cpu_ms / n
        << " ms/img (summed across workers)\n";
    out << "  input upload         : " << result.upload_ms / n
        << " ms/img  (copy and flush, outside kernel time)\n";
    out << "  output download      : " << result.download_ms / n
        << " ms/img  (sync and copy, outside kernel time)\n";
    out << "  wall time            : " << wall_ms_per_image << " ms/img  ("
        << 1000.0 * n / result.wall_ms << " img/s)\n";
    const double busy =
        result.wall_ms <= 0.0 ? 0.0 : result.kernel_ms / result.wall_ms;
    out << "  accelerator busy     : " << 100.0 * busy << " % of wall time\n";

    // Both scopes are reported because a throughput figure is ambiguous
    // without one: kernel-only describes the accelerator, end-to-end describes
    // what this platform delivers including host preparation.
    out << std::setprecision(1);
    out << "  graph ops/img        : " << static_cast<double>(result.graph_ops) / 1.0e9
        << " GOP (1 MAC = 2 ops)\n";
    out << "  avg kernel throughput: "
        << compute_rate(result.graph_ops, kernel_ms_per_image)
        << " GOPS (kernel-only)\n";
    // The minimum is the least contended image in the run, so it is the best
    // available estimate of uncontended accelerator throughput.
    out << "  peak kernel throughput: "
        << compute_rate(result.graph_ops, result.kernel_ms_min)
        << " GOPS (least contended image)\n";
    out << "  end-to-end throughput: "
        << compute_rate(result.graph_ops, wall_ms_per_image)
        << " GOPS (includes host preparation)\n";
  }
  out.flags(flags);
  if (result.cpu_oversubscribed) {
    out << "  NOTE: no core was reserved for submission; kernel-only figures\n"
        << "        above include scheduling delay and understate the device.\n";
  }
  out << "==========================================================\n";
}

} // namespace runtime
} // namespace deeptile
