/**
 * @file dataset_eval.h
 * @brief Full-dataset inference over an open XRT session.
 * @ingroup deeptile_host
 */

#ifndef DEEPTILE_RUNTIME_DATASET_EVAL_H_
#define DEEPTILE_RUNTIME_DATASET_EVAL_H_

#include "backends/xrt_backend.h"
#include "dataset/metrics.h"
#include "dataset/preprocess.h"
#include "runtime/runtime.h"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace deeptile {
namespace runtime {

/** Dataset-evaluation controls, independent of device options. */
struct DatasetEvalOptions {
  std::string dataset_root;
  std::size_t limit = 0;      ///< Images to evaluate; zero means all present.
  int workers = 0;            ///< Preparation threads; zero picks a default.
  std::size_t queue_depth = 0;
  int top_k = 5;
  dataset::PreprocessConfig preprocess;
  std::string predictions_csv;  ///< Optional per-image record.
  bool allow_missing = false;   ///< Evaluate an incomplete listing anyway.
  bool verbose = false;
};

/** Outcome of a dataset run. */
struct DatasetEvalResult {
  bool ok = false;
  std::vector<RuntimeDiagnostic> diagnostics;
  dataset::ClassificationMetrics metrics;

  std::size_t listed = 0;
  std::size_t present = 0;
  std::size_t selected = 0;
  /// SHA-256 over the ordered "<name> <label>\n" lines of the selected images.
  std::string listing_sha256;
  std::size_t evaluated = 0;
  std::size_t prepare_failed = 0;
  int workers = 0;               ///< Effective preparation-worker count.
  std::size_t queue_depth = 0;   ///< Effective prepared-image queue depth.

  /**
   * The wall interval on CLOCK_REALTIME, for out-of-process power alignment.
   *
   * Identical to the interval `wall_ms` measures, so energy integrated over it
   * and the throughput reported from `wall_ms` describe the same work.
   */
  MeasurementWindow window;

  double wall_ms = 0.0;         ///< Whole run, including preparation overlap.
  double kernel_ms = 0.0;       ///< Summed accelerator time.
  double prepare_cpu_ms = 0.0;  ///< Summed across workers, so CPU cost.
  /// Summed host handoff around the launches: copying the packed input into
  /// its buffer and flushing it to the device, and syncing and copying the
  /// outputs back. Excluded from kernel_ms.
  double upload_ms = 0.0;
  double download_ms = 0.0;

  /**
   * Per-image accelerator time spread.
   *
   * Kernel time is host wall-clock around the device wait, so it absorbs
   * scheduling delay whenever preparation threads compete with the submitting
   * thread. A median well above the minimum means the mean is contaminated and
   * kernel-only throughput must not be quoted from this run.
   */
  double kernel_ms_min = 0.0;
  double kernel_ms_median = 0.0;
  bool cpu_oversubscribed = false;
  std::uint64_t graph_macs = 0;  ///< Per image, not for the whole dataset.
  std::uint64_t graph_ops = 0;   ///< Per image, at two operations per MAC.
};

/**
 * Evaluate a dataset end to end on the accelerator.
 *
 * Opens one session for the whole run, so weights and descriptors are uploaded
 * once and each image exchanges only activations. Images are prepared on
 * worker threads and consumed in dataset order, so results do not depend on
 * worker count.
 *
 * @param plan Prepared plan; its input buffer is overwritten per image.
 * @param xrt_options Device, graph, and execution options.
 * @param options Dataset selection, preprocessing, and reporting controls.
 */
DatasetEvalResult run_dataset_evaluation(RuntimeExecutionPlan &plan,
                                         const XrtRunOptions &xrt_options,
                                         const DatasetEvalOptions &options);

void print_dataset_eval_summary(const DatasetEvalResult &result,
                                std::ostream &out);

} // namespace runtime
} // namespace deeptile

#endif // DEEPTILE_RUNTIME_DATASET_EVAL_H_
