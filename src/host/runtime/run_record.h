/**
 * @file run_record.h
 * @brief Declares the minimal DeepTile execution-result record.
 * @ingroup deeptile_host
 */

#ifndef DEEPTILE_RUNTIME_RUN_RECORD_H_
#define DEEPTILE_RUNTIME_RUN_RECORD_H_

#include "runtime/runtime.h"

#include <cstddef>
#include <string>

namespace deeptile {
namespace runtime {

/**
 * Digests of the artifacts a run consumed, so the record can be checked
 * against the packages rather than trusted.
 *
 * These are the values the host has already verified by the time it writes a
 * record: naming an artifact identifies which one was *intended*, while its
 * digest identifies which one actually ran. Each is a bare SHA-256 hex string,
 * or empty when the run had no such artifact — a legacy graph-only fixture has
 * no manifest, and a raw XCLBIN argument carries no TargetPackage.
 */
struct RunArtifactDigests {
  std::string model_sha256;    ///< Of the ModelPackage manifest, which
                               ///< inventories every artifact in turn.
  std::string xclbin_sha256;
  std::string boot_sha256;     ///< The image that programmed the fabric.
};

/** Model, target, and software identity shared by every run mode. */
struct RunRecordContext {
  std::string model;
  std::string target;
  std::string source_revision;

  /**
   * Whether the paths that determine this artifact had uncommitted changes.
   *
   * Scoped deliberately: documentation and paper edits do not affect the
   * executable, and a whole-tree check reports dirty for them, which leaves
   * the flag permanently set and therefore meaningless.
   */
  bool source_dirty = false;
  std::string source_scope;  ///< Paths the dirty check covered.
  std::size_t launches = 0;
  bool fused_depthwise = false;
  bool stem_space_to_depth = false;
  std::string launch_mode;  ///< "direct" or "xrt-run"; see xrt_launch_mode().
  RunArtifactDigests digests;
  std::string schedule_flags;  ///< Non-default lowering switches; usually empty.
};

/** Results and controls specific to one exported-input execution. */
struct SingleRunSummary {
  bool passed = false;
  int warmup_runs = 0;
  int measured_runs = 0;
  bool comparison_performed = false;
  bool exact_reference_match = false;
  std::size_t compared_elements = 0;
  double kernel_ms = 0.0;
  double kernel_gops = 0.0;
  std::string launch_profile;
  bool measure_latency = false;
  double graph_wall_ms = 0.0;
  std::string graph_profile;

  /// Measured-run interval; includes per-run readback when measure_latency.
  MeasurementWindow window;
};

/** Results and controls specific to one dataset evaluation. */
struct DatasetRunSummary {
  bool passed = false;
  std::string dataset;
  std::size_t limit = 0;
  int workers = 0;
  std::size_t queue_depth = 0;
  int top_k = 5;
  std::string preprocessing;
  int short_side = 0;
  std::size_t images = 0;
  double top1_percent = 0.0;
  double topk_percent = 0.0;
  double kernel_ms_per_image = 0.0;
  double kernel_ms_median = 0.0;
  double kernel_ms_min = 0.0;
  double wall_ms_per_image = 0.0;
  double images_per_second = 0.0;
  double accelerator_busy_percent = 0.0;

  double graph_ops = 0.0;
  double avg_kernel_gops = 0.0;
  double peak_kernel_gops = 0.0;
  double end_to_end_gops = 0.0;

  /**
   * SHA-256 over the ordered "<name> <label>\n" lines of the images actually
   * evaluated. One value answers which images a run covered; a full run
   * reproduces the digest the preparation script printed.
   */
  std::string dataset_sha256;

  /// Wall interval the throughput figures above are computed over.
  MeasurementWindow window;

  std::string predictions;
};

/** Derive the common identity from the prepared plan and selected target. */
/**
 * How the XRT backend starts launches. "direct" (the default) writes the
 * compute unit's control registers and polls for completion, costing about
 * 3 µs of host time per launch; "xrt-run" goes through XRT's scheduler, about
 * 65 µs per launch. Kernel time includes this cost, so a RunRecord states the
 * mode. DEEPTILE_XRT_LAUNCH=run selects the scheduler.
 */
std::string xrt_launch_mode();

RunRecordContext make_run_record_context(
    const RuntimeExecutionPlan &plan, const std::string &target_id,
    const RunArtifactDigests &digests = {});

/**
 * Digest the ModelPackage manifest beside a graph file.
 *
 * Returns an empty string when the graph is a manifest-less fixture, which is
 * not an error here: it is the absence of an integrity claim, and the record
 * says so by omitting the field.
 */
std::string model_package_digest(const std::string &graph_path);

/** Write one `tilecnn.run.v1` record atomically. */
bool write_single_run_record(const std::string &path,
                             const RunRecordContext &context,
                             const SingleRunSummary &summary,
                             std::string &error);

/** Write one dataset-mode `tilecnn.run.v1` record atomically. */
bool write_dataset_run_record(const std::string &path,
                              const RunRecordContext &context,
                              const DatasetRunSummary &summary,
                              std::string &error);

} // namespace runtime
} // namespace deeptile

#endif // DEEPTILE_RUNTIME_RUN_RECORD_H_
