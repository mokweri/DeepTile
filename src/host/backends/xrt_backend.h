/**
 * @file xrt_backend.h
 * @brief XRT execution and profiling interface.
 * @ingroup deeptile_host
 */

#ifndef DEEPTILE_RUNTIME_XRT_BACKEND_H_
#define DEEPTILE_RUNTIME_XRT_BACKEND_H_

#include "runtime/runtime.h"

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace deeptile {
namespace runtime {

/** Options controlling device execution, profiling, and comparison. */
struct XrtRunOptions {
  std::string xclbin_path;
  std::string graph_path;
  std::string kernel_name = "tile_conv";
  std::string csv_path;
  int device_index = 0;
  int warmup_runs = 1;
  int measured_runs = 1;
  bool compare_outputs = true;
  bool verbose = false;
  bool measure_latency = false; ///< Read outputs on every batch-one iteration.
};

/** Per-launch timing and throughput sample. */
struct XrtLaunchProfile {
  std::string graph;
  std::string launch_id;
  std::size_t launch_index = 0;
  int run_index = 0;
  double kernel_ms = 0.0;
  std::uint64_t macs = 0;
  std::uint64_t ops = 0;
  double gmacs = 0.0;
  double gops = 0.0;
};

/** Host wall intervals; none of these fields measures device-active cycles. */
struct XrtGraphProfile {
  int run_index = 0;
  double upload_ms = 0.0;
  double execute_ms = 0.0;
  double download_ms = 0.0;
  double wall_ms = 0.0;
};

/**
 * Operation-count convention: one multiply-accumulate counts as two operations.
 *
 * Reported throughput is meaningless without this convention stated alongside
 * it, so it lives in one place rather than being open-coded per call site.
 */
std::uint64_t macs_to_ops(std::uint64_t macs);

/** Giga-operations per second from an operation count and a millisecond span. */
double compute_rate(std::uint64_t operations, double elapsed_ms);

/** Aggregate result of an XRT graph execution. */
struct XrtRunResult {
  bool ok = false;
  std::vector<RuntimeDiagnostic> diagnostics;
  std::vector<XrtLaunchProfile> profiles;
  std::vector<XrtGraphProfile> graph_profiles;
  /**
   * The measured runs' wall interval on CLOCK_REALTIME.
   *
   * Spans measured iterations only. In latency mode each iteration includes
   * output readback; otherwise one final readback follows this window.
   */
  MeasurementWindow window;

  double avg_kernel_latency_ms = 0.0;
  double avg_graph_wall_ms = 0.0;
  double avg_gmacs = 0.0;
  double gops = 0.0;
  std::uint64_t graph_macs = 0;
  std::uint64_t graph_ops = 0;
};

/**
 * A device, XRT-registered XCLBIN, and every buffer a prepared plan needs.
 *
 * Separating session lifetime from execution is what makes repeated inference
 * possible: device buffers, weights, and descriptors are allocated and
 * uploaded once, then each image only exchanges activations. Constructing a
 * session per image would re-upload every weight and dominate the run.
 *
 * A session is bound to the plan it was opened with and must not outlive it.
 */
class XrtSession {
public:
  /**
   * Open a device, register the XCLBIN with XRT, and allocate plan buffers.
   *
   * @return Null on failure, with the reason in `diagnostics`.
   */
  static std::unique_ptr<XrtSession>
  open(RuntimeExecutionPlan &plan, const XrtRunOptions &options,
       std::vector<RuntimeDiagnostic> &diagnostics);

  ~XrtSession();

  XrtSession(const XrtSession &) = delete;
  XrtSession &operator=(const XrtSession &) = delete;

  /** Copy the plan's graph-input host buffers to the device. */
  bool upload_inputs(std::vector<RuntimeDiagnostic> &diagnostics);

  /**
   * Execute every launch once, in plan order.
   *
   * @param profiles Optional per-launch timing sink; nothing is recorded when
   *        null, which is how warm-up runs are excluded.
   * @param run_index Value stamped into collected profiles.
   * @param diagnostics Receives execution failures and validation errors.
   */
  bool execute(std::vector<XrtLaunchProfile> *profiles, int run_index,
               std::vector<RuntimeDiagnostic> &diagnostics);

  /** Copy graph outputs from the device back into the plan's host buffers. */
  bool download_outputs(std::vector<RuntimeDiagnostic> &diagnostics);

  /** Total multiply-accumulates in the graph, for rate reporting. */
  std::uint64_t graph_macs() const;

private:
  struct Impl;
  explicit XrtSession(std::unique_ptr<Impl> impl);
  /** Runs one launch under direct register control and waits for it. */
  void run_direct(std::size_t launch_index);
  std::unique_ptr<Impl> impl_;
};

/** Executes a prepared runtime plan through XRT. */
class XrtBackend {
public:
  explicit XrtBackend(XrtRunOptions options);

  XrtRunResult run(RuntimeExecutionPlan &plan) const;

private:
  XrtRunOptions options_;
};

/** Print a compact or per-run execution summary. */
void print_xrt_run_summary(const XrtRunResult &result, std::ostream &out,
                           bool verbose = false);

bool write_xrt_profile_csv(const XrtRunResult &result,
                           const std::string &csv_path,
                           std::vector<RuntimeDiagnostic> &diagnostics);

bool write_xrt_graph_profile_csv(const XrtRunResult &result,
                                 const std::string &csv_path,
                                 std::vector<RuntimeDiagnostic> &diagnostics);

} // namespace runtime
} // namespace deeptile

#endif // DEEPTILE_RUNTIME_XRT_BACKEND_H_
