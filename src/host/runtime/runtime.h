/**
 * @file runtime.h
 * @brief Runtime-plan preparation and activation-memory planning interface.
 * @ingroup deeptile_host
 */

#ifndef DEEPTILE_RUNTIME_H_
#define DEEPTILE_RUNTIME_H_

#include "runtime/runtime_types.h"

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace deeptile {
namespace runtime {

/**
 * The interval a run reports its throughput over, on CLOCK_REALTIME.
 *
 * Recorded so an out-of-process power sampler can integrate energy over
 * exactly the interval that produced the reported images per second, rather
 * than over the whole process lifetime. Startup, XCLBIN load, and weight
 * upload all draw power and all precede steady-state inference, so a window
 * taken from process boundaries overstates joules per image substantially.
 *
 * CLOCK_REALTIME is chosen because it is the clock the kernel's IIO drivers
 * timestamp samples with; host markers and sensor samples therefore share a
 * time base with no conversion step. It is not monotonic, so a run spanning a
 * clock step records a window that a reader must sanity-check against the
 * separately reported wall time.
 *
 * A default-constructed window is absent, not zero-length, and is omitted from
 * a record rather than written as zeros.
 */
struct MeasurementWindow {
  std::uint64_t start_ns = 0;
  std::uint64_t end_ns = 0;

  bool valid() const { return start_ns != 0 && end_ns > start_ns; }
  double duration_s() const {
    return valid() ? static_cast<double>(end_ns - start_ns) / 1e9 : 0.0;
  }
};

/** Nanoseconds on CLOCK_REALTIME; the time base MeasurementWindow uses. */
std::uint64_t realtime_now_ns();

/** Prepare buffers, launches, and activation storage for a compiled graph. */
RuntimeExecutionPlan
prepare_runtime(const deeptile::graph::GraphExecutionPlan &plan);

/** As above, taking the listed tensors' contents from `overrides`. */
RuntimeExecutionPlan
prepare_runtime(const deeptile::graph::GraphExecutionPlan &plan,
                const TensorOverrides &overrides);

/**
 * Load, compile, and prepare a graph handoff in one operation.
 * A qualifying stride-2 stem is first rewritten by space-to-depth unless
 * DEEPTILE_STEM_S2D=0 (see stem_fold.h); the plan records it in `stem_fold`.
 */
bool load_compile_prepare_runtime(
    const std::string &graph_json_path, RuntimeExecutionPlan &out,
    std::vector<RuntimeDiagnostic> &diagnostics);

/**
 * Pack a CHW int8 activation into the hardware layout.
 *
 * Exposed so callers can pack off the critical path, for example on a worker
 * thread, rather than between inference calls.
 */
std::vector<std::uint8_t> pack_activation_chw(const std::vector<std::int8_t> &chw,
                                              const std::vector<int> &shape);

/**
 * Install an already-packed graph input.
 *
 * The buffer must have been produced by pack_activation_chw() for this input's
 * shape; its size is checked against the prepared layout.
 */
bool set_graph_input_packed(RuntimeExecutionPlan &plan,
                            const std::string &tensor_id,
                            std::vector<std::uint8_t> packed,
                            std::vector<RuntimeDiagnostic> &diagnostics);

/**
 * Replace a graph input's contents with a CHW int8 tensor.
 *
 * Repacks into the hardware activation layout exactly as preparation does, so
 * successive inputs can be pushed through an already-prepared plan without
 * recompiling. The tensor must match the input's declared CHW shape.
 *
 * @return False if the tensor is unknown, is not a prepared graph input, or
 *         has the wrong element count.
 */
bool set_graph_input_chw(RuntimeExecutionPlan &plan,
                         const std::string &tensor_id,
                         const std::vector<std::int8_t> &chw,
                         std::vector<RuntimeDiagnostic> &diagnostics);

/**
 * Unpack a hardware-layout activation buffer back to CHW.
 *
 * Inverse of the packing applied during preparation. Channel padding
 * introduced by the pack is dropped, so the result matches the exporter's
 * canonical layout and can be compared to a reference directly.
 *
 * @return Empty if the buffer is not a CHW activation.
 */
std::vector<std::int8_t>
unpack_packed_activation_to_chw(const RuntimeTensorBuffer &buffer);

/** Return packed activation storage in bytes for a CHW shape. */
std::size_t packed_activation_bytes(const std::vector<int> &shape, int cpack);
/** Return packed weight storage in bytes for a launch. */
std::size_t packed_weight_bytes(const deeptile::tile::ConvLayerParams &params,
                                int cpo, int cpi);
/** Return the padded int32 bias element count. */
std::size_t shifted_bias_count(int channels, int cpo);

const RuntimeTensorStorageBinding *
find_activation_binding(const RuntimeExecutionPlan &plan,
                        const std::string &tensor_id);

const RuntimeActivationSlot *
find_activation_slot(const RuntimeExecutionPlan &plan,
                     const std::string &tensor_id);

RuntimeActivationSlot *find_activation_slot(RuntimeExecutionPlan &plan,
                                            const std::string &tensor_id);

std::size_t activation_tensor_bytes(const RuntimeExecutionPlan &plan,
                                    const std::string &tensor_id);

/** Print activation-slot assignments and lifetimes. */
void print_activation_memory_plan(const RuntimeExecutionPlan &plan,
                                  std::ostream &out);

} // namespace runtime
} // namespace deeptile

#endif // DEEPTILE_RUNTIME_H_
