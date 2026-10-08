/**
 * @file xrt_backend.cpp
 * @brief XRT graph execution, profiling, and buffer management.
 */

#include "backends/xrt_backend.h"
#include "runtime/run_record.h"

#include "common_defs.h"

#include "deeptile/compiler/environment.h"
#include "experimental/xrt_ip.h"
#include "experimental/xrt_xclbin.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace deeptile {
namespace runtime {
namespace {

using Clock = std::chrono::steady_clock;

void add_diag(std::vector<RuntimeDiagnostic> &diagnostics,
              RuntimeDiagnosticSeverity severity, const std::string &context,
              const std::string &message) {
  diagnostics.push_back({severity, context, message});
}

void add_error(std::vector<RuntimeDiagnostic> &diagnostics,
               const std::string &context, const std::string &message) {
  add_diag(diagnostics, RuntimeDiagnosticSeverity::Error, context, message);
}

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

RuntimeTensorBuffer *find_buffer(RuntimeExecutionPlan &plan,
                                 const std::string &id) {
  const auto it = plan.buffer_index_by_id.find(id);
  if (it == plan.buffer_index_by_id.end()) {
    return nullptr;
  }
  return &plan.buffers[it->second];
}

const RuntimeTensorBuffer *find_buffer(const RuntimeExecutionPlan &plan,
                                       const std::string &id) {
  const auto it = plan.buffer_index_by_id.find(id);
  if (it == plan.buffer_index_by_id.end()) {
    return nullptr;
  }
  return &plan.buffers[it->second];
}

std::size_t bo_size(std::size_t size) { return std::max<std::size_t>(1, size); }

xrt::bo make_bytes_bo(xrt::device &device, xrt::kernel &kernel, int arg_index,
                      const std::vector<std::uint8_t> &bytes,
                      bool sync_to_device) {
  xrt::bo bo(device, bo_size(bytes.size()), kernel.group_id(arg_index));
  if (!bytes.empty()) {
    auto *ptr = bo.map<std::uint8_t *>();
    std::memcpy(ptr, bytes.data(), bytes.size());
  }
  if (sync_to_device) {
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  return bo;
}

std::vector<biases_vector_dt>
pack_bias_words(const std::vector<std::int32_t> &bias_i32, int num_bias_vecs) {
  std::vector<biases_vector_dt> words(std::max(1, num_bias_vecs), 0);
  for (int group = 0; group < num_bias_vecs; ++group) {
    biases_vector_dt word = 0;
    for (int lane = 0; lane < OC_TILE_SIZE; ++lane) {
      const std::size_t idx =
          static_cast<std::size_t>(group) * OC_TILE_SIZE + lane;
      const std::uint32_t value =
          idx < bias_i32.size() ? static_cast<std::uint32_t>(bias_i32[idx])
                                : 0u;
      word.range((lane + 1) * 32 - 1, lane * 32) = value;
    }
    words[static_cast<std::size_t>(group)] = word;
  }
  return words;
}

template <typename WordT>
xrt::bo make_word_bo(xrt::device &device, xrt::kernel &kernel, int arg_index,
                     const std::vector<WordT> &words) {
  xrt::bo bo(device, bo_size(words.size() * sizeof(WordT)),
             kernel.group_id(arg_index));
  if (!words.empty()) {
    auto *ptr = bo.map<WordT *>();
    for (std::size_t i = 0; i < words.size(); ++i) {
      ptr[i] = words[i];
    }
  }
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  return bo;
}

template <typename DstT, typename SrcT>
std::vector<DstT> copy_descriptors(const std::vector<SrcT> &src) {
  static_assert(sizeof(DstT) == sizeof(SrcT),
                "XRT descriptor source/destination sizes must match");
  std::vector<DstT> out(src.size());
  if (!src.empty()) {
    std::memcpy(out.data(), src.data(), src.size() * sizeof(DstT));
  }
  return out;
}

std::uint64_t launch_macs(const deeptile::graph::LaunchPlan &launch) {
  const auto &p = launch.params;
  // Standalone depthwise uses one Kh*Kw filter per output
  // channel, no Cin reduction -- p.Cin is the full channel count, not the
  // per-group fan-in, so it must not be multiplied in here.
  std::uint64_t macs =
      p.depthwise
          ? static_cast<std::uint64_t>(p.Cout) * p.output_height() *
                p.output_width() * p.Kh * p.Kw
          : static_cast<std::uint64_t>(p.Cout) * p.output_height() *
                p.output_width() * p.Cin * p.Kh * p.Kw;
  if (launch.dw.enable) {
    // launch.params covers the producer, so add the fused depthwise work.
    macs += static_cast<std::uint64_t>(p.Cout) * launch.dw.Kh * launch.dw.Kw *
            launch.dw.out_h * launch.dw.out_w;
  }
  return macs;
}

bool validate_launch(const RuntimeExecutionPlan &plan, std::size_t index,
                     std::vector<RuntimeDiagnostic> &diagnostics) {
  const auto &binding = plan.launches[index];
  const RuntimeTensorBuffer *ifm = find_buffer(plan, binding.ifm_buffer);
  const RuntimeTensorBuffer *weight = find_buffer(plan, binding.weight_buffer);
  const RuntimeTensorBuffer *bias = find_buffer(plan, binding.bias_buffer);
  const RuntimeTensorBuffer *output = find_buffer(plan, binding.output_buffer);
  if (!ifm || !weight || !bias || !output) {
    add_error(diagnostics, "launch " + binding.id,
              "missing required runtime buffer");
    return false;
  }
  if (ifm->data_bytes.empty() || weight->data_bytes.empty() ||
      bias->int32_data.empty() || output->data_bytes.empty()) {
    add_error(diagnostics, "launch " + binding.id,
              "required runtime buffer has no data");
    return false;
  }
  if (binding.residual_buffer.has_value() &&
      !find_buffer(plan, *binding.residual_buffer)) {
    add_error(diagnostics, "launch " + binding.id,
              "missing residual buffer " + *binding.residual_buffer);
    return false;
  }
  if (!find_activation_binding(plan, binding.ifm_buffer) ||
      !find_activation_binding(plan, binding.output_buffer) ||
      (binding.residual_buffer.has_value() &&
       !find_activation_binding(plan, *binding.residual_buffer))) {
    add_error(diagnostics, "launch " + binding.id,
              "missing activation slot binding");
    return false;
  }
  if (binding.num_xfers != static_cast<int>(binding.transfer_descs.size()) ||
      binding.num_tiles != static_cast<int>(binding.compute_descs.size())) {
    add_error(diagnostics, "launch " + binding.id,
              "descriptor counts do not match launch binding");
    return false;
  }
  if (binding.kernel_size <= 0 || binding.stride <= 0 ||
      binding.spatial_tile_size <= 0 || binding.num_ic_tiles <= 0 ||
      binding.ifm_cig_stride_vecs <= 0) {
    add_error(diagnostics, "launch " + binding.id,
              "invalid launch-level tile_conv scalar arguments");
    return false;
  }
  return true;
}

struct PreparedLaunchBos {
  xrt::bo weight;
  xrt::bo bias;
  xrt::bo xfer_desc;
  xrt::bo comp_desc;
  xrt::bo dw_weights;  // Fused-depthwise compact blob on kernel argument 19.
};

std::size_t slot_index_by_id(const RuntimeExecutionPlan &plan,
                             const std::string &slot_id) {
  for (std::size_t i = 0; i < plan.activation_slots.size(); ++i) {
    if (plan.activation_slots[i].id == slot_id) {
      return i;
    }
  }
  return static_cast<std::size_t>(-1);
}

std::size_t slot_index_for_tensor(const RuntimeExecutionPlan &plan,
                                  const std::string &tensor_id) {
  const RuntimeTensorStorageBinding *binding =
      find_activation_binding(plan, tensor_id);
  if (!binding) {
    return static_cast<std::size_t>(-1);
  }
  return slot_index_by_id(plan, binding->slot_id);
}

} // namespace

std::uint64_t macs_to_ops(std::uint64_t macs) { return macs * 2u; }

double compute_rate(std::uint64_t operations, double elapsed_ms) {
  if (elapsed_ms <= 0.0) {
    return 0.0;
  }
  return static_cast<double>(operations) / elapsed_ms / 1.0e6;
}

struct XrtSession::Impl {
  RuntimeExecutionPlan &plan;
  XrtRunOptions options;

  xrt::device device;
  xrt::kernel kernel;

  std::vector<xrt::bo> activation_bos;
  xrt::bo zero_residual_bo;
  std::vector<PreparedLaunchBos> prepared_launches;

  /** Slot indices resolved once, since they cannot change during a session. */
  struct LaunchSlots {
    std::size_t ifm = static_cast<std::size_t>(-1);
    std::size_t output = static_cast<std::size_t>(-1);
    std::size_t residual = static_cast<std::size_t>(-1);
  };
  std::vector<LaunchSlots> launch_slots;
  /**
   * One run per launch, built once with every argument set. A launch's
   * arguments never change within a session (images differ only in buffer
   * contents), so each inference only restarts these runs. Constructing a run
   * per launch allocated a command buffer and rewrote all 23 arguments, which
   * dominated the fixed per-launch cost.
   */
  std::vector<xrt::run> runs;

  /**
   * Direct control (the default; DEEPTILE_XRT_LAUNCH=run selects XRT's
   * scheduler instead): the host owns the compute
   * unit exclusively, writes each launch's arguments into its control
   * registers and polls for completion, bypassing the driver's scheduler.
   * Register access is memory-mapped, so a launch costs a few register writes
   * instead of a command submission, an interrupt and a thread wake-up.
   */
  bool direct = false;
  xrt::ip ip;
  /** Per launch, (register offset, value) for every argument word. */
  std::vector<std::vector<std::pair<std::uint32_t, std::uint32_t>>> reg_images;
  /** Last value written at each register offset; unwritten words hold 2^32. */
  std::vector<std::uint64_t> reg_shadow;
  /** Interrupt enables found at open, restored when the session closes. */
  std::uint32_t saved_gie = 0;
  std::uint32_t saved_ier = 0;

  std::vector<std::size_t> input_slots;
  std::uint64_t graph_macs = 0;
  bool announced = false;

  Impl(RuntimeExecutionPlan &p, XrtRunOptions o)
      : plan(p), options(std::move(o)) {}

  ~Impl() {
    if (direct) {
      try {
        ip.write_register(kGlobalInterruptEnable, saved_gie);
        ip.write_register(kInterruptEnable, saved_ier);
      } catch (...) {
      }
    }
  }

  static constexpr std::uint32_t kGlobalInterruptEnable = 0x04;
  static constexpr std::uint32_t kInterruptEnable = 0x08;
  static constexpr std::uint32_t kInterruptStatus = 0x0c;
};

XrtSession::XrtSession(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

XrtSession::~XrtSession() = default;

std::unique_ptr<XrtSession>
XrtSession::open(RuntimeExecutionPlan &plan, const XrtRunOptions &options,
                 std::vector<RuntimeDiagnostic> &diagnostics) {
  static_assert(sizeof(deeptile::tile::WeightTransferDescriptor) ==
                    sizeof(uram_xfer_desc_t),
                "UramTransferDescriptor and uram_xfer_desc_t must match");
  static_assert(sizeof(deeptile::tile::TileDescriptor) == sizeof(descriptor_dt),
                "UramTileDescriptor and descriptor_dt must match");
  static_assert(sizeof(pixel_vector_dt) == IFM_VECTOR_WIDTH / 8,
                "pixel_vector_dt size mismatch");
  static_assert(sizeof(weight_vector_dt) == WEIGHT_VECTOR_WIDTH / 8,
                "weight_vector_dt size mismatch");
  static_assert(sizeof(activations_vector_dt) == OFM_VECTOR_WIDTH / 8,
                "activations_vector_dt size mismatch");
  static_assert(sizeof(biases_vector_dt) == BIAS_VECTOR_WIDTH / 8,
                "biases_vector_dt size mismatch");

  if (options.xclbin_path.empty()) {
    add_error(diagnostics, "xrt", "xclbin path is empty");
    return nullptr;
  }
  if (plan.activation_slots.empty()) {
    add_error(diagnostics, "xrt", "runtime activation slot plan is empty");
    return nullptr;
  }
  for (std::size_t i = 0; i < plan.launches.size(); ++i) {
    if (!validate_launch(plan, i, diagnostics)) {
      return nullptr;
    }
  }

  auto impl = std::unique_ptr<Impl>(new Impl(plan, options));
  try {
    impl->device = xrt::device(options.device_index);
    // The UUID is only needed to construct the kernel, so it stays local;
    // holding it as a member would drag libuuid into every host link.
    const auto uuid = impl->device.load_xclbin(options.xclbin_path);
    impl->kernel = xrt::kernel(impl->device, uuid, options.kernel_name);

    impl->activation_bos.reserve(plan.activation_slots.size());
    std::size_t max_activation_bytes = 1;
    // HLS: Guard one 256-bit read beyond an odd-width tile's final row.
    const std::size_t ifm_read_guard =
        std::size_t(2 * P_SPATIAL) * sizeof(pixel_vector_dt);
    for (const auto &slot : plan.activation_slots) {
      max_activation_bytes = std::max(max_activation_bytes, slot.byte_capacity);
      // Activation slots may be read through IFM/residual ports and written
      // through OFM ports across different launches.  Allocate with the IFM
      // memory group and pass the same BO to the other activation arguments.
      impl->activation_bos.emplace_back(
          impl->device, bo_size(slot.byte_capacity + ifm_read_guard),
          impl->kernel.group_id(0));
    }

    std::vector<std::uint8_t> zero_residual(max_activation_bytes, 0);
    impl->zero_residual_bo =
        make_bytes_bo(impl->device, impl->kernel, 4, zero_residual, true);

    impl->prepared_launches.reserve(plan.launches.size());
    impl->launch_slots.reserve(plan.launches.size());
    for (const auto &binding : plan.launches) {
      const RuntimeTensorBuffer *weight =
          find_buffer(plan, binding.weight_buffer);
      const RuntimeTensorBuffer *bias = find_buffer(plan, binding.bias_buffer);
      const auto bias_words =
          pack_bias_words(bias->int32_data, binding.num_bias_vecs);
      const auto xfer_descs =
          copy_descriptors<uram_xfer_desc_t>(binding.transfer_descs);
      const auto comp_descs =
          copy_descriptors<descriptor_dt>(binding.compute_descs);

      // Fused-DW compact blob (kernel arg 19). Non-DW launches still need a
      // valid BO, so default to a single zero word.
      std::vector<std::uint8_t> dw_bytes;
      if (!binding.dw_weight_buffer.empty()) {
        const RuntimeTensorBuffer *dw_blob =
            find_buffer(plan, binding.dw_weight_buffer);
        if (dw_blob) {
          dw_bytes = dw_blob->data_bytes;
        }
      }
      if (dw_bytes.empty()) {
        dw_bytes.assign(WEIGHT_VECTOR_WIDTH / 8, 0);
      }

      PreparedLaunchBos prepared;
      prepared.weight =
          make_bytes_bo(impl->device, impl->kernel, 1, weight->data_bytes, true);
      prepared.bias = make_word_bo(impl->device, impl->kernel, 2, bias_words);
      prepared.xfer_desc =
          make_word_bo(impl->device, impl->kernel, 5, xfer_descs);
      prepared.comp_desc =
          make_word_bo(impl->device, impl->kernel, 7, comp_descs);
      prepared.dw_weights =
          make_bytes_bo(impl->device, impl->kernel, 19, dw_bytes, true);
      impl->prepared_launches.push_back(std::move(prepared));

      Impl::LaunchSlots slots;
      slots.ifm = slot_index_for_tensor(plan, binding.ifm_buffer);
      slots.output = slot_index_for_tensor(plan, binding.output_buffer);
      slots.residual = binding.residual_buffer.has_value()
                           ? slot_index_for_tensor(plan, *binding.residual_buffer)
                           : static_cast<std::size_t>(-1);
      if (slots.ifm == static_cast<std::size_t>(-1) ||
          slots.output == static_cast<std::size_t>(-1) ||
          (binding.residual_buffer.has_value() &&
           slots.residual == static_cast<std::size_t>(-1))) {
        add_error(diagnostics, "launch " + binding.id,
                  "missing activation slot BO");
        return nullptr;
      }
      impl->launch_slots.push_back(slots);
    }

    impl->runs.reserve(plan.launches.size());
    for (std::size_t i = 0; i < plan.launches.size(); ++i) {
      const auto &binding = plan.launches[i];
      const Impl::LaunchSlots &slots = impl->launch_slots[i];
      PreparedLaunchBos &prepared = impl->prepared_launches[i];
      xrt::bo &bo_residual = binding.residual_buffer.has_value()
                                 ? impl->activation_bos[slots.residual]
                                 : impl->zero_residual_bo;
      // Argument order matches the tile_conv signature (kernel_abi.h).
      xrt::run run(impl->kernel);
      run.set_arg(0, impl->activation_bos[slots.ifm]);
      run.set_arg(1, prepared.weight);
      run.set_arg(2, prepared.bias);
      run.set_arg(3, impl->activation_bos[slots.output]);
      run.set_arg(4, bo_residual);
      run.set_arg(5, prepared.xfer_desc);
      run.set_arg(6, binding.num_xfers);
      run.set_arg(7, prepared.comp_desc);
      run.set_arg(8, binding.num_tiles);
      run.set_arg(9, binding.num_bias_vecs);
      run.set_arg(10, binding.shift_out);
      run.set_arg(11, binding.gap_mul);
      run.set_arg(12, binding.kernel_size);
      run.set_arg(13, binding.stride);
      run.set_arg(14, binding.use_ifm_cache ? 1 : 0);
      run.set_arg(15, binding.spatial_tile_size);
      run.set_arg(16, binding.num_ic_tiles);
      run.set_arg(17, binding.ifm_cig_stride_vecs);
      run.set_arg(18, binding.act_max);
      run.set_arg(19, prepared.dw_weights);
      run.set_arg(20, binding.dw_config);
      run.set_arg(21, binding.dw_shift);
      run.set_arg(22, binding.num_writeback_tiles);
      impl->runs.push_back(std::move(run));
    }

    impl->direct = deeptile::runtime::xrt_launch_mode() == "direct";
    if (impl->direct) {
      // Argument register offsets come from the xclbin; pointer arguments are
      // two 32-bit words, low word first.
      const xrt::xclbin xclbin(options.xclbin_path);
      const auto cus = xclbin.get_kernel(options.kernel_name).get_cus();
      if (cus.empty()) {
        add_error(diagnostics, "xrt", "no compute unit for kernel " +
                                          options.kernel_name);
        return nullptr;
      }
      struct ArgReg {
        std::uint32_t offset;
        std::size_t size;
      };
      std::vector<ArgReg> arg_regs(DEEPTILE_KERNEL_ARGUMENT_COUNT, {0, 0});
      std::uint32_t max_offset = 0;
      for (const auto &arg : cus.front().get_args()) {
        const auto index = static_cast<std::size_t>(arg.get_index());
        if (index < arg_regs.size()) {
          arg_regs[index] = {static_cast<std::uint32_t>(arg.get_offset()),
                             static_cast<std::size_t>(arg.get_size())};
          max_offset = std::max(max_offset,
                                static_cast<std::uint32_t>(arg.get_offset() + 8));
        }
      }
      impl->reg_shadow.assign(max_offset / 4 + 1, std::uint64_t(1) << 32);
      for (std::size_t i = 0; i < plan.launches.size(); ++i) {
        const auto &binding = plan.launches[i];
        const Impl::LaunchSlots &slots = impl->launch_slots[i];
        PreparedLaunchBos &prepared = impl->prepared_launches[i];
        const xrt::bo &bo_residual = binding.residual_buffer.has_value()
                                         ? impl->activation_bos[slots.residual]
                                         : impl->zero_residual_bo;
        // Argument order matches the tile_conv signature (kernel_abi.h).
        const std::uint64_t values[DEEPTILE_KERNEL_ARGUMENT_COUNT] = {
            impl->activation_bos[slots.ifm].address(), prepared.weight.address(),
            prepared.bias.address(), impl->activation_bos[slots.output].address(),
            bo_residual.address(), prepared.xfer_desc.address(),
            std::uint64_t(std::uint32_t(binding.num_xfers)), prepared.comp_desc.address(),
            std::uint64_t(std::uint32_t(binding.num_tiles)),
            std::uint64_t(std::uint32_t(binding.num_bias_vecs)),
            std::uint64_t(std::uint32_t(binding.shift_out)),
            std::uint64_t(std::uint32_t(binding.gap_mul)),
            std::uint64_t(std::uint32_t(binding.kernel_size)),
            std::uint64_t(std::uint32_t(binding.stride)),
            std::uint64_t(binding.use_ifm_cache ? 1 : 0),
            std::uint64_t(std::uint32_t(binding.spatial_tile_size)),
            std::uint64_t(std::uint32_t(binding.num_ic_tiles)),
            std::uint64_t(std::uint32_t(binding.ifm_cig_stride_vecs)),
            std::uint64_t(std::uint32_t(binding.act_max)), prepared.dw_weights.address(),
            std::uint64_t(std::uint32_t(binding.dw_config)),
            std::uint64_t(std::uint32_t(binding.dw_shift)),
            std::uint64_t(std::uint32_t(binding.num_writeback_tiles))};
        std::vector<std::pair<std::uint32_t, std::uint32_t>> image;
        for (std::size_t a = 0; a < arg_regs.size(); ++a) {
          if (arg_regs[a].size == 0) {
            add_error(diagnostics, "xrt", "xclbin lacks argument " + std::to_string(a));
            return nullptr;
          }
          image.emplace_back(arg_regs[a].offset,
                             static_cast<std::uint32_t>(values[a] & 0xffffffffu));
          if (arg_regs[a].size == 8) {
            image.emplace_back(arg_regs[a].offset + 4,
                               static_cast<std::uint32_t>(values[a] >> 32));
          }
        }
        impl->reg_images.push_back(std::move(image));
      }
      // The compute unit cannot be shared: release the scheduler's handles,
      // then open it exclusively.
      const std::string cu = cus.front().get_name();
      const std::string cu_instance = cu.substr(cu.find(':') + 1);
      impl->runs.clear();
      impl->kernel = xrt::kernel();
      impl->ip = xrt::ip(impl->device, uuid,
                         options.kernel_name + ":{" + cu_instance + "}");
      // The driver's interrupt handler would service and acknowledge the
      // unit's completion before the host's poll sees it, so the unit's
      // interrupts stay off for the session and are restored afterwards.
      impl->saved_gie = impl->ip.read_register(Impl::kGlobalInterruptEnable);
      impl->saved_ier = impl->ip.read_register(Impl::kInterruptEnable);
      impl->ip.write_register(Impl::kGlobalInterruptEnable, 0);
      impl->ip.write_register(Impl::kInterruptEnable, 0);
      // Status bits are toggle-on-write; writing back the set bits clears them.
      impl->ip.write_register(Impl::kInterruptStatus,
                              impl->ip.read_register(Impl::kInterruptStatus));
    }

    for (const auto &input_id : plan.graph_plan.graph.inputs) {
      const std::size_t slot_index = slot_index_for_tensor(plan, input_id);
      if (slot_index == static_cast<std::size_t>(-1)) {
        add_error(diagnostics, "xrt",
                  "missing graph input activation slot: " + input_id);
        return nullptr;
      }
      impl->input_slots.push_back(slot_index);
    }

    for (const auto &launch : plan.graph_plan.launches) {
      impl->graph_macs += launch_macs(launch);
    }
  } catch (const std::exception &e) {
    add_error(diagnostics, "xrt", e.what());
    return nullptr;
  }

  return std::unique_ptr<XrtSession>(new XrtSession(std::move(impl)));
}

bool XrtSession::upload_inputs(std::vector<RuntimeDiagnostic> &diagnostics) {
  try {
    const auto &inputs = impl_->plan.graph_plan.graph.inputs;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const RuntimeTensorBuffer *input = find_buffer(impl_->plan, inputs[i]);
      if (!input) {
        add_error(diagnostics, "xrt", "missing graph input: " + inputs[i]);
        return false;
      }
      xrt::bo &bo = impl_->activation_bos[impl_->input_slots[i]];
      std::memcpy(bo.map<std::uint8_t *>(), input->data_bytes.data(),
                  input->data_bytes.size());
      bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
  } catch (const std::exception &e) {
    add_error(diagnostics, "xrt", e.what());
    return false;
  }
  return true;
}

void XrtSession::run_direct(std::size_t launch_index) {
  // Control register bits of an HLS ap_ctrl_chain block.
  constexpr std::uint32_t kCtrl = 0x00;
  constexpr std::uint32_t kApStart = 0x1;
  constexpr std::uint32_t kApDone = 0x2;
  constexpr std::uint32_t kApContinue = 0x10;
  // Generous against any single launch; a stuck kernel must not hang the host.
  constexpr auto kTimeout = std::chrono::seconds(10);

  for (const auto &[offset, value] : impl_->reg_images[launch_index]) {
    std::uint64_t &last = impl_->reg_shadow[offset / 4];
    if (last != value) {
      impl_->ip.write_register(offset, value);
      last = value;
    }
  }
  impl_->ip.write_register(kCtrl, kApStart);
  const auto deadline = Clock::now() + kTimeout;
  while ((impl_->ip.read_register(kCtrl) & kApDone) == 0) {
    if (Clock::now() > deadline) {
      throw std::runtime_error("tile_conv did not complete within 10 s");
    }
  }
  // An ap_ctrl_chain block holds ap_done until the host acknowledges it.
  impl_->ip.write_register(kCtrl, kApContinue);
}

bool XrtSession::execute(std::vector<XrtLaunchProfile> *profiles, int run_index,
                         std::vector<RuntimeDiagnostic> &diagnostics) {
  try {
    RuntimeExecutionPlan &plan = impl_->plan;
    for (std::size_t launch_index = 0; launch_index < plan.launches.size();
         ++launch_index) {
      const auto &binding = plan.launches[launch_index];
      const auto &launch = plan.graph_plan.launches[launch_index];

      if (!impl_->announced && impl_->options.verbose) {
        std::cout << "[XRT_LAUNCH] " << launch_index << " " << binding.id
                  << " xfers=" << binding.num_xfers
                  << " tiles=" << binding.num_tiles
                  << " bias_vecs=" << binding.num_bias_vecs
                  << " shift_out=" << binding.shift_out
                  << " gap_mul=" << binding.gap_mul
                  << " k=" << binding.kernel_size
                  << " stride=" << binding.stride
                  << " use_ifm_cache=" << (binding.use_ifm_cache ? 1 : 0)
                  << " spatial_tile_size=" << binding.spatial_tile_size
                  << " num_ic_tiles=" << binding.num_ic_tiles
                  << " cig_stride=" << binding.ifm_cig_stride_vecs << "\n";
      }

      // HLS: tile_conv is ap_ctrl_chain, so a launch must complete before the
      // next starts; each reads the activations its predecessor wrote.
      const auto kernel_start = Clock::now();
      if (impl_->direct) {
        run_direct(launch_index);
      } else {
        xrt::run &run = impl_->runs[launch_index];
        run.start();
        run.wait();
      }
      const auto kernel_end = Clock::now();

      if (profiles != nullptr) {
        XrtLaunchProfile profile;
        profile.graph = impl_->options.graph_path;
        profile.launch_id = binding.id;
        profile.launch_index = launch_index;
        profile.run_index = run_index;
        profile.kernel_ms = elapsed_ms(kernel_start, kernel_end);
        profile.macs = launch_macs(launch);
        profile.ops = macs_to_ops(profile.macs);
        profile.gmacs = compute_rate(profile.macs, profile.kernel_ms);
        profile.gops = compute_rate(profile.ops, profile.kernel_ms);
        profiles->push_back(profile);
      }
    }
    impl_->announced = true;
  } catch (const std::exception &e) {
    add_error(diagnostics, "xrt", e.what());
    return false;
  }
  return true;
}

bool XrtSession::download_outputs(std::vector<RuntimeDiagnostic> &diagnostics) {
  try {
    RuntimeExecutionPlan &plan = impl_->plan;
    for (const auto &output_id : plan.graph_plan.graph.outputs) {
      RuntimeTensorBuffer *output = find_buffer(plan, output_id);
      RuntimeActivationSlot *slot = find_activation_slot(plan, output_id);
      const RuntimeTensorStorageBinding *binding =
          find_activation_binding(plan, output_id);
      const std::size_t slot_index = slot_index_for_tensor(plan, output_id);
      if (!output || !slot || !binding ||
          slot_index == static_cast<std::size_t>(-1)) {
        add_error(diagnostics, "xrt",
                  "missing final output activation slot: " + output_id);
        return false;
      }
      xrt::bo &bo = impl_->activation_bos[slot_index];
      bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      auto *ptr = bo.map<std::uint8_t *>();
      std::memcpy(slot->data_bytes.data(), ptr, binding->byte_size);
      output->data_bytes.assign(ptr, ptr + binding->byte_size);
    }
  } catch (const std::exception &e) {
    add_error(diagnostics, "xrt", e.what());
    return false;
  }
  return true;
}

std::uint64_t XrtSession::graph_macs() const { return impl_->graph_macs; }

XrtBackend::XrtBackend(XrtRunOptions options) : options_(std::move(options)) {}

XrtRunResult XrtBackend::run(RuntimeExecutionPlan &plan) const {
  XrtRunResult result;
  result.ok = false;

  if (options_.measured_runs <= 0) {
    add_error(result.diagnostics, "xrt", "measured runs must be positive");
    return result;
  }
  if (options_.warmup_runs < 0) {
    add_error(result.diagnostics, "xrt", "warmup runs cannot be negative");
    return result;
  }

  std::unique_ptr<XrtSession> session =
      XrtSession::open(plan, options_, result.diagnostics);
  if (!session) {
    return result;
  }

  result.graph_macs = session->graph_macs();
  result.graph_ops = macs_to_ops(result.graph_macs);

  const int total_runs = options_.warmup_runs + options_.measured_runs;
  std::vector<double> graph_kernel_ms(
      static_cast<std::size_t>(options_.measured_runs), 0.0);
  if (options_.measure_latency) {
    result.graph_profiles.reserve(options_.measured_runs);
  }

  for (int iter = 0; iter < total_runs; ++iter) {
    const bool measured = iter >= options_.warmup_runs;
    const int measured_index = iter - options_.warmup_runs;
    if (measured_index == 0) {
      result.window.start_ns = realtime_now_ns();
    }

    const auto upload_start = options_.measure_latency ? Clock::now()
                                                       : Clock::time_point{};
    if (!session->upload_inputs(result.diagnostics)) {
      return result;
    }
    const auto execute_start = options_.measure_latency ? Clock::now()
                                                        : Clock::time_point{};
    const std::size_t first_profile = result.profiles.size();
    if (!session->execute(measured ? &result.profiles : nullptr,
                          measured_index, result.diagnostics)) {
      return result;
    }
    if (options_.measure_latency) {
      const auto download_start = Clock::now();
      if (!session->download_outputs(result.diagnostics)) {
        return result;
      }
      const auto graph_end = Clock::now();
      if (measured) {
        XrtGraphProfile profile;
        profile.run_index = measured_index;
        profile.upload_ms = elapsed_ms(upload_start, execute_start);
        profile.execute_ms = elapsed_ms(execute_start, download_start);
        profile.download_ms = elapsed_ms(download_start, graph_end);
        profile.wall_ms = elapsed_ms(upload_start, graph_end);
        result.avg_graph_wall_ms += profile.wall_ms;
        result.graph_profiles.push_back(profile);
      }
    }
    if (measured) {
      for (std::size_t i = first_profile; i < result.profiles.size(); ++i) {
        graph_kernel_ms[static_cast<std::size_t>(measured_index)] +=
            result.profiles[i].kernel_ms;
      }
    }
  }

  // The power window covers exactly the repeated work, excluding setup.
  result.window.end_ns = realtime_now_ns();

  // Preserve the original throughput path unless batch-one latency is selected.
  if (!options_.measure_latency &&
      !session->download_outputs(result.diagnostics)) {
    return result;
  }
  if (options_.measure_latency) {
    result.avg_graph_wall_ms /= static_cast<double>(options_.measured_runs);
  }

  const double kernel_latency_sum =
      std::accumulate(graph_kernel_ms.begin(), graph_kernel_ms.end(), 0.0);
  result.avg_kernel_latency_ms =
      kernel_latency_sum / static_cast<double>(options_.measured_runs);
  result.avg_gmacs = compute_rate(result.graph_macs, result.avg_kernel_latency_ms);
  result.gops = compute_rate(result.graph_ops, result.avg_kernel_latency_ms);
  result.ok = true;
  return result;
}

void print_xrt_run_summary(const XrtRunResult &result, std::ostream &out,
                           bool verbose) {
  struct LaunchAverage {
    std::string id;
    std::size_t index = 0;
    int samples = 0;
    double kernel_ms = 0.0;
    double gmacs = 0.0;
    double gops = 0.0;
    std::uint64_t macs = 0;
    std::uint64_t ops = 0;
  };

  std::map<std::size_t, LaunchAverage> averages;
  for (const auto &p : result.profiles) {
    auto &avg = averages[p.launch_index];
    avg.id = p.launch_id;
    avg.index = p.launch_index;
    avg.samples += 1;
    avg.kernel_ms += p.kernel_ms;
    avg.gmacs += p.gmacs;
    avg.gops += p.gops;
    avg.macs = p.macs;
    avg.ops = p.ops;
  }

  out << std::fixed << std::setprecision(3);

  if (verbose) {
    out << "\n[XRT_KERNEL_PROFILE] measured launch samples: "
        << result.profiles.size() << "\n";
    out << "[XRT_KERNEL_PROFILE] latency is kernel-only: kernel(...); wait()\n";
    out << "[XRT_KERNEL_PROFILE] GOPS uses 2 ops per MAC\n";
    for (const auto &p : result.profiles) {
      out << "[XRT_KERNEL] run=" << p.run_index
          << " launch=" << p.launch_index << " " << p.launch_id
          << " kernel_ms=" << p.kernel_ms
          << " macs=" << p.macs
          << " ops=" << p.ops
          << " gmac/s=" << p.gmacs
          << " gops=" << p.gops << "\n";
    }

    for (auto &entry : averages) {
      auto &avg = entry.second;
      const double samples = static_cast<double>(avg.samples);
      out << "[XRT_LAUNCH_AVG] launch=" << avg.index << " " << avg.id
          << " samples=" << avg.samples
          << " avg_kernel_ms=" << (avg.kernel_ms / samples)
          << " macs=" << avg.macs
          << " ops=" << avg.ops
          << " avg_gmac/s=" << (avg.gmacs / samples)
          << " avg_gops=" << (avg.gops / samples) << "\n";
    }
  } else {
    // Default to one compact line per launch.
    for (auto &entry : averages) {
      auto &avg = entry.second;
      const double samples = static_cast<double>(avg.samples);
      out << "[LAUNCH] " << avg.index << " " << avg.id
          << " kernel_ms=" << (avg.kernel_ms / samples)
          << " gops=" << (avg.gops / samples) << "\n";
    }
  }

  out << "[XRT_TESTCASE_AVG] graph_kernel_ms=" << result.avg_kernel_latency_ms;
  if (verbose) {
    out << " graph_macs=" << result.graph_macs
        << " graph_ops=" << result.graph_ops
        << " graph_gmac/s=" << result.avg_gmacs;
  }
  out << " graph_gops=" << result.gops;
  if (!result.graph_profiles.empty()) {
    out << " graph_wall_ms=" << result.avg_graph_wall_ms;
  }
  out << "\n"
      << std::defaultfloat;
}

bool write_xrt_profile_csv(const XrtRunResult &result,
                           const std::string &csv_path,
                           std::vector<RuntimeDiagnostic> &diagnostics) {
  if (csv_path.empty()) {
    return true;
  }
  std::ofstream out(csv_path);
  if (!out) {
    add_error(diagnostics, "csv", "cannot open profile CSV: " + csv_path);
    return false;
  }
  out << "graph,launch_index,launch_id,run,kernel_ms,macs,ops,gmacs,gops\n";
  out << std::fixed << std::setprecision(6);
  for (const auto &p : result.profiles) {
    out << p.graph << "," << p.launch_index << "," << p.launch_id << ","
        << p.run_index << "," << p.kernel_ms << "," << p.macs << ","
        << p.ops << "," << p.gmacs << "," << p.gops << "\n";
  }
  return true;
}

bool write_xrt_graph_profile_csv(
    const XrtRunResult &result, const std::string &csv_path,
    std::vector<RuntimeDiagnostic> &diagnostics) {
  std::ofstream out(csv_path);
  if (!out) {
    add_error(diagnostics, "csv", "cannot open graph profile CSV: " + csv_path);
    return false;
  }
  out << "run,upload_ms,execute_ms,download_ms,wall_ms\n"
      << std::fixed << std::setprecision(6);
  for (const auto &p : result.graph_profiles) {
    out << p.run_index << "," << p.upload_ms << "," << p.execute_ms << ","
        << p.download_ms << "," << p.wall_ms << "\n";
  }
  return static_cast<bool>(out);
}

} // namespace runtime
} // namespace deeptile
