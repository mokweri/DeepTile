/**
 * @file pipeline.cpp
 * @brief Implements ordered multi-threaded image preparation.
 *
 * Slots form a ring indexed by sequence number, so a worker writes only the
 * slot it claimed and the consumer reads slots in sequence. Ordering therefore
 * costs no reordering buffer, and a worker blocks only when it would overwrite
 * a slot the consumer has not taken yet.
 */

#include "dataset/pipeline.h"

#include "dataset/image_decode.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace deeptile {
namespace dataset {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

} // namespace

ImageLoader default_image_loader() {
  return [](const DatasetEntry &entry, Image8 &out, std::string &error) {
    return decode_image_rgb(entry.path, out, error);
  };
}

RegionLoader default_region_loader() {
  return [](const DatasetEntry &entry, const RegionChooser &choose,
            ImageRegion &out, std::string &error) {
    return decode_image_rgb_region(entry.path, choose, out, error);
  };
}

struct PreprocessPipeline::Impl {
  const DatasetIndex &index;
  PreprocessConfig config;
  ImageLoader loader;
  TensorPacker packer;
  int direct_lanes = 0;
  RegionLoader region_loader;
  // Built once: the configuration is fixed for the whole pipeline.
  QuantizationTable table;
  int worker_count = 1;
  std::size_t capacity = 1;
  std::size_t total = 0;

  mutable std::mutex mutex;
  std::condition_variable slot_free;   // Consumer released a slot.
  std::condition_variable slot_ready;  // Worker filled a slot.

  std::vector<PreparedImage> slots;
  std::vector<bool> ready;
  std::size_t next_claim = 0;
  std::size_t next_consume = 0;
  bool stopping = false;

  PipelineStats stats;
  std::vector<std::thread> threads;

  Impl(const DatasetIndex &idx, PreprocessConfig cfg, ImageLoader ld,
       TensorPacker pk, int lanes, RegionLoader rl)
      : index(idx), config(std::move(cfg)), loader(std::move(ld)),
        packer(std::move(pk)), direct_lanes(lanes), region_loader(std::move(rl)),
        table(config.frac >= 0 ? build_quantization_table(config)
                               : QuantizationTable{}) {}

  /** Decode only what preprocessing reads, then preprocess from that region. */
  void prepare_region(const DatasetEntry &entry, PreparedImage &prepared) {
    const PreprocessConfig &cfg = config;
    const RegionChooser choose = [&cfg](int w, int h, int &x0, int &y0, int &x1,
                                        int &y1) {
      return preprocess_source_window(w, h, cfg, x0, y0, x1, y1);
    };
    ImageRegion region;
    std::string error;
    const auto decode_start = Clock::now();
    const bool loaded = region_loader(entry, choose, region, error);
    prepared.decode_ms = ms_since(decode_start);
    if (!loaded) {
      prepared.error = error;
      return;
    }
    const auto pre_start = Clock::now();
    prepared.packed = preprocess_rgb_interleaved(
        region.image, region.x, region.y, region.full_width,
        region.full_height, config, table, direct_lanes);
    prepared.preprocess_ms = ms_since(pre_start);
    if (prepared.packed.empty()) {
      prepared.error = "preprocessing failed for " + entry.name;
    } else {
      prepared.ok = true;
    }
  }

  /** Decode the whole image, then preprocess (and pack, if configured). */
  void prepare_full(const DatasetEntry &entry, PreparedImage &prepared) {
    std::string error;
    Image8 image;
    const auto decode_start = Clock::now();
    const bool loaded = loader(entry, image, error);
    prepared.decode_ms = ms_since(decode_start);

    if (!loaded) {
      prepared.error = error;
    } else {
      const auto pre_start = Clock::now();
      if (direct_lanes > 0) {
        // Straight to the packed layout: no CHW tensor is materialized and
        // no separate repack pass runs.
        prepared.packed =
            preprocess_rgb_interleaved(image, config, table, direct_lanes);
        if (prepared.packed.empty()) {
          prepared.error = "preprocessing failed for " + entry.name;
        } else {
          prepared.ok = true;
        }
      } else {
        prepared.tensor = preprocess_rgb(image, config);
        if (prepared.tensor.empty()) {
          prepared.error = "preprocessing failed for " + entry.name;
        } else {
          // Packing belongs here rather than in the consumer: it is
          // per-image work that would otherwise serialize against inference.
          if (packer) {
            prepared.packed = packer(prepared.tensor);
          }
          prepared.ok = true;
        }
      }
      // Covers preprocessing and packing, since both are worker-side cost.
      prepared.preprocess_ms = ms_since(pre_start);
    }
  }

  void worker_loop() {
    while (true) {
      std::size_t seq = 0;
      {
        std::unique_lock<std::mutex> lock(mutex);
        if (stopping || next_claim >= total) {
          return;
        }
        seq = next_claim++;
        // Claim order is monotonic, so the lowest outstanding sequence is
        // never blocked and the pipeline cannot deadlock.
        slot_free.wait(lock, [&] {
          return stopping || seq - next_consume < capacity;
        });
        if (stopping) {
          return;
        }
      }

      const DatasetEntry &entry = index.entries[seq];
      PreparedImage prepared;
      prepared.index = seq;
      prepared.name = entry.name;
      prepared.label = entry.label;

      if (region_loader && direct_lanes > 0) {
        prepare_region(entry, prepared);
      } else {
        prepare_full(entry, prepared);
      }

      {
        std::lock_guard<std::mutex> lock(mutex);
        stats.decode_ms += prepared.decode_ms;
        stats.preprocess_ms += prepared.preprocess_ms;
        if (prepared.ok) {
          ++stats.produced;
        } else {
          ++stats.failed;
        }
        slots[seq % capacity] = std::move(prepared);
        ready[seq % capacity] = true;
      }
      slot_ready.notify_all();
    }
  }
};

PreprocessPipeline::PreprocessPipeline(const DatasetIndex &index,
                                       PreprocessConfig config, int workers,
                                       std::size_t queue_depth,
                                       ImageLoader loader, TensorPacker packer,
                                       int direct_lanes,
                                       RegionLoader region_loader)
    : impl_(new Impl(index, std::move(config), std::move(loader),
                     std::move(packer), direct_lanes,
                     std::move(region_loader))) {
  impl_->worker_count = std::max(1, workers);
  // One slot per worker keeps every worker busy; fewer would idle them.
  impl_->capacity =
      std::max(queue_depth, static_cast<std::size_t>(impl_->worker_count));
  impl_->total = index.entries.size();
  impl_->slots.resize(impl_->capacity);
  impl_->ready.assign(impl_->capacity, false);

  impl_->threads.reserve(static_cast<std::size_t>(impl_->worker_count));
  for (int i = 0; i < impl_->worker_count; ++i) {
    impl_->threads.emplace_back([impl = impl_.get()] { impl->worker_loop(); });
  }
}

PreprocessPipeline::~PreprocessPipeline() {
  stop();
  for (auto &thread : impl_->threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

bool PreprocessPipeline::next(PreparedImage &out) {
  std::unique_lock<std::mutex> lock(impl_->mutex);
  if (impl_->next_consume >= impl_->total) {
    return false;
  }
  const std::size_t slot = impl_->next_consume % impl_->capacity;
  impl_->slot_ready.wait(lock,
                         [&] { return impl_->ready[slot] || impl_->stopping; });
  if (!impl_->ready[slot]) {
    return false;  // stopped before this image was produced
  }

  out = std::move(impl_->slots[slot]);
  impl_->ready[slot] = false;
  ++impl_->next_consume;
  lock.unlock();
  impl_->slot_free.notify_all();
  return true;
}

void PreprocessPipeline::stop() {
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->stopping = true;
  }
  impl_->slot_free.notify_all();
  impl_->slot_ready.notify_all();
}

PipelineStats PreprocessPipeline::stats() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->stats;
}

int PreprocessPipeline::workers() const { return impl_->worker_count; }

} // namespace dataset
} // namespace deeptile
