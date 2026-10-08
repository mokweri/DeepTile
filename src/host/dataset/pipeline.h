/**
 * @file pipeline.h
 * @brief Ordered, bounded, multi-threaded image preparation.
 * @ingroup deeptile_host
 */

#ifndef DEEPTILE_DATASET_PIPELINE_H_
#define DEEPTILE_DATASET_PIPELINE_H_

#include "dataset/dataset.h"
#include "dataset/image.h"
#include "dataset/image_decode.h"
#include "dataset/preprocess.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace deeptile {
namespace dataset {

/** One dataset image prepared for execution. */
struct PreparedImage {
  std::size_t index = 0;  ///< Position in the dataset listing.
  std::string name;
  int label = -1;
  std::vector<std::int8_t> tensor;  ///< int8 CHW, empty when `ok` is false.
  std::vector<std::uint8_t> packed;  ///< Hardware layout, only if a packer is set.
  bool ok = false;
  std::string error;
  double decode_ms = 0.0;
  double preprocess_ms = 0.0;
};

/** Aggregate pipeline cost. */
struct PipelineStats {
  std::size_t produced = 0;
  std::size_t failed = 0;
  double decode_ms = 0.0;      ///< Summed across workers, so CPU cost.
  double preprocess_ms = 0.0;  ///< Summed across workers, so CPU cost.
};

/**
 * Preparation threads used when a caller does not choose.
 *
 * Two workers are enough to keep the accelerator fed on the reference board:
 * preparation drops below inference time, and further workers add scheduling
 * contention without throughput. Networks with much shorter inference than
 * ResNet-50 are preparation-bound and benefit from raising this explicitly.
 */
constexpr int kDefaultWorkers = 2;

/** Loads one dataset entry as RGB; replaceable so tests need no decoder. */
using ImageLoader = std::function<bool(const DatasetEntry &entry, Image8 &out,
                                       std::string &error)>;

/** Default loader: decode the entry's file through the configured backend. */
ImageLoader default_image_loader();

/**
 * Loads the part of one dataset entry that `choose` selects, or more.
 * Lets the decoder skip pixels that preprocessing never reads.
 */
using RegionLoader =
    std::function<bool(const DatasetEntry &entry, const RegionChooser &choose,
                       ImageRegion &out, std::string &error)>;

/** Default region loader: decode_image_rgb_region() on the entry's file. */
RegionLoader default_region_loader();

/**
 * Converts a CHW tensor into the layout the accelerator consumes.
 *
 * Supplying one moves packing onto the worker threads. Packing a full input
 * activation is not cheap, and doing it between inference calls leaves the
 * accelerator idle for that time.
 */
using TensorPacker =
    std::function<std::vector<std::uint8_t>(const std::vector<std::int8_t> &)>;

/**
 * Prepares dataset images on worker threads while the caller consumes them.
 *
 * Board measurements put decode plus preprocessing above single-image
 * inference time, so a serial loop would leave the accelerator idle for more
 * than half of the run. Workers overlap that cost with execution.
 *
 * next() returns images in dataset order regardless of completion order, so
 * worker count affects throughput only, never results or reporting. At most
 * `queue_depth` prepared images are held at once, bounding memory.
 */
class PreprocessPipeline {
public:
  /**
   * @param index Dataset listing to consume, which must outlive the pipeline.
   * @param config Preprocessing parameters applied to every image.
   * @param workers Worker threads; values below one are treated as one.
   * @param queue_depth Prepared images buffered; raised to at least `workers`.
   * @param loader Image source, defaulting to the decoder backend.
   * @param packer Optional hardware packing, performed on the worker thread.
   * @param direct_lanes When positive, produce the packed form directly with
   *        this many channel lanes instead of building a CHW tensor and
   *        repacking it. Valid only when the input fits one channel group.
   * @param region_loader When set together with `direct_lanes`, used in place
   *        of `loader` to decode only the pixels preprocessing reads; the
   *        output is identical.
   */
  PreprocessPipeline(const DatasetIndex &index, PreprocessConfig config,
                     int workers, std::size_t queue_depth,
                     ImageLoader loader = default_image_loader(),
                     TensorPacker packer = nullptr,
                     int direct_lanes = 0,
                     RegionLoader region_loader = nullptr);
  ~PreprocessPipeline();

  PreprocessPipeline(const PreprocessPipeline &) = delete;
  PreprocessPipeline &operator=(const PreprocessPipeline &) = delete;

  /**
   * Retrieve the next prepared image in dataset order.
   *
   * Blocks until it is available. Entries that failed to load or preprocess
   * are still returned, with `ok` false, so the caller can account for them.
   *
   * @return False once every entry has been consumed.
   */
  bool next(PreparedImage &out);

  /** Stop producing; in-flight workers finish their current image. */
  void stop();

  /** Cost accumulated so far. Safe to call while running. */
  PipelineStats stats() const;

  int workers() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace dataset
} // namespace deeptile

#endif // DEEPTILE_DATASET_PIPELINE_H_
