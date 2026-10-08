/**
 * @file pipeline_tests.cpp
 * @brief Tests for ordered multi-threaded image preparation.
 *
 * A synthetic image loader is injected so these run without an image decoder
 * and without touching the filesystem. The properties under test are ordering,
 * independence from worker count, bounded buffering, and failure propagation.
 */

#include "dataset/pipeline.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#define TEST_ASSERT(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "    ASSERT FAILED: " << msg << "\n";                       \
      return false;                                                            \
    }                                                                          \
  } while (0)

namespace {

using deeptile::dataset::DatasetEntry;
using deeptile::dataset::DatasetIndex;
using deeptile::dataset::Image8;
using deeptile::dataset::PreparedImage;
using deeptile::dataset::PreprocessConfig;
using deeptile::dataset::PreprocessPipeline;

DatasetIndex make_index(std::size_t count) {
  DatasetIndex index;
  for (std::size_t i = 0; i < count; ++i) {
    index.entries.push_back(
        {"img_" + std::to_string(i), "/synthetic/" + std::to_string(i),
         static_cast<int>(i % 1000)});
  }
  return index;
}

/** Distinct flat-colour image per entry, so tensors identify their source. */
Image8 synthetic_image(int seed) {
  Image8 img;
  img.width = 40;
  img.height = 32;
  img.channels = 3;
  img.pixels.assign(img.expected_size(),
                    static_cast<std::uint8_t>((seed * 7) % 251));
  return img;
}

int entry_seed(const DatasetEntry &entry) {
  return std::stoi(entry.name.substr(4));
}

PreprocessConfig small_config() {
  PreprocessConfig config;
  config.convention = deeptile::dataset::ResizeConvention::ExportSquash;
  config.target_side = 16;
  return config;
}

/** Order must follow the dataset listing, not completion order. */
bool test_preserves_order() {
  const DatasetIndex index = make_index(64);
  // Delay early entries so out-of-order completion is guaranteed.
  auto loader = [](const DatasetEntry &entry, Image8 &out, std::string &) {
    const int seed = entry_seed(entry);
    if (seed % 8 == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    out = synthetic_image(seed);
    return true;
  };

  PreprocessPipeline pipeline(index, small_config(), 8, 4, loader);
  std::size_t expected = 0;
  PreparedImage image;
  while (pipeline.next(image)) {
    TEST_ASSERT(image.index == expected,
                "out of order: got " << image.index << " want " << expected);
    TEST_ASSERT(image.name == index.entries[expected].name, "name mismatch");
    TEST_ASSERT(image.label == index.entries[expected].label, "label mismatch");
    TEST_ASSERT(image.ok, "expected success for " << image.name);
    ++expected;
  }
  TEST_ASSERT(expected == index.entries.size(),
              "consumed " << expected << " of " << index.entries.size());
  return true;
}

std::vector<std::vector<std::int8_t>> run_with(int workers,
                                               std::size_t queue_depth) {
  const DatasetIndex index = make_index(32);
  auto loader = [](const DatasetEntry &entry, Image8 &out, std::string &) {
    out = synthetic_image(entry_seed(entry));
    return true;
  };
  PreprocessPipeline pipeline(index, small_config(), workers, queue_depth,
                              loader);
  std::vector<std::vector<std::int8_t>> tensors;
  PreparedImage image;
  while (pipeline.next(image)) {
    tensors.push_back(std::move(image.tensor));
  }
  return tensors;
}

/** Worker count is a throughput knob and must never change results. */
bool test_worker_count_does_not_change_output() {
  const auto baseline = run_with(1, 1);
  TEST_ASSERT(baseline.size() == 32, "baseline produced " << baseline.size());
  for (const int workers : {2, 4, 8, 16}) {
    const auto other = run_with(workers, 3);
    TEST_ASSERT(other == baseline,
                "output differs with " << workers << " workers");
  }
  return true;
}

/** Buffering must stay bounded, or a large dataset exhausts memory. */
bool test_queue_depth_is_bounded() {
  const DatasetIndex index = make_index(200);
  std::atomic<int> in_flight{0};
  std::atomic<int> peak{0};

  const std::size_t queue_depth = 4;
  const int workers = 8;
  auto loader = [&](const DatasetEntry &entry, Image8 &out, std::string &) {
    const int now = ++in_flight;
    int previous = peak.load();
    while (now > previous && !peak.compare_exchange_weak(previous, now)) {
    }
    out = synthetic_image(entry_seed(entry));
    --in_flight;
    return true;
  };

  PreprocessPipeline pipeline(index, small_config(), workers,
                              queue_depth, loader);
  std::size_t consumed = 0;
  PreparedImage image;
  while (pipeline.next(image)) {
    // Consume slowly so producers run ahead and hit the bound.
    if (consumed % 32 == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ++consumed;
  }
  TEST_ASSERT(consumed == index.entries.size(), "consumed " << consumed);
  // Capacity is raised to the worker count, so that is the real bound.
  const int bound = workers;
  TEST_ASSERT(peak.load() <= bound,
              "peak in-flight " << peak.load() << " exceeds " << bound);
  return true;
}

/** A failed image must still be delivered, in order, so it can be counted. */
bool test_failures_are_reported_in_order() {
  const DatasetIndex index = make_index(20);
  auto loader = [](const DatasetEntry &entry, Image8 &out,
                   std::string &error) {
    const int seed = entry_seed(entry);
    if (seed % 5 == 0) {
      error = "synthetic decode failure";
      return false;
    }
    out = synthetic_image(seed);
    return true;
  };

  PreprocessPipeline pipeline(index, small_config(), 4, 4, loader);
  std::size_t expected = 0;
  std::size_t failures = 0;
  PreparedImage image;
  while (pipeline.next(image)) {
    TEST_ASSERT(image.index == expected, "failure broke ordering");
    if (expected % 5 == 0) {
      TEST_ASSERT(!image.ok, "expected failure at " << expected);
      TEST_ASSERT(!image.error.empty(), "failure must carry a reason");
      TEST_ASSERT(image.tensor.empty(), "failed image must carry no tensor");
      ++failures;
    } else {
      TEST_ASSERT(image.ok, "unexpected failure at " << expected);
    }
    ++expected;
  }
  TEST_ASSERT(expected == 20, "consumed " << expected);
  TEST_ASSERT(failures == 4, "expected 4 failures, got " << failures);

  const auto stats = pipeline.stats();
  TEST_ASSERT(stats.failed == 4, "stats failed=" << stats.failed);
  TEST_ASSERT(stats.produced == 16, "stats produced=" << stats.produced);
  return true;
}

/** Abandoning a pipeline early must not hang or leak threads. */
bool test_stop_before_exhaustion() {
  const DatasetIndex index = make_index(500);
  auto loader = [](const DatasetEntry &entry, Image8 &out, std::string &) {
    out = synthetic_image(entry_seed(entry));
    return true;
  };
  {
    PreprocessPipeline pipeline(index, small_config(), 4, 4, loader);
    PreparedImage image;
    for (int i = 0; i < 5; ++i) {
      TEST_ASSERT(pipeline.next(image), "expected an image");
    }
  }  // Destructor stops and joins; a deadlock here hangs the test.
  return true;
}

bool test_empty_dataset() {
  const DatasetIndex index = make_index(0);
  auto loader = [](const DatasetEntry &, Image8 &, std::string &) {
    return true;
  };
  PreprocessPipeline pipeline(index, small_config(), 4, 4, loader);
  PreparedImage image;
  TEST_ASSERT(!pipeline.next(image), "empty dataset must yield nothing");
  return true;
}

int passed = 0;
int failed = 0;

void run(const std::string &name, const std::function<bool()> &fn) {
  std::cout << "  " << name << " ... " << std::flush;
  if (fn()) {
    std::cout << "PASS\n";
    ++passed;
  } else {
    std::cout << "FAIL\n";
    ++failed;
  }
}

} // namespace

int main() {
  run("preserves_order", test_preserves_order);
  run("worker_count_does_not_change_output",
      test_worker_count_does_not_change_output);
  run("queue_depth_is_bounded", test_queue_depth_is_bounded);
  run("failures_reported_in_order", test_failures_are_reported_in_order);
  run("stop_before_exhaustion", test_stop_before_exhaustion);
  run("empty_dataset", test_empty_dataset);

  std::cout << "Pipeline tests: " << passed << " passed, " << failed
            << " failed\n";
  return failed == 0 ? 0 : 1;
}
