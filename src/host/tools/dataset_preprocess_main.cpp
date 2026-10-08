/**
 * @file dataset_preprocess_main.cpp
 * @brief Command-line dataset indexing and image preprocessing.
 *
 * Runs the host preprocessing path over a dataset without touching the
 * accelerator, so the decode and preprocessing stages can be validated and
 * timed independently of execution. Preprocessing throughput determines how
 * much of it must overlap inference, so the timing summary is part of the
 * tool's purpose rather than incidental output.
 */

#include "dataset/dataset.h"
#include "dataset/image_decode.h"
#include "dataset/pipeline.h"
#include "dataset/preprocess.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string root;
  std::string out_dir;
  std::size_t limit = 0;
  bool allow_missing = false;
  deeptile::dataset::PreprocessConfig config;
  int workers = 0;  // Zero selects the dataset pipeline default.
  std::size_t queue_depth = 0;
  bool verbose = false;
};



void print_usage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0 << " <dataset-root> [options]\n"
      << "  --limit N            preprocess at most N images (0 = all)\n"
      << "  --allow-missing      proceed when val.txt names absent images\n"
      << "  --convention NAME    'eval' (default) or 'export'\n"
      << "  --side N             network input side, default 224\n"
      << "  --short-side N       eval-convention resize target, default 256\n"
      << "  --frac N             input fractional bits, default 5\n"
      << "  --out DIR            write each tensor as <name>.int8.bin\n"
      << "  --workers N          preparation threads, default 2\n"
      << "  --queue N            prepared images buffered, default 2x workers\n"
      << "  --verbose            print every image\n";
}

bool parse_size(const char *value, std::size_t &out) {
  try {
    out = static_cast<std::size_t>(std::stoull(value));
  } catch (const std::exception &) {
    return false;
  }
  return true;
}

bool parse_int(const char *value, int &out) {
  try {
    out = std::stoi(value);
  } catch (const std::exception &) {
    return false;
  }
  return true;
}

bool parse_cli(int argc, char **argv, Options &out) {
  if (argc < 2) {
    print_usage(argv[0]);
    return false;
  }
  out.root = argv[1];

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << "[ERROR] missing value for " << name << "\n";
        return nullptr;
      }
      return argv[++i];
    };

    if (arg == "--allow-missing") {
      out.allow_missing = true;
    } else if (arg == "--limit") {
      const char *v = value("--limit");
      if (!v || !parse_size(v, out.limit)) {
        return false;
      }
    } else if (arg == "--convention") {
      const char *v = value("--convention");
      if (!v || !deeptile::dataset::parse_resize_convention(
                    v, out.config.convention)) {
        std::cerr << "[ERROR] unknown convention; expected 'eval' or 'export'\n";
        return false;
      }
    } else if (arg == "--side") {
      const char *v = value("--side");
      if (!v || !parse_int(v, out.config.target_side)) {
        return false;
      }
    } else if (arg == "--short-side") {
      const char *v = value("--short-side");
      if (!v || !parse_int(v, out.config.short_side)) {
        return false;
      }
    } else if (arg == "--frac") {
      const char *v = value("--frac");
      if (!v || !parse_int(v, out.config.frac)) {
        return false;
      }
    } else if (arg == "--workers") {
      const char *v = value("--workers");
      if (!v || !parse_int(v, out.workers) || out.workers < 1) {
        std::cerr << "[ERROR] --workers must be at least 1\n";
        return false;
      }
    } else if (arg == "--queue") {
      const char *v = value("--queue");
      if (!v || !parse_size(v, out.queue_depth)) {
        return false;
      }
    } else if (arg == "--out") {
      const char *v = value("--out");
      if (!v) {
        return false;
      }
      out.out_dir = v;
    } else if (arg == "--verbose") {
      out.verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return false;
    } else {
      std::cerr << "[ERROR] unknown argument: " << arg << "\n";
      return false;
    }
  }
  return true;
}

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

} // namespace

int main(int argc, char **argv) {
  Options options;
  if (!parse_cli(argc, argv, options)) {
    return EXIT_FAILURE;
  }

  if (!deeptile::dataset::image_decoding_available()) {
    std::cerr << "[ERROR] this build has no image decoder; rebuild with "
                 "OpenCV available\n";
    return EXIT_FAILURE;
  }

  std::vector<deeptile::runtime::RuntimeDiagnostic> diagnostics;
  deeptile::dataset::DatasetIndex index;
  if (!deeptile::dataset::load_imagenet_index(options.root, options.limit, index,
                                             diagnostics,
                                             options.allow_missing)) {
    for (const auto &d : diagnostics) {
      std::cerr << "[" << deeptile::runtime::severity_name(d.severity) << "] "
                << d.message << "\n";
    }
    return EXIT_FAILURE;
  }
  for (const auto &d : diagnostics) {
    std::cout << "[" << deeptile::runtime::severity_name(d.severity) << "] "
              << d.message << "\n";
  }

  if (options.workers == 0) {
    options.workers = deeptile::dataset::kDefaultWorkers;
  }
  if (options.queue_depth == 0) {
    options.queue_depth = static_cast<std::size_t>(options.workers) * 2;
  }

  std::cout << "[INFO] images=" << index.entries.size()
            << " classes=" << index.class_names.size() << " convention="
            << deeptile::dataset::resize_convention_name(
                   options.config.convention)
            << " side=" << options.config.target_side
            << " frac=" << options.config.frac
            << " workers=" << options.workers
            << " queue=" << options.queue_depth << "\n";

  if (!options.out_dir.empty()) {
    std::filesystem::create_directories(options.out_dir);
  }

  deeptile::dataset::PreprocessPipeline pipeline(
      index, options.config, options.workers, options.queue_depth);

  const auto wall_start = Clock::now();
  std::size_t done = 0;
  std::size_t failed = 0;
  deeptile::dataset::PreparedImage image;

  // Consuming in dataset order keeps results independent of worker count.
  while (pipeline.next(image)) {
    if (!image.ok) {
      std::cerr << "[WARN] " << image.error << "\n";
      ++failed;
      continue;
    }

    if (!options.out_dir.empty()) {
      const std::filesystem::path path =
          std::filesystem::path(options.out_dir) / (image.name + ".int8.bin");
      std::ofstream out(path, std::ios::binary);
      out.write(reinterpret_cast<const char *>(image.tensor.data()),
                static_cast<std::streamsize>(image.tensor.size()));
    }

    if (options.verbose) {
      std::cout << "  " << image.name << " label=" << image.label << " -> "
                << image.tensor.size() << " bytes\n";
    }
    ++done;
  }

  const double wall_ms = ms_since(wall_start);
  const deeptile::dataset::PipelineStats stats = pipeline.stats();

  std::cout << "[SUMMARY] processed=" << done << " failed=" << failed << "\n";
  if (done != 0) {
    const double n = static_cast<double>(done);
    // Per-image costs are summed across workers, so they measure CPU work.
    // Wall-clock throughput is what a dataset run actually sees.
    std::cout << "[TIMING] decode=" << stats.decode_ms / n
              << " ms/img  preprocess=" << stats.preprocess_ms / n
              << " ms/img  cpu-total="
              << (stats.decode_ms + stats.preprocess_ms) / n << " ms/img\n";
    std::cout << "[THROUGHPUT] wall=" << wall_ms / n << " ms/img  "
              << 1000.0 * n / wall_ms << " img/s with " << pipeline.workers()
              << " worker(s)\n";
  }
  return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
