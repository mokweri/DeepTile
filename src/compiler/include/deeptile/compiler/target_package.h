/**
 * @file target_package.h
 * @brief Declares the versioned DeepTile hardware-target contract.
 * @ingroup deeptile_compiler
 */

#ifndef DEEPTILE_TARGET_PACKAGE_H_
#define DEEPTILE_TARGET_PACKAGE_H_

#include "deeptile/tile/hardware_spec.hpp"

#include <string>
#include <vector>

namespace deeptile {
namespace target {

enum class TargetDiagnosticSeverity { Warning, Error };

struct TargetDiagnostic {
  TargetDiagnosticSeverity severity = TargetDiagnosticSeverity::Error;
  std::string context;
  std::string message;
};

struct KernelAbiSpec {
  std::string name;
  int version = 0;
  int argument_count = 0;
  int compute_descriptor_bytes = 0;
  int transfer_descriptor_bytes = 0;
};

struct TargetPackage {
  std::string manifest_path;
  std::string root_dir;
  std::string schema;

  std::string name;
  std::string version;
  std::string id;

  std::string board;
  std::string vitis_platform;
  std::string vitis_mode;

  std::string xclbin;
  std::string xclbin_sha256;
  std::string xclbin_uuid;

  /**
   * Boot image that programs the PL.
   *
   * On a fixed (non-DFX) platform the fabric is programmed at boot from this
   * image, and loading an XCLBIN only registers metadata. The XCLBIN alone
   * therefore does not determine which hardware runs, so a hardware target
   * must declare its boot image for the fabric to be verifiable.
   *
   * Empty when the target declares no boot image, which is expected for
   * emulation modes and for targets packaged before this field existed.
   */
  std::string boot_image;
  std::string boot_sha256;

  bool has_boot_image() const { return !boot_image.empty(); }

  std::string kernel_name;
  std::string compute_unit;
  long long clock_hz = 0;
  KernelAbiSpec abi;

  deeptile::tile::HardwareSpec hardware;
  std::string architecture_fingerprint;

  std::vector<std::string> model_schemas;
  std::string psum_buffering;
  std::string instrumentation;

  std::string source_revision;
  bool source_dirty = false;
  std::string vitis_version;
  std::string provenance_verification = "unverified";
  std::string source_revision_scope;
  std::string build_record_sha256;
};

/** Compute a lowercase SHA-256 digest for text or a file. */
std::string sha256_text(const std::string &text);
bool sha256_file(const std::string &path, std::string &digest,
                 std::string &error);

/** Fingerprint the architecture fields used for compiler compatibility. */
std::string architecture_fingerprint(
    const deeptile::tile::HardwareSpec &hardware);

/** Parse and validate a target manifest against this compiler build. */
bool load_target_package(const std::string &manifest_path, TargetPackage &out,
                         std::vector<TargetDiagnostic> &diagnostics);

bool has_errors(const std::vector<TargetDiagnostic> &diagnostics);
std::string severity_name(TargetDiagnosticSeverity severity);

} // namespace target
} // namespace deeptile

#endif // DEEPTILE_TARGET_PACKAGE_H_
