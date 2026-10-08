/**
 * @file target_artifact.cpp
 * @brief Verifies a target XCLBIN before XRT loads it.
 */

#include "runtime/target_artifact.h"

#include <filesystem>
#include <fstream>

namespace deeptile {
namespace runtime {

bool resolve_target_artifact(
    const deeptile::target::TargetPackage &target, std::string &xclbin_path,
    std::vector<RuntimeDiagnostic> &diagnostics) {
  const std::filesystem::path path =
      std::filesystem::path(target.root_dir) / target.xclbin;
  if (!std::filesystem::is_regular_file(path)) {
    diagnostics.push_back({RuntimeDiagnosticSeverity::Error, "target artifact",
                           "XCLBIN not found: " + path.string()});
    return false;
  }

  std::string actual_hash;
  std::string hash_error;
  if (!deeptile::target::sha256_file(path.string(), actual_hash, hash_error)) {
    diagnostics.push_back({RuntimeDiagnosticSeverity::Error, "target artifact",
                           hash_error});
    return false;
  }
  if (actual_hash != target.xclbin_sha256) {
    diagnostics.push_back(
        {RuntimeDiagnosticSeverity::Error, "target artifact",
         "XCLBIN checksum mismatch: expected " + target.xclbin_sha256 +
             ", got " + actual_hash});
    return false;
  }

  if (target.has_boot_image()) {
    const std::filesystem::path boot_path =
        std::filesystem::path(target.root_dir) / target.boot_image;
    if (!std::filesystem::is_regular_file(boot_path)) {
      diagnostics.push_back(
          {RuntimeDiagnosticSeverity::Error, "target boot image",
           "boot image not found at the package-relative path: " +
               boot_path.string() + "; the running fabric cannot be verified"});
      return false;
    }

    std::string boot_hash;
    std::string boot_error;
    if (!deeptile::target::sha256_file(boot_path.string(), boot_hash,
                                      boot_error)) {
      diagnostics.push_back({RuntimeDiagnosticSeverity::Error,
                             "target boot image", boot_error});
      return false;
    }
    if (boot_hash != target.boot_sha256) {
      diagnostics.push_back(
          {RuntimeDiagnosticSeverity::Error, "target boot image",
           "boot image checksum mismatch: expected " + target.boot_sha256 +
               ", got " + boot_hash +
               ". The fabric is programmed from the boot image, so this "
               "target's hardware is not running. Deploy the full package "
               "and reboot."});
      return false;
    }

    // Full deployment records the Linux boot session that installed the image.
    // If it is still the current session, BOOT.BIN has not yet programmed the
    // fabric even though its on-disk checksum already matches the manifest.
    const std::filesystem::path deploy_boot_id_path =
        std::filesystem::path(target.root_dir) / ".deeptile-deploy-boot-id";
    if (std::filesystem::is_regular_file(deploy_boot_id_path)) {
      std::ifstream deployed_stream(deploy_boot_id_path);
      std::ifstream current_stream("/proc/sys/kernel/random/boot_id");
      std::string deployed_boot_id;
      std::string current_boot_id;
      std::getline(deployed_stream, deployed_boot_id);
      std::getline(current_stream, current_boot_id);
      if (deployed_boot_id.empty() || current_boot_id.empty()) {
        diagnostics.push_back(
            {RuntimeDiagnosticSeverity::Error, "target boot image",
             "cannot verify whether the deployed boot image has been activated"});
        return false;
      }
      if (deployed_boot_id == current_boot_id) {
        diagnostics.push_back(
            {RuntimeDiagnosticSeverity::Error, "target boot image",
             "BOOT.BIN was installed during the current boot session, so the "
             "previous fabric is still active; reboot before execution"});
        return false;
      }
    }
  }

  xclbin_path = path.lexically_normal().string();
  return true;
}

} // namespace runtime
} // namespace deeptile
