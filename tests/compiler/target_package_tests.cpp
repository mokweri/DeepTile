/**
 * @file target_package_tests.cpp
 * @brief Contract tests for TargetPackage parsing and artifact verification.
 */

#include "kernel_abi.h"
#include "runtime/target_artifact.h"
#include "deeptile/compiler/target_package.h"
#include "deeptile/tile/hardware_spec.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace nt = deeptile::target;
namespace nr = deeptile::runtime;

namespace {

int g_failures = 0;

void check(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "  [FAIL] " << message << "\n";
    ++g_failures;
  }
}

void write_text(const fs::path &path, const std::string &text) {
  std::ofstream output(path, std::ios::binary);
  output << text;
}

bool replace_once(std::string &text, const std::string &from,
                  const std::string &to) {
  const std::size_t position = text.find(from);
  if (position == std::string::npos) {
    return false;
  }
  text.replace(position, from.size(), to);
  return true;
}

std::string valid_manifest(const std::string &artifact_hash) {
  const deeptile::tile::HardwareSpec hardware;
  std::ostringstream json;
  json << R"({
  "schema": "deeptile.target.v1",
  "target": {
    "name": "zcu104-p4-pp",
    "version": "test",
    "id": "zcu104-p4-pp@test"
  },
  "platform": {
    "board": "zcu104",
    "vitis_platform": "xilinx_zcu104_base_202420_1"
  },
  "artifact": {
    "vitis_mode": "hw",
    "xclbin": "deeptile.xclbin",
    "sha256": ")"
       << artifact_hash << R"(",
    "uuid": "00000000-0000-0000-0000-000000000000"
  },
  "kernel": {
    "name": "tile_conv",
    "compute_unit": "tile_conv_1",
    "clock_hz": 200000000,
    "abi": {
      "name": ")"
       << DEEPTILE_KERNEL_ABI_NAME << R"(",
      "version": )"
       << DEEPTILE_KERNEL_ABI_VERSION << R"(,
      "argument_count": )"
       << DEEPTILE_KERNEL_ARGUMENT_COUNT << R"(,
      "compute_descriptor_bytes": )"
       << DEEPTILE_COMPUTE_DESCRIPTOR_BYTES << R"(,
      "transfer_descriptor_bytes": )"
       << DEEPTILE_TRANSFER_DESCRIPTOR_BYTES << R"(
    }
  },
  "architecture": {
    "parallelism": {
      "ic_tile_size": )"
       << hardware.ic_tile_size << R"(,
      "oc_tile_size": )"
       << hardware.oc_tile_size << R"(,
      "p_ic": )"
       << hardware.p_ic << R"(,
      "p_spatial": )"
       << hardware.p_spatial << R"(
    },
    "limits": {
      "max_tile_height": )"
       << hardware.max_tile_h << R"(,
      "max_tile_width": )"
       << hardware.max_tile_w << R"(,
      "max_kernel_size": )"
       << hardware.k_max << R"(,
      "ifm_cache_depth": )"
       << hardware.k1_cache_depth << R"(,
      "weight_buffer_depth": )"
       << hardware.weight_buf_depth << R"(,
      "bias_buffer_depth": )"
       << hardware.bias_buf_depth << R"(
    },
    "numeric": {
      "bitwidth": )"
       << BITWIDTH << R"(,
      "gap_scale_frac_bits": )"
       << deeptile::tile::hw_defaults::kGapScaleFracBits << R"(
    },
    "fingerprint": ")"
       << nt::architecture_fingerprint(hardware) << R"("
  },
  "capabilities": {
    "model_schemas": ["deeptile.graph.v1"],
    "residual_add": true,
    "max_pool": true,
    "global_average_pool": true,
    "fused_depthwise": true
  },
  "implementation": {
    "psum_buffering": "ping-pong",
    "instrumentation": "none"
  },
  "provenance": {
    "verification": "verified",
    "deeptile_revision": "test",
    "source_dirty": false,
    "source_revision_scope": "fpga-build-record",
    "vitis_version": "test",
    "build_record_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "architecture_config_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "abi_header_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "kernel_config_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "vivado_impl_config_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "generated_by": "target_package_tests"
  }
})";
  return json.str();
}

void test_hashes() {
  check(nt::sha256_text("abc") ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA-256 matches the standard abc test vector");
}

void test_manifest_and_artifact(const fs::path &directory) {
  const fs::path xclbin = directory / "deeptile.xclbin";
  const fs::path manifest = directory / "target.json";
  write_text(xclbin, "test xclbin\n");

  std::string artifact_hash;
  std::string hash_error;
  check(nt::sha256_file(xclbin.string(), artifact_hash, hash_error),
        "test artifact can be hashed");
  write_text(manifest, valid_manifest(artifact_hash));

  nt::TargetPackage target;
  std::vector<nt::TargetDiagnostic> target_diagnostics;
  check(nt::load_target_package(manifest.string(), target, target_diagnostics),
        "valid target manifest loads");
  check(target.id == "zcu104-p4-pp@test", "target identity is parsed");

  std::string resolved;
  std::vector<nr::RuntimeDiagnostic> runtime_diagnostics;
  check(nr::resolve_target_artifact(target, resolved, runtime_diagnostics),
        "matching target artifact is accepted");
  check(fs::equivalent(resolved, xclbin), "artifact path resolves beside manifest");

  write_text(xclbin, "tampered xclbin\n");
  runtime_diagnostics.clear();
  check(!nr::resolve_target_artifact(target, resolved, runtime_diagnostics),
        "artifact checksum mismatch is rejected");
}

/**
 * The boot image determines the fabric on a fixed platform, so a declared boot
 * image must be present and matching before execution is allowed.
 */
void test_boot_image_verification(const fs::path &directory) {
  const fs::path package = directory / "boot-target";
  fs::create_directories(package);
  const fs::path xclbin = package / "deeptile.xclbin";
  const fs::path boot = package / "BOOT.BIN";
  const fs::path manifest = package / "target.json";
  write_text(xclbin, "boot test xclbin\n");
  write_text(boot, "boot image contents\n");

  std::string artifact_hash;
  std::string boot_hash;
  std::string hash_error;
  check(nt::sha256_file(xclbin.string(), artifact_hash, hash_error),
        "boot-test artifact can be hashed");
  check(nt::sha256_file(boot.string(), boot_hash, hash_error),
        "boot image can be hashed");

  std::string json = valid_manifest(artifact_hash);
  const std::string anchor = "\n  \"kernel\": {";
  const std::string boot_group =
      "\n  \"boot\": {\n    \"image\": \"BOOT.BIN\",\n    \"sha256\": \"" +
      boot_hash + "\"\n  }," + anchor;
  check(replace_once(json, anchor, boot_group),
        "boot group can be inserted into the test manifest");
  write_text(manifest, json);

  nt::TargetPackage target;
  std::vector<nt::TargetDiagnostic> target_diagnostics;
  check(nt::load_target_package(manifest.string(), target, target_diagnostics),
        "manifest declaring a boot image loads");
  check(target.has_boot_image(), "boot image is parsed");
  check(target.boot_sha256 == boot_hash, "boot digest is parsed");

  std::string resolved;
  std::vector<nr::RuntimeDiagnostic> runtime_diagnostics;
  check(nr::resolve_target_artifact(target, resolved, runtime_diagnostics),
        "matching boot image is accepted");

  // A matching file copied during this Linux boot has not programmed the PL.
  // The persistent deployment marker must force a reboot before execution.
  std::ifstream boot_id_stream("/proc/sys/kernel/random/boot_id");
  std::string current_boot_id;
  std::getline(boot_id_stream, current_boot_id);
  if (!current_boot_id.empty()) {
    const fs::path marker = package / ".deeptile-deploy-boot-id";
    write_text(marker, current_boot_id + "\n");
    runtime_diagnostics.clear();
    check(!nr::resolve_target_artifact(target, resolved, runtime_diagnostics),
          "boot image installed in the current session requires a reboot");

    write_text(marker, "an-earlier-boot-session\n");
    runtime_diagnostics.clear();
    check(nr::resolve_target_artifact(target, resolved, runtime_diagnostics),
          "boot image installed before the current session is accepted");
    fs::remove(marker);
  }

  // A different boot image means different hardware is running, which no
  // output comparison can reveal.
  write_text(boot, "a different fabric\n");
  runtime_diagnostics.clear();
  check(!nr::resolve_target_artifact(target, resolved, runtime_diagnostics),
        "boot image checksum mismatch is rejected");

  fs::remove(boot);
  runtime_diagnostics.clear();
  check(!nr::resolve_target_artifact(target, resolved, runtime_diagnostics),
        "declared but absent boot image is rejected");
}

/** Targets without a boot group stay loadable, for emulation and legacy. */
void test_absent_boot_group_is_allowed(const fs::path &directory) {
  const fs::path package = directory / "no-boot-target";
  fs::create_directories(package);
  const fs::path xclbin = package / "deeptile.xclbin";
  const fs::path manifest = package / "target.json";
  write_text(xclbin, "no-boot xclbin\n");

  std::string artifact_hash;
  std::string hash_error;
  check(nt::sha256_file(xclbin.string(), artifact_hash, hash_error),
        "no-boot artifact can be hashed");
  write_text(manifest, valid_manifest(artifact_hash));

  nt::TargetPackage target;
  std::vector<nt::TargetDiagnostic> target_diagnostics;
  check(nt::load_target_package(manifest.string(), target, target_diagnostics),
        "manifest without a boot group loads");
  check(!target.has_boot_image(), "absent boot image is reported as absent");

  std::string resolved;
  std::vector<nr::RuntimeDiagnostic> runtime_diagnostics;
  check(nr::resolve_target_artifact(target, resolved, runtime_diagnostics),
        "target without a declared boot image still resolves");
}

/** A half-declared boot group is worse than none: it implies a check. */
void test_malformed_boot_group_is_rejected(const fs::path &directory) {
  const fs::path manifest = directory / "bad-boot.json";
  std::string json = valid_manifest(std::string(64, '0'));
  const std::string anchor = "\n  \"kernel\": {";
  check(replace_once(json, anchor,
                     "\n  \"boot\": {\n    \"image\": \"BOOT.BIN\",\n"
                     "    \"sha256\": \"not-a-digest\"\n  }," + anchor),
        "malformed boot group can be inserted");
  write_text(manifest, json);

  nt::TargetPackage target;
  std::vector<nt::TargetDiagnostic> diagnostics;
  check(!nt::load_target_package(manifest.string(), target, diagnostics),
        "malformed boot digest is rejected");
  check(nt::has_errors(diagnostics), "malformed boot group reports an error");
}

void test_incompatible_abi(const fs::path &directory) {
  const fs::path manifest = directory / "bad-abi.json";
  std::string json = valid_manifest(std::string(64, '0'));
  const std::string current =
      "\"version\": " + std::to_string(DEEPTILE_KERNEL_ABI_VERSION);
  const std::string incompatible =
      "\"version\": " + std::to_string(DEEPTILE_KERNEL_ABI_VERSION + 1);
  check(replace_once(json, current, incompatible),
        "ABI version exists in test manifest");
  write_text(manifest, json);

  nt::TargetPackage target;
  std::vector<nt::TargetDiagnostic> diagnostics;
  check(!nt::load_target_package(manifest.string(), target, diagnostics),
        "incompatible kernel ABI is rejected");
  check(nt::has_errors(diagnostics), "ABI rejection reports an error");
}

void test_incompatible_architecture(const fs::path &directory) {
  const fs::path manifest = directory / "bad-architecture.json";
  std::string json = valid_manifest(std::string(64, '0'));
  const deeptile::tile::HardwareSpec hardware;
  check(replace_once(json,
                     "\"p_spatial\": " + std::to_string(hardware.p_spatial),
                     "\"p_spatial\": " +
                         std::to_string(hardware.p_spatial * 2)),
        "architecture field exists in test manifest");
  write_text(manifest, json);

  nt::TargetPackage target;
  std::vector<nt::TargetDiagnostic> diagnostics;
  check(!nt::load_target_package(manifest.string(), target, diagnostics),
        "incompatible architecture is rejected");
}

void test_escaping_artifact_path(const fs::path &directory) {
  const fs::path manifest = directory / "bad-artifact-path.json";
  std::string json = valid_manifest(std::string(64, '0'));
  check(replace_once(json, "\"xclbin\": \"deeptile.xclbin\"",
                     "\"xclbin\": \"../deeptile.xclbin\""),
        "artifact path exists in test manifest");
  write_text(manifest, json);

  nt::TargetPackage target;
  std::vector<nt::TargetDiagnostic> diagnostics;
  check(!nt::load_target_package(manifest.string(), target, diagnostics),
        "artifact path outside the package is rejected");
}

void test_legacy_provenance_is_allowed(const fs::path &directory) {
  const fs::path manifest = directory / "legacy-provenance.json";
  std::string json = valid_manifest(std::string(64, '0'));
  check(replace_once(json, "    \"verification\": \"verified\",\n", ""),
        "verification field exists in test manifest");
  write_text(manifest, json);

  nt::TargetPackage target;
  std::vector<nt::TargetDiagnostic> diagnostics;
  check(nt::load_target_package(manifest.string(), target, diagnostics),
        "compatible legacy target remains accepted");
  check(target.provenance_verification == "unverified",
        "legacy target is classified as unverified");
  check(!diagnostics.empty(), "legacy target produces a provenance warning");
}

/**
 * Manifests written before the rename, under the tilecnn.* namespace with the
 * former revision field, load exactly like deeptile.* ones; any other namespace
 * is rejected.
 */
void test_both_schema_namespaces(const fs::path &directory) {
  const fs::path manifest = directory / "tilecnn-namespace.json";
  std::string json = valid_manifest(std::string(64, '0'));
  check(replace_once(json, "\"deeptile.target.v1\"", "\"tilecnn.target.v1\""),
        "target schema exists in test manifest");
  check(replace_once(json, "[\"deeptile.graph.v1\"]", "[\"tilecnn.graph.v1\"]"),
        "model schema list exists in test manifest");
  check(replace_once(json, "\"deeptile_revision\"", "\"tilecnn_revision\""),
        "revision field exists in test manifest");
  write_text(manifest, json);

  nt::TargetPackage target;
  std::vector<nt::TargetDiagnostic> diagnostics;
  check(nt::load_target_package(manifest.string(), target, diagnostics),
        "tilecnn.* target manifest loads");
  check(target.source_revision == "test", "tilecnn_revision is read");

  check(replace_once(json, "\"tilecnn.target.v1\"", "\"other.target.v1\""),
        "target schema can be replaced");
  write_text(manifest, json);
  diagnostics.clear();
  check(!nt::load_target_package(manifest.string(), target, diagnostics),
        "an unknown schema namespace is rejected");
}

} // namespace

int main() {
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path directory =
      fs::temp_directory_path() / ("deeptile-target-test-" + std::to_string(nonce));
  fs::create_directories(directory);

  test_hashes();
  test_manifest_and_artifact(directory);
  test_boot_image_verification(directory);
  test_absent_boot_group_is_allowed(directory);
  test_malformed_boot_group_is_rejected(directory);
  test_incompatible_abi(directory);
  test_incompatible_architecture(directory);
  test_escaping_artifact_path(directory);
  test_legacy_provenance_is_allowed(directory);
  test_both_schema_namespaces(directory);

  fs::remove_all(directory);
  if (g_failures != 0) {
    std::cerr << g_failures << " target-package check(s) failed\n";
    return 1;
  }
  std::cout << "OK: target manifest and artifact contract checks passed\n";
  return 0;
}
