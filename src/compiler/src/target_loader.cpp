/**
 * @file target_loader.cpp
 * @brief Loads target manifests and enforces compiler compatibility.
 */

#include "deeptile/compiler/target_package.h"
#include "deeptile/compiler/schema_ids.h"

#include "kernel_abi.h"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace deeptile {
namespace target {
namespace {

constexpr std::array<std::uint32_t, 64> kSha256RoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

std::uint32_t rotate_right(std::uint32_t value, unsigned int bits) {
  return (value >> bits) | (value << (32U - bits));
}

class Sha256 {
public:
  void update(const std::uint8_t *data, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      block_[block_size_++] = data[i];
      if (block_size_ == block_.size()) {
        transform();
        bit_count_ += 512U;
        block_size_ = 0;
      }
    }
  }

  std::string finish() {
    std::size_t i = block_size_;
    block_[i++] = 0x80U;
    if (i > 56U) {
      while (i < 64U) {
        block_[i++] = 0U;
      }
      transform();
      i = 0;
    }
    while (i < 56U) {
      block_[i++] = 0U;
    }

    bit_count_ += static_cast<std::uint64_t>(block_size_) * 8U;
    for (int shift = 56; shift >= 0; shift -= 8) {
      block_[i++] = static_cast<std::uint8_t>(bit_count_ >> shift);
    }
    transform();

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::uint32_t value : state_) {
      out << std::setw(8) << value;
    }
    return out.str();
  }

private:
  void transform() {
    std::array<std::uint32_t, 64> words = {};
    for (std::size_t i = 0; i < 16; ++i) {
      const std::size_t j = i * 4;
      words[i] = (static_cast<std::uint32_t>(block_[j]) << 24U) |
                 (static_cast<std::uint32_t>(block_[j + 1]) << 16U) |
                 (static_cast<std::uint32_t>(block_[j + 2]) << 8U) |
                 static_cast<std::uint32_t>(block_[j + 3]);
    }
    for (std::size_t i = 16; i < words.size(); ++i) {
      const std::uint32_t s0 = rotate_right(words[i - 15], 7U) ^
                               rotate_right(words[i - 15], 18U) ^
                               (words[i - 15] >> 3U);
      const std::uint32_t s1 = rotate_right(words[i - 2], 17U) ^
                               rotate_right(words[i - 2], 19U) ^
                               (words[i - 2] >> 10U);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t i = 0; i < words.size(); ++i) {
      const std::uint32_t s1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^
                               rotate_right(e, 25U);
      const std::uint32_t choice = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 =
          h + s1 + choice + kSha256RoundConstants[i] + words[i];
      const std::uint32_t s0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^
                               rotate_right(a, 22U);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint8_t, 64> block_ = {};
  std::size_t block_size_ = 0;
  std::uint64_t bit_count_ = 0;
  std::array<std::uint32_t, 8> state_ = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
};

void add_error(std::vector<TargetDiagnostic> &diagnostics,
               const std::string &context, const std::string &message) {
  diagnostics.push_back({TargetDiagnosticSeverity::Error, context, message});
}

void add_warning(std::vector<TargetDiagnostic> &diagnostics,
                 const std::string &context, const std::string &message) {
  diagnostics.push_back({TargetDiagnosticSeverity::Warning, context, message});
}

const rapidjson::Value *object_member(const rapidjson::Value &object,
                                      const char *key,
                                      std::vector<TargetDiagnostic> &diagnostics,
                                      const std::string &context) {
  if (!object.HasMember(key) || !object[key].IsObject()) {
    add_error(diagnostics, context,
              std::string("missing object field: ") + key);
    return nullptr;
  }
  return &object[key];
}

std::string string_member(const rapidjson::Value &object, const char *key,
                          std::vector<TargetDiagnostic> &diagnostics,
                          const std::string &context) {
  if (!object.HasMember(key) || !object[key].IsString()) {
    add_error(diagnostics, context,
              std::string("missing string field: ") + key);
    return {};
  }
  return object[key].GetString();
}

int int_member(const rapidjson::Value &object, const char *key,
               std::vector<TargetDiagnostic> &diagnostics,
               const std::string &context) {
  if (!object.HasMember(key) || !object[key].IsInt()) {
    add_error(diagnostics, context,
              std::string("missing integer field: ") + key);
    return 0;
  }
  return object[key].GetInt();
}

long long int64_member(const rapidjson::Value &object, const char *key,
                       std::vector<TargetDiagnostic> &diagnostics,
                       const std::string &context) {
  if (!object.HasMember(key) || !object[key].IsInt64()) {
    add_error(diagnostics, context,
              std::string("missing integer field: ") + key);
    return 0;
  }
  return object[key].GetInt64();
}

bool bool_member(const rapidjson::Value &object, const char *key,
                 std::vector<TargetDiagnostic> &diagnostics,
                 const std::string &context) {
  if (!object.HasMember(key) || !object[key].IsBool()) {
    add_error(diagnostics, context,
              std::string("missing boolean field: ") + key);
    return false;
  }
  return object[key].GetBool();
}

bool is_sha256(const std::string &value) {
  return value.size() == 64U &&
         !value.empty() &&
         std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return std::isxdigit(c) != 0 &&
                  (std::isdigit(c) != 0 || std::islower(c) != 0);
         });
}

bool has_parent_component(const std::filesystem::path &path) {
  return std::any_of(path.begin(), path.end(), [](const auto &component) {
    return component == "..";
  });
}

void validate_compiled_compatibility(
    const TargetPackage &target, std::vector<TargetDiagnostic> &diagnostics) {
  if (target.abi.name != DEEPTILE_KERNEL_ABI_NAME ||
      target.abi.version != DEEPTILE_KERNEL_ABI_VERSION) {
    add_error(diagnostics, "kernel.abi",
              "target requires " + target.abi.name + " ABI " +
                  std::to_string(target.abi.version) + ", host supports " +
                  DEEPTILE_KERNEL_ABI_NAME + " ABI " +
                  std::to_string(DEEPTILE_KERNEL_ABI_VERSION));
  }
  if (target.abi.argument_count != DEEPTILE_KERNEL_ARGUMENT_COUNT ||
      target.abi.compute_descriptor_bytes !=
          DEEPTILE_COMPUTE_DESCRIPTOR_BYTES ||
      target.abi.transfer_descriptor_bytes !=
          DEEPTILE_TRANSFER_DESCRIPTOR_BYTES) {
    add_error(diagnostics, "kernel.abi",
              "kernel argument or descriptor layout differs from ABI v1");
  }

  const deeptile::tile::HardwareSpec compiled;
  if (deeptile::tile::architecture_key(target.hardware) !=
      deeptile::tile::architecture_key(compiled)) {
    add_error(diagnostics, "architecture",
              "target architecture differs from this compiler build");
  }
  if (target.hardware.supports_residual_add !=
          compiled.supports_residual_add ||
      target.hardware.supports_maxpool != compiled.supports_maxpool ||
      target.hardware.supports_gap != compiled.supports_gap ||
      target.hardware.supports_depthwise != compiled.supports_depthwise) {
    add_error(diagnostics, "capabilities",
              "target capabilities differ from this compiler build");
  }
}

} // namespace

std::string sha256_text(const std::string &text) {
  Sha256 digest;
  digest.update(reinterpret_cast<const std::uint8_t *>(text.data()),
                text.size());
  return digest.finish();
}

bool sha256_file(const std::string &path, std::string &digest,
                 std::string &error) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    error = "cannot open file: " + path;
    return false;
  }
  Sha256 hash;
  std::array<char, 64 * 1024> buffer = {};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0) {
      hash.update(reinterpret_cast<const std::uint8_t *>(buffer.data()),
                  static_cast<std::size_t>(count));
    }
  }
  if (!input.eof()) {
    error = "failed while reading file: " + path;
    return false;
  }
  digest = hash.finish();
  return true;
}

std::string architecture_fingerprint(
    const deeptile::tile::HardwareSpec &hardware) {
  return "sha256:" + sha256_text(deeptile::tile::architecture_key(hardware));
}

bool load_target_package(const std::string &manifest_path, TargetPackage &out,
                         std::vector<TargetDiagnostic> &diagnostics) {
  out = {};
  out.manifest_path = manifest_path;
  out.root_dir = std::filesystem::path(manifest_path).parent_path().string();

  std::ifstream input(manifest_path);
  if (!input) {
    add_error(diagnostics, "target", "cannot open manifest: " + manifest_path);
    return false;
  }
  rapidjson::IStreamWrapper stream(input);
  rapidjson::Document doc;
  doc.ParseStream(stream);
  if (doc.HasParseError()) {
    std::ostringstream message;
    message << "JSON parse error at offset " << doc.GetErrorOffset() << ": "
            << rapidjson::GetParseError_En(doc.GetParseError());
    add_error(diagnostics, "target", message.str());
    return false;
  }
  if (!doc.IsObject()) {
    add_error(diagnostics, "target", "top-level JSON must be an object");
    return false;
  }

  out.schema = string_member(doc, "schema", diagnostics, "target");
  if (!schema::matches(out.schema, "target.v1")) {
    add_error(diagnostics, "target",
              "schema must be deeptile.target.v1 or tilecnn.target.v1");
  }

  const auto *identity = object_member(doc, "target", diagnostics, "target");
  const auto *platform = object_member(doc, "platform", diagnostics, "target");
  const auto *artifact = object_member(doc, "artifact", diagnostics, "target");
  const auto *kernel = object_member(doc, "kernel", diagnostics, "target");
  const auto *architecture =
      object_member(doc, "architecture", diagnostics, "target");
  const auto *capabilities =
      object_member(doc, "capabilities", diagnostics, "target");
  const auto *implementation =
      object_member(doc, "implementation", diagnostics, "target");
  const auto *provenance =
      object_member(doc, "provenance", diagnostics, "target");

  if (identity) {
    out.name = string_member(*identity, "name", diagnostics, "target");
    out.version = string_member(*identity, "version", diagnostics, "target");
    out.id = string_member(*identity, "id", diagnostics, "target");
    if (!out.name.empty() && !out.version.empty() &&
        out.id != out.name + "@" + out.version) {
      add_error(diagnostics, "target", "id must equal name@version");
    }
    if (out.name.empty() || out.version.empty() || out.id.empty()) {
      add_error(diagnostics, "target", "name, version, and id cannot be empty");
    }
  }
  if (platform) {
    out.board = string_member(*platform, "board", diagnostics, "platform");
    out.vitis_platform = string_member(*platform, "vitis_platform", diagnostics,
                                       "platform");
  }
  if (artifact) {
    out.vitis_mode =
        string_member(*artifact, "vitis_mode", diagnostics, "artifact");
    out.xclbin = string_member(*artifact, "xclbin", diagnostics, "artifact");
    out.xclbin_sha256 =
        string_member(*artifact, "sha256", diagnostics, "artifact");
    out.xclbin_uuid =
        string_member(*artifact, "uuid", diagnostics, "artifact");
    const std::filesystem::path xclbin_path(out.xclbin);
    if (xclbin_path.is_absolute() || out.xclbin.empty() ||
        has_parent_component(xclbin_path)) {
      add_error(diagnostics, "artifact", "xclbin must be a relative package path");
    }
    if (!is_sha256(out.xclbin_sha256)) {
      add_error(diagnostics, "artifact", "sha256 must be 64 lowercase hex digits");
    }
    if (out.xclbin_uuid.empty()) {
      add_error(diagnostics, "artifact", "uuid cannot be empty");
    }
    if (out.vitis_mode != "hw" && out.vitis_mode != "hw_emu" &&
        out.vitis_mode != "sw_emu") {
      add_error(diagnostics, "artifact",
                "vitis_mode must be hw, hw_emu, or sw_emu");
    }
  }
  // The boot group is optional: emulation modes have no boot image, and
  // targets packaged before the field existed remain loadable. When it is
  // present it must be complete, because a half-declared boot image is worse
  // than none — it implies a verification that would not happen.
  if (doc.HasMember("boot")) {
    const auto *boot = object_member(doc, "boot", diagnostics, "target");
    if (boot) {
      out.boot_image = string_member(*boot, "image", diagnostics, "boot");
      out.boot_sha256 = string_member(*boot, "sha256", diagnostics, "boot");
      const std::filesystem::path boot_path(out.boot_image);
      if (out.boot_image.empty() || boot_path.is_absolute() ||
          has_parent_component(boot_path)) {
        add_error(diagnostics, "boot",
                  "image must be a relative package path");
      }
      if (!is_sha256(out.boot_sha256)) {
        add_error(diagnostics, "boot",
                  "sha256 must be 64 lowercase hex digits");
      }
    }
  }
  if (kernel) {
    out.kernel_name = string_member(*kernel, "name", diagnostics, "kernel");
    out.compute_unit =
        string_member(*kernel, "compute_unit", diagnostics, "kernel");
    out.clock_hz = int64_member(*kernel, "clock_hz", diagnostics, "kernel");
    if (out.kernel_name.empty() || out.compute_unit.empty() ||
        out.clock_hz <= 0) {
      add_error(diagnostics, "kernel",
                "name, compute_unit, and a positive clock_hz are required");
    }
    const auto *abi = object_member(*kernel, "abi", diagnostics, "kernel");
    if (abi) {
      out.abi.name = string_member(*abi, "name", diagnostics, "kernel.abi");
      out.abi.version =
          int_member(*abi, "version", diagnostics, "kernel.abi");
      out.abi.argument_count =
          int_member(*abi, "argument_count", diagnostics, "kernel.abi");
      out.abi.compute_descriptor_bytes = int_member(
          *abi, "compute_descriptor_bytes", diagnostics, "kernel.abi");
      out.abi.transfer_descriptor_bytes = int_member(
          *abi, "transfer_descriptor_bytes", diagnostics, "kernel.abi");
    }
  }

  if (architecture) {
    const auto *parallelism =
        object_member(*architecture, "parallelism", diagnostics, "architecture");
    const auto *limits =
        object_member(*architecture, "limits", diagnostics, "architecture");
    const auto *numeric =
        object_member(*architecture, "numeric", diagnostics, "architecture");
    if (parallelism) {
      out.hardware.ic_tile_size = int_member(
          *parallelism, "ic_tile_size", diagnostics, "architecture.parallelism");
      out.hardware.oc_tile_size = int_member(
          *parallelism, "oc_tile_size", diagnostics, "architecture.parallelism");
      out.hardware.p_ic =
          int_member(*parallelism, "p_ic", diagnostics, "architecture.parallelism");
      out.hardware.p_spatial = int_member(
          *parallelism, "p_spatial", diagnostics, "architecture.parallelism");
    }
    if (limits) {
      out.hardware.max_tile_h = int_member(
          *limits, "max_tile_height", diagnostics, "architecture.limits");
      out.hardware.max_tile_w = int_member(
          *limits, "max_tile_width", diagnostics, "architecture.limits");
      out.hardware.k_max = int_member(
          *limits, "max_kernel_size", diagnostics, "architecture.limits");
      out.hardware.k1_cache_depth = int_member(
          *limits, "ifm_cache_depth", diagnostics, "architecture.limits");
      out.hardware.weight_buf_depth = int_member(
          *limits, "weight_buffer_depth", diagnostics, "architecture.limits");
      out.hardware.bias_buf_depth = int_member(
          *limits, "bias_buffer_depth", diagnostics, "architecture.limits");
    }
    if (numeric) {
      out.hardware.bitwidth =
          int_member(*numeric, "bitwidth", diagnostics, "architecture.numeric");
      out.hardware.gap_scale_frac_bits = int_member(
          *numeric, "gap_scale_frac_bits", diagnostics, "architecture.numeric");
    }
    const long long max_spatial_tile =
        static_cast<long long>(out.hardware.max_tile_h) *
        out.hardware.max_tile_w;
    if (max_spatial_tile > 0 &&
        max_spatial_tile <= std::numeric_limits<int>::max() &&
        out.hardware.p_spatial > 0 &&
        max_spatial_tile % out.hardware.p_spatial == 0) {
      out.hardware.max_spatial_tile = static_cast<int>(max_spatial_tile);
      out.hardware.psum_depth = deeptile::tile::HardwareSpec::psum_depth_for(
          out.hardware.max_spatial_tile, out.hardware.p_spatial);
    } else {
      out.hardware.max_spatial_tile = 0;
      out.hardware.psum_depth = 0;
      add_error(diagnostics, "architecture",
                "tile area must be representable and divisible by p_spatial");
    }
    out.architecture_fingerprint = string_member(
        *architecture, "fingerprint", diagnostics, "architecture");
    const std::string expected = architecture_fingerprint(out.hardware);
    if (!out.architecture_fingerprint.empty() &&
        out.architecture_fingerprint != expected) {
      add_error(diagnostics, "architecture",
                "fingerprint does not match the architecture fields");
    }
    if (out.architecture_fingerprint.rfind("sha256:", 0) != 0 ||
        !is_sha256(out.architecture_fingerprint.substr(7))) {
      add_error(diagnostics, "architecture",
                "fingerprint must be sha256 followed by 64 lowercase hex digits");
    }
    if (out.hardware.ic_tile_size <= 0 || out.hardware.oc_tile_size <= 0 ||
        out.hardware.p_ic <= 0 || out.hardware.p_spatial <= 0 ||
        out.hardware.max_tile_h <= 0 || out.hardware.max_tile_w <= 0 ||
        out.hardware.k_max <= 0 || out.hardware.k1_cache_depth <= 0 ||
        out.hardware.weight_buf_depth <= 0 ||
        out.hardware.bias_buf_depth <= 0 || out.hardware.bitwidth <= 0 ||
        out.hardware.gap_scale_frac_bits < 0 || out.hardware.p_spatial < 2 ||
        (out.hardware.p_spatial & (out.hardware.p_spatial - 1)) != 0) {
      add_error(diagnostics, "architecture",
                "parallelism, limits, and numeric fields are invalid");
    }
  }

  if (capabilities) {
    if (capabilities->HasMember("model_schemas") &&
        (*capabilities)["model_schemas"].IsArray()) {
      for (const auto &schema : (*capabilities)["model_schemas"].GetArray()) {
        if (schema.IsString()) {
          out.model_schemas.emplace_back(schema.GetString());
        } else {
          add_error(diagnostics, "capabilities",
                    "model_schemas entries must be strings");
        }
      }
    } else {
      add_error(diagnostics, "capabilities",
                "missing array field: model_schemas");
    }
    out.hardware.supports_residual_add = bool_member(
        *capabilities, "residual_add", diagnostics, "capabilities");
    out.hardware.supports_maxpool =
        bool_member(*capabilities, "max_pool", diagnostics, "capabilities");
    out.hardware.supports_gap = bool_member(
        *capabilities, "global_average_pool", diagnostics, "capabilities");
    out.hardware.supports_depthwise = bool_member(
        *capabilities, "fused_depthwise", diagnostics, "capabilities");
  }
  if (implementation) {
    out.psum_buffering = string_member(
        *implementation, "psum_buffering", diagnostics, "implementation");
    out.instrumentation = string_member(
        *implementation, "instrumentation", diagnostics, "implementation");
    if (out.psum_buffering != "ping-pong" &&
        out.psum_buffering != "single") {
      add_error(diagnostics, "implementation",
                "psum_buffering must be ping-pong or single");
    }
    if (out.instrumentation != "none" &&
        out.instrumentation != "profile") {
      add_error(diagnostics, "implementation",
                "instrumentation must be none or profile");
    }
  }
  if (provenance) {
    if (provenance->HasMember("verification") &&
        (*provenance)["verification"].IsString()) {
      out.provenance_verification =
          (*provenance)["verification"].GetString();
      if (out.provenance_verification != "verified" &&
          out.provenance_verification != "unverified") {
        add_error(diagnostics, "provenance",
                  "verification must be verified or unverified");
      }
    } else {
      out.provenance_verification = "unverified";
    }
    // Manifests written before the rename name the field tilecnn_revision.
    const char *revision_key =
        provenance->HasMember(schema::kRevisionField.data())
            ? schema::kRevisionField.data()
            : schema::kFormerRevisionField.data();
    out.source_revision = string_member(
        *provenance, revision_key, diagnostics, "provenance");
    out.source_dirty = bool_member(
        *provenance, "source_dirty", diagnostics, "provenance");
    out.vitis_version = string_member(
        *provenance, "vitis_version", diagnostics, "provenance");
    if (provenance->HasMember("source_revision_scope") &&
        (*provenance)["source_revision_scope"].IsString()) {
      out.source_revision_scope =
          (*provenance)["source_revision_scope"].GetString();
    }
    const std::string kernel_config_sha256 = string_member(
        *provenance, "kernel_config_sha256", diagnostics, "provenance");
    const std::string vivado_config_sha256 = string_member(
        *provenance, "vivado_impl_config_sha256", diagnostics, "provenance");
    std::string architecture_config_sha256;
    std::string abi_header_sha256;
    if (provenance->HasMember("architecture_config_sha256") &&
        (*provenance)["architecture_config_sha256"].IsString()) {
      architecture_config_sha256 =
          (*provenance)["architecture_config_sha256"].GetString();
    }
    if (provenance->HasMember("abi_header_sha256") &&
        (*provenance)["abi_header_sha256"].IsString()) {
      abi_header_sha256 = (*provenance)["abi_header_sha256"].GetString();
    }
    string_member(*provenance, "generated_by", diagnostics, "provenance");
    if (!is_sha256(kernel_config_sha256) ||
        !is_sha256(vivado_config_sha256)) {
      add_error(diagnostics, "provenance",
                "configuration checksums must be 64 lowercase hex digits");
    }
    if (provenance->HasMember("build_record_sha256") &&
        (*provenance)["build_record_sha256"].IsString()) {
      out.build_record_sha256 =
          (*provenance)["build_record_sha256"].GetString();
    }
    if (out.provenance_verification == "verified") {
      if (out.source_revision_scope != "fpga-build-record" ||
          !is_sha256(out.build_record_sha256) ||
          !is_sha256(architecture_config_sha256) ||
          !is_sha256(abi_header_sha256)) {
        add_error(diagnostics, "provenance",
                  "verified provenance requires an FPGA build-record scope "
                  "and complete configuration checksums");
      }
    } else {
      add_warning(diagnostics, "provenance",
                  "XCLBIN origin is unverified; compatibility checks still apply");
    }
  }

  if (std::none_of(out.model_schemas.begin(), out.model_schemas.end(),
                   [](const std::string &id) {
                     return schema::matches(id, "graph.v1");
                   })) {
    add_error(diagnostics, "capabilities",
              "target does not support deeptile.graph.v1 or tilecnn.graph.v1");
  }
  validate_compiled_compatibility(out, diagnostics);
  return !has_errors(diagnostics);
}

bool has_errors(const std::vector<TargetDiagnostic> &diagnostics) {
  for (const auto &diagnostic : diagnostics) {
    if (diagnostic.severity == TargetDiagnosticSeverity::Error) {
      return true;
    }
  }
  return false;
}

std::string severity_name(TargetDiagnosticSeverity severity) {
  return severity == TargetDiagnosticSeverity::Error ? "ERROR" : "WARN";
}

} // namespace target
} // namespace deeptile
