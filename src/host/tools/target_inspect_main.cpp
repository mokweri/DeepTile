/**
 * @file target_inspect_main.cpp
 * @brief Validates and summarizes a DeepTile TargetPackage.
 */

#include "runtime/target_artifact.h"
#include "deeptile/compiler/target_package.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <target.json>\n";
    return EXIT_FAILURE;
  }

  deeptile::target::TargetPackage target;
  std::vector<deeptile::target::TargetDiagnostic> target_diagnostics;
  if (!deeptile::target::load_target_package(argv[1], target,
                                            target_diagnostics)) {
    for (const auto &diagnostic : target_diagnostics) {
      std::cerr << "[" << deeptile::target::severity_name(diagnostic.severity)
                << "] " << diagnostic.context << ": " << diagnostic.message
                << "\n";
    }
    return EXIT_FAILURE;
  }
  for (const auto &diagnostic : target_diagnostics) {
    std::cerr << "[" << deeptile::target::severity_name(diagnostic.severity)
              << "] " << diagnostic.context << ": " << diagnostic.message
              << "\n";
  }

  std::string artifact;
  std::vector<deeptile::runtime::RuntimeDiagnostic> runtime_diagnostics;
  if (!deeptile::runtime::resolve_target_artifact(
          target, artifact, runtime_diagnostics)) {
    for (const auto &diagnostic : runtime_diagnostics) {
      std::cerr << "["
                << deeptile::runtime::severity_name(diagnostic.severity) << "] "
                << diagnostic.context << ": " << diagnostic.message << "\n";
    }
    return EXIT_FAILURE;
  }

  std::cout << "Target:       " << target.id << "\n"
            << "Board:        " << target.board << "\n"
            << "Platform:     " << target.vitis_platform << "\n"
            << "Vitis mode:   " << target.vitis_mode << "\n"
            << "Kernel ABI:   " << target.abi.name << " v"
            << target.abi.version << "\n"
            << "Architecture: " << target.architecture_fingerprint << "\n"
            << "Provenance:   " << target.provenance_verification << "\n"
            << "Artifact:     " << artifact << "\n"
            << "Status:       compatible\n";
  return EXIT_SUCCESS;
}
