/**
 * @file target_artifact.h
 * @brief Resolves and verifies the XCLBIN carried by a TargetPackage.
 * @ingroup deeptile_host
 */

#ifndef DEEPTILE_TARGET_ARTIFACT_H_
#define DEEPTILE_TARGET_ARTIFACT_H_

#include "runtime/runtime_types.h"
#include "deeptile/compiler/target_package.h"

#include <string>
#include <vector>

namespace deeptile {
namespace runtime {

/**
 * @brief Resolves and verifies artifacts declared by a TargetPackage.
 *
 * Checks the XCLBIN, an optional boot image, and the deployer's boot-session
 * marker when present.
 */
bool resolve_target_artifact(
    const deeptile::target::TargetPackage &target, std::string &xclbin_path,
    std::vector<RuntimeDiagnostic> &diagnostics);

} // namespace runtime
} // namespace deeptile

#endif // DEEPTILE_TARGET_ARTIFACT_H_
