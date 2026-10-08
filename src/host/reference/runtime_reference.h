/**
 * @file runtime_reference.h
 * @brief Bit-exact software reference interface for prepared runtime plans.
 * @ingroup deeptile_host
 */

#ifndef DEEPTILE_RUNTIME_REFERENCE_H_
#define DEEPTILE_RUNTIME_REFERENCE_H_

#include "runtime/runtime.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace deeptile {
namespace runtime {

struct RuntimeReferenceTensor {
  std::string id;
  std::vector<int> shape;
  std::vector<std::int8_t> data_chw;
};

struct RuntimeReferenceResult {
  std::map<std::string, RuntimeReferenceTensor> tensors;
  std::vector<RuntimeDiagnostic> diagnostics;
};

/** Generate reference tensors for every launch in a prepared plan. */
RuntimeReferenceResult
generate_bit_exact_reference(const RuntimeExecutionPlan &plan);

const RuntimeReferenceTensor *
find_reference_tensor(const RuntimeReferenceResult &result,
                      const std::string &tensor_id);

} // namespace runtime
} // namespace deeptile

#endif // DEEPTILE_RUNTIME_REFERENCE_H_
