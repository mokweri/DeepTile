/**
 * @file environment.h
 * @brief Declares environment-variable lookup under DeepTile's current and former prefixes.
 * @ingroup deeptile_compiler
 */

#ifndef DEEPTILE_ENVIRONMENT_H_
#define DEEPTILE_ENVIRONMENT_H_

#include <cstdlib>
#include <string>

namespace deeptile {

/**
 * Value of `DEEPTILE_<name>`, or of the former `TILECNN_<name>` when only that
 * is set; null when neither is. The former prefix keeps existing scripts working.
 */
inline const char *env_value(const char *name) {
  const std::string current = std::string("DEEPTILE_") + name;
  if (const char *value = std::getenv(current.c_str())) {
    return value;
  }
  const std::string former = std::string("TILECNN_") + name;
  return std::getenv(former.c_str());
}

/** True when the variable is set to anything other than empty or "0". */
inline bool env_flag(const char *name) {
  const char *value = env_value(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

}  // namespace deeptile

#endif  // DEEPTILE_ENVIRONMENT_H_
