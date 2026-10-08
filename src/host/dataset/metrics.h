/**
 * @file metrics.h
 * @brief Classification metrics over quantized logits.
 * @ingroup deeptile_host
 */

#ifndef DEEPTILE_DATASET_METRICS_H_
#define DEEPTILE_DATASET_METRICS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace deeptile {
namespace dataset {

/**
 * Rank classes by logit, highest first.
 *
 * Logits are int8, so their value space is small and exact ties between
 * classes are common rather than exceptional. Ties break towards the lower
 * class index, deterministically, because an unspecified tie rule makes top-k
 * accuracy irreproducible and incomparable with published baselines.
 *
 * @param logits One value per class.
 * @param k Number of classes to return; clamped to the class count.
 */
std::vector<int> top_k(const std::vector<std::int8_t> &logits, int k);

/** Running top-1 and top-k counts over a dataset. */
struct ClassificationMetrics {
  int k = 5;
  std::size_t evaluated = 0;
  std::size_t top1_correct = 0;
  std::size_t topk_correct = 0;
  std::size_t unlabelled = 0;

  /** Accumulate one prediction. A negative label counts as unlabelled. */
  void accumulate(const std::vector<std::int8_t> &logits, int label);

  double top1_accuracy() const;
  double topk_accuracy() const;
};

/** Agreement between accelerator output and a reference, per image. */
struct AgreementMetrics {
  std::size_t compared = 0;
  std::size_t logits_identical = 0;
  std::size_t top1_identical = 0;

  void accumulate(const std::vector<std::int8_t> &observed,
                  const std::vector<std::int8_t> &reference);

  double logit_agreement() const;
  double top1_agreement() const;
};

/** Format a class name for reporting, falling back to the bare index. */
std::string class_label(const std::vector<std::string> &class_names, int index);

} // namespace dataset
} // namespace deeptile

#endif // DEEPTILE_DATASET_METRICS_H_
