/**
 * @file metrics.cpp
 * @brief Implements classification and agreement metrics.
 */

#include "dataset/metrics.h"

#include <algorithm>
#include <numeric>

namespace deeptile {
namespace dataset {

std::vector<int> top_k(const std::vector<std::int8_t> &logits, int k) {
  const int classes = static_cast<int>(logits.size());
  if (classes == 0 || k <= 0) {
    return {};
  }
  const int wanted = std::min(k, classes);

  std::vector<int> order(static_cast<std::size_t>(classes));
  std::iota(order.begin(), order.end(), 0);
  // Stable ordering on index makes the tie rule explicit rather than incidental
  // to the sort implementation.
  std::partial_sort(order.begin(), order.begin() + wanted, order.end(),
                    [&logits](int a, int b) {
                      const int va = logits[static_cast<std::size_t>(a)];
                      const int vb = logits[static_cast<std::size_t>(b)];
                      if (va != vb) {
                        return va > vb;
                      }
                      return a < b;
                    });
  order.resize(static_cast<std::size_t>(wanted));
  return order;
}

void ClassificationMetrics::accumulate(const std::vector<std::int8_t> &logits,
                                       int label) {
  if (label < 0) {
    ++unlabelled;
    return;
  }
  const std::vector<int> ranked = top_k(logits, k);
  if (ranked.empty()) {
    return;
  }
  ++evaluated;
  if (ranked.front() == label) {
    ++top1_correct;
  }
  if (std::find(ranked.begin(), ranked.end(), label) != ranked.end()) {
    ++topk_correct;
  }
}

double ClassificationMetrics::top1_accuracy() const {
  return evaluated == 0 ? 0.0
                        : static_cast<double>(top1_correct) /
                              static_cast<double>(evaluated);
}

double ClassificationMetrics::topk_accuracy() const {
  return evaluated == 0 ? 0.0
                        : static_cast<double>(topk_correct) /
                              static_cast<double>(evaluated);
}

void AgreementMetrics::accumulate(const std::vector<std::int8_t> &observed,
                                  const std::vector<std::int8_t> &reference) {
  ++compared;
  if (observed.size() == reference.size() && observed == reference) {
    ++logits_identical;
  }
  const std::vector<int> a = top_k(observed, 1);
  const std::vector<int> b = top_k(reference, 1);
  if (!a.empty() && !b.empty() && a.front() == b.front()) {
    ++top1_identical;
  }
}

double AgreementMetrics::logit_agreement() const {
  return compared == 0 ? 0.0
                       : static_cast<double>(logits_identical) /
                             static_cast<double>(compared);
}

double AgreementMetrics::top1_agreement() const {
  return compared == 0 ? 0.0
                       : static_cast<double>(top1_identical) /
                             static_cast<double>(compared);
}

std::string class_label(const std::vector<std::string> &class_names,
                        int index) {
  if (index < 0) {
    return "unlabelled";
  }
  if (static_cast<std::size_t>(index) < class_names.size()) {
    return class_names[static_cast<std::size_t>(index)];
  }
  return std::to_string(index);
}

} // namespace dataset
} // namespace deeptile
