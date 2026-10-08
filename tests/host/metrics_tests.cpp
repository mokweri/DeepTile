/**
 * @file metrics_tests.cpp
 * @brief Tests for classification and agreement metrics.
 *
 * Logits are int8, so ties between classes are routine. The tie rule is the
 * main thing under test here: an unspecified rule makes top-k accuracy
 * irreproducible and incomparable with published baselines.
 */

#include "dataset/metrics.h"

#include <functional>
#include <iostream>
#include <string>
#include <vector>

#define TEST_ASSERT(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "    ASSERT FAILED: " << msg << "\n";                       \
      return false;                                                            \
    }                                                                          \
  } while (0)

namespace {

using deeptile::dataset::AgreementMetrics;
using deeptile::dataset::ClassificationMetrics;
using deeptile::dataset::top_k;

std::vector<std::int8_t> logits(std::initializer_list<int> values) {
  std::vector<std::int8_t> out;
  for (const int v : values) {
    out.push_back(static_cast<std::int8_t>(v));
  }
  return out;
}

bool test_ranks_by_value() {
  const auto ranked = top_k(logits({3, 9, -2, 7, 0}), 3);
  TEST_ASSERT(ranked.size() == 3, "expected 3 results");
  TEST_ASSERT(ranked[0] == 1 && ranked[1] == 3 && ranked[2] == 0,
              "wrong ranking: " << ranked[0] << "," << ranked[1] << ","
                                << ranked[2]);
  return true;
}

/** Ties resolve to the lower class index, always. */
bool test_ties_break_to_lower_index() {
  const auto ranked = top_k(logits({5, 5, 5, 5}), 2);
  TEST_ASSERT(ranked.size() == 2, "expected 2 results");
  TEST_ASSERT(ranked[0] == 0 && ranked[1] == 1,
              "tie broke to " << ranked[0] << "," << ranked[1]);

  // A tie straddling the top-k boundary must resolve the same way.
  const auto boundary = top_k(logits({9, 4, 4, 1}), 2);
  TEST_ASSERT(boundary[0] == 0 && boundary[1] == 1,
              "boundary tie broke to " << boundary[0] << "," << boundary[1]);
  return true;
}

bool test_k_is_clamped() {
  TEST_ASSERT(top_k(logits({1, 2}), 5).size() == 2, "k must clamp to classes");
  TEST_ASSERT(top_k(logits({1, 2}), 0).empty(), "k=0 yields nothing");
  TEST_ASSERT(top_k({}, 5).empty(), "no classes yields nothing");
  return true;
}

bool test_accuracy_counts() {
  ClassificationMetrics metrics;
  metrics.k = 3;
  metrics.accumulate(logits({9, 1, 2, 3}), 0);  // top-1 hit
  metrics.accumulate(logits({9, 1, 2, 3}), 3);  // in top-3, not top-1
  metrics.accumulate(logits({9, 1, 2, 3}), 1);  // rank 4, miss
  TEST_ASSERT(metrics.evaluated == 3, "evaluated=" << metrics.evaluated);
  TEST_ASSERT(metrics.top1_correct == 1, "top1=" << metrics.top1_correct);
  TEST_ASSERT(metrics.topk_correct == 2, "topk=" << metrics.topk_correct);
  TEST_ASSERT(metrics.top1_accuracy() > 0.33 && metrics.top1_accuracy() < 0.34,
              "top1 accuracy " << metrics.top1_accuracy());
  return true;
}

/** Unlabelled images must not silently count as wrong. */
bool test_unlabelled_excluded() {
  ClassificationMetrics metrics;
  metrics.accumulate(logits({9, 1}), -1);
  metrics.accumulate(logits({9, 1}), 0);
  TEST_ASSERT(metrics.unlabelled == 1, "unlabelled=" << metrics.unlabelled);
  TEST_ASSERT(metrics.evaluated == 1, "evaluated=" << metrics.evaluated);
  TEST_ASSERT(metrics.top1_accuracy() == 1.0, "accuracy must ignore unlabelled");
  return true;
}

bool test_empty_metrics_do_not_divide_by_zero() {
  ClassificationMetrics metrics;
  TEST_ASSERT(metrics.top1_accuracy() == 0.0, "empty top1");
  TEST_ASSERT(metrics.topk_accuracy() == 0.0, "empty topk");
  AgreementMetrics agreement;
  TEST_ASSERT(agreement.logit_agreement() == 0.0, "empty logit agreement");
  return true;
}

/** Identical top-1 with differing logits is a real and useful distinction. */
bool test_agreement_separates_logits_from_top1() {
  AgreementMetrics agreement;
  agreement.accumulate(logits({9, 1, 2}), logits({9, 1, 2}));
  agreement.accumulate(logits({9, 1, 2}), logits({8, 1, 2}));
  agreement.accumulate(logits({1, 9, 2}), logits({9, 1, 2}));
  TEST_ASSERT(agreement.compared == 3, "compared=" << agreement.compared);
  TEST_ASSERT(agreement.logits_identical == 1,
              "logits_identical=" << agreement.logits_identical);
  TEST_ASSERT(agreement.top1_identical == 2,
              "top1_identical=" << agreement.top1_identical);
  return true;
}

bool test_class_label() {
  const std::vector<std::string> names = {"tench", "goldfish"};
  TEST_ASSERT(deeptile::dataset::class_label(names, 1) == "goldfish", "name");
  TEST_ASSERT(deeptile::dataset::class_label(names, 7) == "7", "index fallback");
  TEST_ASSERT(deeptile::dataset::class_label(names, -1) == "unlabelled",
              "unlabelled");
  return true;
}

int passed = 0;
int failed = 0;

void run(const std::string &name, const std::function<bool()> &fn) {
  std::cout << "  " << name << " ... " << std::flush;
  if (fn()) {
    std::cout << "PASS\n";
    ++passed;
  } else {
    std::cout << "FAIL\n";
    ++failed;
  }
}

} // namespace

int main() {
  run("ranks_by_value", test_ranks_by_value);
  run("ties_break_to_lower_index", test_ties_break_to_lower_index);
  run("k_is_clamped", test_k_is_clamped);
  run("accuracy_counts", test_accuracy_counts);
  run("unlabelled_excluded", test_unlabelled_excluded);
  run("empty_metrics_safe", test_empty_metrics_do_not_divide_by_zero);
  run("agreement_separates_logits_from_top1",
      test_agreement_separates_logits_from_top1);
  run("class_label", test_class_label);

  std::cout << "Metrics tests: " << passed << " passed, " << failed
            << " failed\n";
  return failed == 0 ? 0 : 1;
}
