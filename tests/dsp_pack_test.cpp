// dsp_pack_test.cpp — standalone bit-exactness proof for the dual-INT8 DSP
// packing scheme used in src/kernel/conv_engine.cpp (dsp_dual_mul + the
// OC_PAIR reduction in mac_tile).
//
// Mirrors the kernel arithmetic exactly but runs as plain C++ (no HLS
// synthesis), so the single-pair check can be exhaustive over all 2^24
// (a, w0, w1) triples. Validates:
//   1. Both packed lanes equal independent int multiplies (all triples).
//   2. The 35-bit product never truncates vs. a 45-bit reference.
//   3. The 16-term, deferred-popcount-correction lane reduction is exact,
//      including the saturating corners (all -128; mixed +-128/127).
//   4. Negative control: the lane-UNSAFE packed-word tree is shown to fail,
//      so the reduction test genuinely discriminates correct lane handling.
//
// Build & run (needs only the Vitis/HLS ap_int headers):
//   g++ -O2 -std=c++14 -I/opt/Xilinx/Vitis/2024.2/include \
//       tests/dsp_pack_test.cpp -o /tmp/dsp_pack_test && /tmp/dsp_pack_test
//
// Exit code 0 on success, 1 on any failure.

#include <ap_int.h>
#include <cstdio>
#include <cstdlib>

typedef ap_int<8>  pixel_t;
typedef ap_int<8>  weight_t;
typedef ap_int<32> acc_t;

// ── Under test: identical to dsp_dual_mul in conv_engine.cpp ──────────────
// Native 8-bit activation (no fabric widening); 27s x 8s -> 35s.
static inline ap_int<35> dsp_dual_mul(pixel_t a, weight_t w0, weight_t w1) {
  const ap_int<27> packed_w = (ap_int<27>(w1) << 18) + ap_int<27>(w0);
  return packed_w * a;
}

// ── Part 1: exhaustive single-pair + 35-bit width safety ──────────────────
static long long part1(long long &width_fail) {
  long long fails = 0;
  width_fail = 0;
  for (int a = -128; a <= 127; a++)
    for (int w0 = -128; w0 <= 127; w0++)
      for (int w1 = -128; w1 <= 127; w1++) {
        const ap_int<35> pr = dsp_dual_mul(pixel_t(a), weight_t(w0), weight_t(w1));
        // 45-bit reference detects any truncation in the 35-bit product.
        const ap_int<45> ref =
            ((ap_int<27>(w1) << 18) + ap_int<27>(w0)) * ap_int<18>(a);
        if (pr != ref) width_fail++;

        const acc_t lo = acc_t(ap_int<16>(pr.range(15, 0)));
        const acc_t hi = acc_t(ap_int<16>(pr.range(33, 18))) + acc_t(pr[15]);
        if ((int)lo != a * w0 || (int)hi != a * w1) {
          if (fails < 5)
            printf("PAIR MISMATCH a=%d w0=%d w1=%d  lo=%d(exp %d) hi=%d(exp %d)\n",
                   a, w0, w1, (int)lo, a * w0, (int)hi, a * w1);
          fails++;
        }
      }
  return fails;
}

// ── The kernel's lane reduction: extract raw, two trees, deferred popcount ─
static void lane_reduce(const pixel_t a[16], const weight_t w0[16],
                        const weight_t w1[16], acc_t &lane0, acc_t &lane1) {
  acc_t p0[16], p1[16];
  ap_uint<1> lo_neg[16];
  for (int c = 0; c < 16; c++) {
    const ap_int<35> pr = dsp_dual_mul(a[c], w0[c], w1[c]);
    p0[c] = acc_t(ap_int<16>(pr.range(15, 0)));
    p1[c] = acc_t(ap_int<16>(pr.range(33, 18)));  // raw (uncorrected) high lane
    lo_neg[c] = pr[15];
  }
  for (int step = 8; step > 0; step >>= 1)
    for (int i = 0; i < step; i++) { p0[i] += p0[i + step]; p1[i] += p1[i + step]; }
  ap_uint<5> nb = 0;
  for (int c = 0; c < 16; c++) nb += lo_neg[c];
  lane0 = p0[0];
  lane1 = p1[0] + acc_t(nb);  // restore the 16 deferred borrows at once
}

// ── Negative control: the lane-unsafe packed-word tree rejected in the Phase 2 note.
// Sums groups of 4 packed 35-bit products, THEN extracts lanes. Unsafe: a
// negative low product stores 2^18+lo, so the low-field sum leaks n_neg*2^18
// into the high lane. Must fail heavily — proves the reduction test above is
// actually discriminating, not vacuously passing.
static void lane_reduce_packed_BROKEN(const pixel_t a[16], const weight_t w0[16],
                                      const weight_t w1[16],
                                      acc_t &lane0, acc_t &lane1) {
  ap_int<35> pr[16];
  for (int c = 0; c < 16; c++) pr[c] = dsp_dual_mul(a[c], w0[c], w1[c]);
  ap_int<45> psum[4];
  for (int g = 0; g < 4; g++)
    psum[g] = (pr[4 * g] + pr[4 * g + 1]) + (pr[4 * g + 2] + pr[4 * g + 3]);
  lane0 = 0; lane1 = 0;
  for (int g = 0; g < 4; g++) {
    lane0 += acc_t(ap_int<18>(psum[g].range(17, 0)));
    lane1 += acc_t(ap_int<18>(psum[g].range(35, 18)));
  }
}

// ── Depthwise orientation (P5 / mac_tile_dw): the SAME DSP, roles swapped ──
// Pack two SPATIAL activations on the 27-bit port and broadcast ONE weight on
// the 8-bit port: dsp_dual_mul(w, a0, a1). Low lane = a0·w (spatial pos p0),
// high lane = a1·w (spatial pos p1). There is no reduction tree, so the borrow
// is restored PER PRODUCT (+pr[15]) as the two products accumulate into their
// independent spatial accumulators over K² kernel positions. This mirrors the
// mac_tile_dw inner loop exactly; it must equal two independent integer MAC
// streams for all inputs.
static void dw_lane_accum(const weight_t w[49], const pixel_t a0[49],
                          const pixel_t a1[49], int k2, acc_t &acc0, acc_t &acc1) {
  // The packing is dsp_dual_mul(a=w, w0=a0, w1=a1) — the broadcast weight is the
  // shared 8-bit operand; the two activations ride the packed 27-bit port.
  acc0 = 0;
  acc1 = 0;
  for (int k = 0; k < k2; k++) {
    const ap_int<35> pr = dsp_dual_mul(w[k], a0[k], a1[k]);
    const acc_t prod0 = acc_t(ap_int<16>(pr.range(15, 0)));
    const acc_t prod1 = acc_t(ap_int<16>(pr.range(33, 18))) + acc_t(pr[15]);
    acc0 += prod0;
    acc1 += prod1;
  }
}

static int fill_vec(int it, pixel_t a[16], weight_t w0[16], weight_t w1[16],
                    int &e0, int &e1) {
  e0 = e1 = 0;
  for (int c = 0; c < 16; c++) {
    int av, x0, x1;
    if      (it < 1000) { av = -128; x0 = -128; x1 = -128; }  // worst-case corner
    else if (it < 2000) { av = -128; x0 =  127; x1 = -128; }
    else if (it < 3000) { av =  127; x0 = -128; x1 =  127; }
    else { av = (rand() % 256) - 128; x0 = (rand() % 256) - 128; x1 = (rand() % 256) - 128; }
    a[c] = av; w0[c] = x0; w1[c] = x1;
    e0 += av * x0; e1 += av * x1;
  }
  return 0;
}

int main() {
  long long width_fail = 0;
  const long long f1 = part1(width_fail);
  printf("Part1 single-pair (all 2^24 triples): %s  (%lld mismatches, %lld truncations)\n",
         (f1 || width_fail) ? "FAIL" : "PASS", f1, width_fail);

  const int N = 5000000;

  srand(7);
  long long f2 = 0;
  for (int it = 0; it < N; it++) {
    pixel_t a[16]; weight_t w0[16], w1[16]; int e0, e1;
    fill_vec(it, a, w0, w1, e0, e1);
    acc_t l0, l1; lane_reduce(a, w0, w1, l0, l1);
    if ((int)l0 != e0 || (int)l1 != e1) {
      if (f2 < 5) printf("RED MISMATCH it=%d l0=%d(exp %d) l1=%d(exp %d)\n",
                         it, (int)l0, e0, (int)l1, e1);
      f2++;
    }
  }
  printf("Part2 16-term reduction (5M vectors + corners): %s  (%lld mismatches)\n",
         f2 ? "FAIL" : "PASS", f2);

  // Negative control: the broken packed-word tree must mismatch heavily.
  srand(7);
  long long fbroken = 0;
  for (int it = 0; it < N; it++) {
    pixel_t a[16]; weight_t w0[16], w1[16]; int e0, e1;
    fill_vec(it, a, w0, w1, e0, e1);
    acc_t l0, l1; lane_reduce_packed_BROKEN(a, w0, w1, l0, l1);
    if ((int)l0 != e0 || (int)l1 != e1) fbroken++;
  }
  printf("Part3 negative control (packed-word tree must fail): %s  (%lld/%d mismatch, %.2f%%)\n",
         fbroken > N / 2 ? "PASS" : "FAIL", fbroken, N, 100.0 * fbroken / N);

  // Part4 — depthwise orientation (mac_tile_dw): two spatial activations packed,
  // one broadcast weight, per-product folded borrow, accumulated over K².
  srand(11);
  long long f4 = 0;
  for (int it = 0; it < N; it++) {
    weight_t w[49]; pixel_t a0[49], a1[49];
    const int k2 = (it < 3000) ? 9 : 1 + (rand() % 49);  // mostly K=3², some 1..49
    long long e0 = 0, e1 = 0;
    for (int k = 0; k < k2; k++) {
      int wv, x0, x1;
      if      (it < 1000) { wv = -128; x0 = -128; x1 = -128; }   // worst-case corner
      else if (it < 2000) { wv = -128; x0 =  127; x1 = -128; }
      else if (it < 3000) { wv =  127; x0 = -128; x1 =  127; }
      else { wv = (rand()%256)-128; x0 = (rand()%256)-128; x1 = (rand()%256)-128; }
      w[k] = wv; a0[k] = x0; a1[k] = x1;
      e0 += (long long)x0 * wv; e1 += (long long)x1 * wv;
    }
    acc_t acc0, acc1; dw_lane_accum(w, a0, a1, k2, acc0, acc1);
    if ((long long)acc0 != e0 || (long long)acc1 != e1) {
      if (f4 < 5) printf("DW MISMATCH it=%d k2=%d acc0=%d(exp %lld) acc1=%d(exp %lld)\n",
                         it, k2, (int)acc0, e0, (int)acc1, e1);
      f4++;
    }
  }
  printf("Part4 depthwise accumulation (5M K^2 streams + corners): %s  (%lld mismatches)\n",
         f4 ? "FAIL" : "PASS", f4);

  const bool ok = !(f1 || width_fail || f2 || f4) && (fbroken > N / 2);
  printf("\nOVERALL: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
