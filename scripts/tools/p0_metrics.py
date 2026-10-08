#!/usr/bin/env python3
"""Recompute P0 cost-model metrics from recorded validation measurements.

The embedded measurements originate from HLS co-simulation, XSIM waveforms,
and XRT profiles. Derived outputs include:

  * weight_load per-bank burst overhead  Wbank = (epoch_meas - 16*depth)/16
  * array (MAC-feed) efficiency           ideal_mac_iters / cosim_cycles
  * achieved GOPS and the structural peak
  * DDR traffic, operational intensity and roofline (I, GOPS) coordinates
  * predicted-vs-measured residual per test case

Uses only the Python standard library.
"""

import os
import re
import sys
from dataclasses import dataclass, field

# Read architecture constants from the hardware source of truth.
_DEFS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "src", "common", "arch_config.h")


def _arch(name):
    with open(_DEFS) as fh:
        m = re.search(rf"^#define\s+{name}\s+(\d+)", fh.read(), re.M)
    if m is None:
        sys.exit(f"error: could not read #define {name} from {_DEFS}")
    return int(m.group(1))


P_SPATIAL = _arch("P_SPATIAL")
P_IC = _arch("P_IC")
OC_TILE_SIZE = _arch("OC_TILE_SIZE")
CLOCK_MHZ = 200.0
MACS_PER_CYCLE = P_SPATIAL * P_IC * OC_TILE_SIZE            # 1024
PEAK_GMACS = MACS_PER_CYCLE * CLOCK_MHZ / 1000.0           # 204.8 GMAC/s
PEAK_GOPS = 2.0 * PEAK_GMACS                                # 409.6 GOP/s
HP_BW_MBS = 3200.0                                          # per HP port (gmem0/1/3)

# DDR constants are derived from the TC1 HP1 XRT trace.
BYTES_PER_CYCLE = HP_BW_MBS / CLOCK_MHZ                     # 16 B/cy (3200/200)
XRT_TRANSFER_BYTES = 256                                    # avg HP1 transfer size
XRT_TRANSFER_NS = 75.0                                      # single-read occupancy


def ceil_div(a, b):
    return (a + b - 1) // b


@dataclass
class TC:
    name: str
    K: int
    S: int
    Cin: int
    Cout: int
    H: int          # input H (=W assumed square)
    OFM: int        # output H (=W)
    cached: bool    # K=1 URAM IFM cache path used
    cosim_cy: int   # HLS cosim total cycles
    board_cy: int   # XRT board cycles (200 MHz)
    weight_epoch_meas: int          # measured weight_load cycles / epoch (XSIM)
    padded: int = 0                 # padded tile side (K>1) ; 0 -> derive
    residual: bool = False
    board_epoch_us: float = 0.0     # measured first weight-epoch span (XRT waveform), 0=n/a
    board_ifm_bytes: int = 0        # board-measured HP0 ifm read bytes/call (XRT), 0=use estimate
    label: str = ""

    # Derived geometry and performance metrics.
    @property
    def num_cig(self):
        return ceil_div(self.Cin, P_IC)

    @property
    def num_cog(self):
        return ceil_div(self.Cout, OC_TILE_SIZE)

    @property
    def num_outputs(self):
        return self.OFM * self.OFM

    @property
    def num_groups(self):            # spatial groups (kernel: conv_engine.cpp:107)
        return ceil_div(self.num_outputs, P_SPATIAL)

    @property
    def K2(self):
        return self.K * self.K

    @property
    def depth_count(self):           # log convention: per-bank depth = num_cig*K^2
        return self.num_cig * self.K2

    @property
    def compute_per_cog(self):       # mac_tile MAC_FLAT iterations per COG
        return self.num_cig * self.num_groups * self.K2

    @property
    def ideal_mac_iters(self):       # whole layer, II=1 -> 1 cycle each
        return self.num_cog * self.compute_per_cog

    @property
    def weight_formula(self):        # OC_TILE_SIZE * depth_count
        return OC_TILE_SIZE * self.depth_count

    @property
    def wbank(self):
        return (self.weight_epoch_meas - self.weight_formula) / OC_TILE_SIZE

    @property
    def weight_vectors(self):        # total 128-bit weight vectors for the layer
        return self.weight_bytes // P_IC

    @property
    def weight_read_cy_bw(self):     # BW-bound: 1 vector (16 B) per cycle
        return self.weight_bytes / BYTES_PER_CYCLE

    @property
    def weight_frac_of_call(self):   # total weight-read time as fraction of the call
        return self.weight_read_cy_bw / self.cosim_cy

    @property
    def array_eff(self):             # fraction of cycles the 1024-MAC array is fed
        return self.ideal_mac_iters / self.cosim_cy

    @property
    def useful_macs(self):           # dense, valid-output MACs
        return self.Cout * self.Cin * self.K2 * self.num_outputs

    @property
    def achieved_gmacs(self):
        return self.useful_macs / (self.board_cy / (CLOCK_MHZ * 1e6)) / 1e9

    @property
    def achieved_gops(self):
        return 2.0 * self.achieved_gmacs

    # DDR traffic uses one byte per INT8 element.
    @property
    def ifm_side(self):
        return self.padded if self.padded else self.H

    @property
    def weight_bytes(self):
        return self.Cout * self.Cin * self.K2

    @property
    def ifm_useful_bytes(self):      # minimal: each input pixel read once, no pad/reload
        return self.Cin * self.H * self.H

    @property
    def ifm_bytes(self):
        if self.board_ifm_bytes:     # prefer board-measured HP0 read traffic
            return self.board_ifm_bytes
        base = self.Cin * self.ifm_side * self.ifm_side
        # Cached inputs load once; uncached inputs reload for each COG.
        return base if self.cached else base * self.num_cog

    @property
    def ifm_overfetch(self):
        return self.ifm_bytes / self.ifm_useful_bytes

    @property
    def ifm_read_cy(self):           # BW-bound HP0 read time (16 B/cy)
        return self.ifm_bytes / BYTES_PER_CYCLE

    @property
    def ofm_bytes(self):
        return self.Cout * self.num_outputs

    @property
    def ddr_bytes(self):
        b = self.weight_bytes + self.ifm_bytes + self.ofm_bytes
        if self.residual:
            b += self.ofm_bytes        # residual read ~= OFM size
        return b

    @property
    def op_intensity(self):          # MAC per DDR byte
        return self.useful_macs / self.ddr_bytes


# Measured calibration data retained for reproducibility.
TCS = [
    TC("TC1", K=3, S=1, Cin=128, Cout=128, H=28, OFM=28, cached=False,
       cosim_cy=125807, board_cy=126475, weight_epoch_meas=4748, padded=30,
       board_ifm_bytes=802032, label="best_case_3x3_s1 (MAC)"),
    TC("TC2", K=1, S=1, Cin=1024, Cout=256, H=14, OFM=14, cached=True,
       cosim_cy=80105, board_cy=80302, weight_epoch_meas=4624,
       label="deep_k1_s1 (MAC)"),
    TC("TC3", K=1, S=1, Cin=2048, Cout=512, H=7, OFM=7, cached=True,
       cosim_cy=114983, board_cy=115648, weight_epoch_meas=32800, board_epoch_us=164.0,
       label="late_k1_s1 (Weight)"),
    TC("TC4", K=1, S=2, Cin=256, Cout=512, H=56, OFM=28, cached=True,
       cosim_cy=156005, board_cy=156215, weight_epoch_meas=4236,
       label="proj_bottleneck_1x1_s2 (MAC+fill+emit)"),
    TC("TC5", K=1, S=2, Cin=1024, Cout=2048, H=14, OFM=7, cached=True,
       cosim_cy=214181, board_cy=214784, weight_epoch_meas=65676, board_epoch_us=327.0,
       label="small_proj_1x1_s2 (Weight)"),
    TC("TC6", K=3, S=2, Cin=128, Cout=128, H=56, OFM=28, cached=False,
       cosim_cy=230525, board_cy=227600, weight_epoch_meas=4748, padded=29,
       board_ifm_bytes=3268608, label="stride2_3x3_s2 (IFM)"),
    TC("TC7c", K=1, S=1, Cin=512, Cout=2048, H=7, OFM=7, cached=True,
       cosim_cy=109068, board_cy=109684, weight_epoch_meas=32908, residual=True,
       label="tail_gap conv3 (Weight)"),
    TC("TC7f", K=1, S=1, Cin=2048, Cout=1008, H=1, OFM=1, cached=True,
       cosim_cy=145385, board_cy=146041, weight_epoch_meas=65676,
       label="tail_gap FC (Weight, inter-epoch stall)"),
]


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    hr("Architecture constants")
    print(f"  MACs/cycle = P_SPATIAL*P_IC*OC_TILE_SIZE = {MACS_PER_CYCLE}")
    print(f"  Structural peak @ {CLOCK_MHZ:.0f} MHz = {PEAK_GMACS:.1f} GMAC/s = "
          f"{PEAK_GOPS:.1f} GOP/s")
    print(f"  HP port BW (gmem0/1/3) = {HP_BW_MBS:.0f} MB/s each")

    hr("Weight reads are BANDWIDTH-bound (board-grounded, TC1 HP1 XRT trace)")
    print(f"  transfer = {XRT_TRANSFER_BYTES} B in {XRT_TRANSFER_NS:.0f} ns "
          f"({XRT_TRANSFER_BYTES/XRT_TRANSFER_NS*1000:.0f} MB/s ~ port max) -> "
          f"{BYTES_PER_CYCLE:.0f} B/cy = 1 weight-vector/cy")
    print(f"{'TC':6}{'wt_vectors':>11}{'bw_total_cy':>12}{'/2 epochs':>10}"
          f"{'epoch_meas':>11}{'ratio':>7}{'wt%call':>9}")
    for t in TCS:
        per_ep = t.weight_read_cy_bw / 2.0
        ratio = t.weight_epoch_meas / per_ep if per_ep else 0.0
        print(f"{t.name:6}{t.weight_vectors:>11}{t.weight_read_cy_bw:>12.0f}"
              f"{per_ep:>10.0f}{t.weight_epoch_meas:>11}{ratio:>7.2f}"
              f"{t.weight_frac_of_call*100:>8.0f}%")
    print("  => epoch_meas ~= weight_bytes/16/2 for ALL weight-bound cases (ratio ~1.0):")
    print("     the '65676 floor' / 'non-monotonic Wbank' was an ARTIFACT of the")
    print("     per-COG depth_count decomposition. Real model: BW-bound at 16 B/cy.")
    print("     Every weight-bound layer is 2 epochs; TC3's old cosim 3,400 was a mis-read")
    print("     (board shows 2x164us = 32,800 cy/epoch).")
    print("  Board first-epoch span (XRT waveform) vs model weight_bytes/16/2:")
    for t in TCS:
        if t.board_epoch_us:
            board_cy = t.board_epoch_us * CLOCK_MHZ           # us * MHz = cycles
            model_cy = t.weight_read_cy_bw / 2.0
            print(f"     {t.name}: {t.board_epoch_us:.0f} us = {board_cy:.0f} cy "
                  f"vs model {model_cy:.0f} cy ({board_cy/model_cy:.3f}x), "
                  f"cosim {t.weight_epoch_meas}")

    hr("IFM reads — BW-bound (16 B/cy); K>1 over-fetch (board where available)")
    print(f"{'TC':6}{'ifm_B/call':>11}{'overfetch':>10}{'ifm_read_cy':>12}"
          f"{'%call':>7}{'compute_cy':>11}{'bound':>11}{'src':>7}")
    for t in TCS:
        compute = t.ideal_mac_iters
        bound = "IFM" if t.ifm_read_cy > compute else "MAC/wt"
        src = "board" if t.board_ifm_bytes else "est"
        print(f"{t.name:6}{t.ifm_bytes:>11,.0f}{t.ifm_overfetch:>9.1f}x"
              f"{t.ifm_read_cy:>12,.0f}{t.ifm_read_cy/t.cosim_cy*100:>6.0f}%"
              f"{compute:>11,}{bound:>11}{src:>7}")
    print("  TC1 vs TC6: same K=3, same ~8x over-fetch, same compute (112,896 cy);")
    print("    TC6's 56x56 input = 4x ifm bytes/COG -> ifm_read 204k > compute -> IFM-bound,")
    print("    while TC1's 28x28 -> ifm_read 50k < compute -> over-fetch stays hidden (MAC-bound).")

    hr("Array (MAC-feed) efficiency and achieved throughput")
    print(f"{'TC':6}{'ideal_it':>10}{'cosim_cy':>10}{'arr_eff':>9}"
          f"{'GMAC/s':>9}{'GOP/s':>9}{'%peak':>7}")
    for t in TCS:
        print(f"{t.name:6}{t.ideal_mac_iters:>10}{t.cosim_cy:>10}"
              f"{t.array_eff*100:>8.1f}%{t.achieved_gmacs:>9.1f}"
              f"{t.achieved_gops:>9.1f}{t.achieved_gops/PEAK_GOPS*100:>6.0f}%")

    hr("DDR traffic (INT8 bytes) and operational intensity")
    print(f"{'TC':6}{'wt_B':>10}{'ifm_B':>11}{'ofm_B':>10}{'tot_B':>11}"
          f"{'I(MAC/B)':>10}")
    for t in TCS:
        print(f"{t.name:6}{t.weight_bytes:>10}{t.ifm_bytes:>11}{t.ofm_bytes:>10}"
              f"{t.ddr_bytes:>11}{t.op_intensity:>10.2f}")
    print("  K>1 uncached reloads IFM num_cog x (TC1, TC6, board). TC6's large 56x56")
    print("  input makes its over-fetch dominate -> I=32.9 (memory-bound); TC1's smaller")
    print("  tile keeps it compute-bound despite the same 8x reload (I=110).")

    hr("Roofline coordinates (operational intensity, achieved GOP/s)")
    print(f"{'TC':6}{'I(MAC/B)':>10}{'GOP/s':>9}  knee@I*BW(1 port, GOP/s)")
    for t in TCS:
        mem_roof = 2.0 * t.op_intensity * HP_BW_MBS / 1000.0  # GOP/s at 1-port BW
        print(f"{t.name:6}{t.op_intensity:>10.2f}{t.achieved_gops:>9.1f}"
              f"   {min(mem_roof, PEAK_GOPS):>8.1f}")

    hr("Cosim vs board residual (sanity: cost model uses cosim as board proxy)")
    print(f"{'TC':6}{'cosim_cy':>10}{'board_cy':>10}{'err%':>8}")
    for t in TCS:
        err = (t.cosim_cy - t.board_cy) / t.board_cy * 100.0
        print(f"{t.name:6}{t.cosim_cy:>10}{t.board_cy:>10}{err:>7.2f}%")


if __name__ == "__main__":
    main()
