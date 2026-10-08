/**
 * @file cost_model_v2_1.cpp
 * @brief Max-plus cost evaluator for the instantiated kernel dataflow graph.
 */

#include "deeptile/compiler/environment.h"
#include "deeptile/tile/cost_model_v2_1.hpp"

#include "deeptile/tile/hardware_spec.hpp"  // Architecture fingerprint inputs.

// Embed the strict synthesis manifest so runtime lookup is unnecessary.
#include "kernel_manifest_embedded.inc"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace deeptile {
namespace tile {

namespace {
inline int  ceil_div(int a, int b) { return b > 0 ? (a + b - 1) / b : 0; }
inline double dmax(double a, double b) { return a > b ? a : b; }
inline long  lround_(double v) { return static_cast<long>(std::llround(v)); }

// ABI: A compact depthwise block holds K*K taps and four bias words, padded
// to 16 words for 3x3 and 32 for 5x5 (arch_config.h).
constexpr int kDwTaps3 = DW_TAPS_3;
} // namespace

// Environment / manifest seeds
AxiPortProfile axi_profile_cosim() {
    AxiPortProfile p;
    p.name = "cosim";
    p.physical_port_bits = 256;   // xsim full-width AXI model → 1 beat / 256b word
    p.first_beat_latency = 225.0; // Calibrated 16-bank epoch residual.
    p.data_beat_ii       = 1.0;
    return p;
}
AxiPortProfile axi_profile_board() {
    AxiPortProfile p;
    p.name = "board";
    p.physical_port_bits = 128;   // HP port is physically 128b → 2 beats / 256b word
    p.first_beat_latency = 9.0;   // outstanding transactions hide DRAM latency
    p.data_beat_ii       = 1.0;
    return p;
}

// Seed values are used only when an extracted synthesis manifest is absent.
KernelManifest default_manifest_csynth() {
    KernelManifest m;
    m.provenance = "csynth-seed";
    auto L = [](double ii, int first, int tail) { return LoopProfile{ii, first, tail}; };

    m.schedule_xfer     = L(1.0, 6, 1);
    m.schedule_tile     = L(1.0, 6, 1);
    m.bias_preload      = L(1.0, 8, 1);

    m.weight_depth      = L(1.0, 10, 2);
    m.weight_depth_flat = false;            // Each bank restarts the depth loop.
    m.mac_dense         = L(1.0, 14, 1);
    m.mac_task_entry    = 8;
    m.cig_restart       = 4;                // Per-CIG continuous-pipeline bubble.

    m.ifm_load_raw      = L(1.0, 32, 1);
    m.ifm_stream        = L(1.0, 20, 1);
    m.k1_fill           = L(1.0, 52, 1);
    m.k1_reuse          = L(1.0, 4, 0);

    m.acc_group         = L(1.0, 10, 2);
    m.emit              = L(1.0, 54, 1);    // Emit depth plus bias/quantize entry.

    m.residual          = L(1.0, 12, 1);
    m.pool_row          = L(1.0, 14, 1);
    m.gap_acc           = L(1.0, 10, 1);
    m.dw_preload        = L(1.0, 10, 1);
    m.dw_local_weight   = L(1.0, 12, 1);
    m.dw_tile_pix       = L(1.0, 16, 1);    // 3x3 depthwise pipeline fill.
    m.store_row         = L(1.0, 22, 1);

    m.pixel_fifo_depth           = 32;
    m.acc_fifo_depth             = 32;
    m.intermediate_ofm_fifo_depth = 32;
    m.final_store_fifo_depth     = 512;
    return m;
}

// Compile-time fingerprint of the topology-affecting constants; the extractor
// (scripts/tools/extract_kernel_manifest.py) builds the identical string from
// common_defs.h. Any divergence means the manifest was taken from a different
// architecture and must be rejected.
std::string architecture_hash_compiled() {
    return cost_model_architecture_key(HardwareSpec{});
}

// File and embedded manifests use the same parser and strict validation.
static KernelManifest parse_manifest(std::istream& in, bool strict) {
    // key -> LoopProfile slot in the manifest (every synthesized stage).
    KernelManifest m;
    m.provenance = "extracted";
    std::map<std::string, LoopProfile*> slot = {
        {"schedule_xfer", &m.schedule_xfer}, {"schedule_tile", &m.schedule_tile},
        {"bias_preload", &m.bias_preload},   {"weight_depth", &m.weight_depth},
        {"mac_dense", &m.mac_dense},         {"ifm_load_raw", &m.ifm_load_raw},
        {"ifm_stream", &m.ifm_stream},       {"k1_fill", &m.k1_fill},
        {"k1_reuse", &m.k1_reuse},           {"acc_group", &m.acc_group},
        {"emit", &m.emit},                   {"residual", &m.residual},
        {"pool_row", &m.pool_row},           {"gap_acc", &m.gap_acc},
        {"dw_preload", &m.dw_preload},       {"dw_local_weight", &m.dw_local_weight},
        {"dw_tile_pix", &m.dw_tile_pix},     {"store_row", &m.store_row},
    };
    std::map<std::string, int*> fifo = {
        {"fifo_pixel", &m.pixel_fifo_depth}, {"fifo_acc", &m.acc_fifo_depth},
        {"fifo_intermediate_ofm", &m.intermediate_ofm_fifo_depth},
        {"fifo_final_store", &m.final_store_fifo_depth},
    };
    std::map<std::string, bool> seen;  // which loop keys got at least one field
    std::string manifest_arch, line;
    bool have_entry = false, have_cig = false;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string key, val;
        ls >> key >> val;
        if (key.empty()) continue;
        if (key == "manifest.source_commit") m.source_commit = val;
        else if (key == "manifest.hls_version") m.hls_version = val;
        else if (key == "manifest.target_clock_ns") m.target_clock_ns = std::atof(val.c_str());
        else if (key == "manifest.arch_hash") { manifest_arch = val; m.architecture_hash = val; }
        else if (key == "manifest.provenance" || key == "manifest.part" || key == "manifest.top") {}
        else if (key == "mac_task_entry") { m.mac_task_entry = std::atoi(val.c_str()); have_entry = true; }
        else if (key == "cig_restart") { m.cig_restart = std::atoi(val.c_str()); have_cig = true; }
        else if (fifo.count(key)) *fifo[key] = std::atoi(val.c_str());
        else {
            const auto dot = key.rfind('.');
            if (dot == std::string::npos) continue;
            const std::string base = key.substr(0, dot), field = key.substr(dot + 1);
            if (field == "flat") {
                const bool flat = std::atoi(val.c_str()) != 0;
                if (base == "weight_depth") m.weight_depth_flat = flat;
                // A flattened II=1 nest still has an explicit measured CIG restart.
                continue;
            }
            auto it = slot.find(base);
            if (it == slot.end()) continue;  // unknown/new loop (e.g. k1_prefetch, schedule_group) — ignore
            LoopProfile& lp = *it->second;
            if      (field == "ii")    lp.ii = std::atof(val.c_str());
            else if (field == "first") lp.first = std::atoi(val.c_str());
            else if (field == "tail")  lp.tail = std::atoi(val.c_str());
            seen[base] = true;
        }
    }

    if (strict) {
        if (!have_entry)
            throw std::runtime_error("cost_model_v2_1: manifest missing 'mac_task_entry'");
        if (!have_cig)
            throw std::runtime_error("cost_model_v2_1: manifest missing 'cig_restart'");
        const std::string want = architecture_hash_compiled();
        if (manifest_arch != want)
            throw std::runtime_error("cost_model_v2_1: manifest arch_hash '" + manifest_arch +
                                     "' != compiled '" + want + "' (stale or wrong architecture)");
        static const char* required[] = {
            "schedule_tile", "weight_depth", "mac_dense", "ifm_load_raw", "ifm_stream",
            "k1_fill", "k1_reuse", "acc_group", "emit", "store_row",
        };
        for (const char* r : required)
            if (!seen.count(r))
                throw std::runtime_error(std::string("cost_model_v2_1: manifest missing required loop '") + r + "'");
    }
    return m;
}

KernelManifest load_kernel_manifest(const std::string& path, bool strict) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cost_model_v2_1: cannot open manifest '" + path + "'");
    return parse_manifest(in, strict);
}

const KernelManifest& default_manifest() {
    // Parse once and reject architecture mismatches instead of silently mispredicting.
    static const KernelManifest m = [] () -> KernelManifest {
        if (const char* env = deeptile::env_value("MANIFEST"))
            return load_kernel_manifest(env);
        std::istringstream in(kEmbeddedKernelManifest);
        return parse_manifest(in, /*strict=*/true);
    }();
    return m;
}

namespace {

// Model a burst chain with overlapped outstanding requests and a steady beat stream.
double ddr_chain(const AxiPortProfile& a, long words, int word_bits, long num_bursts) {
    const int    bpw   = (word_bits + a.physical_port_bits - 1) / a.physical_port_bits;
    const double beats = static_cast<double>(words) * bpw;
    // First-beat latency is exposed per outstanding-request group, then beats stream.
    const double stream = std::max(a.data_beat_ii * beats,
                                   a.address_issue_ii * static_cast<double>(num_bursts));
    return a.first_beat_latency + stream;
}

struct TaskModel {
    int  cigs = 1;
    int  num_windows = 0;
    int  num_groups = 0;
    long store_vecs = 0;
    bool writeback = false;
    double s_ifm_ddr = 0.0;  // DDR-port busy for this task's IFM traffic
    double s_ld = 0.0;       // K>1 raw load service (DDR + loop)
    double s_st = 0.0;       // K>1 stream service
    double s_prod = 0.0;     // K=1 fill/replay service
    double s_mac = 0.0;      // MAC service (all CIGs)
    double mac_ideal = 0.0;  // II·trips (MAC floor)
    double s_acc = 0.0;      // accumulate-pass service (all CIGs)
};

// Geometry for one partial-sum accumulation and writeback unit.
struct WbUnit {
    size_t last_task;        // index of the writeback task
    int    num_windows = 0;  // EMIT tokens (conv-tile pixels)
    long   conv_h = 0, conv_w = 0;
    long   pooled_h = 0, pooled_w = 0;   // final store tile
    long   store_vecs = 0;
    double mac_finish = 0.0; // when the MAC for this unit completes
    double s_acc = 0.0;      // accumulate service for this unit
};

} // namespace

// Evaluator
CostV2_1 estimate_cost_v2_1(const SchedulePlan& plan, const PostOpConfig& post,
                            const KernelManifest& mf, const AxiPortProfile& axi,
                            double clock_mhz) {
    CostV2_1 c;
    if (!plan.valid || plan.compute_tasks.empty() || plan.weight_epochs.empty())
        return c;

    const ConvLayerParams& L = plan.layer;
    const int  p_spatial = plan.hw.p_spatial;
    const int  banks     = plan.hw.oc_tile_size;
    const int  K2        = L.Kh * L.Kw;
    const bool k1_path   = (L.Kh == 1 && L.Kw == 1) && plan.stats.uses_k1_cache;
    const bool flatK1    = (L.Kh == 1 && L.Kw == 1);
    const int  num_cig   = L.num_cig();
    const size_t T       = plan.compute_tasks.size();
    const int    E       = static_cast<int>(plan.weight_epochs.size());
    // Standalone depthwise layers use diagonal lowering on the dense datapath.
    const LoopProfile& mac_lp = mf.mac_dense;

    const bool has_resid = post.residual_add_enable;
    const bool has_pool  = post.maxpool_enable;
    const bool has_gap   = post.gap_enable;
    const bool has_dw    = post.dw_enable;
    const int  dw_stride = post.dw_stride > 0 ? post.dw_stride : 1;
    const int  dw_k      = post.dw_kernel == 5 ? 5 : 3;

    // Build per-task geometry and service curves.
    std::vector<TaskModel> tm(T);
    for (size_t i = 0; i < T; ++i) {
        const ComputeTask& t = plan.compute_tasks[i];
        const SpatialTile& st = plan.spatial_tiles[static_cast<size_t>(t.spatial_idx)];
        const InputWindow& w  = st.ifm_window;
        TaskModel& m = tm[i];

        m.cigs        = t.cig_count > 0 ? t.cig_count : 1;
        m.num_windows = st.conv_tile_oh * st.conv_tile_ow;
        m.num_groups  = flatK1
            ? ceil_div(m.num_windows, p_spatial)                       // flat (K=1)
            : st.conv_tile_oh * ceil_div(st.conv_tile_ow, p_spatial);  // row-aligned
        m.writeback   = L.depthwise || (t.cig_begin + t.cig_count >= num_cig);
        m.store_vecs  = static_cast<long>(st.pooled_tile_oh) * st.pooled_tile_ow;

        // The continuous MAC pipeline pays one fill plus a restart per extra CIG.
        const long trips_all = static_cast<long>(m.num_groups) * K2 * m.cigs;
        m.mac_ideal = mac_lp.ii * static_cast<double>(trips_all);
        m.s_mac     = mf.mac_task_entry + static_cast<double>(mac_lp.done(trips_all))
                    + static_cast<double>(mf.cig_restart) * (m.cigs - 1);

        // accumulate pass: II=1/group, per-CIG restart (ACC_CIG_LOOP flatten off).
        m.s_acc = static_cast<double>(mf.acc_group.done(m.num_groups)) * m.cigs;

        if (k1_path) {
            if (!t.reuse_ifm) {
                // Fill reads all CIG planes: one burst per (CIG,row), pipelined.
                const long half_w = ceil_div(w.read_cols, 2);
                const long words  = static_cast<long>(st.conv_tile_oh) * half_w * m.cigs;
                const long bursts = static_cast<long>(st.conv_tile_oh) * m.cigs;
                m.s_ifm_ddr = ddr_chain(axi, words, 256, bursts);
                m.s_prod    = mf.k1_fill.first + mf.k1_fill.tail + m.s_ifm_ddr;
            } else {
                m.s_prod = static_cast<double>(mf.k1_reuse.done(
                    static_cast<long>(m.num_groups) * m.cigs));
            }
        } else {
            const long rows   = w.read_rows;
            const long half_w = ceil_div(w.read_cols, 2);
            const long words  = rows * half_w * m.cigs;
            const long bursts = rows * m.cigs;
            m.s_ifm_ddr = ddr_chain(axi, words, 256, bursts);
            // Load time is the maximum of loop service and DDR service, not their sum.
            m.s_ld = std::max(static_cast<double>(mf.ifm_load_raw.done(words)), m.s_ifm_ddr);
            // Window generator is a continuous pipeline across CIGs (fill once).
            m.s_st = static_cast<double>(
                mf.ifm_stream.done(static_cast<long>(m.num_groups) * K2 * m.cigs))
                + static_cast<double>(mf.cig_restart) * (m.cigs - 1);
        }
    }

    // Model weight-epoch service and DDR occupancy.
    std::vector<double> s_wl(static_cast<size_t>(E), 0.0);
    for (int e = 0; e < E; ++e) {
        const WeightEpoch& ep = plan.weight_epochs[static_cast<size_t>(e)];
        // One burst per bank pipelines through the outstanding-transaction window.
        const double ddr = ddr_chain(axi, static_cast<long>(banks) * ep.weight_vectors,
                                     128, banks);
        // Compute side: per-bank depth-loop restart unless the nest is flattened.
        const double loop = mf.weight_depth_flat
            ? mf.weight_depth.done(static_cast<long>(banks) * ep.weight_vectors)
            : static_cast<double>(banks) * mf.weight_depth.done(ep.weight_vectors);
        s_wl[static_cast<size_t>(e)] = std::max(loop, ddr);
        c.weight_busy_cy += lround_(ddr);
    }

    // Evaluate weight, IFM, and MAC ping-pong recurrence.
    const double t0 = static_cast<double>(mf.schedule_xfer.done(std::max(1, E))
                                          + mf.bias_preload.done(num_cig));
    c.critical_path_cy[static_cast<size_t>(CriticalCause::Launch)] = lround_(t0);

    std::vector<double> wl_finish(static_cast<size_t>(E), 0.0);
    wl_finish[0] = t0 + s_wl[0];
    c.critical_path_cy[static_cast<size_t>(CriticalCause::WeightPrime)] = lround_(s_wl[0]);
    int cur_epoch = 0;

    double pp_busy = k1_path ? 0.0 : t0 + tm[0].s_ld;
    double prod_busy = t0;
    double mac_free = 0.0;

    std::vector<double> mac_finish_of(T, 0.0);

    for (size_t i = 0; i < T; ++i) {
        const ComputeTask& t = plan.compute_tasks[i];
        const TaskModel&   m = tm[i];

        if (t.epoch_idx > cur_epoch && t.epoch_idx < E) {
            const int e = t.epoch_idx;
            const double late = wl_finish[static_cast<size_t>(e)] - mac_free;
            if (late > 0.0)
                c.critical_path_cy[static_cast<size_t>(CriticalCause::InterEpoch)] += lround_(late);
            const double iter_start = dmax(wl_finish[static_cast<size_t>(e)], mac_free);
            if (e + 1 < E)
                wl_finish[static_cast<size_t>(e + 1)] = iter_start + s_wl[static_cast<size_t>(e + 1)];
            cur_epoch = e;
        } else if (t.epoch_idx == 0 && i == 0 && E > 1) {
            wl_finish[1] = wl_finish[0] + s_wl[1];
        }
        const double epoch_gate = wl_finish[static_cast<size_t>(std::min(t.epoch_idx, E - 1))];

        // Producer (IFM).
        double data_start, producer_finish;
        if (!k1_path) {
            const double st_start = pp_busy;
            data_start      = st_start + mf.ifm_stream.first;
            producer_finish = st_start + m.s_st;
        } else {
            const double p_start = dmax(prod_busy, t0);
            data_start      = p_start + (t.reuse_ifm ? mf.k1_reuse.first : mf.k1_fill.first);
            producer_finish = p_start + m.s_prod;
            prod_busy       = p_start;
        }

        const double mac_start = dmax(dmax(mac_free, epoch_gate), data_start);
        if (data_start > dmax(mac_free, epoch_gate))
            c.critical_path_cy[static_cast<size_t>(CriticalCause::IfmReady)] +=
                lround_(data_start - dmax(mac_free, epoch_gate));

        const double cand_serv = mac_start + m.s_mac;
        const double cand_data = producer_finish + mac_lp.first;  // rate-locked producer
        double mac_finish = dmax(cand_serv, cand_data);
        if (cand_data > cand_serv)
            c.critical_path_cy[static_cast<size_t>(CriticalCause::IfmReady)] +=
                lround_(cand_data - cand_serv);

        c.mac_floor_cy += lround_(m.mac_ideal);
        c.mac_busy_cy  += lround_(m.s_mac);

        if (!k1_path) {
            const double st_finish = dmax(producer_finish, mac_finish - mac_lp.first);
            double ld_next = 0.0;
            if (i + 1 < T) ld_next = pp_busy + tm[i + 1].s_ld;
            pp_busy = dmax(st_finish, ld_next);
        } else {
            prod_busy = dmax(producer_finish, mac_finish - mac_lp.first);
        }

        mac_finish_of[i] = mac_finish;
        mac_free = mac_finish;
        c.ifm_busy_cy += lround_(m.s_ifm_ddr);
    }

    // Build partial-sum writeback units.
    std::vector<WbUnit> units;
    for (size_t i = 0; i < T; ++i) {
        if (!tm[i].writeback) continue;
        const ComputeTask& t = plan.compute_tasks[i];
        const SpatialTile& st = plan.spatial_tiles[static_cast<size_t>(t.spatial_idx)];
        WbUnit u;
        u.last_task   = i;
        u.num_windows = tm[i].num_windows;
        u.conv_h = st.conv_tile_oh; u.conv_w = st.conv_tile_ow;
        u.pooled_h = st.pooled_tile_oh; u.pooled_w = st.pooled_tile_ow;
        u.store_vecs = tm[i].store_vecs;
        u.mac_finish = mac_finish_of[i];
        u.s_acc = tm[i].s_acc;
        units.push_back(u);
    }

    // Fused-DW compact-weight preload happens once, before the DATAFLOW loop.
    double dw_ready = t0;
    if (has_dw) {
        const int num_cogs = ceil_div(L.Cout, banks);
        const int words_per_cog = dw_k == 5 ? DW_WORDS_PER_COG_5 : DW_WORDS_PER_COG_3;
        dw_ready = t0 + mf.dw_preload.done(static_cast<long>(num_cogs) * words_per_cog);
    }

    // Two PSUM buffers alternate; unit u+2 waits for unit u to emit.
    double buf_free[2] = {0.0, 0.0};
    double emit_free   = 0.0;    // single emit engine (serialises emits)
    double store_free  = 0.0;    // store engine
    double sink        = 0.0;
    long   store_busy  = 0;
    const int fifo = mf.final_store_fifo_depth;

    for (size_t k = 0; k < units.size(); ++k) {
        const WbUnit& u = units[k];
        const int b = static_cast<int>(k % 2);

        // Accumulate: MAC-fed, into buffer b (free after its previous emit).
        const double acc_ready = dmax(u.mac_finish + mf.acc_group.first, buf_free[b]);
        const double acc_finish = dmax(acc_ready, u.mac_finish) + u.s_acc;

        // Emit: single engine, after accumulate and after previous emit.
        const double emit_start = dmax(acc_finish, emit_free);
        double emit_finish = emit_start + static_cast<double>(mf.emit.done(u.num_windows));
        emit_free = emit_finish;
        c.emit_busy_cy += lround_(mf.emit.done(u.num_windows));

        // Output pipeline: emit → residual → maxpool → gap → fused-DW → store.
        // Adjacent II=1 stages overlap; each contributes its prime once. The DW
        // stage changes token count (padded raster in, DW-tile out).
        double post_first = 0.0;   // summed pipeline primes (fill of each stage)
        long   post_tokens = u.store_vecs;   // tokens entering the store FIFO
        if (has_resid) post_first += mf.residual.first;
        if (has_pool)  post_first += mf.pool_row.first;
        if (has_gap)   post_first += mf.gap_acc.first;
        double dw_service = 0.0;
        if (has_dw) {
            const long padded_h = u.conv_h + dw_k - 1;   // same-pad window height
            const long padded_w = u.conv_w + dw_k - 1;
            // A 5x5 block loads 16 more tap words per tile than a 3x3 one.
            dw_service = mf.dw_local_weight.first + (dw_k * dw_k - kDwTaps3)
                       + static_cast<double>(mf.dw_tile_pix.done(padded_h * padded_w));
            post_first += mf.dw_tile_pix.first;
            const long out_h = (padded_h - dw_k) / dw_stride + 1;
            const long out_w = (padded_w - dw_k) / dw_stride + 1;
            post_tokens = out_h * out_w;
            c.dw_busy_cy += lround_(dw_service);
        }

        // dw_stage output feeds the 512-deep FIFO, then store. Store overlaps the
        // pipeline unless it stalls longer than the FIFO can absorb.
        const double pipe_ready = dmax(emit_start, dw_ready) + post_first + dw_service;
        const double store_service = static_cast<double>(mf.store_row.done(post_tokens));
        const double store_start = dmax(store_free, pipe_ready - std::min<double>(fifo, post_tokens));
        const double store_finish = dmax(store_start + store_service,
                                         pipe_ready + mf.store_row.first);
        store_free = store_finish;
        store_busy += lround_(store_service);

        buf_free[b] = emit_finish;   // buffer released when its emit completes
        sink = dmax(sink, dmax(emit_finish, store_finish));
    }
    c.store_busy_cy = store_busy;

    // Drain: tail after the last MAC.
    const double last_mac = mac_finish_of[T - 1];
    c.critical_path_cy[static_cast<size_t>(CriticalCause::Drain)] = lround_(sink - last_mac);
    c.critical_path_cy[static_cast<size_t>(CriticalCause::MacService)] =
        std::max<long>(0, lround_(last_mac) - c.critical_path_cy[static_cast<size_t>(CriticalCause::Launch)]
                          - c.critical_path_cy[static_cast<size_t>(CriticalCause::WeightPrime)]);

    c.predicted_cy = lround_(sink);
    c.predicted_ms = static_cast<double>(c.predicted_cy) / (clock_mhz * 1.0e3);

    // Classify the resource-demand roofline bound.
    struct { ResourceBound b; long v; } demand[] = {
        {ResourceBound::MAC,    c.mac_floor_cy},
        {ResourceBound::Weight, c.weight_busy_cy},
        {ResourceBound::IFM,    c.ifm_busy_cy},
        {ResourceBound::PostOp, c.emit_busy_cy + c.dw_busy_cy},
        {ResourceBound::Store,  c.store_busy_cy},
    };
    c.resource_bound = demand[0].b;
    long best = demand[0].v;
    for (const auto& d : demand) if (d.v > best) { best = d.v; c.resource_bound = d.b; }

    // Preserve the diagnostic model's weight/IFM/MAC verdict vocabulary.
    if (c.weight_busy_cy >= c.ifm_busy_cy && c.weight_busy_cy >= c.mac_floor_cy)
        c.bound_by = ScheduleCost::Bound::Weight;
    else if (c.ifm_busy_cy >= c.weight_busy_cy && c.ifm_busy_cy >= c.mac_floor_cy)
        c.bound_by = ScheduleCost::Bound::IFM;
    else
        c.bound_by = ScheduleCost::Bound::MAC;

    // The largest critical-path contribution defines the critical cause.
    {
        size_t arg = 0; long mx = -1;
        for (size_t j = 0; j < c.critical_path_cy.size(); ++j)
            if (c.critical_path_cy[j] > mx) { mx = c.critical_path_cy[j]; arg = j; }
        c.critical_bound = static_cast<CriticalCause>(arg);
    }

    // Derive useful throughput.
    const long useful = L.depthwise
        ? static_cast<long>(L.Cout) * K2 * L.output_height() * L.output_width()
        : static_cast<long>(L.Cout) * L.Cin * K2 * L.output_height() * L.output_width();
    c.useful_macs = useful;
    c.mac_array_active_fraction = c.predicted_cy > 0
        ? static_cast<double>(c.mac_floor_cy) / static_cast<double>(c.predicted_cy) : 0.0;
    c.predicted_gops = c.predicted_cy > 0
        ? 2.0 * static_cast<double>(useful) / static_cast<double>(c.predicted_cy) * clock_mhz / 1.0e3
        : 0.0;
    return c;
}

// Reporting
const char* resource_bound_name(ResourceBound b) {
    switch (b) {
        case ResourceBound::MAC:    return "MAC";
        case ResourceBound::Weight: return "Weight";
        case ResourceBound::IFM:    return "IFM";
        case ResourceBound::PostOp: return "PostOp";
        case ResourceBound::Store:  return "Store";
    }
    return "?";
}
const char* critical_cause_name(CriticalCause c) {
    switch (c) {
        case CriticalCause::Controller:  return "controller";
        case CriticalCause::BiasPreload: return "bias_preload";
        case CriticalCause::WeightPrime: return "weight_prime";
        case CriticalCause::InterEpoch:  return "inter_epoch";
        case CriticalCause::IfmReady:    return "ifm_ready";
        case CriticalCause::MacService:  return "mac_service";
        case CriticalCause::AccPsum:     return "acc_psum";
        case CriticalCause::Emit:        return "emit";
        case CriticalCause::PostOp:      return "post_op";
        case CriticalCause::FusedDw:     return "fused_dw";
        case CriticalCause::Store:       return "store";
        case CriticalCause::Drain:       return "drain";
        case CriticalCause::Launch:      return "launch";
        default: return "?";
    }
}

void print_cost_v2_1(const SchedulePlan& plan, const CostV2_1& c,
                     const AxiPortProfile& axi) {
    std::cout << "\nCost model V2.1 (topology max-plus, axi=" << axi.name << ")\n";
    for (int i = 0; i < 66; ++i) std::cout << '-';
    std::cout << '\n';
    if (!plan.valid) { std::cout << "  [plan invalid]\n"; return; }
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  predicted:   " << c.predicted_cy << " cy (" << c.predicted_ms
              << " ms)   " << c.predicted_gops << " GOPS   eff "
              << c.mac_array_active_fraction * 100.0 << "%\n";
    std::cout << "  resource:    " << resource_bound_name(c.resource_bound)
              << "   critical: " << critical_cause_name(c.critical_bound) << "\n";
    std::cout << "  busy:        mac=" << c.mac_busy_cy << " wt=" << c.weight_busy_cy
              << " ifm=" << c.ifm_busy_cy << " emit=" << c.emit_busy_cy
              << " dw=" << c.dw_busy_cy << " store=" << c.store_busy_cy << "\n";
    std::cout << "  crit-path:   launch="
              << c.critical_path_cy[static_cast<size_t>(CriticalCause::Launch)]
              << " wprime=" << c.critical_path_cy[static_cast<size_t>(CriticalCause::WeightPrime)]
              << " inter_ep=" << c.critical_path_cy[static_cast<size_t>(CriticalCause::InterEpoch)]
              << " ifm=" << c.critical_path_cy[static_cast<size_t>(CriticalCause::IfmReady)]
              << " mac=" << c.critical_path_cy[static_cast<size_t>(CriticalCause::MacService)]
              << " drain=" << c.critical_path_cy[static_cast<size_t>(CriticalCause::Drain)] << "\n";
    std::cout << std::setprecision(6);
}

} // namespace tile
} // namespace deeptile
