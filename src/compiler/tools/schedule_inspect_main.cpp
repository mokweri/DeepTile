/**
 * @file schedule_inspect_main.cpp
 * @brief CLI for inspecting schedule plans, descriptors, and cost estimates.
 */

#include "deeptile/tile/descriptor_emitter.hpp"
#include "deeptile/tile/schedule_builder.hpp"
#include "deeptile/tile/schedule.hpp"
#include "deeptile/tile/schedule_options.hpp"
#include "deeptile/tile/cost_model.hpp"
#include "deeptile/tile/cost_model_v2_1.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace deeptile::tile;

// Reference layer definitions (ResNet-50 reference set)

struct LayerDef {
    ConvLayerParams params;
    TileConfig      tile_cfg;
    PostOpConfig    post_ops;
};

static LayerDef make_layer(int H, int W, int Cin, int Cout,
                           int K, int S, int pad,
                           int T_oh, int T_ow,
                           bool use_relu = true,
                           PostOpConfig post = {}) {
    LayerDef d;
    auto& p = d.params;
    p.H = H; p.W = W; p.Cin = Cin; p.Cout = Cout;
    p.Kh = K; p.Kw = K; p.Sh = S; p.Sw = S;
    p.pad_top = pad; p.pad_bottom = pad;
    p.pad_left = pad; p.pad_right = pad;
    p.Cpi = 16; p.Cpo = 16;
    p.elem_bytes = 1;
    p.frac_in = 5; p.frac_w = 8; p.frac_b = 5; p.frac_out = 3;
    p.use_relu = use_relu;

    auto& cfg = d.tile_cfg;
    cfg.T_oh = T_oh; cfg.T_ow = T_ow;
    cfg.T_cog = 1; cfg.T_cig = 1;
    cfg.Cpi = 16; cfg.Cpo = 16;

    d.post_ops = post;
    return d;
}

static std::unordered_map<std::string, LayerDef> build_layer_table() {
    std::unordered_map<std::string, LayerDef> t;

    // best_case_3x3_s1: ResNet layer1_0_conv2 — 56×56, 64→64, K=3 s1
    t["best_case_3x3_s1"] = make_layer(56, 56, 64, 64,  3, 1, 1, 32, 64);

    // stride2_3x3_s2: ResNet layer2_0_conv1 — 56×56, 64→128, K=3 s2
    t["stride2_3x3_s2"]   = make_layer(56, 56, 64, 128, 3, 2, 1, 16, 32);

    // proj_bottleneck_1x1_s2: ResNet layer2_0_downsample — 56×56, 64→128, K=1 s2
    t["proj_bottleneck_1x1_s2"] = make_layer(56, 56, 64, 128, 1, 2, 0, 28, 28);

    // deep_k1_s1: ResNet layer3 K=1 — 14×14, 256→64, K=1 (bottleneck squeeze)
    t["deep_k1_s1"]       = make_layer(14, 14, 256, 64,  1, 1, 0, 14, 14);

    // late_k1_s1: ResNet layer4 K=1 expand — 7×7, 512→2048
    t["late_k1_s1"]       = make_layer(7,  7,  512, 2048, 1, 1, 0, 7, 7);

    // tail_gap_fc: ResNet final K=1 with GAP — 7×7, 2048→1000
    {
        PostOpConfig gap;
        gap.gap_enable = true;
        t["tail_gap_fc"]  = make_layer(7,  7,  2048, 1000, 1, 1, 0, 7, 7,
                                       /*relu=*/false, gap);
    }

    // layer4_0_conv2: K=3 stride-2, 14×14→7×7, 512→512 — COG-batching target
    t["layer4_0_conv2"]   = make_layer(14, 14, 512, 512, 3, 2, 1, 7, 7);

    return t;
}

// ResNet-50 full-network layer table (54 layers, network order)

static std::vector<std::pair<std::string, LayerDef>> make_resnet50_defs() {
    std::vector<std::pair<std::string, LayerDef>> defs;
    defs.reserve(54);

    auto conv_out = [](int d, int K, int S, int pad) {
        return (d + 2 * pad - K) / S + 1;
    };
    // Fill tile width first, then height within the shared spatial-area budget.
    const int WM_W = 64, WM_H = 32, WM_A = WM_W * WM_H;
    auto wm = [WM_W, WM_H, WM_A](int oh, int ow) {
        const int t_ow = std::min(ow, WM_W);
        const int t_oh = std::min(oh, std::min(WM_H, WM_A / t_ow));
        return std::pair<int, int>(t_oh, t_ow);
    };
    auto add_k1 = [&](const std::string& name, int H, int W, int Cin, int Cout,
                       int S = 1, bool relu = true, PostOpConfig post = {}) {
        int oh = conv_out(H, 1, S, 0);
        int ow = conv_out(W, 1, S, 0);
        auto t = wm(oh, ow);
        defs.push_back({name, make_layer(H, W, Cin, Cout, 1, S, 0, t.first, t.second, relu, post)});
    };
    auto add_k3 = [&](const std::string& name, int H, int W, int Cin, int Cout, int S = 1) {
        int oh = conv_out(H, 3, S, 1);
        int ow = conv_out(W, 3, S, 1);
        auto t = wm(oh, ow);
        defs.push_back({name, make_layer(H, W, Cin, Cout, 3, S, 1, t.first, t.second)});
    };
    auto add_k7 = [&](const std::string& name, int H, int W, int Cin, int Cout, int S) {
        int oh = conv_out(H, 7, S, 3);
        int ow = conv_out(W, 7, S, 3);
        auto t = wm(oh, ow);
        defs.push_back({name, make_layer(H, W, Cin, Cout, 7, S, 3, t.first, t.second)});
    };

    add_k7("stem_conv", 224, 224, 3, 64, 2);

    auto append_stage = [&](const std::string& stage, int num_blocks,
                            int Cin_stage, int C_bn, int Cout_stage,
                            int H, int W, int first_stride) {
        int h = H, w = W, block_Cin = Cin_stage;
        for (int b = 0; b < num_blocks; ++b) {
            const std::string bn = stage + "_b" + std::to_string(b);
            const int stride = (b == 0) ? first_stride : 1;
            const int oh = conv_out(h, 3, stride, 1);
            const int ow = conv_out(w, 3, stride, 1);
            add_k1(bn + "_conv1", h,  w,  block_Cin,  C_bn);
            add_k3(bn + "_conv2", h,  w,  C_bn,       C_bn,       stride);
            add_k1(bn + "_conv3", oh, ow, C_bn,       Cout_stage);
            if (b == 0)
                add_k1(bn + "_proj", h, w, block_Cin, Cout_stage, stride);
            h = oh; w = ow;
            block_Cin = Cout_stage;
        }
    };

    append_stage("conv2_x", 3, 64,   64,   256,  56, 56, 1);
    append_stage("conv3_x", 4, 256, 128,   512,  56, 56, 2);
    append_stage("conv4_x", 6, 512, 256,  1024,  28, 28, 2);
    append_stage("conv5_x", 3, 1024, 512, 2048,  14, 14, 2);

    {
        PostOpConfig gap;
        gap.gap_enable = true;
        add_k1("fc_gap", 7, 7, 2048, 1000, 1, /*relu=*/false, gap);
    }

    return defs;
}

// Printing helpers

static void print_separator(char c = '-', int width = 60) {
    for (int i = 0; i < width; ++i) std::cout << c;
    std::cout << '\n';
}

static std::string decode_flags(uint8_t flags) {
    std::string s;
    if (flags & ACCUM_RESET)      s += "RST|";
    if (flags & ACCUM_WRITE_BACK) s += "WB|";
    if (flags & RELU_ON)          s += "RELU|";
    if (flags & IS_LAST_TILE)     s += "LAST|";
    if (flags & ADD_ENABLE)       s += "ADD|";
    if (flags & POST_ADD_RELU)    s += "ARELU|";
    if (flags & POOL_ENABLE)      s += "POOL|";
    if (flags & GAP_ENABLE)       s += "GAP|";
    if (!s.empty()) s.pop_back();
    return s.empty() ? "0" : s;
}

static void print_descriptor_table(const DescriptorBundle& bundle) {
    std::cout << "\nWeight transfer descriptors (" << bundle.transfers.size() << ")\n";
    print_separator();
    std::cout << "  idx  buf  ddr_off   wt_vecs  depth  compute  last\n";
    for (size_t i = 0; i < bundle.transfers.size(); ++i) {
        const auto& d = bundle.transfers[i];
        std::cout << "  [" << i << "]"
                  << "  buf" << static_cast<int>(d.buffer_id)
                  << "  off=" << d.ddr_offset_vecs
                  << "  len=" << d.ddr_length_vecs
                  << "  depth=" << d.depth_count
                  << "  compute=" << d.compute_count
                  << ((d.flags & XFER_LAST) ? "  LAST" : "") << "\n";
    }

    std::cout << "\nCompute descriptors (" << bundle.compute_descs.size() << ")\n";
    print_separator();
    std::cout << "  idx  xf  uram   ifm_off  ofm_off  bias  ncig  reuse  flags\n";

    int cursor = 0;
    for (size_t i = 0; i < bundle.transfers.size(); ++i) {
        const int count = static_cast<int>(bundle.transfers[i].compute_count);
        for (int d = 0; d < count; ++d) {
            const auto& cd = bundle.compute_descs[static_cast<size_t>(cursor + d)];
            std::cout << "  [" << cursor + d << "]"
                      << "  x" << i
                      << "  u=" << cd.uram_base_depth
                      << "  ifm=" << cd.ifm_offset_vecs
                      << "  ofm=" << cd.ofm_offset_vecs
                      << "  b=" << cd.bias_offset_idx
                      << "  nc=" << static_cast<int>(cd.num_cigs)
                      << (cd.reuse_ifm ? "  reuse" : "       ")
                      << "  " << decode_flags(cd.flags)
                      << "\n";
        }
        cursor += count;
    }
}

static void print_plan(const SchedulePlan& plan, bool detail) {
    const auto& p   = plan.layer;
    const auto& cfg = plan.tile_cfg;

    print_separator('=');
    std::cout << "SchedulePlan — "
              << p.H << "×" << p.W
              << " Cin=" << p.Cin << " Cout=" << p.Cout
              << " K=" << p.Kh << " S=" << p.Sh << "\n";
    print_separator('=');

    if (!plan.valid) {
        std::cout << "[INVALID]\n";
        for (const auto& iss : plan.issues)
            std::cout << "  ! " << iss << '\n';
        return;
    }

    std::cout << "\nInputs\n";
    print_separator();
    std::cout << "  Layer:     " << p.H << "×" << p.W
              << "  Cin=" << p.Cin << "  Cout=" << p.Cout
              << "  K=" << p.Kh << "  S=" << p.Sh
              << "  pad=" << p.pad_top << "\n";
    std::cout << "  Output:    "
              << p.output_height() << "×" << p.output_width()
              << "  num_cig=" << p.num_cig()
              << "  num_cog=" << p.num_cog() << "\n";
    std::cout << "  Tile:      T_oh=" << cfg.T_oh << "  T_ow=" << cfg.T_ow
              << "  Cpi=" << cfg.Cpi << "  Cpo=" << cfg.Cpo << "\n";
    std::cout << "  HW:        p_spatial=" << plan.hw.p_spatial
              << "  weight_buf_depth=" << plan.hw.weight_buf_depth
              << "  " << plan.hw.macs_per_cycle() << " MACs/cycle\n";

    std::cout << "\nSchedule stats\n";
    print_separator();
    const auto& s = plan.stats;
    std::cout << "  Weight epochs:    " << s.num_epochs << "\n";
    std::cout << "  Spatial tiles:    " << s.num_spatial_tiles << "\n";
    std::cout << "  Compute tasks:    " << s.num_compute_tasks << "\n";
    std::cout << "  CIG chaining:     " << (s.uses_cig_chaining ? "YES" : "no") << "\n";
    std::cout << "  K=1 cache:        " << (s.uses_k1_cache ? "YES" : "no") << "\n";
    std::cout << "  COG batching:     " << (s.uses_cog_batching ? "YES" : "no") << "\n";
    std::cout << "  Single-tile path: " << (s.single_spatial_tile_cache_path ? "YES" : "no") << "\n";

    std::cout << "\nWeight epochs\n";
    print_separator();
    std::cout << "  idx  buf   oc_range       depth  wt_vecs  compute_tasks\n";
    for (const auto& ep : plan.weight_epochs) {
        int ct = 0;
        for (const auto& task : plan.compute_tasks)
            if (task.epoch_idx == ep.epoch_index) ++ct;
        std::cout << "  [" << ep.epoch_index << "]"
                  << "  buf" << (ep.buffer_a ? "A" : "B")
                  << "  oc[" << ep.oc_tile_begin
                  << ".." << ep.oc_tile_begin + ep.oc_tile_count - 1 << "]"
                  << "  depth=" << ep.depth_count
                  << "  wtvec=" << ep.weight_vectors
                  << "  tasks=" << ct << "\n";
    }

    std::cout << "\nSpatial tiles (" << plan.spatial_tiles.size() << " total)\n";
    print_separator();
    if (!plan.spatial_tiles.empty()) {
        const int show = detail ? static_cast<int>(plan.spatial_tiles.size()) : 4;
        for (int i = 0; i < std::min(show, static_cast<int>(plan.spatial_tiles.size())); ++i) {
            const auto& st = plan.spatial_tiles[static_cast<size_t>(i)];
            const auto& w  = st.ifm_window;
            std::cout << "  [" << i << "]  oh=" << st.oh << "  ow=" << st.ow
                      << "  conv=" << st.conv_tile_oh << "×" << st.conv_tile_ow
                      << "  pool=" << st.pooled_tile_oh << "×" << st.pooled_tile_ow
                      << "  ifm_ddr=" << w.read_rows << "×" << w.read_cols
                      << "@(" << w.ih_start_actual << "," << w.iw_start_actual << ")"
                      << "  pad=[" << (int)w.pad_top << "," << (int)w.pad_bottom
                      << "," << (int)w.pad_left << "," << (int)w.pad_right << "]\n";
        }
        if (!detail && static_cast<int>(plan.spatial_tiles.size()) > show)
            std::cout << "  ... (" << (plan.spatial_tiles.size() - static_cast<size_t>(show))
                      << " more; use --detail to show all)\n";
    }

    std::cout << "\nCompute tasks (" << plan.compute_tasks.size() << " total)\n";
    print_separator();
    if (!plan.compute_tasks.empty()) {
        const int show = detail ? static_cast<int>(plan.compute_tasks.size()) : 6;
        for (int i = 0; i < std::min(show, static_cast<int>(plan.compute_tasks.size())); ++i) {
            const auto& t = plan.compute_tasks[static_cast<size_t>(i)];
            std::cout << "  [" << i << "]"
                      << "  ep=" << t.epoch_idx
                      << "  sp=" << t.spatial_idx
                      << "  cog=" << t.cog
                      << "  cig[" << t.cig_begin << ".." << t.cig_begin + t.cig_count - 1 << "]";
            if (t.reuse_ifm)   std::cout << "  reuse";
            if (t.is_cache_fill) std::cout << "  FILL";
            std::cout << '\n';
        }
        if (!detail && static_cast<int>(plan.compute_tasks.size()) > show)
            std::cout << "  ... (" << (plan.compute_tasks.size() - static_cast<size_t>(show))
                      << " more; use --detail to show all)\n";
    }

    print_separator('=');
    std::cout << "Plan: " << (plan.valid ? "VALID" : "INVALID") << "\n";
}

// main

// Score each feasible epoch count and retain the packed and minimum-cost points.
struct EpochSweep { int packed_E; long packed_cy; int best_E; long best_cy; };

static EpochSweep sweep_epoch_depth(const LayerDef& def, ScheduleOptions opts,
                                    bool print_curve) {
    const int cap = std::min(def.params.num_cog(), 32);
    EpochSweep r{0, 0, 0, std::numeric_limits<long>::max()};

    ScheduleOptions base = opts;
    base.optimize_epoch_depth = false; base.forced_epochs = 0;  // packed baseline
    auto p0 = build_schedule(def.params, def.tile_cfg, HardwareSpec{}, base, 0, 0, 0, def.post_ops);
    if (p0.valid) { r.packed_E = p0.stats.num_epochs; r.packed_cy = estimate_cost(p0).predicted_cy; }

    if (print_curve) std::cout << "   E    pred_cy   pred_ms  bound\n";
    int last_E = -1;
    for (int E = 1; E <= cap; ++E) {
        ScheduleOptions o = opts;
        o.optimize_epoch_depth = false; o.forced_epochs = E;
        auto p = build_schedule(def.params, def.tile_cfg, HardwareSpec{}, o, 0, 0, 0, def.post_ops);
        if (!p.valid) continue;
        const int aE = p.stats.num_epochs;
        if (aE == last_E) continue;     // skip clamp duplicates (E < natural)
        last_E = aE;
        const ScheduleCost c = estimate_cost(p);
        if (c.predicted_cy < r.best_cy) { r.best_cy = c.predicted_cy; r.best_E = aE; }
        if (print_curve) {
            std::ostringstream ms; ms << std::fixed << std::setprecision(3) << c.predicted_ms;
            std::cout << "  " << std::setw(3) << aE << "  " << std::setw(9) << c.predicted_cy
                      << "  " << std::setw(8) << ms.str() << "  " << bound_name(c.bound_by)
                      << (aE == r.packed_E ? "   (packed)" : "") << "\n";
        }
    }
    if (r.best_cy == std::numeric_limits<long>::max()) { r.best_cy = r.packed_cy; r.best_E = r.packed_E; }
    return r;
}

int main(int argc, char* argv[]) {
    std::string layer_name;
    SchedulePolicy policy = SchedulePolicy::DynamicWeightEpochs;
    bool detail           = false;
    bool show_descriptors = false;
    bool resnet_mode      = false;
    bool show_cost        = false;
    bool show_cost_v2     = false;  // v2: max-plus schedule model (board + cosim DDR profiles)
    std::string profile_path;       // v2: optional kernel_profile.txt (csynth-extracted)
    bool csv_mode         = false;
    bool opt_epochs       = false;  // Enable cost-driven epoch selection.
    bool sweep_epochs     = false;  // Print predicted cycles by epoch count.
    int  forced_epochs    = 0;      // Positive value forces an epoch count.

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--detail") {
            detail = true;
        } else if (arg == "--descriptors") {
            show_descriptors = true;
        } else if (arg == "--cost") {
            show_cost = true;
        } else if (arg == "--cost-v2") {
            show_cost_v2 = true;
        } else if (arg == "--profile" && i + 1 < argc) {
            profile_path = argv[++i];
        } else if (arg == "--csv") {
            csv_mode = true;
            show_cost = true;   // CSV is the cost report in machine-readable form
        } else if (arg == "--resnet") {
            resnet_mode = true;
        } else if (arg == "--opt-epochs") {
            opt_epochs = true;
        } else if (arg == "--sweep-epochs") {
            sweep_epochs = true;
        } else if (arg == "--epochs" && i + 1 < argc) {
            forced_epochs = std::atoi(argv[++i]);
        } else if (arg == "--policy" && i + 1 < argc) {
            const std::string pol = argv[++i];
            if      (pol == "Baseline")    policy = SchedulePolicy::Baseline;
            else if (pol == "K1Cache")     policy = SchedulePolicy::K1Cache;
            else if (pol == "CigChained")  policy = SchedulePolicy::CigChained;
            else if (pol == "Phase2Default") policy = SchedulePolicy::Phase2Default;
            else if (pol == "CogBatchedExperimental")
                policy = SchedulePolicy::CogBatchedExperimental;
            else if (pol == "DynamicWeightEpochs")
                policy = SchedulePolicy::DynamicWeightEpochs;
            else {
                std::cerr << "Unknown policy: " << pol << "\n";
                return 1;
            }
        } else if (arg[0] != '-') {
            layer_name = arg;
        }
    }

    ScheduleOptions opts = options_for_policy(policy);
    if (opt_epochs)        opts.optimize_epoch_depth = true;
    if (forced_epochs > 0) opts.forced_epochs        = forced_epochs;

    if (resnet_mode && sweep_epochs) {
        const auto defs = make_resnet50_defs();
        std::cout << "Epoch-depth sweep (Pε, predicted)  policy=" << policy_name(policy) << "\n";
        print_separator('=', 80);
        std::cout << std::left << std::setw(34) << "  Name" << std::right
                  << std::setw(7) << "packE" << std::setw(11) << "packed_cy"
                  << std::setw(7) << "bestE" << std::setw(11) << "best_cy"
                  << std::setw(9) << "save%" << "\n";
        print_separator('-', 80);
        long tot_packed = 0, tot_best = 0;
        for (const auto& kv : defs) {
            const EpochSweep s = sweep_epoch_depth(kv.second, opts, false);
            if (s.packed_cy <= 0) continue;
            tot_packed += s.packed_cy;
            tot_best   += s.best_cy;
            const double save = 100.0 * static_cast<double>(s.packed_cy - s.best_cy)
                              / static_cast<double>(s.packed_cy);
            std::cout << std::left << std::setw(34) << ("  " + kv.first) << std::right
                      << std::setw(7) << s.packed_E << std::setw(11) << s.packed_cy
                      << std::setw(7) << s.best_E << std::setw(11) << s.best_cy
                      << std::setw(8) << std::fixed << std::setprecision(1) << save << "\n";
        }
        print_separator('=', 80);
        const double net = tot_packed > 0
            ? 100.0 * static_cast<double>(tot_packed - tot_best) / static_cast<double>(tot_packed)
            : 0.0;
        std::cout << "  network predicted: packed " << tot_packed << " cy  ->  best "
                  << tot_best << " cy   (" << std::fixed << std::setprecision(2)
                  << net << "% saved)\n";
        return 0;
    }

    if (resnet_mode && csv_mode) {
        const auto defs = make_resnet50_defs();
        std::cout << "name,K,S,out_h,out_w,bound,intensity_mac_per_byte,pred_gops,"
                     "pred_ms,ifm_overfetch,mac_floor_cy,weight_read_cy,ifm_read_cy\n";
        std::cout << std::fixed;
        for (const auto& kv : defs) {
            const LayerDef& def = kv.second;
            auto plan = build_schedule(def.params, def.tile_cfg,
                                       HardwareSpec{}, opts, 0, 0, 0, def.post_ops);
            if (!plan.valid) continue;
            const ScheduleCost c = estimate_cost(plan);
            std::cout << kv.first << ',' << def.params.Kh << ',' << def.params.Sh << ','
                      << plan.stats.final_oh << ',' << plan.stats.final_ow << ','
                      << bound_name(c.bound_by) << ','
                      << std::setprecision(4) << c.op_intensity << ','
                      << std::setprecision(2) << c.predicted_gops << ','
                      << std::setprecision(4) << c.predicted_ms << ','
                      << std::setprecision(3) << c.ifm_overfetch << ','
                      << c.mac_floor_cy << ',' << c.weight_read_cy << ',' << c.ifm_read_cy
                      << '\n';
        }
        return 0;
    }

    if (resnet_mode) {
        const auto defs = make_resnet50_defs();
        std::cout << "ResNet-50 full-network schedule validation ("
                  << defs.size() << " layers)  policy=" << policy_name(policy) << "\n";
        print_separator('=', 80);
        std::cout << std::left
                  << std::setw(36) << "  Name"
                  << std::setw(9)  << "Out"
                  << std::setw(4)  << "K"
                  << std::setw(4)  << "S"
                  << std::setw(7)  << "Xfers"
                  << std::setw(7)  << "Descs";
        if (show_cost)
            std::cout << std::setw(8) << "Bound" << std::setw(10) << "ms";
        std::cout << "Status\n";
        print_separator('-', 80);
        int pass = 0, fail = 0;
        for (const auto& kv : defs) {
            const LayerDef& def = kv.second;
            auto plan = build_schedule(def.params, def.tile_cfg,
                                       HardwareSpec{}, opts, 0, 0, 0, def.post_ops);
            int xfers = 0, descs = 0;
            DescriptorBundle bundle;
            if (plan.valid) {
                bundle = emit_descriptors(plan);
                xfers  = static_cast<int>(bundle.transfers.size());
                descs  = static_cast<int>(bundle.compute_descs.size());
            }
            const bool ok = plan.valid && bundle.valid && descs > 0;
            if (ok) ++pass; else ++fail;
            const std::string out = plan.valid
                ? std::to_string(plan.stats.final_oh) + "x" + std::to_string(plan.stats.final_ow)
                : "?x?";
            std::cout << std::left
                      << std::setw(36) << ("  " + kv.first)
                      << std::setw(9)  << out
                      << std::setw(4)  << def.params.Kh
                      << std::setw(4)  << def.params.Sh
                      << std::setw(7)  << xfers
                      << std::setw(7)  << descs;
            if (show_cost && plan.valid) {
                const ScheduleCost cost = estimate_cost(plan);
                std::ostringstream ms;
                ms << std::fixed << std::setprecision(3) << cost.predicted_ms;
                std::cout << std::setw(8) << bound_name(cost.bound_by)
                          << std::setw(10) << ms.str();
            } else if (show_cost) {
                std::cout << std::setw(8) << "-" << std::setw(10) << "-";
            }
            std::cout << (ok ? "OK" : "FAIL") << "\n";
        }
        print_separator('=', 80);
        std::cout << "Passed: " << pass << "/" << defs.size() << "\n";
        return (fail == 0) ? 0 : 1;
    }

    const auto layers = build_layer_table();

    if (layer_name.empty()) {
        std::cerr << "Usage: schedule_inspect <layer-name> [--policy <P>] [--detail] [--descriptors] [--cost] [--cost-v2] [--profile <kernel_profile.txt>]\n";
        std::cerr << "       schedule_inspect --resnet [--policy <P>] [--cost]\n";
        std::cerr << "\nAvailable layers:\n";
        for (const auto& kv : layers)
            std::cerr << "  " << kv.first << '\n';
        std::cerr << "\nPolicies: DynamicWeightEpochs (default), Phase2Default, Baseline, "
                     "K1Cache, CigChained, CogBatchedExperimental\n";
        return 1;
    }

    auto it = layers.find(layer_name);
    if (it == layers.end()) {
        std::cerr << "Unknown layer: " << layer_name << "\n";
        return 1;
    }

    const LayerDef& def = it->second;

    if (sweep_epochs) {
        std::cout << "Epoch-depth sweep: " << layer_name
                  << "  policy=" << policy_name(policy) << "\n";
        const EpochSweep s = sweep_epoch_depth(def, opts, /*print_curve=*/true);
        const double save = s.packed_cy > 0
            ? 100.0 * static_cast<double>(s.packed_cy - s.best_cy) / static_cast<double>(s.packed_cy)
            : 0.0;
        std::cout << "  packed E=" << s.packed_E << " (" << s.packed_cy << " cy)"
                  << "   best E*=" << s.best_E << " (" << s.best_cy << " cy)"
                  << "   saving " << std::fixed << std::setprecision(1) << save << "%\n";
        return 0;
    }

    std::cout << "Policy: " << policy_name(policy) << "\n\n";

    SchedulePlan plan = build_schedule(
        def.params, def.tile_cfg,
        HardwareSpec{}, opts,
        0, 0, 0,
        def.post_ops);

    print_plan(plan, detail);

    if (show_cost && plan.valid) {
        const ScheduleCost cost = estimate_cost(plan);
        print_cost(plan, cost);
    }

    if (show_cost_v2 && plan.valid) {
        // V2.1 (strict synthesis manifest); --profile overrides the committed manifest.
        const KernelManifest mf = profile_path.empty()
            ? default_manifest()
            : load_kernel_manifest(profile_path);
        const AxiPortProfile board = axi_profile_board();
        const AxiPortProfile cosim = axi_profile_cosim();
        print_cost_v2_1(plan, estimate_cost_v2_1(plan, plan.post_ops, mf, board), board);
        print_cost_v2_1(plan, estimate_cost_v2_1(plan, plan.post_ops, mf, cosim), cosim);
    }

    if (show_descriptors && plan.valid) {
        const DescriptorBundle bundle = emit_descriptors(plan);
        if (bundle.valid)
            print_descriptor_table(bundle);
        else {
            std::cerr << "Descriptor emission failed\n";
            for (const auto& iss : bundle.issues)
                std::cerr << "  ! " << iss << "\n";
        }
    }

    return plan.valid ? 0 : 1;
}
