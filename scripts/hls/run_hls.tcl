# Drives DeepTile C simulation, synthesis, co-simulation, and IP export.
#
# Environment inputs:
#   DEEPTILE_HLS_EXEC=<mode>     flow mode, 1-6 (default 1)
#   DEEPTILE_CASE=<name>         case under data/phase1_testcases (default deep_k1_s1)
#   DEEPTILE_GRAPH=<path>        explicit graph.json, overrides DEEPTILE_CASE
#   DEEPTILE_RESET_PROJECT=0     reuse the existing HLS component
#   DEEPTILE_TRACE_LEVEL=<lvl>   co-simulation trace level: all, port, or none
#   DEEPTILE_DEBUG=1             compile the kernel with -DHLS_RUNTIME_DEBUG
#   DEEPTILE_SKIP_TRIPCOUNTS=1   skip graph_plan (use fallback defaults)
#   DEEPTILE_DISABLE_K1_CACHE=1  disable K=1 cache scheduling in the testbench
#   DEEPTILE_DISABLE_CIG_CHAIN=1 disable CIG chaining in the testbench
#   DEEPTILE_NO_PSUM_PP=1        synthesize the single-buffer PSUM variant
#   DEEPTILE_HLS_COMPONENT=name use a separate component for an experiment
#   DEEPTILE_DATAFLOW_PROFILE=1 collect RTL FIFO/process stall observations
#   DEEPTILE_HLS_REPEATS=N      repeat the complete graph in the testbench
#   DEEPTILE_HLS_EPOCH_SWEEP=N,...  repeat with each requested transfer count
#
# Modes:
#   1  C simulation
#   2  C simulation and synthesis
#   3  C simulation, synthesis, and co-simulation
#   4  Synthesis and IP export
#   5  C-simulation regression sweep
#   6  Co-simulation using an existing synthesis result

# The former TILECNN_ prefix still works: it seeds each unset DEEPTILE_ variable.
foreach former [array names ::env TILECNN_*] {
	set current "DEEPTILE_[string range $former 8 end]"
	if {![info exists ::env($current)]} {
		set ::env($current) $::env($former)
	}
}

set PROJ_ROOT [file normalize [file dirname [info script]]/../..]
set HLS_BUILD_DIR "$PROJ_ROOT/build/hls"
set HLS_COMPONENT "deeptile_hls_proj"
if {[info exists ::env(DEEPTILE_HLS_COMPONENT)]} {
	set HLS_COMPONENT $::env(DEEPTILE_HLS_COMPONENT)
	if {![regexp {^[A-Za-z0-9_-]+$} $HLS_COMPONENT]} {
		error "DEEPTILE_HLS_COMPONENT must be a simple component name"
	}
}
file mkdir $HLS_BUILD_DIR
cd $HLS_BUILD_DIR

# HLS component
set reset_project 1
if {[info exists ::env(DEEPTILE_RESET_PROJECT)]} {
	set reset_project $::env(DEEPTILE_RESET_PROJECT)
}
if {$reset_project} {
	open_component -reset $HLS_COMPONENT -flow_target vitis
} else {
	open_component $HLS_COMPONENT -flow_target vitis
}

# Test cases
#
# DEEPTILE_CASE accepts any fixture directory; all_cases defines the mode-5
# regression set.
set TESTCASE_DIR "$PROJ_ROOT/data/phase1_testcases"

set all_cases [list \
	best_case_3x3_s1 \
	stride2_3x3_s2 \
	proj_bottleneck_1x1_s2 \
	small_proj_1x1_s2 \
	deep_k1_s1 \
	late_k1_s1
]

proc case_graph {name} {
	global TESTCASE_DIR
	set path "$TESTCASE_DIR/$name/graph.json"
	if {![file exists $path]} {
		error "Case '$name' not found: $path"
	}
	return $path
}

# DEEPTILE_GRAPH takes precedence over the named case.
set case_name "deep_k1_s1"
if {[info exists ::env(DEEPTILE_CASE)]} {
	set case_name $::env(DEEPTILE_CASE)
}
if {[info exists ::env(DEEPTILE_GRAPH)]} {
	set graph_arg [file normalize $::env(DEEPTILE_GRAPH)]
} else {
	set graph_arg [case_graph $case_name]
}

# Co-simulation waveform detail for modes 3 and 6.
set trace_level "none"
if {[info exists ::env(DEEPTILE_TRACE_LEVEL)]} {
	set trace_level $::env(DEEPTILE_TRACE_LEVEL)
}

# Compile flags
set CFLAGS "-DAP_INT_MAX_W=4096 -I$PROJ_ROOT/src/common -I$PROJ_ROOT/src/kernel"
if {[info exists ::env(DEEPTILE_DEBUG)] && $::env(DEEPTILE_DEBUG)} {
	set CFLAGS "$CFLAGS -DHLS_RUNTIME_DEBUG"
}
# Empty and zero values select the default ping-pong PSUM implementation.
if {[info exists ::env(DEEPTILE_NO_PSUM_PP)] &&
    $::env(DEEPTILE_NO_PSUM_PP) ne "" && $::env(DEEPTILE_NO_PSUM_PP) ne "0"} {
	set CFLAGS "$CFLAGS -DDEEPTILE_NO_PSUM_PP"
	puts "\[run_hls\] DEEPTILE_NO_PSUM_PP=1 -> single-buffer PSUM (Bitstream B)"
}
set TB_CFLAGS "$CFLAGS -std=c++17 -I$PROJ_ROOT/src/host -I$PROJ_ROOT/src/compiler/include"
set LDFLAGS "-lstdc++fs"

# Per-case trip-count and AXI-depth macros
#
# HLS: Compute these flags before add_files because Vitis snapshots -cflags at
# that call. Missing AXI depth overrides can make the co-simulation wrapper copy
# beyond the smaller testbench buffers.
set tc_flags ""
if {![info exists ::env(DEEPTILE_SKIP_TRIPCOUNTS)]} {
	set tc_script "$PROJ_ROOT/scripts/compute_tripcounts.py"
	if {[file exists $tc_script]} {
		if {[catch {
			set tc_flags [exec python3 $tc_script $graph_arg \
			              --build-dir $PROJ_ROOT/build/tools]
		} err]} {
			puts "NOTE: per-case tripcounts unavailable — synthesis uses default bounds."
			puts "      Build graph_plan once from a system terminal (not inside Vitis):"
			puts "        cmake -S $PROJ_ROOT/scripts -B $PROJ_ROOT/build/tools -DCMAKE_CXX_COMPILER=/usr/bin/g++"
			puts "        cmake --build $PROJ_ROOT/build/tools --target graph_plan"
		}
	} else {
		puts "WARNING: $tc_script not found — skipping per-case tripcounts."
	}
}
set CFLAGS "$CFLAGS $tc_flags"
set TB_CFLAGS "$TB_CFLAGS $tc_flags"

# HLS and testbench sources. The computed flags must already be final.
add_files $PROJ_ROOT/src/common/common_defs.h
add_files $PROJ_ROOT/src/kernel/tile_conv.h
add_files $PROJ_ROOT/src/kernel/tile_conv.cpp        -cflags $CFLAGS
add_files $PROJ_ROOT/src/kernel/data_orch.h
add_files $PROJ_ROOT/src/kernel/data_orch.cpp       -cflags $CFLAGS
add_files $PROJ_ROOT/src/kernel/conv_engine.h
add_files $PROJ_ROOT/src/kernel/conv_engine.cpp     -cflags $CFLAGS
add_files $PROJ_ROOT/src/kernel/output_postproc.h
add_files $PROJ_ROOT/src/kernel/output_postproc.cpp -cflags $CFLAGS
add_files $PROJ_ROOT/src/kernel/dw_stage.h
add_files $PROJ_ROOT/src/kernel/dw_stage.cpp        -cflags $CFLAGS

add_files -tb $PROJ_ROOT/src/compiler/src/schedule_builder.cpp            -cflags $TB_CFLAGS
add_files -tb $PROJ_ROOT/src/compiler/src/cost_model.cpp                  -cflags $TB_CFLAGS
add_files -tb $PROJ_ROOT/src/compiler/src/descriptor_emitter.cpp          -cflags $TB_CFLAGS
add_files -tb $PROJ_ROOT/src/compiler/src/graph_loader.cpp                -cflags $TB_CFLAGS
add_files -tb $PROJ_ROOT/src/compiler/src/model_package.cpp               -cflags $TB_CFLAGS
add_files -tb $PROJ_ROOT/src/compiler/src/target_loader.cpp               -cflags $TB_CFLAGS
add_files -tb $PROJ_ROOT/src/compiler/src/graph_compiler.cpp              -cflags $TB_CFLAGS
add_files -tb $PROJ_ROOT/src/host/runtime/runtime.cpp                      -cflags $TB_CFLAGS
add_files -tb $PROJ_ROOT/src/host/runtime/stem_fold.cpp                    -cflags $TB_CFLAGS
add_files -tb $PROJ_ROOT/src/host/backends/hls_adapter.cpp                -cflags $TB_CFLAGS
add_files -tb $PROJ_ROOT/tests/integration/hls/tile_conv_tb.cpp           -cflags $TB_CFLAGS

set_top tile_conv

# Device and clock
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 5


# Helpers
proc banner {msg} {
	puts ""
	puts "================================================="
	puts "  $msg"
	puts "================================================="
}

proc run_cosim {graph ldflags trace} {
	set options [list -argv $graph -ldflags $ldflags]
	if {$trace != "none"} {
		lappend options -trace_level $trace
	}
	if {[info exists ::env(DEEPTILE_DATAFLOW_PROFILE)] &&
	    $::env(DEEPTILE_DATAFLOW_PROFILE) eq "1"} {
		lappend options -enable_dataflow_profiling
	}
	cosim_design {*}$options
}

# Mode dispatch
set hls_exec 1
if {[info exists ::env(DEEPTILE_HLS_EXEC)]} {
	set hls_exec $::env(DEEPTILE_HLS_EXEC)
}

if {$hls_exec == 1} {
	banner "C-sim case: $case_name"
	csim_design -argv "$graph_arg" -ldflags "$LDFLAGS"

} elseif {$hls_exec == 2} {
	banner "C-sim + C-synth case: $case_name"
	csim_design -argv "$graph_arg" -ldflags "$LDFLAGS"
	csynth_design

} elseif {$hls_exec == 3} {
	banner "C-sim + C-synth + Cosim case: $case_name  (trace=$trace_level)"
	csim_design -argv "$graph_arg" -ldflags "$LDFLAGS"
	csynth_design
	run_cosim $graph_arg $LDFLAGS $trace_level

} elseif {$hls_exec == 4} {
	banner "export_design"
	csynth_design
	export_design

} elseif {$hls_exec == 5} {
	foreach c $all_cases {
		banner "C-sim regression case: $c"
		csim_design -argv "[case_graph $c]" -ldflags "$LDFLAGS"
	}

} elseif {$hls_exec == 6} {
	banner "Cosim-only case: $case_name  (trace=$trace_level, reuse synth)"
	run_cosim $graph_arg $LDFLAGS $trace_level

} else {
	error "Unknown DEEPTILE_HLS_EXEC '$hls_exec' (valid: 1-6)"
}

exit
