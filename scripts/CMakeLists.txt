# Native compiler, host, inspection-tool, and test targets.

cmake_minimum_required(VERSION 3.16)

project(DeepTile LANGUAGES CXX)

get_filename_component(DEEPTILE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(DEEPTILE_COMPILER_DIR ${DEEPTILE_ROOT}/src/compiler)
set(DEEPTILE_COMPILER_INCLUDE ${DEEPTILE_COMPILER_DIR}/include)
set(DEEPTILE_HOST_DIR ${DEEPTILE_ROOT}/src/host)

# Graph import, scheduling, and descriptor lowering.
add_library(deeptile_compiler STATIC
  ${DEEPTILE_COMPILER_DIR}/src/schedule_builder.cpp
  ${DEEPTILE_COMPILER_DIR}/src/descriptor_emitter.cpp
  ${DEEPTILE_COMPILER_DIR}/src/cost_model.cpp
  ${DEEPTILE_COMPILER_DIR}/src/graph_loader.cpp
  ${DEEPTILE_COMPILER_DIR}/src/model_package.cpp
  ${DEEPTILE_COMPILER_DIR}/src/target_loader.cpp
  ${DEEPTILE_COMPILER_DIR}/src/graph_compiler.cpp
)
target_include_directories(deeptile_compiler PUBLIC
  ${DEEPTILE_COMPILER_INCLUDE}
  ${DEEPTILE_ROOT}/src/common
)
target_compile_options(deeptile_compiler PRIVATE -Wall -Wextra -Wpedantic)

# Runtime planning, tensor packing, and the software reference path.
add_library(deeptile_host STATIC
  ${DEEPTILE_HOST_DIR}/runtime/runtime.cpp
  ${DEEPTILE_HOST_DIR}/runtime/stem_fold.cpp
  ${DEEPTILE_HOST_DIR}/runtime/run_record.cpp
  ${DEEPTILE_HOST_DIR}/runtime/target_artifact.cpp
  ${DEEPTILE_HOST_DIR}/reference/runtime_reference.cpp
)
target_include_directories(deeptile_host PUBLIC ${DEEPTILE_HOST_DIR})
target_link_libraries(deeptile_host PUBLIC deeptile_compiler)
target_compile_options(deeptile_host PRIVATE -O2 -Wall -Wextra -Wpedantic)

execute_process(
  COMMAND git rev-parse --short=12 HEAD
  WORKING_DIRECTORY ${DEEPTILE_ROOT}
  OUTPUT_VARIABLE DEEPTILE_GIT_REVISION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(NOT DEEPTILE_GIT_REVISION)
  set(DEEPTILE_GIT_REVISION unknown)
endif()
# Scoped to the paths that determine the artifact; see scripts/hw/Makefile for
# why a whole-repo check is the wrong question to ask.
set(DEEPTILE_SOURCE_SCOPE "src scripts")
execute_process(
  COMMAND git status --porcelain -- src scripts
  WORKING_DIRECTORY ${DEEPTILE_ROOT}
  OUTPUT_VARIABLE DEEPTILE_GIT_STATUS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(DEEPTILE_GIT_STATUS)
  set(DEEPTILE_SOURCE_DIRTY 1)
else()
  set(DEEPTILE_SOURCE_DIRTY 0)
endif()
target_compile_definitions(deeptile_host PRIVATE
  DEEPTILE_GIT_REVISION="${DEEPTILE_GIT_REVISION}"
  DEEPTILE_SOURCE_DIRTY=${DEEPTILE_SOURCE_DIRTY}
  DEEPTILE_SOURCE_SCOPE="${DEEPTILE_SOURCE_SCOPE}")

# Dataset indexing, image preprocessing, and evaluation metrics.
#
# Resizing must reproduce the exporter's Pillow output bit-exactly, so it is
# implemented here rather than delegated to OpenCV. OpenCV supplies only the
# image decoder and stays optional: a native build without it still compiles
# and tests the whole deterministic path through a stub decoder.
# Discovery order: CMake package config, then pkg-config, then a direct header
# and library search. Distributions ship these three independently, so relying
# on any single one silently disables decoding on a machine that has OpenCV.
set(DEEPTILE_OPENCV_FOUND FALSE)
set(DEEPTILE_OPENCV_INCLUDE_DIRS "")
set(DEEPTILE_OPENCV_LIBRARIES "")
set(DEEPTILE_OPENCV_LIBRARY_DIRS "")

find_package(OpenCV QUIET COMPONENTS core imgcodecs)
if(OpenCV_FOUND)
  set(DEEPTILE_OPENCV_FOUND TRUE)
  set(DEEPTILE_OPENCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
  set(DEEPTILE_OPENCV_LIBRARIES ${OpenCV_LIBS})
endif()

if(NOT DEEPTILE_OPENCV_FOUND)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(DEEPTILE_PC_OPENCV QUIET opencv4)
    if(DEEPTILE_PC_OPENCV_FOUND)
      set(DEEPTILE_OPENCV_FOUND TRUE)
      set(DEEPTILE_OPENCV_INCLUDE_DIRS ${DEEPTILE_PC_OPENCV_INCLUDE_DIRS})
      set(DEEPTILE_OPENCV_LIBRARIES ${DEEPTILE_PC_OPENCV_LIBRARIES})
      set(DEEPTILE_OPENCV_LIBRARY_DIRS ${DEEPTILE_PC_OPENCV_LIBRARY_DIRS})
    endif()
  endif()
endif()

if(NOT DEEPTILE_OPENCV_FOUND)
  # Only core and imgcodecs are needed; resizing is DeepTile-owned.
  find_path(DEEPTILE_OPENCV_INCLUDE_DIR opencv2/imgcodecs.hpp
    PATH_SUFFIXES opencv4)
  find_library(DEEPTILE_OPENCV_CORE_LIB opencv_core)
  find_library(DEEPTILE_OPENCV_IMGCODECS_LIB opencv_imgcodecs)
  if(DEEPTILE_OPENCV_INCLUDE_DIR AND DEEPTILE_OPENCV_CORE_LIB
     AND DEEPTILE_OPENCV_IMGCODECS_LIB)
    set(DEEPTILE_OPENCV_FOUND TRUE)
    set(DEEPTILE_OPENCV_INCLUDE_DIRS ${DEEPTILE_OPENCV_INCLUDE_DIR})
    set(DEEPTILE_OPENCV_LIBRARIES
      ${DEEPTILE_OPENCV_CORE_LIB} ${DEEPTILE_OPENCV_IMGCODECS_LIB})
  endif()
endif()

set(DEEPTILE_DATASET_SRCS
  ${DEEPTILE_HOST_DIR}/dataset/resize.cpp
  ${DEEPTILE_HOST_DIR}/dataset/preprocess.cpp
  ${DEEPTILE_HOST_DIR}/dataset/dataset.cpp
  ${DEEPTILE_HOST_DIR}/dataset/evaluation.cpp
  ${DEEPTILE_HOST_DIR}/dataset/pipeline.cpp
  ${DEEPTILE_HOST_DIR}/dataset/metrics.cpp
)

if(DEEPTILE_OPENCV_FOUND)
  list(APPEND DEEPTILE_DATASET_SRCS
    ${DEEPTILE_HOST_DIR}/dataset/image_decode_opencv.cpp)
else()
  list(APPEND DEEPTILE_DATASET_SRCS
    ${DEEPTILE_HOST_DIR}/dataset/image_decode_stub.cpp)
  message(STATUS
    "OpenCV not found - dataset image decoding is DISABLED in this build; "
    "preprocessing arithmetic is still built and tested")
endif()

find_package(Threads REQUIRED)

add_library(deeptile_dataset STATIC ${DEEPTILE_DATASET_SRCS})
target_include_directories(deeptile_dataset PUBLIC ${DEEPTILE_HOST_DIR})
target_link_libraries(deeptile_dataset PUBLIC deeptile_host Threads::Threads)
target_compile_options(deeptile_dataset PRIVATE -O2 -Wall -Wextra -Wpedantic)
if(DEEPTILE_OPENCV_FOUND)
  target_include_directories(deeptile_dataset PRIVATE ${DEEPTILE_OPENCV_INCLUDE_DIRS})
  target_link_libraries(deeptile_dataset PUBLIC ${DEEPTILE_OPENCV_LIBRARIES})
  target_link_directories(deeptile_dataset PUBLIC ${DEEPTILE_OPENCV_LIBRARY_DIRS})
  # libjpeg(-turbo), OpenCV's own JPEG backend, lets the dataset path decode
  # only the region preprocessing reads. Without it every image is decoded in
  # full through OpenCV, with identical results.
  find_path(DEEPTILE_JPEG_INCLUDE_DIR jpeglib.h)
  find_library(DEEPTILE_JPEG_LIB jpeg)
  if(DEEPTILE_JPEG_INCLUDE_DIR AND DEEPTILE_JPEG_LIB)
    target_include_directories(deeptile_dataset PRIVATE ${DEEPTILE_JPEG_INCLUDE_DIR})
    target_link_libraries(deeptile_dataset PUBLIC ${DEEPTILE_JPEG_LIB})
    target_compile_definitions(deeptile_dataset PRIVATE DEEPTILE_HAVE_LIBJPEG)
  endif()
endif()

# Command-line inspection tools.
add_executable(graph_plan ${DEEPTILE_COMPILER_DIR}/tools/graph_plan_main.cpp)
target_link_libraries(graph_plan PRIVATE deeptile_compiler)
target_compile_options(graph_plan PRIVATE -Wall -Wextra -Wpedantic)

add_executable(graph_schedule ${DEEPTILE_COMPILER_DIR}/tools/graph_schedule_main.cpp)
target_link_libraries(graph_schedule PRIVATE deeptile_compiler)
target_compile_options(graph_schedule PRIVATE -Wall -Wextra -Wpedantic)

add_executable(descriptor_dump ${DEEPTILE_COMPILER_DIR}/tools/descriptor_dump_main.cpp)
target_link_libraries(descriptor_dump PRIVATE deeptile_compiler)
target_compile_options(descriptor_dump PRIVATE -Wall -Wextra -Wpedantic)

add_executable(ifm_traffic ${DEEPTILE_COMPILER_DIR}/tools/ifm_traffic_main.cpp)
target_link_libraries(ifm_traffic PRIVATE deeptile_compiler)
target_compile_options(ifm_traffic PRIVATE -Wall -Wextra -Wpedantic)

add_executable(schedule_inspect
  ${DEEPTILE_COMPILER_DIR}/tools/schedule_inspect_main.cpp
  ${DEEPTILE_COMPILER_DIR}/src/cost_model_v2_1.cpp
)
target_link_libraries(schedule_inspect PRIVATE deeptile_compiler)
target_compile_options(schedule_inspect PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)

add_executable(runtime_dry_run ${DEEPTILE_HOST_DIR}/tools/runtime_dry_run_main.cpp)
target_link_libraries(runtime_dry_run PRIVATE deeptile_host)
target_compile_options(runtime_dry_run PRIVATE -Wall -Wextra -Wpedantic)

add_executable(reference_dump ${DEEPTILE_HOST_DIR}/tools/reference_dump_main.cpp)
target_link_libraries(reference_dump PRIVATE deeptile_host)
target_compile_options(reference_dump PRIVATE -Wall -Wextra -Wpedantic)

add_executable(target_inspect
  ${DEEPTILE_HOST_DIR}/tools/target_inspect_main.cpp)
target_link_libraries(target_inspect PRIVATE deeptile_host)
target_compile_options(target_inspect PRIVATE -Wall -Wextra -Wpedantic)

add_executable(dataset_preprocess
  ${DEEPTILE_HOST_DIR}/tools/dataset_preprocess_main.cpp)
target_link_libraries(dataset_preprocess PRIVATE deeptile_dataset)
target_compile_options(dataset_preprocess PRIVATE -O2 -Wall -Wextra -Wpedantic)

# Native tests.
enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME profiling_tools_tests COMMAND ${Python3_EXECUTABLE} -B -m unittest
  discover -s ${DEEPTILE_ROOT}/tests/profiling -v)
add_test(NAME fixture_generator_tests COMMAND ${Python3_EXECUTABLE} -B -m unittest
  discover -s ${DEEPTILE_ROOT}/tests/integration/fixtures -v)

add_executable(compiler_tests ${DEEPTILE_ROOT}/tests/compiler/graph_compiler_tests.cpp)
target_link_libraries(compiler_tests PRIVATE deeptile_compiler)
target_compile_definitions(compiler_tests PRIVATE DEEPTILE_REPO_ROOT="${DEEPTILE_ROOT}")
target_compile_options(compiler_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME compiler_tests COMMAND compiler_tests)

add_executable(chained_schedule_tests ${DEEPTILE_ROOT}/tests/compiler/chained_schedule_tests.cpp)
target_link_libraries(chained_schedule_tests PRIVATE deeptile_compiler)
target_compile_options(chained_schedule_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME chained_schedule_tests COMMAND chained_schedule_tests)

add_executable(host_tests ${DEEPTILE_ROOT}/tests/host/runtime_tests.cpp)
target_link_libraries(host_tests PRIVATE deeptile_host)
target_compile_definitions(host_tests PRIVATE DEEPTILE_REPO_ROOT="${DEEPTILE_ROOT}")
target_compile_options(host_tests PRIVATE -Wall -Wextra -Wpedantic)
add_dependencies(host_tests runtime_dry_run)
add_test(NAME host_tests COMMAND host_tests)

add_executable(run_record_tests
  ${DEEPTILE_ROOT}/tests/host/run_record_tests.cpp)
target_link_libraries(run_record_tests PRIVATE deeptile_host)
target_compile_options(run_record_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME run_record_tests COMMAND run_record_tests)

add_executable(preprocess_tests ${DEEPTILE_ROOT}/tests/host/preprocess_tests.cpp)
target_link_libraries(preprocess_tests PRIVATE deeptile_dataset)
target_compile_definitions(preprocess_tests PRIVATE DEEPTILE_REPO_ROOT="${DEEPTILE_ROOT}")
target_compile_options(preprocess_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME preprocess_tests COMMAND preprocess_tests)

add_executable(metrics_tests ${DEEPTILE_ROOT}/tests/host/metrics_tests.cpp)
target_link_libraries(metrics_tests PRIVATE deeptile_dataset)
target_compile_options(metrics_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME metrics_tests COMMAND metrics_tests)

add_executable(pipeline_tests ${DEEPTILE_ROOT}/tests/host/pipeline_tests.cpp)
target_link_libraries(pipeline_tests PRIVATE deeptile_dataset)
target_compile_options(pipeline_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME pipeline_tests COMMAND pipeline_tests)

add_executable(dataset_tests ${DEEPTILE_ROOT}/tests/host/dataset_tests.cpp)
target_link_libraries(dataset_tests PRIVATE deeptile_dataset)
target_compile_definitions(dataset_tests PRIVATE DEEPTILE_REPO_ROOT="${DEEPTILE_ROOT}")
target_compile_options(dataset_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME dataset_tests COMMAND dataset_tests)

add_executable(abi_tests ${DEEPTILE_ROOT}/tests/compiler/abi_tests.cpp)
target_link_libraries(abi_tests PRIVATE deeptile_compiler)
add_test(NAME abi_tests COMMAND abi_tests)

add_executable(target_package_tests
  ${DEEPTILE_ROOT}/tests/compiler/target_package_tests.cpp)
target_link_libraries(target_package_tests PRIVATE deeptile_host)
target_compile_options(target_package_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME target_package_tests COMMAND target_package_tests)

add_executable(model_package_tests
  ${DEEPTILE_ROOT}/tests/compiler/model_package_tests.cpp)
target_link_libraries(model_package_tests PRIVATE deeptile_compiler)
target_compile_options(model_package_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME model_package_tests COMMAND model_package_tests)

add_executable(cost_model_tests ${DEEPTILE_ROOT}/tests/compiler/cost_model_tests.cpp)
target_link_libraries(cost_model_tests PRIVATE deeptile_compiler)
add_test(NAME cost_model_tests COMMAND cost_model_tests)

add_executable(depthwise_tests ${DEEPTILE_ROOT}/tests/compiler/depthwise_tests.cpp)
target_link_libraries(depthwise_tests PRIVATE deeptile_compiler)
add_test(NAME depthwise_tests COMMAND depthwise_tests)

add_executable(reference_equivalence_tests
  ${DEEPTILE_ROOT}/tests/integration/reference_equivalence_tests.cpp)
target_link_libraries(reference_equivalence_tests PRIVATE deeptile_host)
target_compile_definitions(reference_equivalence_tests PRIVATE
  DEEPTILE_REPO_ROOT="${DEEPTILE_ROOT}")
target_compile_options(reference_equivalence_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME reference_equivalence_tests COMMAND reference_equivalence_tests)
# The default folds a stride-2 RGB stem by space-to-depth; the same gate with
# the stem as exported keeps the unfolded path covered too.
add_test(NAME reference_equivalence_unfolded_stem_tests
  COMMAND reference_equivalence_tests)
set_tests_properties(reference_equivalence_unfolded_stem_tests PROPERTIES
  ENVIRONMENT "DEEPTILE_STEM_S2D=0")

# Pins the exporter's image-preprocessing convention against a checked-in
# package. Registered only when a decoder exists, since the anchor is a JPEG.
add_executable(preprocess_anchor_test
  ${DEEPTILE_ROOT}/tests/integration/preprocess_anchor_test.cpp)
target_link_libraries(preprocess_anchor_test PRIVATE deeptile_dataset)
target_compile_definitions(preprocess_anchor_test PRIVATE
  DEEPTILE_REPO_ROOT="${DEEPTILE_ROOT}")
target_compile_options(preprocess_anchor_test PRIVATE -Wall -Wextra -Wpedantic)
if(DEEPTILE_OPENCV_FOUND)
  add_test(NAME preprocess_anchor_test COMMAND preprocess_anchor_test)
else()
  message(STATUS
    "No image decoder - skipping preprocess_anchor_test; the exporter "
    "preprocessing convention is UNVERIFIED in this build")
endif()

add_test(NAME schedule_inspect_smoke COMMAND schedule_inspect best_case_3x3_s1)
add_test(NAME schedule_inspect_resnet COMMAND schedule_inspect --resnet)

# HLS header discovery for the descriptor-ABI bridge and optional XRT backend.
set(DEEPTILE_HLS_INCLUDE_DIR "")
if(DEFINED ENV{XILINX_VITIS_HLS})
  set(DEEPTILE_HLS_INCLUDE_DIR "$ENV{XILINX_VITIS_HLS}/include")
elseif(DEFINED ENV{XILINX_HLS})
  set(DEEPTILE_HLS_INCLUDE_DIR "$ENV{XILINX_HLS}/include")
elseif(DEFINED ENV{XILINX_VIVADO})
  set(DEEPTILE_HLS_INCLUDE_DIR "$ENV{XILINX_VIVADO}/include")
endif()

if(DEEPTILE_HLS_INCLUDE_DIR)
  add_executable(common_defs_bridge_check ${DEEPTILE_ROOT}/tests/common_defs_bridge_check.cpp)
  target_include_directories(common_defs_bridge_check PRIVATE
    ${DEEPTILE_ROOT}/src/common
    ${DEEPTILE_COMPILER_INCLUDE}
    ${DEEPTILE_HLS_INCLUDE_DIR}
  )
  add_test(NAME common_defs_bridge_check COMMAND common_defs_bridge_check)

  # Alignment regressions exercise the actual private loader implementation.
  # The test uses explicit comparisons (also active in Release builds).
  add_executable(ifm_load_raw_tests ${DEEPTILE_ROOT}/tests/kernel/ifm_load_raw_tb.cpp)
  target_include_directories(ifm_load_raw_tests PRIVATE
    ${DEEPTILE_ROOT}/src/common
    ${DEEPTILE_ROOT}/src/kernel
    ${DEEPTILE_HLS_INCLUDE_DIR})
  target_compile_options(ifm_load_raw_tests PRIVATE -Wno-unknown-pragmas)
  add_test(NAME ifm_load_raw_tests COMMAND ifm_load_raw_tests)

  # The fused depthwise stage alone, against the software reference: 3x3 and
  # 5x5 kernels, both strides, row-halo seams and 72 channel groups.
  add_executable(dw_stage_tests
    ${DEEPTILE_ROOT}/tests/kernel/dw_stage_tb.cpp
    ${DEEPTILE_ROOT}/src/kernel/dw_stage.cpp)
  target_link_libraries(dw_stage_tests PRIVATE deeptile_host)
  target_include_directories(dw_stage_tests PRIVATE
    ${DEEPTILE_ROOT}/src/kernel ${DEEPTILE_HLS_INCLUDE_DIR})
  target_compile_options(dw_stage_tests PRIVATE -Wno-unknown-pragmas)
  target_compile_definitions(dw_stage_tests PRIVATE AP_INT_MAX_W=4096)
  foreach(dw_case pwdw_rowtiled pwdw_stride2 pwdw_k5_rowtiled pwdw_k5_stride2
                  pwdw_k5_wide pwdw_k5_deep)
    add_test(NAME dw_stage_${dw_case}
             COMMAND dw_stage_tests
                     ${DEEPTILE_ROOT}/data/dw_testcases/${dw_case}/graph.json)
  endforeach()

  # Optimized C-model runner for full-network correctness checks. This is NOT
  # cycle simulation; RTL/board timing is measured apart. It is part of the
  # default build so it can never lag the kernel and compiler sources: a stale
  # copy once validated an earlier kernel without any sign of it.
  add_executable(deeptile_csim
    ${DEEPTILE_ROOT}/tests/integration/hls/tile_conv_tb.cpp
    ${DEEPTILE_HOST_DIR}/backends/hls_adapter.cpp
    ${DEEPTILE_ROOT}/src/kernel/tile_conv.cpp
    ${DEEPTILE_ROOT}/src/kernel/data_orch.cpp
    ${DEEPTILE_ROOT}/src/kernel/conv_engine.cpp
    ${DEEPTILE_ROOT}/src/kernel/output_postproc.cpp
    ${DEEPTILE_ROOT}/src/kernel/dw_stage.cpp)
  target_link_libraries(deeptile_csim PRIVATE deeptile_host)
  target_include_directories(deeptile_csim PRIVATE
    ${DEEPTILE_ROOT}/src/kernel ${DEEPTILE_HLS_INCLUDE_DIR})
  target_compile_options(deeptile_csim PRIVATE -Wno-unknown-pragmas)
  target_compile_definitions(deeptile_csim PRIVATE AP_INT_MAX_W=4096)
else()
  message(STATUS
    "HLS headers not found - skipping common_defs_bridge_check; "
    "the kernel ABI bridge is UNVERIFIED in this build")
endif()

if(DEFINED ENV{XILINX_XRT} AND DEEPTILE_HLS_INCLUDE_DIR)
  add_library(deeptile_xrt EXCLUDE_FROM_ALL
    ${DEEPTILE_HOST_DIR}/backends/xrt_backend.cpp
    ${DEEPTILE_HOST_DIR}/backends/dataset_eval.cpp
  )
  target_include_directories(deeptile_xrt PUBLIC
    ${DEEPTILE_HOST_DIR}
    ${DEEPTILE_ROOT}/src/common
    ${DEEPTILE_ROOT}/src/kernel
    $ENV{XILINX_XRT}/include
    ${DEEPTILE_HLS_INCLUDE_DIR}
  )
  target_link_directories(deeptile_xrt PUBLIC $ENV{XILINX_XRT}/lib)
  target_link_libraries(deeptile_xrt PUBLIC deeptile_host deeptile_dataset
    xrt_coreutil)
  target_compile_options(deeptile_xrt PRIVATE -O2 -Wall -Wextra -Wpedantic)

  add_executable(tilecnn EXCLUDE_FROM_ALL ${DEEPTILE_HOST_DIR}/main.cpp)
  set_target_properties(tilecnn PROPERTIES OUTPUT_NAME deeptile)
  target_link_libraries(tilecnn PRIVATE deeptile_xrt)
  target_compile_options(tilecnn PRIVATE -O2 -Wall -Wextra -Wpedantic)
endif()
