#!/usr/bin/env bash
# Runs schedule-policy comparisons on a deployed ZCU104 package.
#
# Policy variables are evaluated by the host-side graph compiler, so all cells
# share one FPGA binary. Logs and profile CSVs are retrieved over SSH.
#
#   ./scripts/hw/run_ablation.sh                 # all 8
#   POLICIES=p1 NETS=resnet50 ./scripts/hw/run_ablation.sh   # one cell
#
# Requires an SSH host alias (default `zcu104`) and a prior deploy.
set -uo pipefail

BOARD="${BOARD:-zcu104}"
REMOTE_DIR="${REMOTE_DIR:-/run/media/boot-mmcblk0p1}"
RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-2}"
POLICIES="${POLICIES:-p0 p1 p2 p3}"
NETS="${NETS:-resnet50 mobilenetv2}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/build/ablation}"
mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.csv"

# Clear every policy variable before each cell because P3 is defined by their
# absence and the remote login environment may already export them.
POLICY_VARS="DEEPTILE_DISABLE_K1_CACHE DEEPTILE_DISABLE_CIG_CHAIN \
DEEPTILE_PACK_EPOCHS DEEPTILE_FORCE_EPOCHS DEEPTILE_FUSE_DW DEEPTILE_ABLATION"

env_clear_args() {
    local v out=""
    # Clear the former TILECNN_ names too; the host still reads them.
    for v in ${POLICY_VARS}; do out="${out} -u ${v} -u TILECNN_${v#DEEPTILE_}"; done
    echo "${out}"
}

# P3 is the default configuration and therefore sets no policy variables.
policy_env() {
    case "$1" in
        p0) echo "DEEPTILE_DISABLE_K1_CACHE=1 DEEPTILE_DISABLE_CIG_CHAIN=1 DEEPTILE_PACK_EPOCHS=1" ;;
        p1) echo "DEEPTILE_DISABLE_CIG_CHAIN=1 DEEPTILE_PACK_EPOCHS=1" ;;
        p2) echo "DEEPTILE_PACK_EPOCHS=1" ;;
        p3) echo "" ;;
        *)  echo "__INVALID__" ;;
    esac
}

net_graph() {
    case "$1" in
        resnet50)    echo "data/resnet50_int8_tilecnn/graph.json" ;;
        mobilenetv2) echo "data/mobilenet_v2_int8_tilecnn/graph.json" ;;
        *)           echo "__INVALID__" ;;
    esac
}

# Enable fused pointwise-depthwise lowering only for MobileNetV2.
net_fuse() { [ "$1" = mobilenetv2 ] && echo "DEEPTILE_FUSE_DW=1" || echo ""; }

# Board validation
echo "Checking ${BOARD}:${REMOTE_DIR}"
preflight_files="deeptile deeptile.xclbin target.json"
for net in ${NETS}; do preflight_files="${preflight_files} $(net_graph "${net}")"; done
if ! ssh "${BOARD}" "cd '${REMOTE_DIR}' && for f in ${preflight_files}; do \
        [ -f \"\$f\" ] || { echo \"MISSING: \$f\"; exit 1; }; done && echo OK"; then
    echo "Error: board is not ready.  Deploy first:" >&2
    echo "  make -C scripts hw-package VITIS_MODE=hw && ./scripts/hw/deploy_zcu104.sh" >&2
    echo "Note: deploy_zcu104.sh never copies data/ — graph fixtures must already" >&2
    echo "be on the board under ${REMOTE_DIR}/data/." >&2
    exit 1
fi

# Preserve prior scoped runs unless FRESH=1 requests a new summary.
if [ "${FRESH:-0}" = 1 ] || [ ! -f "${SUMMARY}" ]; then
    echo "net,policy,status,compare,avg_kernel_ms,gops" > "${SUMMARY}"
fi

run_one() {
    local net="$1" policy="$2"
    local graph envs fuse tag remote_csv log
    graph="$(net_graph "${net}")"
    envs="$(policy_env "${policy}")"
    fuse="$(net_fuse "${net}")"
    tag="${net}_${policy}"
    remote_csv="ablation_${tag}.csv"
    log="${OUT_DIR}/${tag}.log"

    # A concurrent host process contends for DDR and invalidates timing comparisons.
    local busy
    busy=$(ssh -o ConnectTimeout=10 "${BOARD}" "pgrep -c deeptile || true" 2>/dev/null)
    if [ -n "${busy}" ] && [ "${busy}" -gt 0 ] 2>/dev/null; then
        echo "    WARNING: ${busy} deeptile process(es) already running on ${BOARD}" >&2
        echo "    Timing for this cell will not be comparable." >&2
    fi

    echo "[$(date +%H:%M:%S)] ${net} ${policy} ..."
    # Bound remote execution because a kernel deadlock otherwise blocks wait().
    # The log records the effective policy environment for result provenance.
    local clear
    clear="$(env_clear_args)"
    timeout "${RUN_TIMEOUT:-300}" \
    ssh "${BOARD}" "cd '${REMOTE_DIR}' && \
        export LD_LIBRARY_PATH=/mnt:/tmp:\${LD_LIBRARY_PATH:-} && \
        export XILINX_XRT=/usr && \
        echo \"[ENV] \$(env ${clear} ${envs} ${fuse} sh -c 'env | grep ^DEEPTILE_ | sort | tr \"\\n\" \" \"')\" && \
        echo \"[ENV] (none above means P3/shipped)\" && \
        env ${clear} ${envs} ${fuse} ./deeptile ./target.json ${graph} \
            --runs ${RUNS} --warmup ${WARMUP} --csv ${remote_csv}" \
        > "${log}" 2>&1
    local rc=$?

    # A timeout leaves the FPGA unusable until reboot, so later cells must not run.
    if [ "${rc}" -eq 124 ]; then
        echo "${net},${policy},HANG,NONE,NA,NA" >> "${SUMMARY}"
        echo "    -> HANG (kernel did not complete in ${RUN_TIMEOUT:-300}s)"
        echo "    The FPGA is left wedged. Reboot before any further run:" >&2
        echo "      ssh ${BOARD} reboot" >&2
        ssh -o ConnectTimeout=10 "${BOARD}" "pkill -f deeptile" >/dev/null 2>&1
        return 124
    fi

    # Record timing only alongside the bit-exact comparison status.
    local compare=NONE
    if grep -q "\[COMPARE\]\[FAIL\]\|\[COMPARE\]\[ERROR\]" "${log}"; then
        compare=FAIL
    elif grep -q "\[COMPARE\]\[PASS\]" "${log}"; then
        compare=PASS
    fi

    local avg_ms gops status
    avg_ms=$(grep -oP "\[XRT_TESTCASE_AVG\] graph_kernel_ms=\K[0-9.]+" "${log}" | tail -1)
    avg_ms=${avg_ms:-NA}
    gops=$(grep -oP "\[XRT_TESTCASE_AVG\].*graph_gops=\K[0-9.]+" "${log}" | tail -1)
    gops=${gops:-NA}

    if [ "${rc}" -eq 0 ] && [ "${compare}" = PASS ]; then status=PASS; else status=FAIL; fi
    echo "${net},${policy},${status},${compare},${avg_ms},${gops}" >> "${SUMMARY}"
    echo "    -> ${status} (compare=${compare} avg_kernel_ms=${avg_ms} gops=${gops})"

    scp -q "${BOARD}:${REMOTE_DIR}/${remote_csv}" "${OUT_DIR}/${tag}_profile.csv" 2>/dev/null \
        || echo "    (no per-launch profile CSV retrieved)"
}

for net in ${NETS}; do
    for policy in ${POLICIES}; do
        [ "$(policy_env "${policy}")" = "__INVALID__" ] && { echo "unknown policy ${policy}" >&2; exit 1; }
        [ "$(net_graph "${net}")" = "__INVALID__" ] && { echo "unknown net ${net}" >&2; exit 1; }
        run_one "${net}" "${policy}" || {
            echo "Aborting sweep: board needs a reboot before results are trustworthy." >&2
            column -s, -t "${SUMMARY}"
            exit 1
        }
    done
done

echo "=== ABLATION COMPLETE ==="
column -s, -t "${SUMMARY}"
echo "Artifacts: ${OUT_DIR}"
