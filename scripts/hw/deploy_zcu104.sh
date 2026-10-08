#!/usr/bin/env bash
# Deploy a DeepTile TargetPackage or complete SD-card package to the ZCU104.
# The remote mount and backing device are verified before any files are copied.
set -euo pipefail

# Override settings when needed, for example:
#   BOARD=my-board REMOTE_BOOT_DIR=/run/media/boot-mmcblk0p1 \
#       ./scripts/hw/deploy_zcu104.sh
BOARD="${BOARD:-zcu104}"
VITIS_MODE="${VITIS_MODE:-hw}"
REMOTE_BOOT_DIR="${REMOTE_BOOT_DIR:-/run/media/boot-mmcblk0p1}"
EXPECTED_BOOT_DEVICE="${EXPECTED_BOOT_DEVICE:-/dev/mmcblk0p1}"
TARGET_DIR="${TARGET_DIR:-}"
SD_CARD_DIR="${SD_CARD_DIR:-}"
DEPLOY_BOOT_ID_FILE=".deeptile-deploy-boot-id"

# Explicit paths avoid relying on target auto-discovery. Environment overrides
# remain supported for existing callers.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-dir|--sd-card-dir)
            if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
                echo "Error: $1 requires a directory." >&2
                exit 2
            fi
            if [[ "$1" == --target-dir ]]; then
                TARGET_DIR="$2"
            else
                SD_CARD_DIR="$2"
            fi
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--target-dir DIR | --sd-card-dir DIR]"
            echo "Verifies the boot mount, deploys the selected package, and stamps its boot ID."
            echo "Full deployments require a subsequent board reboot."
            exit 0
            ;;
        *)
            echo "Error: unknown deployment argument: $1" >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ -n "${TARGET_DIR}" && -n "${SD_CARD_DIR}" ]]; then
    echo "Error: set TARGET_DIR or SD_CARD_DIR, not both." >&2
    exit 2
fi

if [[ -z "${TARGET_DIR}" && -z "${SD_CARD_DIR}" ]]; then
    shopt -s nullglob
    manifests=("${REPO_ROOT}"/build/hw/*/target.json)
    shopt -u nullglob
    if [[ ${#manifests[@]} -ne 1 ]]; then
        echo "Error: found ${#manifests[@]} target manifests under build/hw." >&2
        echo "Set TARGET_DIR=build/hw/<target> explicitly." >&2
        exit 2
    fi
    TARGET_DIR="$(dirname -- "${manifests[0]}")"
fi

deployment_mode="runtime"
source_dir="${TARGET_DIR}"
required_files=(deeptile deeptile.xclbin target.json)

if [[ -n "${SD_CARD_DIR}" ]]; then
    deployment_mode="full"
    source_dir="${SD_CARD_DIR}"
    required_files=(BOOT.BIN Image boot.scr deeptile deeptile.xclbin target.json)
elif [[ -d "${TARGET_DIR}/sd_card_${VITIS_MODE}/sd_card" ]]; then
    deployment_mode="full"
    source_dir="${TARGET_DIR}/sd_card_${VITIS_MODE}/sd_card"
    required_files=(BOOT.BIN Image boot.scr deeptile deeptile.xclbin target.json)
fi

if [[ ! -d "${source_dir}" ]]; then
    echo "Error: deployment source not found: ${source_dir}" >&2
    exit 1
fi

for file in "${required_files[@]}"; do
    if [[ ! -f "${source_dir}/${file}" ]]; then
        echo "Error: required deployment file not found: ${source_dir}/${file}" >&2
        exit 1
    fi
done

echo "Verifying ${BOARD}:${REMOTE_BOOT_DIR} is the boot partition"
resolved_boot_dir=$(
    ssh "${BOARD}" sh -s -- "${REMOTE_BOOT_DIR}" "${EXPECTED_BOOT_DEVICE}" <<'REMOTE_CHECK'
set -eu

requested_dir=$1
expected_device=$2
resolved_dir=$(readlink -f "${requested_dir}")

test -d "${resolved_dir}"
test -w "${resolved_dir}"

mounted_device=$(awk -v path="${resolved_dir}" '$2 == path { print $1; exit }' /proc/mounts)
test "${mounted_device}" = "${expected_device}"

printf '%s\n' "${resolved_dir}"
REMOTE_CHECK
) || {
    echo "Error: refusing to deploy to ${REMOTE_BOOT_DIR}." >&2
    echo "It must be a writable mount backed by ${EXPECTED_BOOT_DEVICE}." >&2
    exit 1
}

echo "Confirmed boot partition: ${BOARD}:${resolved_boot_dir}"
if [[ "${deployment_mode}" == "full" ]]; then
    echo "Copying boot and DeepTile files from ${source_dir}"
    tar --exclude='./data' -C "${source_dir}" -cf - . |
        ssh "${BOARD}" "tar -C '${resolved_boot_dir}' -xf -"
    current_boot_id=$(ssh "${BOARD}" "cat /proc/sys/kernel/random/boot_id")
    ssh "${BOARD}" "printf '%s\n' '${current_boot_id}' > '${resolved_boot_dir}/${DEPLOY_BOOT_ID_FILE}'"
    # Remove the marker written under the project's former name, so a stale one
    # can never stand in for this deployment.
    ssh "${BOARD}" "rm -f '${resolved_boot_dir}/.tilecnn-deploy-boot-id'"
    echo "Full deployment complete. Reboot ${BOARD} to activate updated boot files."
else
    # BOOT.BIN programs this fixed platform during boot. Runtime-only updates
    # must therefore match a boot image that has already been activated.
    declared_boot="$("${PYTHON:-python3}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("boot",{}).get("sha256",""))' "${source_dir}/target.json" 2>/dev/null || true)"
    if [[ -n "${declared_boot}" ]]; then
        board_boot="$(ssh "${BOARD}" "sha256sum '${resolved_boot_dir}/BOOT.BIN' 2>/dev/null | cut -d' ' -f1" || true)"
        if [[ "${declared_boot}" != "${board_boot}" ]]; then
            echo "Error: this target's boot image is not the one on the board." >&2
            echo "  target BOOT.BIN: ${declared_boot}" >&2
            echo "  board  BOOT.BIN: ${board_boot:-<absent>}" >&2
            echo "The fabric is programmed from BOOT.BIN, so a runtime deployment" >&2
            echo "would leave the previous hardware running. Package this target" >&2
            echo "and deploy the full SD-card set, then reboot." >&2
            exit 1
        fi

        deployed_boot_id=$(ssh "${BOARD}" "cat '${resolved_boot_dir}/${DEPLOY_BOOT_ID_FILE}' 2>/dev/null" || true)
        current_boot_id=$(ssh "${BOARD}" "cat /proc/sys/kernel/random/boot_id" || true)
        if [[ -n "${deployed_boot_id}" && "${deployed_boot_id}" == "${current_boot_id}" ]]; then
            echo "Error: this boot image was installed during the current boot session." >&2
            echo "Reboot ${BOARD} before deploying or running this target." >&2
            exit 1
        fi
    fi

    echo "Copying DeepTile runtime files from ${source_dir}"
    tar -C "${source_dir}" -cf - deeptile deeptile.xclbin target.json |
        ssh "${BOARD}" "tar -C '${resolved_boot_dir}' -xf -"
    echo "Runtime deployment complete; the boot image is active with no pending reboot."
fi

ssh "${BOARD}" "sync; sha256sum '${resolved_boot_dir}/deeptile' '${resolved_boot_dir}/deeptile.xclbin' '${resolved_boot_dir}/target.json'"
