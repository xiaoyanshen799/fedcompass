#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR_DEFAULT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_DEFAULT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)/.venv/bin/python"

NUM_CLIENTS=10
CLIENT_CONFIG="config/client_1.yaml"
WORKDIR="${WORKDIR_DEFAULT}"
PYTHON_BIN="${PYTHON_DEFAULT}"
UNIT_PREFIX="fedcompass-client"
DEFAULT_CPU_QUOTA="50%"
USE_SUDO=1
DRY_RUN=0
LOGGING_OUTPUT_DIR=""
LOGGING_OUTPUT_FILENAME=""
DATA_OUTPUT_DIR=""
DATA_OUTPUT_FILENAME=""
SAVE_MANIFEST=""

# Optional per-client overrides.
# Leave these arrays empty to use the defaults for every client.
# Example:
# CLIENT_CPU_QUOTAS=("30%" "30%" "50%")
# CLIENT_CPU_AFFINITIES=("0" "1" "2-3")
# CLIENT_CPU_QUOTAS=("45%","45%","45%","48%","50%","52%","54%","56%","58%","60%","62%","64%","66%","68%","70%","72%","74%","76%","78%","80%")
CLIENT_CPU_AFFINITIES=("1","2","3","4","5","6","7","8","9","10")
CLIENT_CPU_QUOTAS=("60%","60%","60%","60%","60%","60%","60%","60%","60%","60%")
normalize_csv_array() {
    local array_name="$1"
    declare -n array_ref="${array_name}"

    if (( ${#array_ref[@]} == 1 )) && [[ "${array_ref[0]}" == *,* ]]; then
        local joined="${array_ref[0]}"
        joined="${joined//, /,}"
        IFS=',' read -r -a array_ref <<< "${joined}"
    fi
}

normalize_cpu_quota_value() {
    local value="$1"
    value="${value// /}"
    if [[ -z "${value}" ]]; then
        echo ""
        return
    fi
    local numeric="${value%\%}"
    local integer_part="${numeric%%.*}"
    if [[ -z "${integer_part}" ]]; then
        integer_part="0"
    fi
    if [[ "${integer_part}" -lt 1 ]]; then
        integer_part="1"
    fi
    echo "${integer_part}%"
}

normalize_cpu_quota_array() {
    local array_name="$1"
    declare -n array_ref="${array_name}"
    local normalized=()
    local item
    for item in "${array_ref[@]}"; do
        normalized+=("$(normalize_cpu_quota_value "${item}")")
    done
    array_ref=("${normalized[@]}")
}

usage() {
    cat <<'EOF'
Usage:
  launch_clients_systemd.sh [options]

Options:
  --num-clients N       Number of gRPC clients to launch. Default: 20
  --config PATH         Client config path relative to examples/. Default: config/client_1.yaml
  --workdir PATH        Working directory for systemd-run. Default: repo/examples
  --python PATH         Python binary. Default: repo/.venv/bin/python
  --unit-prefix PREFIX  Prefix for transient systemd unit names. Default: fedcompass-client
  --cpu-quota PERCENT   Default CPUQuota for clients without an explicit override. Default: 50%
  --cpu-quotas CSV      Comma-separated per-client CPUQuota overrides for this run.
  --logging-output-dir PATH
                        Override client log output directory for this run.
  --logging-output-filename NAME
                        Override client log output base filename for this run.
  --data-output-dir PATH
                        Override dataset visualization output directory for this run.
  --data-output-filename NAME
                        Override dataset visualization output filename for this run.
  --save-manifest PATH  Save launched client quota/affinity metadata to CSV.
  --no-sudo             Call systemd-run directly instead of sudo systemd-run
  --dry-run             Print commands without launching
  -h, --help            Show this message

Notes:
  1. This script launches detached transient services on purpose.
     `--wait --pipe` is not used because it blocks per client and is not suitable
     for starting 20 clients concurrently from one command.
  2. To change CPU quota or CPU affinity for specific clients, edit the
     CLIENT_CPU_QUOTAS / CLIENT_CPU_AFFINITIES arrays near the top of this script.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-clients)
            NUM_CLIENTS="$2"
            shift 2
            ;;
        --config)
            CLIENT_CONFIG="$2"
            shift 2
            ;;
        --workdir)
            WORKDIR="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --unit-prefix)
            UNIT_PREFIX="$2"
            shift 2
            ;;
        --cpu-quota)
            DEFAULT_CPU_QUOTA="$2"
            shift 2
            ;;
        --cpu-quotas)
            CLIENT_CPU_QUOTAS=("$2")
            shift 2
            ;;
        --logging-output-dir)
            LOGGING_OUTPUT_DIR="$2"
            shift 2
            ;;
        --logging-output-filename)
            LOGGING_OUTPUT_FILENAME="$2"
            shift 2
            ;;
        --data-output-dir)
            DATA_OUTPUT_DIR="$2"
            shift 2
            ;;
        --data-output-filename)
            DATA_OUTPUT_FILENAME="$2"
            shift 2
            ;;
        --save-manifest)
            SAVE_MANIFEST="$2"
            shift 2
            ;;
        --no-sudo)
            USE_SUDO=0
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -d "${WORKDIR}" ]]; then
    echo "Working directory does not exist: ${WORKDIR}" >&2
    exit 1
fi

if [[ ! -f "${WORKDIR}/${CLIENT_CONFIG}" && ! -f "${CLIENT_CONFIG}" ]]; then
    echo "Client config does not exist: ${CLIENT_CONFIG}" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python binary is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

if ! command -v systemd-run >/dev/null 2>&1; then
    echo "systemd-run is not available in PATH." >&2
    exit 1
fi

normalize_csv_array CLIENT_CPU_QUOTAS
normalize_csv_array CLIENT_CPU_AFFINITIES
DEFAULT_CPU_QUOTA="$(normalize_cpu_quota_value "${DEFAULT_CPU_QUOTA}")"
normalize_cpu_quota_array CLIENT_CPU_QUOTAS

if (( ${#CLIENT_CPU_QUOTAS[@]} > 0 && ${#CLIENT_CPU_QUOTAS[@]} < NUM_CLIENTS )); then
    echo "CLIENT_CPU_QUOTAS must be empty or contain at least ${NUM_CLIENTS} entries." >&2
    exit 1
fi

if (( ${#CLIENT_CPU_AFFINITIES[@]} > 0 && ${#CLIENT_CPU_AFFINITIES[@]} < NUM_CLIENTS )); then
    echo "CLIENT_CPU_AFFINITIES must be empty or contain at least ${NUM_CLIENTS} entries." >&2
    exit 1
fi

CPU_COUNT="$(nproc)"
SYSTEMD_RUN_CMD=(systemd-run)
if (( USE_SUDO )); then
    SYSTEMD_RUN_CMD=(sudo systemd-run)
fi

echo "Launching ${NUM_CLIENTS} clients from ${WORKDIR}/${CLIENT_CONFIG}"
echo "Python: ${PYTHON_BIN}"
echo "Working directory: ${WORKDIR}"

if [[ -n "${LOGGING_OUTPUT_DIR}" ]]; then
    mkdir -p "${LOGGING_OUTPUT_DIR}"
fi
if [[ -n "${DATA_OUTPUT_DIR}" ]]; then
    mkdir -p "${DATA_OUTPUT_DIR}"
fi
if [[ -z "${SAVE_MANIFEST}" && -n "${LOGGING_OUTPUT_DIR}" ]]; then
    SAVE_MANIFEST="${LOGGING_OUTPUT_DIR}/client_launch_manifest.csv"
fi
if [[ -n "${SAVE_MANIFEST}" ]]; then
    mkdir -p "$(dirname "${SAVE_MANIFEST}")"
    printf 'client_id,unit_name,cpu_quota,cpu_affinity,config,workdir,logging_output_dir,logging_output_filename\n' > "${SAVE_MANIFEST}"
fi

for ((client_id = 0; client_id < NUM_CLIENTS; client_id++)); do
    unit_name="${UNIT_PREFIX}-${client_id}"
    cpu_quota="${DEFAULT_CPU_QUOTA}"
    cpu_affinity="$((client_id % CPU_COUNT))"

    if (( ${#CLIENT_CPU_QUOTAS[@]} > 0 )); then
        cpu_quota="${CLIENT_CPU_QUOTAS[client_id]}"
    fi
    if (( ${#CLIENT_CPU_AFFINITIES[@]} > 0 )); then
        cpu_affinity="${CLIENT_CPU_AFFINITIES[client_id]}"
    fi

    cmd=(
        "${SYSTEMD_RUN_CMD[@]}"
        --unit "${unit_name}"
        --collect
        --service-type=exec
        -p "CPUQuota=${cpu_quota}"
        -p "WorkingDirectory=${WORKDIR}"
        env
        OMP_NUM_THREADS=1
        MKL_NUM_THREADS=1
        OPENBLAS_NUM_THREADS=1
        NUMEXPR_NUM_THREADS=1
        taskset -c "${cpu_affinity}"
        "${PYTHON_BIN}"
        grpc/run_client.py
        --config "${CLIENT_CONFIG}"
        --client-id "${client_id}"
        --num-clients "${NUM_CLIENTS}"
    )

    if [[ -n "${LOGGING_OUTPUT_DIR}" ]]; then
        cmd+=(--logging-output-dir "${LOGGING_OUTPUT_DIR}")
    fi
    if [[ -n "${LOGGING_OUTPUT_FILENAME}" ]]; then
        cmd+=(--logging-output-filename "${LOGGING_OUTPUT_FILENAME}")
    fi
    if [[ -n "${DATA_OUTPUT_DIR}" ]]; then
        cmd+=(--data-output-dir "${DATA_OUTPUT_DIR}")
    fi
    if [[ -n "${DATA_OUTPUT_FILENAME}" ]]; then
        cmd+=(--data-output-filename "${DATA_OUTPUT_FILENAME}")
    fi

    if (( DRY_RUN )); then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        continue
    fi

    printf 'Launching %s with CPUQuota=%s CPU=%s\n' "${unit_name}" "${cpu_quota}" "${cpu_affinity}"
    "${cmd[@]}"
    if [[ -n "${SAVE_MANIFEST}" ]]; then
        printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$((client_id + 1))" \
            "${unit_name}" \
            "${cpu_quota}" \
            "${cpu_affinity}" \
            "${CLIENT_CONFIG}" \
            "${WORKDIR}" \
            "${LOGGING_OUTPUT_DIR}" \
            "${LOGGING_OUTPUT_FILENAME}" \
            >> "${SAVE_MANIFEST}"
    fi
done

cat <<EOF
Submitted ${NUM_CLIENTS} transient services.

Useful commands:
  systemctl status ${UNIT_PREFIX}-0
  journalctl -u ${UNIT_PREFIX}-0 -f
  systemctl list-units '${UNIT_PREFIX}-*'
EOF
