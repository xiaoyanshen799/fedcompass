#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXAMPLES_DIR="${REPO_ROOT}/examples"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

SEEDS=("${@}")
if (( ${#SEEDS[@]} == 0 )); then
    SEEDS=(456 789 1011)
fi

FEDAVG_DIR="normal-10-fedavg-dirichelet"
FEDCOMPASS_DIR="normal-10-fedcompass-dirichelet"
CLIENT_CONFIG="config/client_5.yaml"
SUDO_KEEPALIVE_PID=""

run_prepare() {
    local seed="$1"
    "${PYTHON_BIN}" examples/prepare_fed_compare.py \
        --seed "${seed}" \
        --client-output-fedavg "${FEDAVG_DIR}" \
        --client-output-fedcompass "${FEDCOMPASS_DIR}"
}

read_cpu_csv() {
    local seed="$1"
    local json_path="${EXAMPLES_DIR}/output/prepared_runs/generated_client_cpu_quotas_normal_seed${seed}.json"
    "${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cpu_quota_csv"])' "${json_path}"
}

stop_sudo_keepalive() {
    if [[ -n "${SUDO_KEEPALIVE_PID}" ]] && kill -0 "${SUDO_KEEPALIVE_PID}" 2>/dev/null; then
        kill "${SUDO_KEEPALIVE_PID}" 2>/dev/null || true
        wait "${SUDO_KEEPALIVE_PID}" 2>/dev/null || true
    fi
}

start_sudo_keepalive() {
    if ! command -v sudo >/dev/null 2>&1; then
        return
    fi
    if sudo -n true 2>/dev/null; then
        :
    else
        echo "Refreshing sudo credentials once for all client launches..."
        sudo -v
    fi
    (
        while true; do
            sleep 60
            sudo -n -v >/dev/null 2>&1 || exit 0
        done
    ) &
    SUDO_KEEPALIVE_PID=$!
    trap stop_sudo_keepalive EXIT
}

wait_for_target_update() {
    local csv_path="$1"
    local target_update="$2"
    local server_pid="$3"

    while true; do
        if [[ -f "${csv_path}" ]]; then
            local last_line
            last_line="$(tail -n 1 "${csv_path}" | tr -d '\r')"
            if [[ -n "${last_line}" && "${last_line}" != Global\ Update* ]]; then
                local current_update
                current_update="$(printf '%s\n' "${last_line}" | cut -d',' -f1 | tr -d ' ')"
                if [[ "${current_update}" =~ ^[0-9]+$ ]] && (( current_update >= target_update )); then
                    break
                fi
            fi
        fi

        if ! kill -0 "${server_pid}" 2>/dev/null; then
            echo "Server process ${server_pid} exited before reaching target update ${target_update}." >&2
            return 1
        fi
        sleep 10
    done
}

wait_for_client_units_to_finish() {
    local unit_prefix="$1"
    local num_clients="$2"
    local server_pid="$3"

    while true; do
        local active_units=0
        local failed_units=()
        local client_id

        for ((client_id = 0; client_id < num_clients; client_id++)); do
            local unit_name="${unit_prefix}-${client_id}"
            local state

            state="$(systemctl is-active "${unit_name}" 2>/dev/null || true)"
            case "${state}" in
                active|activating|reloading)
                    active_units=$((active_units + 1))
                    ;;
                failed)
                    failed_units+=("${unit_name}")
                    ;;
            esac
        done

        if (( ${#failed_units[@]} > 0 )); then
            printf 'Client units failed before completion: %s\n' "${failed_units[*]}" >&2
            return 1
        fi

        if (( active_units == 0 )); then
            break
        fi

        if ! kill -0 "${server_pid}" 2>/dev/null; then
            echo "Server process ${server_pid} exited while waiting for client units to finish." >&2
            return 1
        fi
        sleep 10
    done
}

stop_server() {
    local pid="$1"
    if kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
    fi
    local _attempt
    for _attempt in {1..10}; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            wait "${pid}" 2>/dev/null || true
            return
        fi
        sleep 1
    done
    if kill -0 "${pid}" 2>/dev/null; then
        kill -9 "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
}

run_one_algo() {
    local seed="$1"
    local algo_name="$2"
    local config_path="$3"
    local output_dir="$4"
    local target_updates="$5"
    local cpu_csv="$6"
    local wait_mode="$7"
    local num_clients="$8"

    local base_name="result_seed${seed}_${algo_name}"
    local server_csv="${EXAMPLES_DIR}/${output_dir}/${base_name}_server.csv"
    local server_stdout="${EXAMPLES_DIR}/${output_dir}/${base_name}_server.stdout.log"
    local manifest_path="${EXAMPLES_DIR}/${output_dir}/client_launch_manifest_seed${seed}_${algo_name}.csv"
    local vis_name="visualization_seed${seed}_${algo_name}.pdf"
    local unit_prefix="${algo_name}-seed${seed}-client"

    mkdir -p "${EXAMPLES_DIR}/${output_dir}"
    rm -f \
        "${EXAMPLES_DIR}/${output_dir}/${base_name}"_server*.csv \
        "${EXAMPLES_DIR}/${output_dir}/${base_name}"_server*.txt \
        "${EXAMPLES_DIR}/${output_dir}/${base_name}"_Client*.csv \
        "${EXAMPLES_DIR}/${output_dir}/${base_name}"_Client*.txt \
        "${EXAMPLES_DIR}/${output_dir}/${vis_name%.*}"*.pdf \
        "${server_stdout}" \
        "${manifest_path}"

    echo "=== ${algo_name} seed ${seed}: starting server ==="
    (
        cd "${EXAMPLES_DIR}"
        exec "${PYTHON_BIN}" grpc/run_server.py \
            --config "${config_path}" \
            --logging-output-dir "${output_dir}" \
            --logging-output-filename "${base_name}"
    ) > "${server_stdout}" 2>&1 &
    local server_pid=$!

    sleep 5

    echo "=== ${algo_name} seed ${seed}: launching clients ==="
    (
        cd "${EXAMPLES_DIR}"
        ./grpc/launch_clients_systemd.sh \
            --config "${CLIENT_CONFIG}" \
            --num-clients "${num_clients}" \
            --unit-prefix "${unit_prefix}" \
            --logging-output-dir "${output_dir}" \
            --logging-output-filename "${base_name}" \
            --data-output-dir "${output_dir}" \
            --data-output-filename "${vis_name}" \
            --save-manifest "${manifest_path}" \
            --cpu-quotas "${cpu_csv}"
    )

    if [[ "${wait_mode}" == "client-units" ]]; then
        echo "=== ${algo_name} seed ${seed}: waiting for client units to finish ==="
        wait_for_client_units_to_finish "${unit_prefix}" "${num_clients}" "${server_pid}"
    else
        echo "=== ${algo_name} seed ${seed}: waiting for update ${target_updates} ==="
        wait_for_target_update "${server_csv}" "${target_updates}" "${server_pid}"
    fi

    echo "=== ${algo_name} seed ${seed}: target reached, stopping server ==="
    stop_server "${server_pid}"
    sleep 5
}

main() {
    cd "${REPO_ROOT}"
    local num_clients=10
    start_sudo_keepalive
    for seed in "${SEEDS[@]}"; do
        echo "=============================="
        echo "Preparing seed ${seed}"
        echo "=============================="
        run_prepare "${seed}"
        cpu_csv="$(read_cpu_csv "${seed}")"
        echo "CPU quotas for seed ${seed}: ${cpu_csv}"

        run_one_algo "${seed}" "fedavg" "config/server_fedavg.yaml" "${FEDAVG_DIR}" 50 "${cpu_csv}" "target-update" "${num_clients}"
        run_one_algo "${seed}" "fedcompass" "config/server_fedcompass.yaml" "${FEDCOMPASS_DIR}" 600 "${cpu_csv}" "client-units" "${num_clients}"
    done

    echo "All requested seeds finished."
}

main "$@"
