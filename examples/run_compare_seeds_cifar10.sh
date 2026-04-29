#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NUM_CLIENTS="${NUM_CLIENTS:-10}"
export CLIENT_CONFIG="${CLIENT_CONFIG:-config/client_5_cifar10.yaml}"
export SERVER_FEDAVG_CONFIG="${SERVER_FEDAVG_CONFIG:-config/server_fedavg_cifar10.yaml}"
export SERVER_FEDCOMPASS_CONFIG="${SERVER_FEDCOMPASS_CONFIG:-config/server_fedcompass_cifar10.yaml}"
export FEDAVG_DIR="${FEDAVG_DIR:-normal-${NUM_CLIENTS}-fedavg-cifar10-byclass}"
export FEDCOMPASS_DIR="${FEDCOMPASS_DIR:-normal-${NUM_CLIENTS}-fedcompass-cifar10-byclass}"

exec "${SCRIPT_DIR}/run_compare_seeds.sh" "$@"
