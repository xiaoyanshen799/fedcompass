#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/launch_clients_systemd.sh" \
    --config "config/client_5_cifar10.yaml" \
    "$@"
