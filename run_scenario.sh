#!/bin/bash

# Usage:
#   ./machina-reinstall-sim-rerun.sh [--scenario <scenario_directory>] [additional arguments...]
#
# Description:
#   This script re-installs machina and runs the scenario based on the provided '--scenario' argument.
#   If '--scenario' is not specified, it defaults to "scenarios/livesim/hrr". Any additional
#   arguments are passed on to the 'machina/scripts/dev/kind-local-agent-reinstall.sh' script.
#   If no additional arguments are provided, it defaults to "--values dev/local-dev-values.yaml --values db/postgres.yaml"
#   If the MACHINA_DIR environment variable is set, that value will be used as the machina directory.
#   Otherwise, it defaults to /home/ubuntu/machina.
#
# Example Usages:
#   1. Use the default scenario and no additional arguments:
#         ./machina-reinstall-sim-rerun.sh
#
#   2. Specify a custom scenario directory (relative to machina-sim directory):
#         ./machina-reinstall-sim-rerun.sh --scenario scenarios/e2e/periodic_revisit_tle
#
#   3. Specify a custom scenario directory and pass arguments to the reinstall script:
#         ./machina-reinstall-sim-rerun.sh --scenario scenarios/e2e/periodic_revisit_tle --values dev/dev-il2.yaml --values db/postgres.yaml
#
#   4. Pass arguments to the reinstall script while using the default scenario:
#         ./machina-reinstall-sim-rerun.sh --values dev/dev-il5.yaml
#
#   5. Specify the machina directory using the --machina-dir argument or MACHINA_DIR env var:
#         ./machina-reinstall-sim-rerun.sh --machina-dir /path/to/machina
#

# Path to machina dir
MACHINA_DIR=${MACHINA_DIR:-/home/ubuntu/machina}

# Default scenario directory (relative to machina-sim)
SCENARIO_DIR="scenarios/livesim/hrr"

# Array to collect additional arguments for the reinstall script
REINSTALL_ARGS=()

# Process command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --scenario)
            if [[ -n "$2" ]]; then
                SCENARIO_DIR="$2"
                shift 2
            else
                echo "Error: --scenario requires an argument."
                exit 1
            fi
            ;;
        --machina-dir)
            if [[ -n "$2" ]]; then
                MACHINA_DIR="$2"
                shift 2
            else
                echo "Error: --machina-dir requires an argument."
                exit 1
            fi
            ;;
        *)
            REINSTALL_ARGS+=("$1")
            shift
            ;;
    esac
done


if [ ${#REINSTALL_ARGS[@]} -eq 0 ]; then
    REINSTALL_ARGS=(--values /home/ubuntu/machina/ml.yaml --values db/postgres.yaml)
fi


# The machina-sim project root directory
MACHINA_SIM_DIR=/home/ubuntu/machina-sim

# Echo info with color
function info() {
    echo -e "\033[1;32m$1\033[0m"
}

# Echo message with dim color
function echodim() {
    echo -e "\033[2m$1\033[0m"
}

function kube_wait_deployment_log_shows() {
    if [ "$#" -ne 2 ]; then
        echo "Usage: kube_wait_deployment_log_shows <deployment_prefix> <search_string>"
        return 1
    fi  

    local prefix="$1"
    local search="$2"

    # List deployments in the 'machina' namespace matching the prefix.
    local deployments
    deployments=$(kubectl -n machina get deployments -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | grep "^${prefix}")
    
    if [ -z "$deployments" ]; then
        echo "No deployments found with prefix '${prefix}'."
        exit 1
    fi

    # For each matching deployment, wait until the log contains the search string.
    for deployment in $deployments; do
        echodim "Waiting for deployment '${deployment}' to show log containing '${search}'"
        while true; do
            if kubectl -n machina logs deployment/"${deployment}" | grep -q "${search}"; then
                echodim "Found string '${search}' in deployment '${deployment}' logs."
                break
            fi  
            sleep 1
        done
    done
}

# Patch ui service to set NodePort 30003
kubectl patch -n machina service machina-marimo-ui --patch-file /home/ubuntu/patch-service-machina-marimo-ui.yaml

export UV_INDEX_NEXUS_USERNAME=marco.kobayashi
export UV_INDEX_NEXUS_USERNAME=807b52b654e3d670421d6a5a73e96cc9

# Run the scenario using the provided scenario directory
cd "$MACHINA_SIM_DIR" && uv run machina-lab run "$SCENARIO_DIR"
