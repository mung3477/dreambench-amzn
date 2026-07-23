#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCHMARK_BASE_DIR="${BASE_DIR}/detail_benchmark"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <variant> <destination>

Copies a dataset variant from ${BENCHMARK_BASE_DIR}/<variant> to the specified destination.
Overwrites existing contents if the destination directory already exists.

Options:
  -n, --dry-run    Perform a dry run without copying files.
  -h, --help       Show this help message and exit.

Examples:
  $(basename "$0") v1 /path/to/destination
  $(basename "$0") --dry-run v1 /path/to/destination
EOF
    exit 1
}

DRY_RUN=false
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Error: Unknown option $1" >&2
            usage
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#POSITIONAL_ARGS[@]} -ne 2 ]]; then
    echo "Error: Invalid number of arguments." >&2
    usage
fi

VARIANT="${POSITIONAL_ARGS[0]}"
DEST="${POSITIONAL_ARGS[1]}"

SOURCE="${BENCHMARK_BASE_DIR}/${VARIANT}"

if [[ ! -d "${SOURCE}" ]]; then
    echo "Error: Source dataset directory does not exist: ${SOURCE}" >&2
    exit 1
fi

echo "Source:      ${SOURCE}"
echo "Destination: ${DEST}"
if [[ "${DRY_RUN}" == true ]]; then
    echo "Mode:        DRY RUN (no changes will be made)"
else
    echo "Mode:        LIVE (destination contents will be overwritten)"
fi

RSYNC_FLAGS=("-a" "--delete")

if [[ "${DRY_RUN}" == true ]]; then
    RSYNC_FLAGS+=("--dry-run" "-v")
fi

if [[ "${DRY_RUN}" != true ]]; then
    mkdir -p "${DEST}"
fi

echo "Running copy operation..."
rsync "${RSYNC_FLAGS[@]}" "${SOURCE}/" "${DEST}/"

echo "Completed successfully!"
