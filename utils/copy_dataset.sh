#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_SOURCE="${BASE_DIR}/detail_benchmark/wordy"
DEFAULT_DEST="/root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn/original"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [SOURCE_DIR] [DEST_DIR]

Copies/syncs updated dataset from SOURCE_DIR to DEST_DIR, preserving all existing *_results.json eval files.

Defaults:
  SOURCE_DIR: ${DEFAULT_SOURCE}
  DEST_DIR:   ${DEFAULT_DEST}

Options:
  -n, --dry-run    Perform a dry run without modifying any files.
  -h, --help       Show this help message and exit.

Examples:
  $(basename "$0")
  $(basename "$0") --dry-run
  $(basename "$0") /path/to/custom_source /path/to/custom_dest
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

SOURCE="${POSITIONAL_ARGS[0]:-${DEFAULT_SOURCE}}"
DEST="${POSITIONAL_ARGS[1]:-${DEFAULT_DEST}}"

if [[ ! -d "${SOURCE}" ]]; then
    echo "Error: Source dataset directory does not exist: ${SOURCE}" >&2
    exit 1
fi

echo "Source:      ${SOURCE}"
echo "Destination: ${DEST}"
echo "Exclude:     *_results.json"
if [[ "${DRY_RUN}" == true ]]; then
    echo "Mode:        DRY RUN (no changes will be made)"
else
    echo "Mode:        LIVE (destination contents will be updated, preserving *_results.json)"
fi

RSYNC_FLAGS=("-a" "--delete" "--exclude=*_results.json")

if [[ "${DRY_RUN}" == true ]]; then
    RSYNC_FLAGS+=("--dry-run" "-v")
fi

if [[ "${DRY_RUN}" != true ]]; then
    mkdir -p "${DEST}"
fi

echo "Running copy operation..."
rsync "${RSYNC_FLAGS[@]}" "${SOURCE}/" "${DEST}/"

echo "Completed successfully!"
