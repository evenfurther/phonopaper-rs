#!/usr/bin/env bash
# Shared helpers for PhonoPaper ML automation. Source this file; do not execute it.

set -euo pipefail

SCRIPT_LIB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ML_ROOT=$(cd -- "$SCRIPT_LIB_DIR/.." && pwd -P)
REPO_ROOT=$(cd -- "$ML_ROOT/.." && pwd -P)
PYTHON=${PYTHON:-python3}

absolute_from_ml() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$ML_ROOT" "$1" ;;
    esac
}

# Succeeds only for the dataset contract required by the current trainer.
dataset_is_compatible() {
    local dataset=$1
    local source_scale=$2
    "$PYTHON" - "$dataset/manifest.json" "$source_scale" <<'PY'
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
expected_scale = int(sys.argv[2])
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)

if manifest.get("format_version") != 2:
    raise SystemExit(1)
config = manifest.get("config")
if not isinstance(config, dict) or config.get("source_scale") != expected_scale:
    raise SystemExit(1)
if not (manifest_path.parent / "labels.csv").is_file():
    raise SystemExit(1)
PY
}

prepare_dataset() {
    local dataset=$1
    local count=$2
    local source_scale=$3
    local size=$4
    local seed=$5
    local positive_ratio=$6
    shift 6
    local -a generator=("$@")

    if dataset_is_compatible "$dataset" "$source_scale"; then
        echo "== reusing compatible v2 dataset: $dataset (source_scale=$source_scale)"
        return
    fi

    if [[ -e "$dataset" ]]; then
        [[ -n "$dataset" && "$dataset" != "/" && "$dataset" != "$ML_ROOT" ]] || {
            echo "error: refusing to replace unsafe dataset path '$dataset'" >&2
            exit 2
        }
        if [[ ! -f "$dataset/manifest.json" && ! -f "$dataset/labels.csv" ]]; then
            echo "error: refusing to replace non-dataset path '$dataset'" >&2
            echo "       remove it explicitly or choose DATASET=another-directory" >&2
            exit 2
        fi
        echo "== dataset is incompatible (requires format v2, source_scale=$source_scale)" >&2
        echo "== removing and rebuilding recognized dataset: $dataset" >&2
        rm -rf -- "$dataset"
    fi

    mkdir -p -- "$(dirname -- "$dataset")"
    "${generator[@]}" --output "$dataset" --count "$count" --size "$size" \
        --seed "$seed" --positive-ratio "$positive_ratio" --source-scale "$source_scale"
    dataset_is_compatible "$dataset" "$source_scale" || {
        echo "error: generated dataset failed compatibility validation: $dataset" >&2
        exit 1
    }
}
