#!/usr/bin/env bash
# Run deterministic inference fixtures and optionally compare JSONL with a baseline.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

if (($# == 0)); then
    echo "usage: ARTIFACTS=artifacts-local [EXPECTED=baseline.jsonl] $0 IMAGE..." >&2
    exit 2
fi

ARTIFACTS=$(absolute_from_ml "${ARTIFACTS:-artifacts-local}")
BACKEND=${BACKEND:-flex}
THRESHOLD=${THRESHOLD:-0.5}
OUTPUT=${OUTPUT:-$ML_ROOT/infer-results.jsonl}
EXPECTED=${EXPECTED:-}

case "$BACKEND" in
    flex) FEATURE_ARGS=() ;;
    cuda | wgpu) FEATURE_ARGS=(--no-default-features --features "$BACKEND") ;;
    *) echo "error: BACKEND must be flex, cuda, or wgpu, got '$BACKEND'" >&2; exit 2 ;;
esac
[[ -s "$ARTIFACTS/model.bpk" && -s "$ARTIFACTS/model.json" ]] || {
    echo "error: $ARTIFACTS does not contain model.bpk and model.json" >&2
    exit 1
}

mkdir -p -- "$(dirname -- "$OUTPUT")"
(
    cd -- "$ML_ROOT"
    cargo run --quiet --release -p phonopaper-train "${FEATURE_ARGS[@]}" -- \
        infer --artifacts "$ARTIFACTS" --threshold "$THRESHOLD" "$@"
) | tee "$OUTPUT"

if [[ -n "$EXPECTED" ]]; then
    EXPECTED=$(absolute_from_ml "$EXPECTED")
    cmp --silent "$EXPECTED" "$OUTPUT" || {
        echo "error: inference output differs from $EXPECTED" >&2
        diff -u -- "$EXPECTED" "$OUTPUT" || true
        exit 1
    }
    echo "== inference output matches $EXPECTED"
else
    echo "== wrote $OUTPUT (set EXPECTED to compare a checked baseline)"
fi
