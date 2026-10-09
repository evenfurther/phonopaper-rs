#!/usr/bin/env bash
# Safely refresh the library's embedded model from a completed training run.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

ARTIFACTS=$(absolute_from_ml "${ARTIFACTS:-artifacts-local}")
SOURCE_MODEL="$ML_ROOT/phonopaper-train/src/model.rs"
DESTINATION="$REPO_ROOT/phonopaper-rs/src/decode/nn"

for file in model.bpk model.json; do
    [[ -s "$ARTIFACTS/$file" ]] || {
        echo "error: missing or empty $ARTIFACTS/$file; finish training/export first" >&2
        exit 1
    }
done
[[ -s "$SOURCE_MODEL" ]] || { echo "error: missing $SOURCE_MODEL" >&2; exit 1; }
[[ -d "$DESTINATION" ]] || { echo "error: missing destination $DESTINATION" >&2; exit 1; }

stage=$(mktemp -d "$DESTINATION/.embed-model.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT
cp -- "$ARTIFACTS/model.bpk" "$stage/model.bpk"
cp -- "$ARTIFACTS/model.json" "$stage/model.json"
cp -- "$SOURCE_MODEL" "$stage/model.rs"

# Validate the staged architecture/weights through the repository's embedded-model test.
backup=$(mktemp -d "$DESTINATION/.embed-backup.XXXXXX")
trap 'rm -rf -- "$stage" "$backup"' EXIT
for file in model.bpk model.json model.rs; do
    cp -- "$DESTINATION/$file" "$backup/$file"
done
restore() {
    for file in model.bpk model.json model.rs; do
        cp -- "$backup/$file" "$DESTINATION/$file"
    done
}
trap 'restore; rm -rf -- "$stage" "$backup"' ERR INT TERM
for file in model.bpk model.json model.rs; do
    cp -- "$stage/$file" "$DESTINATION/$file"
done

(
    cd -- "$ML_ROOT"
    cargo test -p phonopaper-train --test embedded
)

trap - ERR INT TERM
rm -rf -- "$stage" "$backup"
trap - EXIT
echo "== embedded and validated: $ARTIFACTS/{model.bpk,model.json} and model.rs"
