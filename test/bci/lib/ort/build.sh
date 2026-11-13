#!/bin/sh
# Usage: build.sh <preprocessing.py> <ort_dir> <out lib> [extra typhon args]
# ort_dir: ORT package root with lib/ (macOS/iOS: linked). For Android (dlopen) pass "-" as ort_dir.
set -e
SRC=$1; ORT=$2; OUT=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
B=$(mktemp -d)
cp "$HERE/bci_model.py" "$HERE/ort_api.py" "$B/"
cp "$SRC" "$B/preprocessing.py"
TY=${TYPHON:-typhon}
if [ "$ORT" = "-" ]; then
  "$TY" build -release -lib -o "$OUT" "$@" "$B/bci_model.py"
else
  "$TY" build -release -lib -o "$OUT" "$@" "$B/bci_model.py" \
    "-linker-flags=-L$ORT/lib -lonnxruntime -Wl,-rpath,$ORT/lib"
fi
rm -rf "$B"
