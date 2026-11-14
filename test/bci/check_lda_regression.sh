#!/bin/sh
# check_lda_regression.sh [baseline.dylib]
# Builds the variant B lib (host + android-arm64) from the user's preprocessing.py with the current typhon,
# checks exports/ELF, then runs check_lda_regression.py (features/labels/proba/calib, bit-for-bit vs baseline).
set -e
T=$(cd "$(dirname "$0")/../.." && pwd)
TY=${TYPHON:-$T/target/release/typhon}
PRE=${BCI_PREPROCESSING_PY:-/Users/mind/orca/codon/experiments/bci-research-lab/research/src/models/log_power_lda_model/preprocessing.py}
W=${W:-/Users/mind/.jcode/scratch/lda-regression/guard}
PY=${PY:-/Users/mind/.jcode/scratch/.venv/bin/python}
RE=${READELF:-/opt/homebrew/opt/llvm@22/bin/llvm-readelf}
BASE=${1:-/Users/mind/.jcode/scratch/lda-regression/libbci_base.dylib}
mkdir -p "$W"
TYPHON=$TY sh "$T/examples/bci_flutter/native/build.sh" "$PRE" "$W/libbci.dylib"
TYPHON=$TY sh "$T/examples/bci_flutter/native/build.sh" "$PRE" "$W/libbci.so" -target android-arm64
n=$(nm -gU "$W/libbci.dylib" | grep -c ' _bci_'); echo "host bci exports: $n"; [ "$n" = 17 ]
n=$("$RE" --dyn-syms "$W/libbci.so" | grep -c ' bci_'); echo "android bci exports: $n"; [ "$n" = 17 ]
"$RE" -l "$W/libbci.so" | awk '$1=="LOAD"{print "LOAD align", $NF; if ($NF!="0x4000") bad=1} END{exit bad}'
need=$("$RE" -d "$W/libbci.so" | grep NEEDED | sed 's/.*\[\(.*\)\]/\1/' | sort | tr '\n' ' '); echo "NEEDED: $need"
[ "$need" = "libc.so libdl.so libm.so " ]
ls -l "$W/libbci.dylib" "$W/libbci.so"
if [ -f "$BASE" ]; then set -- "$W/libbci.dylib" "$BASE"; else set -- "$W/libbci.dylib"; fi
PYTHONPATH=${ORTPKG:-/Users/mind/.jcode/scratch/bci-flutter/ortpkg} "$PY" "$T/test/bci/check_lda_regression.py" "$@"
