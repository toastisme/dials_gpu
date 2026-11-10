#!/usr/bin/env bash
# Build dials_gpu from source: compile the two C++ extensions and install the
# package (editable) into the active DIALS Python environment.
#
# Prerequisites: DIALS and a CUDA build of PyTorch are already installed, and the
# DIALS environment is active (e.g. `source /path/to/dials/dials`), or PYTHON
# points at its interpreter.
#
# Usage:
#   ./build.sh                 # compile extensions + pip install -e .
#   ./build.sh --ext-only      # compile extensions only
#   PYTHON=/path/to/python CXX=g++ CXXFLAGS="-O2" ./build.sh
set -euo pipefail

EXT_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --ext-only) EXT_ONLY=1 ;;
        -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- Locate the DIALS Python environment ---------------------------------------
if [[ -z "${PYTHON:-}" ]]; then
    for cand in libtbx.python python3 python; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import dials' >/dev/null 2>&1; then
            PYTHON="$(command -v "$cand")"
            break
        fi
    done
fi
if [[ -z "${PYTHON:-}" ]]; then
    echo "error: no Python with DIALS found. Activate DIALS first (source /path/to/dials/dials) or set PYTHON." >&2
    exit 1
fi

# Collect the paths we need from the environment itself.
eval "$("$PYTHON" - <<'EOF'
import os, sys, sysconfig, importlib.util
import dials, dxtbx, scitbx
def q(s): return "'" + s.replace("'", "'\\''") + "'"
print("PREFIX=" + q(sys.prefix))
print("PYINC=" + q(sysconfig.get_paths()["include"]))
print("PYVER=" + q(f"{sys.version_info.major}{sys.version_info.minor}"))
# .../modules/dials/src/dials/__init__.py -> .../modules/dials/src
print("DIALS_SRC=" + q(os.path.dirname(os.path.dirname(dials.__file__))))
print("DXTBX_SRC=" + q(os.path.dirname(os.path.dirname(dxtbx.__file__))))
# cctbx/scitbx headers ship inside site-packages
print("CCTBX_INC=" + q(os.path.dirname(os.path.dirname(scitbx.__file__))))
print("HAVE_TORCH=" + ("1" if importlib.util.find_spec("torch") else "0"))
EOF
)"

LIBDIR="$PREFIX/lib"
if [[ -z "${CXX:-}" ]]; then
    CXX="$(ls "$PREFIX"/bin/*-linux-gnu-c++ 2>/dev/null | head -n1 || true)"
    [[ -n "$CXX" ]] || CXX="$(command -v c++ || command -v g++)"
fi
CXXFLAGS="${CXXFLAGS:--O3 -march=native}"

echo "Python   : $PYTHON"
echo "Prefix   : $PREFIX"
echo "DIALS    : $DIALS_SRC"
echo "Compiler : $CXX"
[[ "$HAVE_TORCH" == 1 ]] || echo "warning: torch is not importable from $PYTHON; dials_gpu will not run until it is installed." >&2

for f in "$PREFIX/include/libdeflate.h" "$PREFIX/include/boost/python.hpp" \
         "$CCTBX_INC/cctbx/miller.h" "$LIBDIR/libboost_python$PYVER.so"; do
    [[ -e "$f" ]] || { echo "error: required file not found: $f" >&2; exit 1; }
done

# Build into a temp dir and move into place only on success, so a failed build
# never leaves a broken .so behind.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

COMMON=(-std=c++17 -fPIC -shared -pthread $CXXFLAGS)
RPATH=(-L"$LIBDIR" -Wl,-rpath,"$LIBDIR")

echo "==> Building spot_finding/_cbf_read_ext.so"
"$CXX" "${COMMON[@]}" -isystem "$PREFIX/include" \
    src/dials_gpu/spot_finding/_cbf_read.cpp \
    "${RPATH[@]}" -ldeflate -lz \
    -o "$TMP/_cbf_read_ext.so"

echo "==> Building integration/_shoebox_pack_ext.so (this takes a minute)"
"$CXX" "${COMMON[@]}" \
    -DBOOST_PYTHON_DYN_LINK -DBOOST_PYTHON_NO_LIB \
    -DBOOST_BIND_GLOBAL_PLACEHOLDERS -DBOOST_ALLOW_DEPRECATED_HEADERS \
    -I"$DIALS_SRC" -isystem "$DXTBX_SRC" -isystem "$CCTBX_INC" \
    -isystem "$PREFIX/include" -isystem "$PYINC" \
    src/dials_gpu/integration/_shoebox_pack.cpp \
    "${RPATH[@]}" -lboost_python"$PYVER" -lcctbx -lscitbx_boost_python \
    -o "$TMP/_shoebox_pack_ext.so"

mv -f "$TMP/_cbf_read_ext.so" src/dials_gpu/spot_finding/_cbf_read_ext.so
mv -f "$TMP/_shoebox_pack_ext.so" src/dials_gpu/integration/_shoebox_pack_ext.so

echo "==> Checking extensions load"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - <<'EOF'
import ctypes, os
import dials.array_family.flex  # registers the Boost.Python converters the extension needs
import dials_gpu.spot_finding as sf
from dials_gpu.integration import _shoebox_pack_ext
ctypes.CDLL(os.path.join(os.path.dirname(sf.__file__), "_cbf_read_ext.so")).read_cbf_batch_omp
print("    _cbf_read_ext and _shoebox_pack_ext OK")
EOF

if [[ "$EXT_ONLY" == 0 ]]; then
    echo "==> Installing dials_gpu (editable)"
    "$PYTHON" -m pip install --no-deps --no-build-isolation -e "$ROOT"
fi

echo "Done."
