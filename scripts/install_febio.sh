#!/usr/bin/env bash
# Build FEBio 4 from source in user space (no root required).
#
# Usage:
#   bash scripts/install_febio.sh                 # default tag, prefix ~/opt/febio
#   FEBIO_TAG=v4.13 FEBIO_PREFIX=$HOME/opt/febio JOBS=4 bash scripts/install_febio.sh
#
# Requirements: git, a C++17 compiler (gcc >= 9), CMake >= 3.10 (if the system
# CMake is missing or too old, `pip install cmake` provides one).
#
# Optional third-party libraries (Intel MKL/Pardiso, HYPRE, MMG, levmar, NLopt,
# FFTW) are disabled. Without MKL, FEBio uses its built-in skyline direct
# solver, which is exact but slower and memory-hungry on large models; it is
# appropriate for the small/medium cross-verification meshes used by craniofea.
#
# After a successful build the executable is at $FEBIO_PREFIX/bin/febio4.
# Point craniofea at it with `export FEBIO=$FEBIO_PREFIX/bin/febio4` or put
# $FEBIO_PREFIX/bin on PATH.
set -euo pipefail

FEBIO_TAG="${FEBIO_TAG:-v4.13}"
FEBIO_PREFIX="${FEBIO_PREFIX:-$HOME/opt/febio}"
SRC_DIR="${FEBIO_SRC:-$HOME/src/FEBio}"
BUILD_DIR="${SRC_DIR}/cbuild-release"
JOBS="${JOBS:-4}"

for tool in git g++ cmake make; do
  command -v "$tool" >/dev/null || { echo "error: '$tool' not found on PATH" >&2; exit 1; }
done

if [ ! -d "$SRC_DIR/.git" ]; then
  git clone --depth 1 --branch "$FEBIO_TAG" https://github.com/febiosoftware/FEBio.git "$SRC_DIR"
fi

cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_MKL=OFF -DUSE_HYPRE=OFF -DUSE_MMG=OFF -DUSE_LEVMAR=OFF -DUSE_NLOPT=OFF \
  -DUSE_FFTW=OFF -DUSE_SUPERLU_MT=OFF -DUSE_PDL=OFF -DUSE_ZLIB=OFF \
  -DUSE_STATIC_STDLIBS=OFF -DSET_DEVCOMMIT=OFF \
  -DCMAKE_SHARED_LINKER_FLAGS=-fopenmp -DCMAKE_EXE_LINKER_FLAGS=-fopenmp
# FEBio compiles every library with -fopenmp but, without MKL, only links the
# OpenMP runtime into some of them; the explicit linker flags above fix the
# resulting "undefined reference to GOMP_*" errors when linking febio4.

cmake --build "$BUILD_DIR" --target febio4 -j "$JOBS"

mkdir -p "$FEBIO_PREFIX/bin" "$FEBIO_PREFIX/lib"
cp "$BUILD_DIR/bin/febio4" "$FEBIO_PREFIX/bin/"
# Shared FEBio libraries (if built as .so) must sit next to the executable or on the rpath.
find "$BUILD_DIR/lib" -maxdepth 1 -name '*.so*' -exec cp -P {} "$FEBIO_PREFIX/lib/" \; || true
if [ -f "$BUILD_DIR/bin/febio.xml" ]; then cp "$BUILD_DIR/bin/febio.xml" "$FEBIO_PREFIX/bin/"; fi

if ldd "$FEBIO_PREFIX/bin/febio4" | grep -q "not found"; then
  echo "error: unresolved shared libraries:" >&2
  ldd "$FEBIO_PREFIX/bin/febio4" | grep "not found" >&2
  exit 1
fi
echo "FEBio installed: $FEBIO_PREFIX/bin/febio4"
echo "Run models with: $FEBIO_PREFIX/bin/febio4 -i model.feb -silent"
