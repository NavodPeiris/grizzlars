#!/usr/bin/env bash
set -euo pipefail

echo "[vendor_tbb] Searching for system libtbb..."
shopt -s nullglob

LIB_CANDIDATES=(/usr/lib64/libtbb.so* /usr/lib/x86_64-linux-gnu/libtbb.so* /usr/lib/libtbb.so* /usr/local/lib/libtbb.so*)
found_libs=()
for p in "${LIB_CANDIDATES[@]}"; do
  for f in $p; do
    if [ -e "$f" ]; then
      found_libs+=("$f")
    fi
  done
done

if [ ${#found_libs[@]} -eq 0 ]; then
  echo "[vendor_tbb] No libtbb found on builder image; skipping vendorization"
  exit 0
fi

command -v patchelf >/dev/null || { echo "[vendor_tbb] patchelf not found" >&2; exit 1; }
command -v unzip >/dev/null || { echo "[vendor_tbb] unzip not found" >&2; exit 1; }
command -v zip >/dev/null || { echo "[vendor_tbb] zip not found" >&2; exit 1; }

echo "[vendor_tbb] Found libs: ${found_libs[*]}"

for whl in wheelhouse/*.whl; do
  echo "[vendor_tbb] Processing $whl"
  tmpdir=$(mktemp -d)
  unzip -q "$whl" -d "$tmpdir"

  # Find all extension .so files and vendor libtbb next to them
  mapfile -t so_files < <(find "$tmpdir" -name "*.so")
  if [ ${#so_files[@]} -eq 0 ]; then
    echo "[vendor_tbb] No .so files found in $whl"
  fi

  for so in "${so_files[@]}"; do
    dir=$(dirname "$so")
    libsdir="$dir/.libs"
    mkdir -p "$libsdir"
    for lib in "${found_libs[@]}"; do
      cp -v "$lib" "$libsdir/"
    done
    echo "[vendor_tbb] Setting rpath for $so"
    patchelf --set-rpath '\$ORIGIN/.libs' "$so"
  done

  # Repack wheel
  newwhl="$tmpdir/$(basename "$whl")"
  (cd "$tmpdir" && zip -q -r "$newwhl" .)
  mv -f "$newwhl" "$whl"
  rm -rf "$tmpdir"
  echo "[vendor_tbb] Updated $whl"
done

echo "[vendor_tbb] Done."
