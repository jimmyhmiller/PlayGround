#!/bin/sh
set -eu

# Build the exact llhttp release used by coil.http.server. The release branch
# contains generated C, so building it requires only a C compiler and ar.
LLHTTP_VERSION=9.4.3
LLHTTP_SHA256=50fe0797029e3cdf3230ae2510bfa12b4467ec3bc24243601f4b1d3ef42a9445

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
machine=$(uname -m)
system=$(uname -s)

case "$system" in
  Darwin) target="$machine-macos" ;;
  Linux) target="$machine-linux" ;;
  *) echo "build-llhttp: $system is not supported" >&2; exit 1 ;;
esac

output_dir="$repo_dir/build/bin/native/llhttp/$target"
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT HUP INT TERM
archive="$work_dir/llhttp.tar.gz"
source_dir="$work_dir/source"
object_dir="$work_dir/objects"

mkdir -p "$source_dir" "$object_dir" "$output_dir"
curl -fsSL \
  "https://github.com/nodejs/llhttp/archive/refs/heads/release/v$LLHTTP_VERSION.tar.gz" \
  -o "$archive"

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'
  fi
}

[ "$(sha256 "$archive")" = "$LLHTTP_SHA256" ] || {
  echo "build-llhttp: archive checksum mismatch" >&2
  exit 1
}

tar -xzf "$archive" -C "$source_dir" --strip-components=1

cflags="-std=c99 -Os -fvisibility=hidden -ffunction-sections -fdata-sections"
cc $cflags -I"$source_dir/include" -c "$source_dir/src/llhttp.c" -o "$object_dir/llhttp.o"
cc $cflags -I"$source_dir/include" -c "$source_dir/src/api.c" -o "$object_dir/api.o"
cc $cflags -I"$source_dir/include" -c "$source_dir/src/http.c" -o "$object_dir/http.o"
cc $cflags -I"$source_dir/include" -c "$repo_dir/scripts/native/llhttp_shim.c" -o "$object_dir/shim.o"
ar rcs "$output_dir/libllhttp.a" \
  "$object_dir/llhttp.o" "$object_dir/api.o" "$object_dir/http.o" "$object_dir/shim.o"

echo "built $output_dir/libllhttp.a"
ls -lh "$output_dir/libllhttp.a"
