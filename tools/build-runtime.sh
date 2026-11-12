#!/usr/bin/env bash
# Cross-build libtyphonrt.a for mobile (and host) Rust targets.
#   tools/build-runtime.sh <rust-triple>... [-- extra cargo args]
# Output: ${CARGO_TARGET_DIR:-typhon/target}/<triple>/release/libtyphonrt.a
# Android: NDK from $ANDROID_NDK_HOME, else newest $ANDROID_HOME/ndk/*; API ${TYPHON_ANDROID_API:-24}.
# TYPHON_CARGO_SUBCMD=rustc with `-- --crate-type staticlib -- --print native-static-libs` lists link libs.
# iOS: Xcode SDKs via xcrun; min version ${IPHONEOS_DEPLOYMENT_TARGET:-16.4} (Accelerate ILP64 LAPACK).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
targets=(); extra=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; extra=("$@"); break; fi
  targets+=("$1"); shift
done
[ ${#targets[@]} -gt 0 ] || { echo "usage: $0 <rust-triple>... [-- cargo args]" >&2; exit 2; }
api="${TYPHON_ANDROID_API:-24}"
ios_min="${IPHONEOS_DEPLOYMENT_TARGET:-16.4}"
# Cross targets need rustup's std for them; a Homebrew rust earlier in PATH has only the host.
if command -v rustup >/dev/null; then
  tc_bin="$(dirname "$(cd "$root" && rustup which cargo 2>/dev/null || true)")"
  [ -x "$tc_bin/cargo" ] && export PATH="$tc_bin:$PATH" RUSTC="$tc_bin/rustc"
fi

find_ndk() {
  if [ -n "${ANDROID_NDK_HOME:-}" ] && [ -d "$ANDROID_NDK_HOME" ]; then echo "$ANDROID_NDK_HOME"; return; fi
  local sdk n
  for sdk in "${ANDROID_HOME:-}" "${ANDROID_SDK_ROOT:-}" /opt/homebrew/share/android-commandlinetools "$HOME/Library/Android/sdk"; do
    [ -n "$sdk" ] && [ -d "$sdk/ndk" ] || continue
    # newest NDK that actually has the LLVM toolchain (partial installs are skipped)
    n="$(ls -d "$sdk"/ndk/*/toolchains/llvm/prebuilt 2>/dev/null | sort -V | tail -1)"
    n="${n%/toolchains/llvm/prebuilt}"
    [ -n "$n" ] && { echo "${n%/}"; return; }
  done
  echo "error: Android NDK not found (set ANDROID_NDK_HOME)" >&2; exit 1
}

for t in "${targets[@]}"; do
  u="$(echo "$t" | tr '[:lower:]-' '[:upper:]_')"; l="$(echo "$t" | tr '-' '_')"
  (
  case "$t" in
    *-linux-android*)
      ndk="$(find_ndk)"; host_tag="$(ls "$ndk/toolchains/llvm/prebuilt" | head -1)"
      bin="$ndk/toolchains/llvm/prebuilt/$host_tag/bin"
      case "$t" in
        aarch64-*) abi=arm64-v8a; cc="$bin/aarch64-linux-android${api}-clang";;
        x86_64-*) abi=x86_64; cc="$bin/x86_64-linux-android${api}-clang";;
        armv7-*) abi=armeabi-v7a; cc="$bin/armv7a-linux-androideabi${api}-clang";;
        i686-*) abi=x86; cc="$bin/i686-linux-android${api}-clang";;
        *) echo "unsupported Android target $t" >&2; exit 2;;
      esac
      export "CC_$l=$cc" "AR_$l=$bin/llvm-ar" "CARGO_TARGET_${u}_LINKER=$cc" "CARGO_TARGET_${u}_AR=$bin/llvm-ar"
      export "CFLAGS_$l=-DLARGE_CONFIG -fPIC"
      # bdwgc-alloc-sys runs bindgen on gc.h: give libclang the NDK sysroot.
      export "BINDGEN_EXTRA_CLANG_ARGS_$l=--sysroot=$bin/../sysroot"
      export "CMAKE_TOOLCHAIN_FILE_$l=$root/tools/cmake/android/android.toolchain.cmake"
      export TYPHON_ANDROID_NDK="$ndk" TYPHON_ANDROID_ABI="$abi" TYPHON_ANDROID_API="$api" ANDROID_NDK_HOME="$ndk"
      ;;
    *-apple-ios*)
      case "$t" in
        aarch64-apple-ios) sdk=iphoneos; min="-mios-version-min=$ios_min";;
        *) sdk=iphonesimulator; min="-mios-simulator-version-min=$ios_min";;
      esac
      sysroot="$(xcrun --sdk $sdk --show-sdk-path)"
      arch=arm64; [ "${t%%-*}" = x86_64 ] && arch=x86_64
      export IPHONEOS_DEPLOYMENT_TARGET="$ios_min"
      export "CC_$l=$(xcrun --sdk $sdk -f clang)" "AR_$l=$(xcrun --sdk $sdk -f ar)"
      export "CFLAGS_$l=-DLARGE_CONFIG $min -isysroot $sysroot"
      export "BINDGEN_EXTRA_CLANG_ARGS_$l=-isysroot $sysroot"
      export "CMAKE_TOOLCHAIN_FILE_$l=$root/tools/cmake/ios.toolchain.cmake"
      export TYPHON_IOS_SYSROOT="$sysroot" TYPHON_IOS_ARCH="$arch" TYPHON_IOS_MIN="$ios_min"
      ;;
  esac
  cd "$root"
  echo "== building libtyphonrt for $t" >&2
  cargo ${TYPHON_CARGO_SUBCMD:-build} --release -p typhon-runtime --target "$t" ${extra[@]+"${extra[@]}"}
  )
done
