# Wrapper around the NDK toolchain file (used by tools/build-runtime.sh for bdwgc via
# cmake-rs, which can only pass CMAKE_TOOLCHAIN_FILE_<triple> through the environment).
# The file name must stay `android.toolchain.cmake`: cmake-rs keys its NDK mode on it.
set(ANDROID_ABI "$ENV{TYPHON_ANDROID_ABI}")
set(ANDROID_PLATFORM "android-$ENV{TYPHON_ANDROID_API}")
include("$ENV{TYPHON_ANDROID_NDK}/build/cmake/android.toolchain.cmake")
