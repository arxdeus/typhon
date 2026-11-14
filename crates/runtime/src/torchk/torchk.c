/*
 * C-ABI kernels for stdlib/torch on aarch64 (macOS, iOS, Android), matching torch 2.15.0 CPU
 * results bit for bit. Wraps the original C libraries torch 2.15.0 links, vendored at the
 * commits pinned in its third_party/ (see crates/runtime/vendor/ and NOTICE):
 *   SLEEF 5a1d179 (Boost-1.0): ATen Vectorized<float> exp/expm1/log on arm64 call
 *     Sleef_expf4_u10 / Sleef_expm1f4_u10 / Sleef_logf4_u10 (AdvSIMD).
 *   NNPACK c07e3a0 (BSD-2) + pthreadpool, cpuinfo (BSD-2), psimd, FP16, FXdiv (MIT): ATen's
 *     NnpackSpatial backend calls nnp_convolution_output(nnp_convolution_algorithm_auto, ...)
 *     per group on contiguous group sub-tensors (aten/src/ATen/native/NNPACK.cpp).
 *
 * Threads: ATen (NNPACK.cpp) passes caffe2::pthreadpool_() sized to the intra-op thread count
 * (NULL-equivalent at 1 thread). NNPACK fixes its tile/block sizes from the problem shape and
 * cache sizes only; every pthreadpool_parallelize_* call splits independent output tiles (input
 * transform per (channel, sample), GEMM per (out-channel block, batch block) with the input
 * channel blocks reduced in a sequential outer loop, output transform per (sample, out block)).
 * So each output float is computed by the same op sequence regardless of pool size: results
 * are bit-identical for any nthreads (verified 1/2/4/5 x 100 reps against torch, see tests).
 *
 * GC: pool workers are plain pthreads (this file does not include gc.h, so pthread_create is
 * not redirected to GC_pthread_create). They only read/write the x, w, b, y buffers passed in
 * and NNPACK's own malloc'd workspace, never allocate GC memory, and never hold the only
 * reference to a GC object (the caller keeps x/w/b/y alive and blocks until the call returns).
 * bdwgc scans/stops only registered threads, so no registration is needed.
 */
#include <pthread.h>
#include <pthreadpool.h>
#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <nnpack.h>

float32x4_t Sleef_expf4_u10advsimd(float32x4_t);
float32x4_t Sleef_expm1f4_u10advsimd(float32x4_t);
float32x4_t Sleef_logf4_u10advsimd(float32x4_t);

/* Full 4-lane blocks only: processes floor(n/4)*4 elements and returns that count. The caller
 * handles the tail (ATen's vectorized map uses scalar code there). in and out may alias. */
#define TYPHON_SLEEF_V(name, fn)                                                  \
    int64_t name(const float *in, float *out, int64_t n) {                        \
        int64_t m = n & ~(int64_t)3;                                              \
        for (int64_t i = 0; i < m; i += 4) vst1q_f32(out + i, fn(vld1q_f32(in + i))); \
        return m;                                                                 \
    }
TYPHON_SLEEF_V(typhon_sleef_expf_v, Sleef_expf4_u10advsimd)
TYPHON_SLEEF_V(typhon_sleef_expm1f_v, Sleef_expm1f4_u10advsimd)
TYPHON_SLEEF_V(typhon_sleef_logf_v, Sleef_logf4_u10advsimd)

/* y[N,O,OH,OW] = conv(x[N,C,H,W], w[O,C,KH,KW]) + b[O], stride 1, no dilation, symmetric
 * padding (pad_top = pad_bottom, pad_left = pad_right) as ATen passes it. One group: the
 * caller splits groups. Returns the nnp_status (0 = success). */
static enum nnp_status conv(int64_t N, int64_t C, int64_t O, int64_t H, int64_t W, int64_t KH,
                            int64_t KW, int64_t pad_top, int64_t pad_left, const float *x,
                            const float *w, const float *b, float *y, pthreadpool_t pool) {
    enum nnp_status st = nnp_initialize(); /* pthread_once inside, cheap after the first call */
    if (st != nnp_status_success) return st;
    struct nnp_size in_size = {.width = (size_t)W, .height = (size_t)H};
    struct nnp_size k_size = {.width = (size_t)KW, .height = (size_t)KH};
    struct nnp_padding pad = {.top = (size_t)pad_top, .bottom = (size_t)pad_top,
                              .left = (size_t)pad_left, .right = (size_t)pad_left};
    return nnp_convolution_output(nnp_convolution_algorithm_auto, (size_t)N, (size_t)C, (size_t)O,
                                  in_size, pad, k_size, x, w, b, y, NULL, NULL,
                                  nnp_activation_identity, NULL, pool, NULL);
}

int64_t typhon_nnp_conv_output(int64_t N, int64_t C, int64_t O, int64_t H, int64_t W,
                               int64_t KH, int64_t KW, int64_t pad_top, int64_t pad_left,
                               const float *x, const float *w, const float *b, float *y) {
    return conv(N, C, O, H, W, KH, KW, pad_top, pad_left, x, w, b, y, NULL);
}

/* Shared lazily created pool, recreated when nthreads changes. The mutex is held for the whole
 * call so a resize never races a running conv (concurrent callers serialize, as they would on
 * torch's single intra-op pool). nthreads <= 1 runs on the caller thread without locking. */
static pthread_mutex_t pool_mu = PTHREAD_MUTEX_INITIALIZER;
static pthreadpool_t pool;
static int64_t pool_n;

int64_t typhon_nnp_conv_output_mt(int64_t N, int64_t C, int64_t O, int64_t H, int64_t W,
                                  int64_t KH, int64_t KW, int64_t pad_top, int64_t pad_left,
                                  const float *x, const float *w, const float *b, float *y,
                                  int64_t nthreads) {
    if (nthreads <= 1) return conv(N, C, O, H, W, KH, KW, pad_top, pad_left, x, w, b, y, NULL);
    pthread_mutex_lock(&pool_mu);
    if (pool_n != nthreads) {
        if (pool) pthreadpool_destroy(pool);
        pool = pthreadpool_create((size_t)nthreads);
        pool_n = pool ? nthreads : 0;
    }
    int64_t st = pool ? conv(N, C, O, H, W, KH, KW, pad_top, pad_left, x, w, b, y, pool)
                      : nnp_status_out_of_memory;
    pthread_mutex_unlock(&pool_mu);
    return st;
}
