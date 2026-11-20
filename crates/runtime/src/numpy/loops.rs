//! numpy ufunc loops.
//!
//! Elementwise math uses the Highway kernels ported in `hmath.rs` (as Codon
//! does), evaluated in blocks LLVM auto-vectorizes; out-of-range blocks and
//! tails use libm with Codon's per-128-bit-vector granularity, so results match
//! Codon bit-for-bit on aarch64. Strides are in bytes.

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn binary_loop<T: Copy>(
    in1: *const T, is1: usize, in2: *const T, is2: usize, out: *mut T, os: usize, n: usize, f: impl Fn(T, T) -> T,
) {
    let sz = core::mem::size_of::<T>();
    let apart = |p: *const T| (out as usize) >= (p as usize) + n * sz || (p as usize) >= (out as usize) + n * sz;
    unsafe {
        if is1 == sz && is2 == sz && os == sz && apart(in1) && apart(in2) {
            // Disjoint slices give LLVM noalias (no alias check / scalar loop copy).
            let (a, b) = (core::slice::from_raw_parts(in1, n), core::slice::from_raw_parts(in2, n));
            for ((o, &x), &y) in core::slice::from_raw_parts_mut(out, n).iter_mut().zip(a).zip(b) {
                *o = f(x, y);
            }
        } else if is1 == sz && is2 == sz && os == sz && out as *const T == in1 && apart(in2) {
            // In-place into the first operand (fused temporaries): still vectorizable.
            let b = core::slice::from_raw_parts(in2, n);
            for (o, &y) in core::slice::from_raw_parts_mut(out, n).iter_mut().zip(b) {
                *o = f(*o, y);
            }
        } else if is1 == sz && is2 == sz && os == sz && out as *const T == in2 && apart(in1) {
            let a = core::slice::from_raw_parts(in1, n);
            for (o, &x) in core::slice::from_raw_parts_mut(out, n).iter_mut().zip(a) {
                *o = f(x, *o);
            }
        } else if is1 == sz && is2 == sz && os == sz {
            for i in 0..n {
                out.add(i).write_unaligned(f(in1.add(i).read_unaligned(), in2.add(i).read_unaligned()));
            }
        } else if is1 == 0 && is2 == sz && os == sz {
            let a = in1.read_unaligned();
            for i in 0..n {
                out.add(i).write_unaligned(f(a, in2.add(i).read_unaligned()));
            }
        } else if is1 == sz && is2 == 0 && os == sz {
            let b = in2.read_unaligned();
            for i in 0..n {
                out.add(i).write_unaligned(f(in1.add(i).read_unaligned(), b));
            }
        } else {
            let (a, b, o) = (in1 as *const u8, in2 as *const u8, out as *mut u8);
            for i in 0..n {
                let x = (a.wrapping_add(i.wrapping_mul(is1)) as *const T).read_unaligned();
                let y = (b.wrapping_add(i.wrapping_mul(is2)) as *const T).read_unaligned();
                (o.wrapping_add(i.wrapping_mul(os)) as *mut T).write_unaligned(f(x, y));
            }
        }
    }
}

/// Block size: the range check and the vectorized map run per block.
const W: usize = 256;

/// Fast path over whole `W` blocks of disjoint, aligned slices: range-check a
/// block, then map it (out of line so the vectorized body owns the registers).
/// Lanes failing `ok` fall back per 128-bit group, as in Codon. Returns the
/// number of elements processed.
#[inline(never)]
fn contiguous<T: Copy>(x: &[T], y: &mut [T], v: &impl Fn(T) -> T, ok: &impl Fn(T) -> bool, lib: fn(T) -> T) -> usize {
    let g = 16 / core::mem::size_of::<T>();
    let n = x.len().min(y.len()) / W * W;
    for (xs, ys) in x[..n].chunks_exact(W).zip(y[..n].chunks_exact_mut(W)) {
        if xs.iter().fold(true, |a, &e| a & ok(e)) {
            for (o, &e) in ys.iter_mut().zip(xs) {
                *o = v(e);
            }
        } else {
            fallback(xs.as_ptr(), ys.as_mut_ptr(), g, ok, v, lib);
        }
    }
    n
}

/// In-place variant of `contiguous` (`input == out`, as emitted by the numpy
/// fusion pass): no stack round trip. Element `k` is read before it is written.
#[inline(never)]
fn contiguous_inplace<T: Copy>(y: &mut [T], v: &impl Fn(T) -> T, ok: &impl Fn(T) -> bool, lib: fn(T) -> T) -> usize {
    let g = 16 / core::mem::size_of::<T>();
    let n = y.len() / W * W;
    for ys in y[..n].chunks_exact_mut(W) {
        if ys.iter().fold(true, |a, &e| a & ok(e)) {
            for o in ys.iter_mut() {
                *o = v(*o);
            }
        } else {
            let p = ys.as_mut_ptr();
            fallback(p, p, g, ok, v, lib);
        }
    }
    n
}

#[cold]
#[inline(never)]
fn fallback<T: Copy>(p: *const T, o: *mut T, g: usize, ok: &impl Fn(T) -> bool, v: &impl Fn(T) -> T, lib: fn(T) -> T) {
    for b in (0..W).step_by(g) {
        let good = (b..b + g).all(|k| ok(unsafe { p.add(k).read_unaligned() }));
        for k in b..b + g {
            let x = unsafe { p.add(k).read_unaligned() };
            unsafe { o.add(k).write_unaligned(if good { v(x) } else { lib(x) }) };
        }
    }
}

/// Highway-style unary loop (Codon `UnaryLoop`): whole blocks go through the
/// branchless port `v` (strided or in-place input goes through stack blocks);
/// vectors with a lane failing `ok` and the scalar tail use libm `lib`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn hwy_unary<T: Copy + Default>(
    input: *const T, is: usize, out: *mut T, os: usize, n: usize, v: impl Fn(T) -> T, ok: impl Fn(T) -> bool,
    lib: fn(T) -> T,
) {
    let (ib, ob) = (input as *const u8, out as *mut u8);
    let ld = |i: usize| unsafe { (ib.wrapping_add(i.wrapping_mul(is)) as *const T).read_unaligned() };
    let st = |i: usize, x: T| unsafe { (ob.wrapping_add(i.wrapping_mul(os)) as *mut T).write_unaligned(x) };
    let sz = core::mem::size_of::<T>();
    let mut i = 0;
    let disjoint = (out as usize) >= (input as usize) + n * sz || (input as usize) >= (out as usize) + n * sz;
    if is == sz && os == sz && disjoint {
        // Disjoint slices give LLVM noalias: no runtime alias check / scalar copy.
        let (x, y) = unsafe { (core::slice::from_raw_parts(input, n), core::slice::from_raw_parts_mut(out, n)) };
        i = contiguous(x, y, &v, &ok, lib);
    } else if is == sz && os == sz && input == out as *const T {
        let y = unsafe { core::slice::from_raw_parts_mut(out, n) };
        i = contiguous_inplace(y, &v, &ok, lib);
    } else {
        let mut buf = [T::default(); W];
        while i + W <= n {
            for (j, b) in buf.iter_mut().enumerate() {
                *b = ld(i + j);
            }
            let mut res = [T::default(); W];
            contiguous(&buf, &mut res, &v, &ok, lib);
            for (j, &b) in res.iter().enumerate() {
                st(i + j, b);
            }
            i += W;
        }
    }
    // Remaining whole 128-bit vectors, then the scalar tail (as in Codon).
    let g = 16 / sz;
    while i + g <= n {
        let good = (i..i + g).all(|k| ok(ld(k)));
        for k in i..i + g {
            st(k, if good { v(ld(k)) } else { lib(ld(k)) });
        }
        i += g;
    }
    for k in i..n {
        st(k, lib(ld(k)));
    }
}

macro_rules! hwy_unary {
    ($($name:ident: $t:ty => $v:expr, $lib:expr $(, ok = $ok:expr)?;)*) => {$(
        /// # Safety
        /// `input`/`out` valid for `n` elements at the given byte strides.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(input: *const $t, is: usize, out: *mut $t, os: usize, n: usize) {
            #[allow(unused_variables)]
            let ok = |x: $t| true;
            $(let ok = $ok;)?
            unsafe { hwy_unary(input, is, out, os, n, $v, ok, $lib) }
        }
    )*};
}

/// Codon functor wrappers around the Highway kernels, generic over f32/f64.
mod f {
    use super::hm::{self, Hf, MulAdd};
    pub trait F: Hf + MulAdd + Default {}
    impl<T: Hf + MulAdd + Default> F for T {}
    #[inline(always)]
    fn sel<T: F>(m: bool, a: T, b: T) -> T { hm::fsel(m, a, b) }
    #[inline(always)]
    fn nan<T: F>() -> T { T::c(f64::NAN) }
    #[inline(always)]
    fn inf<T: F>() -> T { T::c(f64::INFINITY) }
    #[inline(always)]
    pub fn acosh<T: F>(x: T) -> T { sel(x == inf(), x, sel(x < T::c(1.0), nan(), hm::acosh(x))) }
    #[inline(always)]
    pub fn asinh<T: F>(x: T) -> T { sel(x.is_nan() || x == inf() || x == -inf::<T>(), x, hm::asinh(x)) }
    #[inline(always)]
    pub fn atan<T: F>(x: T) -> T {
        let h = T::c(core::f64::consts::FRAC_PI_2);
        sel(x == inf(), h, sel(x == -inf::<T>(), -h, hm::atan(x)))
    }
    #[inline(always)]
    pub fn atanh<T: F>(x: T) -> T {
        let one = T::c(1.0);
        sel(x == one, inf(), sel(x == -one, -inf::<T>(), sel(x > one || x < -one, nan(), hm::atanh(x))))
    }
    /// exp/exp2/expm1: NaN passes through, `x >= lim` -> +inf.
    #[inline(always)]
    pub fn expw<T: F>(x: T, lim: T, f: impl Fn(T) -> T) -> T { sel(x.is_nan(), x, sel(x >= lim, inf(), f(x))) }
    #[inline(always)]
    pub fn log1p<T: F>(x: T) -> T {
        let m1 = T::c(-1.0);
        sel(x == m1, -inf::<T>(), sel(x < m1, nan(), sel(x == inf() || x.is_nan(), x, hm::log1p(x))))
    }
    #[inline(always)]
    pub fn lim<T: F>(l: T) -> impl Fn(T) -> bool { move |x: T| x >= -l && x <= l }
}

use super::hmath as hm;
use core::f64::consts::{LOG10_E as LOG10E, LOG2_E as LOG2E};

hwy_unary! {
    cnp_acos_float32: f32 => hm::acos, f32::acos;
    cnp_acos_float64: f64 => hm::acos, f64::acos;
    cnp_asin_float32: f32 => hm::asin, f32::asin;
    cnp_asin_float64: f64 => hm::asin, f64::asin;
    cnp_acosh_float32: f32 => f::acosh, f32::acosh;
    cnp_acosh_float64: f64 => f::acosh, f64::acosh;
    cnp_asinh_float32: f32 => f::asinh, f32::asinh;
    cnp_asinh_float64: f64 => f::asinh, f64::asinh;
    cnp_atan_float32: f32 => f::atan, f32::atan;
    cnp_atan_float64: f64 => f::atan, f64::atan;
    cnp_atanh_float32: f32 => f::atanh, f32::atanh;
    cnp_atanh_float64: f64 => f::atanh, f64::atanh;
    cnp_cos_float32: f32 => hm::sincos::<f32, true>, f32::cos, ok = f::lim(2.63e7f32);
    cnp_cos_float64: f64 => hm::sincos::<f64, true>, f64::cos, ok = f::lim(3.37e9);
    // Codon uses 5.30e8 for f32 sin, but Highway's f32 reduction is garbage above
    // ~1e7 (e.g. sin(2.68e8) = 2922); use cos's limit so those go to libm.
    cnp_sin_float32: f32 => hm::sincos::<f32, false>, f32::sin, ok = f::lim(2.63e7f32);
    cnp_sin_float64: f64 => hm::sincos::<f64, false>, f64::sin, ok = f::lim(6.74e9);
    cnp_sinh_float32: f32 => hm::sinh, f32::sinh, ok = f::lim(88.7228f32);
    cnp_sinh_float64: f64 => hm::sinh, f64::sinh, ok = f::lim(709.0);
    cnp_tanh_float32: f32 => hm::tanh, f32::tanh;
    cnp_tanh_float64: f64 => hm::tanh, f64::tanh;
    cnp_exp_float32: f32 => |x| f::expw(x, 128.0, hm::exp), f32::exp;
    cnp_exp_float64: f64 => |x| f::expw(x, 1000.0, hm::exp), f64::exp;
    cnp_exp2_float32: f32 => |x| f::expw(x, 128.0, hm::exp2), f32::exp2;
    cnp_exp2_float64: f64 => |x| f::expw(x, 2048.0, hm::exp2), f64::exp2;
    cnp_expm1_float32: f32 => |x| f::expw(x, 128.0, hm::expm1), f32::exp_m1;
    cnp_expm1_float64: f64 => |x| f::expw(x, 1000.0, hm::expm1), f64::exp_m1;
    cnp_log_float32: f32 => |x| hm::log_codon(x, 1.0), f32::ln;
    cnp_log_float64: f64 => |x| hm::log_codon(x, 1.0), f64::ln;
    cnp_log2_float32: f32 => |x| hm::log_codon(x, LOG2E as _), f32::log2;
    cnp_log2_float64: f64 => |x| hm::log_codon(x, LOG2E as _), f64::log2;
    cnp_log10_float32: f32 => |x| hm::log_codon(x, LOG10E as _), f32::log10;
    cnp_log10_float64: f64 => |x| hm::log_codon(x, LOG10E as _), f64::log10;
    cnp_log1p_float32: f32 => f::log1p, f32::ln_1p;
    cnp_log1p_float64: f64 => f::log1p, f64::ln_1p;
}

/// Binary Highway loop: the per-element kernel is branchless, so the existing
/// contiguous/broadcast fast paths vectorize directly.
macro_rules! hwy_binary {
    ($($name:ident: $t:ty => $f:path;)*) => {$(
        /// # Safety
        /// Inputs/output valid for `n` elements at the given byte strides.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            in1: *const $t, is1: usize, in2: *const $t, is2: usize, out: *mut $t, os: usize, n: usize,
        ) {
            unsafe { binary_loop(in1, is1, in2, is2, out, os, n, |a, b| $f(a, b)) }
        }
    )*};
}

hwy_binary! {
    cnp_atan2_float32: f32 => hm::atan2;
    cnp_atan2_float64: f64 => hm::atan2;
    cnp_hypot_float32: f32 => hm::hypot;
    cnp_hypot_float64: f64 => hm::hypot;
}


macro_rules! complex_abs {
    ($($name:ident: $t:ty => $h:path),*) => {$(
        /// `|z|` over interleaved complex input (`is` = byte stride per complex value).
        ///
        /// # Safety
        /// `input` valid for `n` complex values, `out` for `n` reals, at byte strides.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(input: *const $t, is: usize, out: *mut $t, os: usize, n: usize) {
            let (ib, ob) = (input as *const u8, out as *mut u8);
            let sz = core::mem::size_of::<$t>();
            if is == 2 * sz && os == sz {
                let (z, o) = unsafe { (core::slice::from_raw_parts(input, 2 * n), core::slice::from_raw_parts_mut(out, n)) };
                for (o, z) in o.iter_mut().zip(z.chunks_exact(2)) {
                    *o = $h(z[0], z[1]);
                }
                return;
            }
            for i in 0..n {
                unsafe {
                    let z = ib.wrapping_add(i.wrapping_mul(is)) as *const $t;
                    let r = $h(z.read_unaligned(), z.add(1).read_unaligned());
                    (ob.wrapping_add(i.wrapping_mul(os)) as *mut $t).write_unaligned(r);
                }
            }
        }
    )*};
}

complex_abs!(cnp_abs_complex64: f32 => hm::hypot, cnp_abs_complex128: f64 => hm::hypot);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_and_strided() {
        let x = [0.0f64, 1.0, 2.0, 3.0, 4.0];
        let mut y = [0.0f64; 5];
        unsafe { cnp_exp_float64(x.as_ptr(), 8, y.as_mut_ptr(), 8, 5) };
        for i in 0..5 {
            assert!((y[i] - x[i].exp()).abs() < 1e-12);
        }
        // every other element
        let mut z = [0.0f32; 3];
        let xs = [1.0f32, 99.0, 4.0, 99.0, 9.0];
        unsafe { cnp_log2_float32(xs.as_ptr(), 8, z.as_mut_ptr(), 4, 3) };
        assert_eq!(z, [0.0, 2.0, 9f32.log2()]);
    }

    #[test]
    fn binary_broadcast() {
        let a = [3.0f64];
        let b = [4.0f64, 0.0, 5.0];
        let mut o = [0.0f64; 3];
        unsafe { cnp_hypot_float64(a.as_ptr(), 0, b.as_ptr(), 8, o.as_mut_ptr(), 8, 3) };
        assert_eq!(o[0], 5.0);
        assert_eq!(o[1], 3.0);
        unsafe { cnp_atan2_float64(b.as_ptr(), 8, a.as_ptr(), 0, o.as_mut_ptr(), 8, 3) };
        assert!((o[0] - 4f64.atan2(3.0)).abs() < 1e-15);
        let v = [0.0, -0.0, 1.0, -1.0, 0.3, -7.5, 1e-300, 1e300, f64::INFINITY, f64::NEG_INFINITY];
        for &y in &v {
            for &x in &v {
                let (a, b) = (hm::atan2(y, x), y.atan2(x));
                assert!(a == b || (a - b).abs() <= 4e-16 * b.abs().max(1e-300), "atan2({y},{x}) {a} vs {b}");
                assert_eq!(a.is_sign_negative(), b.is_sign_negative(), "sign atan2({y},{x})");
            }
        }
        assert!(hm::atan2(f64::NAN, 1.0).is_nan());
        for k in -2000..2000 {
            let x = k as f64 * 0.01237 + (k as f64).powi(3) * 1e-3;
            assert!((hm::sincos::<f64, false>(x) - x.sin()).abs() < 1e-15, "sin {x}");
            assert!((hm::sincos::<f64, true>(x) - x.cos()).abs() < 1e-15, "cos {x}");
        }
        let v = [0.5, -1e12, f64::NAN, f64::INFINITY, 3.0, 1.0, 2.0, -0.0, 7.0];
        let mut o = [0.0; 9];
        unsafe { cnp_sin_float64(v.as_ptr(), 8, o.as_mut_ptr(), 8, 9) };
        for (a, b) in o.iter().zip(v) {
            assert!(a == &b.sin() || (a.is_nan() && b.sin().is_nan()));
        }
    }

    #[test]
    fn large_trig_range_reduction() {
        // mpmath at 100 decimal digits, using the exact binary64 input
        // 558072812.39732873439788818359375 (0x1.0a1c1f632dbabp+29).
        // An unfused first reduction gave cos = -0.3195696683783391 on
        // baseline x86-64, whereas both macOS and UCRT libm were correct.
        let x = 558072812.3973287_f64;
        let cos = -0.31956961189919546071044005546429405_f64;
        let sin = -0.94756280169210821074012989020700785_f64;
        type Kernel = unsafe extern "C" fn(*const f64, usize, *mut f64, usize, usize);
        for sign in [1.0, -1.0] {
            for (kernel, reference) in [(cnp_cos_float64 as Kernel, cos), (cnp_sin_float64 as Kernel, sign * sin)] {
                // Whole block, remaining 128-bit vector, and scalar libm tail.
                let input = [sign * x; 2 * (W + 3)];
                for stride in [8, 16] {
                    let mut output = [f64::NAN; W + 3];
                    unsafe { kernel(input.as_ptr(), stride, output.as_mut_ptr(), 8, output.len()) };
                    for y in output {
                        assert!((y - reference).abs() <= 1e-15 * reference.abs(), "trig({sign} * {x}) = {y}, expected {reference}");
                    }
                }
            }
        }
    }

    /// Every Highway-backed unary kernel vs libm over a wide range and special
    /// values (contiguous and strided, long enough to hit the block path and the
    /// per-vector fallback). Tolerance ~4 ulp (Highway's documented max error),
    /// plus an absolute floor near zero crossings; specials must match exactly.
    #[test]
    fn all_unary_vs_libm() {
        type K64 = unsafe extern "C" fn(*const f64, usize, *mut f64, usize, usize);
        type K32 = unsafe extern "C" fn(*const f32, usize, *mut f32, usize, usize);
        let cases: [(&str, K64, K32, fn(f64) -> f64); 17] = [
            ("acos", cnp_acos_float64, cnp_acos_float32, f64::acos),
            ("asin", cnp_asin_float64, cnp_asin_float32, f64::asin),
            ("atan", cnp_atan_float64, cnp_atan_float32, f64::atan),
            ("acosh", cnp_acosh_float64, cnp_acosh_float32, f64::acosh),
            ("asinh", cnp_asinh_float64, cnp_asinh_float32, f64::asinh),
            ("atanh", cnp_atanh_float64, cnp_atanh_float32, f64::atanh),
            ("cos", cnp_cos_float64, cnp_cos_float32, f64::cos),
            ("sin", cnp_sin_float64, cnp_sin_float32, f64::sin),
            ("sinh", cnp_sinh_float64, cnp_sinh_float32, f64::sinh),
            ("tanh", cnp_tanh_float64, cnp_tanh_float32, f64::tanh),
            ("exp", cnp_exp_float64, cnp_exp_float32, f64::exp),
            ("exp2", cnp_exp2_float64, cnp_exp2_float32, f64::exp2),
            ("expm1", cnp_expm1_float64, cnp_expm1_float32, f64::exp_m1),
            ("log", cnp_log_float64, cnp_log_float32, f64::ln),
            ("log2", cnp_log2_float64, cnp_log2_float32, f64::log2),
            ("log10", cnp_log10_float64, cnp_log10_float32, f64::log10),
            ("log1p", cnp_log1p_float64, cnp_log1p_float32, f64::ln_1p),
        ];
        let special = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 1e-310, 1e-20, 1e300, -1e300, 700.0, 710.0, -745.0,
            -1001.0, 88.0, 89.0, -104.5, 3.0e9, 7e9, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 268435457.0, 18.8];
        let mut xs: Vec<f64> = (0..6000).map(|k| {
            let t = k as f64 / 6000.0;
            match k % 4 { 0 => t * 40.0 - 20.0, 1 => t * 2.0 - 1.0, 2 => (t * 60.0 - 30.0).exp(), _ => -(t * 10.0).exp() }
        }).collect();
        xs.extend(special);
        xs.extend((0..300).map(|k| k as f64 * 0.01 - 1.5));
        let close = |a: f64, b: f64, ulp: f64, floor: f64| {
            (a.is_nan() && b.is_nan()) || a == b || (a - b).abs() <= ulp * b.abs() + floor
        };
        for (name, k64, k32, lib) in cases {
            let mut o = vec![0.0; xs.len()];
            unsafe { k64(xs.as_ptr(), 8, o.as_mut_ptr(), 8, xs.len()) };
            for (&x, &y) in xs.iter().zip(&o) {
                // atanh near |x| = 1 inherits the cancellation in (2|x|)/(1-|x|) (same in Codon).
                let tol = if name == "atanh" { 1e-13 } else { 1e-15 };
                assert!(close(y, lib(x), tol, 1e-300), "{name}_f64({x}) = {y}, libm {}", lib(x));
            }
            // strided: every other element
            let mut s = vec![0.0; xs.len() / 2];
            unsafe { k64(xs.as_ptr(), 16, s.as_mut_ptr(), 8, s.len()) };
            for (i, &y) in s.iter().enumerate() {
                // fallback groups shift with the stride (as in Codon), so compare loosely
                assert!(close(y, o[2 * i], 1e-13, 1e-300), "{name} strided");
            }
            let xf: Vec<f32> = xs.iter().map(|&x| x as f32).collect();
            let mut of = vec![0.0f32; xf.len()];
            unsafe { k32(xf.as_ptr(), 4, of.as_mut_ptr(), 4, xf.len()) };
            for (&x, &y) in xf.iter().zip(&of) {
                let r = lib(x as f64) as f32;
                // f32 sin/cos far from 0 lose absolute accuracy in Highway's reduction.
                let floor = if matches!(name, "sin" | "cos") { 1e-6 * (x.abs() as f64).max(1.0) } else { 1e-37 };
                let tol = if name == "atanh" { 1e-4 } else { 1e-6 };
                assert!(close(y as f64, r as f64, tol, floor), "{name}_f32({x}) = {y}, libm {r}");
            }
        }
    }

    #[test]
    fn hypot_extremes() {
        let v = [0.0, -0.0, 1.0, -3.0, 1e-320, 5e-310, 1e-200, 1e200, 1.7e308, f64::MAX, f64::INFINITY,
            f64::NEG_INFINITY, f64::NAN, 2.2250738585072014e-308];
        for &a in &v {
            for &b in &v {
                let (h, r) = (hm::hypot(a, b), a.hypot(b));
                assert!(h == r || (h.is_nan() && r.is_nan()) || (h - r).abs() <= 2.3e-16 * r, "hypot({a},{b}) {h} vs {r}");
                let (hf, rf) = (hm::hypot(a as f32, b as f32), (a as f32).hypot(b as f32));
                assert!(hf == rf || (hf.is_nan() && rf.is_nan()) || (hf - rf).abs() <= 1.2e-7 * rf, "hypotf({a},{b})");
            }
        }
    }

    #[test]
    fn in_place_matches_out_of_place() {
        let x: Vec<f64> = (0..1000).map(|k| k as f64 * 0.37 - 150.0).collect();
        let mut a = vec![0.0; x.len()];
        unsafe { cnp_tanh_float64(x.as_ptr(), 8, a.as_mut_ptr(), 8, x.len()) };
        let mut b = x.clone();
        unsafe { cnp_tanh_float64(b.as_ptr(), 8, b.as_mut_ptr(), 8, b.len()) };
        assert_eq!(a, b);
        let mut c = vec![0.0; x.len()];
        unsafe { cnp_hypot_float64(x.as_ptr(), 8, a.as_ptr(), 8, c.as_mut_ptr(), 8, x.len()) };
        unsafe { cnp_hypot_float64(x.as_ptr(), 8, a.as_ptr(), 8, a.as_mut_ptr(), 8, x.len()) };
        assert_eq!(a, c);
    }

    #[test]
    fn complex_abs_values() {
        let z = [3.0f32, 4.0, 5.0, 12.0];
        let mut o = [0.0f32; 2];
        unsafe { cnp_abs_complex64(z.as_ptr(), 8, o.as_mut_ptr(), 4, 2) };
        assert_eq!(o, [5.0, 13.0]);
    }
}
