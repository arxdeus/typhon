//! numpy ufunc loops.
//!
//! Scalar loops over std/libm functions with a contiguous fast path that LLVM can
//! auto-vectorize. Results are correctly rounded libm values, so outputs may
//! differ from SIMD approximations in the last ulp. Strides are in bytes.

macro_rules! unary {
    ($($name:ident: $t:ty => $f:expr),* $(,)?) => {$(
        /// # Safety
        /// `input`/`out` valid for `n` elements at the given byte strides.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(input: *const $t, is: usize, out: *mut $t, os: usize, n: usize) {
            let f: fn($t) -> $t = $f;
            unsafe { unary_loop(input, is, out, os, n, f) }
        }
    )*};
}

macro_rules! binary {
    ($($name:ident: $t:ty => $f:expr),* $(,)?) => {$(
        /// # Safety
        /// Inputs/output valid for `n` elements at the given byte strides.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            in1: *const $t, is1: usize, in2: *const $t, is2: usize, out: *mut $t, os: usize, n: usize,
        ) {
            let f: fn($t, $t) -> $t = $f;
            unsafe { binary_loop(in1, is1, in2, is2, out, os, n, f) }
        }
    )*};
}

#[inline(always)]
unsafe fn unary_loop<T: Copy>(input: *const T, is: usize, out: *mut T, os: usize, n: usize, f: fn(T) -> T) {
    let sz = core::mem::size_of::<T>();
    unsafe {
        if is == sz && os == sz {
            for i in 0..n {
                out.add(i).write_unaligned(f(input.add(i).read_unaligned()));
            }
        } else {
            let (ib, ob) = (input as *const u8, out as *mut u8);
            for i in 0..n {
                let x = (ib.wrapping_add(i.wrapping_mul(is)) as *const T).read_unaligned();
                (ob.wrapping_add(i.wrapping_mul(os)) as *mut T).write_unaligned(f(x));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn binary_loop<T: Copy>(
    in1: *const T, is1: usize, in2: *const T, is2: usize, out: *mut T, os: usize, n: usize, f: fn(T, T) -> T,
) {
    let sz = core::mem::size_of::<T>();
    unsafe {
        if is1 == sz && is2 == sz && os == sz {
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

unary! {
    cnp_acos_float32: f32 => f32::acos,
    cnp_acos_float64: f64 => f64::acos,
    cnp_acosh_float32: f32 => f32::acosh,
    cnp_acosh_float64: f64 => f64::acosh,
    cnp_asin_float32: f32 => f32::asin,
    cnp_asin_float64: f64 => f64::asin,
    cnp_asinh_float32: f32 => f32::asinh,
    cnp_asinh_float64: f64 => f64::asinh,
    cnp_atan_float32: f32 => f32::atan,
    cnp_atan_float64: f64 => f64::atan,
    cnp_atanh_float32: f32 => f32::atanh,
    cnp_atanh_float64: f64 => f64::atanh,
    cnp_cos_float32: f32 => f32::cos,
    cnp_exp_float32: f32 => f32::exp,
    cnp_exp_float64: f64 => f64::exp,
    cnp_exp2_float32: f32 => f32::exp2,
    cnp_exp2_float64: f64 => f64::exp2,
    cnp_expm1_float32: f32 => f32::exp_m1,
    cnp_expm1_float64: f64 => f64::exp_m1,
    cnp_log_float32: f32 => f32::ln,
    cnp_log_float64: f64 => f64::ln,
    cnp_log10_float32: f32 => f32::log10,
    cnp_log10_float64: f64 => f64::log10,
    cnp_log1p_float32: f32 => f32::ln_1p,
    cnp_log1p_float64: f64 => f64::ln_1p,
    cnp_log2_float32: f32 => f32::log2,
    cnp_log2_float64: f64 => f64::log2,
    cnp_sin_float32: f32 => f32::sin,
    cnp_sinh_float32: f32 => f32::sinh,
    cnp_sinh_float64: f64 => f64::sinh,
    cnp_tanh_float32: f32 => f32::tanh,
    cnp_tanh_float64: f64 => f64::tanh,
}

/// Sin/cos in vectorizable 8-wide chunks (port of Highway's `Sin`/`Cos` as used by
/// Codon's numpy runtime, bit-identical results). A chunk containing an argument
/// outside `[-limit, limit]` (or NaN) falls back to libm, like Codon.
#[inline(always)]
unsafe fn trig_loop<const COS: bool>(input: *const f64, is: usize, out: *mut f64, os: usize, n: usize) {
    const W: usize = 8;
    let (limit, lib): (f64, fn(f64) -> f64) = if COS { (3.37e9, f64::cos) } else { (6.74e9, f64::sin) };
    let (ib, ob) = (input as *const u8, out as *mut u8);
    let mut i = 0;
    while i + W <= n {
        let mut buf = [0.0f64; W];
        let mut ok = true;
        for (j, b) in buf.iter_mut().enumerate() {
            *b = unsafe { (ib.wrapping_add((i + j).wrapping_mul(is)) as *const f64).read_unaligned() };
            ok &= b.abs() <= limit;
        }
        let mut res = [0.0f64; W];
        if ok {
            for j in 0..W {
                res[j] = if COS { cos_poly(buf[j]) } else { sin_poly(buf[j]) };
            }
        } else {
            for j in 0..W {
                res[j] = lib(buf[j]);
            }
        }
        for (j, r) in res.iter().enumerate() {
            unsafe { (ob.wrapping_add((i + j).wrapping_mul(os)) as *mut f64).write_unaligned(*r) };
        }
        i += W;
    }
    while i < n {
        unsafe {
            let x = (ib.wrapping_add(i.wrapping_mul(is)) as *const f64).read_unaligned();
            (ob.wrapping_add(i.wrapping_mul(os)) as *mut f64).write_unaligned(lib(x));
        }
        i += 1;
    }
}

#[inline(always)]
fn sincos_core(x: f64, q: i32, half: bool, sign: u64) -> f64 {
    const P: [f64; 4] = [-3.1415926218032836914, -3.1786509424591713469e-8, -1.2246467864107188502e-16, -1.2736634327021899816e-24];
    let (qf, s) = (q as f64, if half { 0.5 } else { 1.0 });
    let mut x = x;
    for p in P {
        x = fma(qf, s * p, x);
    }
    let x = f64::from_bits(x.to_bits() ^ sign);
    const K: [f64; 9] = [
        -0.166666666666666657414808, 0.00833333333333332974823815, -0.000198412698412696162806809,
        2.75573192239198747630416e-6, -2.50521083763502045810755e-8, 1.60590430605664501629054e-10,
        -7.64712219118158833288484e-13, 2.81009972710863200091251e-15, -7.97255955009037868891952e-18,
    ];
    let y = x * x;
    let (y2, y4) = (y * y, y * y * y * y);
    let y8 = y4 * y4;
    let p = |i: usize| fma(K[i + 1], y, K[i]);
    let poly = fma(K[8], y8, fma(fma(p(6), y2, p(4)), y4, fma(p(2), y2, p(0))));
    fma(poly, y * x, x)
}

#[inline(always)]
fn sin_poly(x: f64) -> f64 {
    let ax = x.abs();
    let q = fma(ax, core::f64::consts::FRAC_1_PI, 0.5) as i32;
    let sign = (((q & 1) as u64) << 63) ^ ((ax.to_bits() ^ x.to_bits()) & (1 << 63));
    sincos_core(ax, q, false, sign)
}

#[inline(always)]
fn cos_poly(x: f64) -> f64 {
    let y = x.abs();
    let q = ((y * core::f64::consts::FRAC_1_PI) as i32).wrapping_shl(1).wrapping_add(1);
    let sign = (((!q) & 2) as i64 as u64) << 62;
    sincos_core(y, q, true, sign)
}

/// # Safety
/// `input`/`out` valid for `n` elements at the given byte strides.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnp_sin_float64(input: *const f64, is: usize, out: *mut f64, os: usize, n: usize) {
    unsafe { trig_loop::<false>(input, is, out, os, n) }
}

/// # Safety
/// `input`/`out` valid for `n` elements at the given byte strides.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnp_cos_float64(input: *const f64, is: usize, out: *mut f64, os: usize, n: usize) {
    unsafe { trig_loop::<true>(input, is, out, os, n) }
}

#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    if cfg!(any(target_arch = "aarch64", target_feature = "fma")) { a.mul_add(b, c) } else { a * b + c }
}

/// Branchless atan2 (port of Highway's `Atan2`, as used by Codon's numpy runtime):
/// polynomial on [0, 1] so the contiguous loop auto-vectorizes (scalar libm `atan2`
/// is ~1.6x slower). Max error ~1-2 ulp; handles signed zeros, infinities and NaN.
#[inline(always)]
fn atan2_poly(y: f64, x: f64) -> f64 {
    use core::f64::consts::{FRAC_PI_2, PI};
    const K: [f64; 19] = [
        -0.333333333333311110369124, 0.199999999996591265594148, -0.14285714266771329383765,
        0.111111105648261418443745, -0.090908995008245008229153, 0.0769219538311769618355029,
        -0.0666573579361080525984562, 0.0587666392926673580854313, -0.0523674852303482457616113,
        0.0466667150077840625632675, -0.0407629191276836500001934, 0.0337852580001353069993897,
        -0.0254517624932312641616861, 0.016599329773529201970117, -0.00889896195887655491740809,
        0.00370026744188713119232403, -0.00110611831486672482563471, 0.000209850076645816976906797,
        -1.88796008463073496563746e-5,
    ];
    // atan(r), r = min(|y|,|x|) / max(|y|,|x|) in [0, 1]: one division.
    let (ay, ax) = (y.abs(), x.abs());
    let big = ay > ax;
    let r = if big { ax / ay } else { ay / ax };
    let r2 = r * r;
    let r4 = r2 * r2;
    let r8 = r4 * r4;
    let r16 = r8 * r8;
    let p = |i: usize| fma(K[i + 1], r2, K[i]);
    let e0 = fma(fma(p(6), r4, p(4)), r8, fma(p(2), r4, p(0)));
    let e1 = fma(fma(p(14), r4, p(12)), r8, fma(p(10), r4, p(8)));
    let e2 = fma(K[18], r4, p(16));
    let poly = fma(fma(e2, r16, e1), r16, e0);
    let a = fma(poly, r2 * r, r);
    let a = if big { FRAC_PI_2 - a } else { a };
    let xneg = x < 0.0 || (x == 0.0 && x.is_sign_negative());
    let mut t = if x < 0.0 { PI - a } else { a };
    // special cases: zeros and infinities (sign applied at the end)
    let yinf_val = if x.is_infinite() { if x < 0.0 { 0.75 * PI } else { 0.25 * PI } } else { FRAC_PI_2 };
    t = if x.is_infinite() { if x < 0.0 { PI } else { 0.0 } } else { t };
    t = if x == 0.0 { FRAC_PI_2 } else { t };
    t = if y.is_infinite() { yinf_val } else { t };
    t = if y == 0.0 { if xneg { PI } else { 0.0 } } else { t };
    if y.is_nan() || x.is_nan() { f64::NAN } else { t.copysign(y) }
}

binary! {
    cnp_atan2_float32: f32 => |y, x| atan2_poly(y as f64, x as f64) as f32,
    cnp_atan2_float64: f64 => atan2_poly,
    cnp_hypot_float32: f32 => libm::hypotf,
    cnp_hypot_float64: f64 => libm::hypot,
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

complex_abs!(cnp_abs_complex64: f32 => libm::hypotf, cnp_abs_complex128: f64 => libm::hypot);

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
                let (a, b) = (atan2_poly(y, x), y.atan2(x));
                assert!(a == b || (a - b).abs() <= 4e-16 * b.abs().max(1e-300), "atan2({y},{x}) {a} vs {b}");
                assert_eq!(a.is_sign_negative(), b.is_sign_negative(), "sign atan2({y},{x})");
            }
        }
        assert!(atan2_poly(f64::NAN, 1.0).is_nan());
        for k in -2000..2000 {
            let x = k as f64 * 0.01237 + (k as f64).powi(3) * 1e-3;
            assert!((sin_poly(x) - x.sin()).abs() < 1e-15, "sin {x}");
            assert!((cos_poly(x) - x.cos()).abs() < 1e-15, "cos {x}");
        }
        let v = [0.5, -1e12, f64::NAN, f64::INFINITY, 3.0, 1.0, 2.0, -0.0, 7.0];
        let mut o = [0.0; 9];
        unsafe { cnp_sin_float64(v.as_ptr(), 8, o.as_mut_ptr(), 8, 9) };
        for (a, b) in o.iter().zip(v) {
            assert!(a == &b.sin() || (a.is_nan() && b.sin().is_nan()));
        }
    }

    #[test]
    fn complex_abs_values() {
        let z = [3.0f32, 4.0, 5.0, 12.0];
        let mut o = [0.0f32; 2];
        unsafe { cnp_abs_complex64(z.as_ptr(), 8, o.as_mut_ptr(), 4, 2) };
        assert_eq!(o, [5.0, 13.0]);
    }
}
