//! Port of `codon/runtime/numpy/loops.cpp` (Highway SIMD ufunc loops).
//!
//! PORT-NOTE: Highway's dynamic-dispatch vector math (`hwy/contrib/math`) is replaced by
//! scalar loops over std/libm functions with a contiguous fast path that LLVM can
//! auto-vectorize. Results are correctly rounded libm values rather than Highway's
//! approximations (which are within a few ulp), so outputs may differ in the last ulp.
//! Strides are in bytes, as in the C++ API.

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
    cnp_cos_float64: f64 => f64::cos,
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
    cnp_sin_float64: f64 => f64::sin,
    cnp_sinh_float32: f32 => f32::sinh,
    cnp_sinh_float64: f64 => f64::sinh,
    cnp_tanh_float32: f32 => f32::tanh,
    cnp_tanh_float64: f64 => f64::tanh,
}

binary! {
    cnp_atan2_float32: f32 => f32::atan2,
    cnp_atan2_float64: f64 => f64::atan2,
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
    }

    #[test]
    fn complex_abs_values() {
        let z = [3.0f32, 4.0, 5.0, 12.0];
        let mut o = [0.0f32; 2];
        unsafe { cnp_abs_complex64(z.as_ptr(), 8, o.as_mut_ptr(), 4, 2) };
        assert_eq!(o, [5.0, 13.0]);
    }
}
