//! Single-precision complex math wrappers that
//! sidestep ABI issues with `complex<float>` returns.

/// C `float _Complex` layout (two floats; HFA on aarch64, packed SSE on x86-64).
#[repr(C)]
#[derive(Clone, Copy)]
struct CFloatComplex {
    re: f32,
    im: f32,
}

macro_rules! cfunc {
    ($($name:ident => $libm:ident),* $(,)?) => {
        unsafe extern "C" {
            $(fn $libm(z: CFloatComplex) -> CFloatComplex;)*
        }
        $(
        /// # Safety
        /// `z` must point to two writable floats.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(r: f32, i: f32, z: *mut f32) {
            let y = unsafe { $libm(CFloatComplex { re: r, im: i }) };
            unsafe {
                *z = y.re;
                *z.add(1) = y.im;
            }
        }
    )*};
}

// PORT-NOTE: C99 libm complex functions are used (instead of a Rust complex crate).
// Inverse trig real-axis values below are handled explicitly because UCRT does
// not consistently honour the signed-zero side of their branch cuts.
// bionic only has clogf from API 26 (minimum is 24): log|z| + i*arg(z) (atan2 keeps the
// signed-zero branch cut; hypot avoids overflow).
#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnp_clogf(r: f32, i: f32, z: *mut f32) {
    let (rd, id) = (r as f64, i as f64);
    unsafe {
        *z = libm::log(libm::hypot(rd, id)) as f32;
        *z.add(1) = libm::atan2(id, rd) as f32;
    }
}
#[cfg(not(target_os = "android"))]
cfunc! { cnp_clogf => clogf }

cfunc! {
    cnp_cexpf => cexpf,
    cnp_csqrtf => csqrtf,
    cnp_ccoshf => ccoshf,
    cnp_csinhf => csinhf,
    cnp_ctanhf => ctanhf,
    cnp_cacoshf => cacoshf,
    cnp_casinhf => casinhf,
    cnp_catanhf => catanhf,
    cnp_ccosf => ccosf,
    cnp_csinf => csinf,
    cnp_ctanf => ctanf,
    cnp_catanf => catanf,
}

unsafe extern "C" {
    fn casinf(z: CFloatComplex) -> CFloatComplex;
    fn cacosf(z: CFloatComplex) -> CFloatComplex;
}

// asin(x +/- i0) has imaginary part +/- acosh(|x|) outside [-1, 1].
// acos has the opposite imaginary sign. Evaluate the real axis directly,
// including endpoints, infinities and signed zeros, without perturbing the
// input off the cut or relying on the platform's complex branch convention.
fn inverse_real_axis(r: f32, i: f32, acos: bool) -> CFloatComplex {
    let a = r.abs();
    let (re, magnitude) = if a <= 1.0 {
        (if acos { libm::acosf(r) } else { libm::asinf(r) }, 0.0)
    } else {
        let re = if acos {
            if r.is_sign_negative() { core::f32::consts::PI } else { 0.0 }
        } else {
            core::f32::consts::FRAC_PI_2.copysign(r)
        };
        (re, libm::acosh(r.abs() as f64) as f32)
    };
    CFloatComplex { re, im: magnitude.copysign(if acos { -i } else { i }) }
}

/// # Safety
/// `z` must point to two writable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnp_casinf(r: f32, i: f32, z: *mut f32) {
    let y = if i == 0.0 && !r.is_nan() {
        inverse_real_axis(r, i, false)
    } else {
        unsafe { casinf(CFloatComplex { re: r, im: i }) }
    };
    unsafe { *z = y.re; *z.add(1) = y.im; }
}

/// # Safety
/// `z` must point to two writable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnp_cacosf(r: f32, i: f32, z: *mut f32) {
    let y = if i == 0.0 && !r.is_nan() {
        inverse_real_axis(r, i, true)
    } else {
        unsafe { cacosf(CFloatComplex { re: r, im: i }) }
    };
    unsafe { *z = y.re; *z.add(1) = y.im; }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(f: unsafe extern "C" fn(f32, f32, *mut f32), r: f32, i: f32) -> (f32, f32) {
        let mut z = [0f32; 2];
        unsafe { f(r, i, z.as_mut_ptr()) };
        (z[0], z[1])
    }

    fn close(a: (f32, f32), b: (f32, f32)) -> bool {
        (a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5
    }

    #[test]
    fn zmath() {
        let pi = core::f32::consts::PI;
        assert!(close(call(cnp_cexpf, 0.0, pi), (-1.0, 0.0)));
        assert!(close(call(cnp_clogf, -1.0, 0.0), (0.0, pi)));
        assert!(close(call(cnp_csqrtf, -4.0, 0.0), (0.0, 2.0)));
        assert!(close(call(cnp_csinf, 0.0, 0.0), (0.0, 0.0)));
        assert!(close(call(cnp_ccosf, 0.0, 0.0), (1.0, 0.0)));
        assert!(close(call(cnp_catanf, 1.0, 0.0), (pi / 4.0, 0.0)));
        assert!(close(call(cnp_ctanhf, 0.0, 0.0), (0.0, 0.0)));
        let (r, i) = call(cnp_casinf, 0.5, 0.25);
        assert!(close(call(cnp_csinf, r, i), (0.5, 0.25)));
        let (r, i) = call(cnp_cacoshf, 2.0, 1.0);
        assert!(close(call(cnp_ccoshf, r, i), (2.0, 1.0)));
        let (r, i) = call(cnp_casinhf, 0.3, -0.7);
        assert!(close(call(cnp_csinhf, r, i), (0.3, -0.7)));
        let (r, i) = call(cnp_catanhf, 0.2, 0.1);
        assert!(close(call(cnp_ctanhf, r, i), (0.2, 0.1)));
        let (r, i) = call(cnp_cacosf, 0.2, 0.1);
        assert!(close(call(cnp_ccosf, r, i), (0.2, 0.1)));
        assert!(close(call(cnp_ctanf, 0.0, 0.0), (0.0, 0.0)));
    }
}

#[cfg(test)]
mod branch_cut_tests {
    use super::*;
    #[test]
    fn asin_acos_real_axis_signed_zero_imag() {
        let mut z = [0f32; 2];
        for r in [f32::NEG_INFINITY, -f32::MAX, -2.0, -1.0, -0.5, -0.0,
            0.0, 0.5, 1.0, 2.0, f32::MAX, f32::INFINITY] {
            for i in [0.0_f32, -0.0] {
                let magnitude = if r.abs() <= 1.0 { 0.0 } else { libm::acosh(r.abs() as f64) as f32 };
                unsafe { cnp_casinf(r, i, z.as_mut_ptr()) };
                let re = if r.abs() <= 1.0 { libm::asinf(r) } else { core::f32::consts::FRAC_PI_2.copysign(r) };
                assert_eq!(z[0].to_bits(), re.to_bits(), "asin real({r}, {i})");
                assert_eq!(z[1].abs(), magnitude, "asin imag magnitude({r}, {i})");
                assert_eq!(z[1].is_sign_negative(), i.is_sign_negative(), "asin imag sign({r}, {i})");
                unsafe { cnp_cacosf(r, i, z.as_mut_ptr()) };
                let re = if r.abs() <= 1.0 { libm::acosf(r) } else if r.is_sign_negative() { core::f32::consts::PI } else { 0.0 };
                assert_eq!(z[0].to_bits(), re.to_bits(), "acos real({r}, {i})");
                assert_eq!(z[1].abs(), magnitude, "acos imag magnitude({r}, {i})");
                assert_eq!(z[1].is_sign_negative(), !i.is_sign_negative(), "acos imag sign({r}, {i})");
            }
        }
        // Independent known value: acosh(2) = log(2 + sqrt(3)).
        unsafe { cnp_casinf(2.0, 0.0, z.as_mut_ptr()) };
        assert!((z[1] - 1.3169578969248167_f32).abs() < 2e-7);
    }
}
