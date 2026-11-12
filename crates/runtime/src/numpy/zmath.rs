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

// PORT-NOTE: C99 libm complex functions are used (instead of a Rust complex crate) so
// branch cuts honour signed zeros like libc++ `std::complex`, which C++ wraps.
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
    cnp_cacosf => cacosf,
    cnp_casinf => casinf,
    cnp_catanf => catanf,
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
    fn asin_acos_real_above_one_positive_zero_imag() {
        let mut z = [0f32; 2];
        unsafe { cnp_casinf(2.0, 0.0, z.as_mut_ptr()) };
        assert!(z[1] > 0.0);
        unsafe { cnp_cacosf(2.0, 0.0, z.as_mut_ptr()) };
        assert!(z[1] < 0.0);
    }
}
