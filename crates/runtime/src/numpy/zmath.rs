//! Port of `codon/runtime/numpy/zmath.cpp`: single-precision complex math wrappers that
//! sidestep ABI issues with `complex<float>` returns.

use num_complex::Complex32;

macro_rules! cfunc {
    ($($name:ident => $method:ident),* $(,)?) => {$(
        /// # Safety
        /// `z` must point to two writable floats.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(r: f32, i: f32, z: *mut f32) {
            let y = Complex32::new(r, i).$method();
            unsafe {
                *z = y.re;
                *z.add(1) = y.im;
            }
        }
    )*};
}

// PORT-NOTE: num-complex formulas may differ from libc++ std::complex in the last ulp
// and in some branch-cut / infinity special cases.
cfunc! {
    cnp_cexpf => exp,
    cnp_clogf => ln,
    cnp_csqrtf => sqrt,
    cnp_ccoshf => cosh,
    cnp_csinhf => sinh,
    cnp_ctanhf => tanh,
    cnp_cacoshf => acosh,
    cnp_casinhf => asinh,
    cnp_catanhf => atanh,
    cnp_ccosf => cos,
    cnp_csinf => sin,
    cnp_ctanf => tan,
    cnp_cacosf => acos,
    cnp_casinf => asin,
    cnp_catanf => atan,
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
