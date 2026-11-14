//! torch kernels (src/torchk/torchk.c, aarch64 only): outputs must equal torch 2.15.0 CPU
//! bit for bit. Golden bits come from torch.exp/expm1/log in torch 2.15.0 (macOS arm64).
#![cfg(target_arch = "aarch64")]

extern crate typhonrt;

unsafe extern "C" {
    fn typhon_sleef_expf_v(i: *const f32, o: *mut f32, n: i64) -> i64;
    fn typhon_sleef_expm1f_v(i: *const f32, o: *mut f32, n: i64) -> i64;
    fn typhon_sleef_logf_v(i: *const f32, o: *mut f32, n: i64) -> i64;
    fn typhon_nnp_conv_output(
        n: i64, c: i64, o: i64, h: i64, w: i64, kh: i64, kw: i64, pt: i64, pl: i64,
        x: *const f32, k: *const f32, b: *const f32, y: *mut f32,
    ) -> i64;
}

const X: [u32; 16] = [
    0x0, 0x80000000, 0x3f800000, 0xbf800000, 0x3f000000, 0x42b16666, 0xc2ae999a, 0xc2d00000,
    0x33d6bf95, 0x40533333, 0xc1480000, 0x40e80000, 0x7f800000, 0x40000000, 0x42c80000, 0x6ce3ee,
];
const EXP: [u32; 16] = [
    0x3f800000, 0x3f800000, 0x402df854, 0x3ebc5ab2, 0x3fd3094c, 0x7f7a37fc, 0x84c38b, 0x0,
    0x3f800001, 0x41d8e6af, 0x367a176a, 0x44b0035b, 0x7f800000, 0x40ec7326, 0x7f800000, 0x3f800000,
];
const EXPM1: [u32; 16] = [
    0x0, 0x80000000, 0x3fdbf0a9, 0xbf21d2a7, 0x3f261299, 0x7f7a37fc, 0xbf800000, 0xbf800000,
    0x33d6bf96, 0x41d0e6af, 0xbf7fffc1, 0x44afe35b, 0x7f800000, 0x40cc7326, 0x7f800000, 0x6ce3ee,
];
const LOG: [u32; 16] = [
    0xff800000, 0xff800000, 0x0, 0x7fc00000, 0xbf317218, 0x408f8740, 0x7fc00000, 0x7fc00000,
    0xc180f1dc, 0x3f98d273, 0x7fc00000, 0x3ffd9175, 0x7f800000, 0x3f317218, 0x40935d8e, 0xc2aeff18,
];

fn check(f: unsafe extern "C" fn(*const f32, *mut f32, i64) -> i64, want: &[u32; 16]) {
    let x: Vec<f32> = X.iter().map(|&b| f32::from_bits(b)).collect();
    let mut y = vec![-7.0f32; 18];
    // n = 18: 16 processed (four full blocks), the 2-element tail is left to the caller.
    let mut xin = x.clone();
    xin.extend([1.0, 1.0]);
    let m = unsafe { f(xin.as_ptr(), y.as_mut_ptr(), 18) };
    assert_eq!(m, 16);
    assert_eq!(y[16], -7.0);
    for i in 0..16 {
        let (g, w) = (y[i], f32::from_bits(want[i]));
        assert!(g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()), "lane {i}: {:#x} != {:#x}", g.to_bits(), want[i]);
    }
}

#[test]
fn sleef_matches_torch() {
    check(typhon_sleef_expf_v, &EXP);
    check(typhon_sleef_expm1f_v, &EXPM1);
    check(typhon_sleef_logf_v, &LOG);
}

#[test]
fn nnpack_conv_matches_direct() {
    // 16 x 3 x 1 x 37 input, 5 outputs, 1x5 kernel, pad 2: compare against a float64 direct
    // conv (tolerance only; exact torch parity is checked by the BCI regression test).
    let (n, c, o, w, k, p) = (16usize, 3usize, 5usize, 37usize, 5usize, 2usize);
    let mut s = 12345u32;
    let mut r = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    };
    let x: Vec<f32> = (0..n * c * w).map(|_| r()).collect();
    let kw: Vec<f32> = (0..o * c * k).map(|_| r()).collect();
    let b: Vec<f32> = (0..o).map(|_| r()).collect();
    let mut y = vec![0f32; n * o * w];
    let st = unsafe {
        typhon_nnp_conv_output(n as i64, c as i64, o as i64, 1, w as i64, 1, k as i64, 0, p as i64,
            x.as_ptr(), kw.as_ptr(), b.as_ptr(), y.as_mut_ptr())
    };
    assert_eq!(st, 0);
    for ni in 0..n {
        for oi in 0..o {
            for t in 0..w {
                let mut acc = b[oi] as f64;
                for ci in 0..c {
                    for ki in 0..k {
                        let ti = t as isize + ki as isize - p as isize;
                        if ti >= 0 && (ti as usize) < w {
                            acc += x[(ni * c + ci) * w + ti as usize] as f64 * kw[(oi * c + ci) * k + ki] as f64;
                        }
                    }
                }
                let g = y[(ni * o + oi) * w + t] as f64;
                assert!((g - acc).abs() < 1e-4, "n{ni} o{oi} t{t}: {g} vs {acc}");
            }
        }
    }
}
