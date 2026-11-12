//! Port of Highway's `hwy/contrib/math` (as used by Codon's numpy runtime) to
//! scalar, branchless Rust that LLVM auto-vectorizes when applied over fixed
//! chunks (see `loops.rs`). Same polynomials, reductions and evaluation order
//! (Estrin trees, fused multiply-adds), so results match Codon bit-for-bit on
//! targets where Highway uses FMA (aarch64, x86 AVX2+).
#![allow(clippy::excessive_precision, clippy::approx_constant)]

use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Shl, Shr, Sub};

/// Same-width unsigned integer (keeps f32 lanes 32-bit wide when vectorized).
pub trait Ui: Copy + PartialEq + BitAnd<Output = Self> + BitOr<Output = Self> + BitXor<Output = Self> + Not<Output = Self>
    + Shl<u32, Output = Self> + Shr<u32, Output = Self> {
    fn u(x: u64) -> Self;
    fn i(x: i32) -> Self;
    fn to_i32(self) -> i32;
    fn wadd(self, o: Self) -> Self;
    /// Reinterpret as signed and convert to float of the same width.
    fn sf<F: Hf<U = Self>>(self) -> F;
}
macro_rules! ui { ($($t:ty: $s:ty),*) => {$(impl Ui for $t {
    #[inline(always)] fn u(x: u64) -> Self { x as $t }
    #[inline(always)] fn i(x: i32) -> Self { x as $t }
    #[inline(always)] fn to_i32(self) -> i32 { self as i32 }
    #[inline(always)] fn wadd(self, o: Self) -> Self { self.wrapping_add(o) }
    #[inline(always)] fn sf<F: Hf<U = Self>>(self) -> F { F::from_sint(self as $s as i64) }
})*} }
ui!(u32: i32, u64: i64);

pub trait Hf:
    Copy + PartialOrd + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self>
{
    /// Mantissa bits, exponent bias, total bits.
    const MANT: u32;
    const BIAS: i64;
    const BITS: u32;
    /// `static_cast<T>(double)`.
    fn c(x: f64) -> Self;
    /// Per-type constant: `f` for f32, `d` for f64.
    fn pick(f: f32, d: f64) -> Self;
    fn sqrt(self) -> Self;
    type U: Ui;
    fn bits(self) -> Self::U;
    fn from_bits(b: Self::U) -> Self;
    fn is_nan(self) -> bool;
    /// Truncating, saturating conversion (NaN -> 0), like `ConvertTo`/`DemoteTo`.
    fn to_i32(self) -> i32;
    fn nearest_i32(self) -> i32;
    fn from_i32(i: i32) -> Self;
    fn from_sint(i: i64) -> Self;
    fn exp_poly(x: Self) -> Self;
    fn log_poly(x: Self) -> Self;
    fn asin_poly(x2: Self) -> Self;
    fn atan_poly(x: Self) -> Self;
    fn sincos_poly(x: Self) -> Self;
    /// (pi-part constants, cos limit, sin limit)
    const PI_PARTS: [f64; 4];
}

#[inline(always)]
pub fn fma<T: Hf + MulAdd>(a: T, b: T, c: T) -> T {
    if cfg!(any(target_arch = "aarch64", target_feature = "fma")) { a.mul_add_(b, c) } else { a * b + c }
}

pub trait MulAdd {
    fn mul_add_(self, b: Self, c: Self) -> Self;
}

/// Highway's `Estrin` evaluation tree over `c[0] + c[1] x + ...`.
#[inline(always)]
fn estrin<T: Hf + MulAdd, const N: usize>(x: T, c: [T; N]) -> T {
    let mut v = c;
    let mut len = N;
    // pairs, then pairwise combination with x^2, x^4, ... (odd leftover passes through)
    let mut p = x;
    while len > 1 {
        let half = len / 2;
        for i in 0..half {
            v[i] = fma(p, v[2 * i + 1], v[2 * i]);
        }
        if len % 2 == 1 {
            v[half] = v[len - 1];
        }
        len = half + len % 2;
        p = p * p;
    }
    v[0]
}

macro_rules! hf_impl {
    ($t:ty, $u:ty, $mant:expr, $bias:expr, $bits:expr, $pi:expr, $nearest:expr) => {
        impl MulAdd for $t {
            #[inline(always)]
            fn mul_add_(self, b: Self, c: Self) -> Self { self.mul_add(b, c) }
        }
        impl Hf for $t {
            const MANT: u32 = $mant;
            const BIAS: i64 = $bias;
            const BITS: u32 = $bits;
            const PI_PARTS: [f64; 4] = $pi;
            #[inline(always)] fn c(x: f64) -> Self { x as $t }
            #[inline(always)] fn pick(f: f32, d: f64) -> Self { if $bits == 32 { f as $t } else { d as $t } }
            #[inline(always)] fn sqrt(self) -> Self { <$t>::sqrt(self) }
            type U = $u;
            #[inline(always)] fn bits(self) -> $u { self.to_bits() }
            #[inline(always)] fn from_bits(b: $u) -> Self { <$t>::from_bits(b) }
            #[inline(always)] fn is_nan(self) -> bool { <$t>::is_nan(self) }
            // Via i64 so f64 lanes convert with one vector instruction; lanes out of
            // i32 range only occur where the kernel masks the result anyway.
            #[inline(always)] fn to_i32(self) -> i32 { self as i64 as i32 }
            #[inline(always)] fn nearest_i32(self) -> i32 { let f: fn($t) -> i32 = $nearest; f(self) }
            #[inline(always)] fn from_i32(i: i32) -> Self { i as $t }
            #[inline(always)] fn from_sint(i: i64) -> Self { if $bits == 32 { i as i32 as $t } else { i as $t } }
            #[inline(always)] fn exp_poly(x: Self) -> Self { fma(estrin(x, Polys::<$t>::EXP), x * x, x) }
            #[inline(always)] fn log_poly(x: Self) -> Self { Polys::<$t>::log(x) }
            #[inline(always)] fn asin_poly(x2: Self) -> Self { estrin(x2, Polys::<$t>::ASIN) }
            #[inline(always)] fn atan_poly(x: Self) -> Self { let y = x * x; fma(estrin(y, Polys::<$t>::ATAN), y * x, x) }
            #[inline(always)] fn sincos_poly(x: Self) -> Self { let y = x * x; fma(estrin(y, Polys::<$t>::SINCOS), y * x, x) }
        }
    };
}

struct Polys<T>(T);

impl Polys<f32> {
    const EXP: [f32; 6] = [0.5, 0.166666671633720397949219, 0.0416664853692054748535156, 0.00833336077630519866943359,
        0.00139304355252534151077271, 0.000198527617612853646278381];
    const ASIN: [f32; 5] = [0.1666677296, 0.07495029271, 0.04547423869, 0.02424046025, 0.04197454825];
    const ATAN: [f32; 8] = [-0.333331018686294555664062, 0.199926957488059997558594, -0.142027363181114196777344,
        0.106347933411598205566406, -0.0748900920152664184570312, 0.0425049886107444763183594,
        -0.0159569028764963150024414, 0.00282363896258175373077393];
    const SINCOS: [f32; 4] = [-1.66666597127914428710938e-1, 8.33307858556509017944336e-3, -1.981069071916863322258e-4,
        2.6083159809786593541503e-6];
    #[inline(always)]
    fn log(x: f32) -> f32 {
        let (x2, x4) = (x * x, x * x * (x * x));
        fma(fma(0.28498786688, x4, 0.66666662693), x2, fma(0.24279078841, x4, 0.40000972152) * x4)
    }
}

impl Polys<f64> {
    const EXP: [f64; 11] = [0.5, 0.166666666666666851703837, 0.0416666666666665047591422, 0.00833333333331652721664984,
        0.00138888888889774492207962, 0.000198412698960509205564975, 2.4801587159235472998791e-5,
        2.75572362911928827629423e-6, 2.75573911234900471893338e-7, 2.51112930892876518610661e-8,
        2.08860621107283687536341e-9];
    const ASIN: [f64; 12] = [0.1666666666666497543, 0.07500000000378581611, 0.04464285681377102438, 0.03038195928038132237,
        0.02237176181932048341, 0.01735956991223614604, 0.01388715184501609218, 0.01215360525577377331,
        0.006606077476277170610, 0.01929045477267910674, -0.01581918243329996643, 0.03161587650653934628];
    const ATAN: [f64; 19] = [-0.333333333333311110369124, 0.199999999996591265594148, -0.14285714266771329383765,
        0.111111105648261418443745, -0.090908995008245008229153, 0.0769219538311769618355029,
        -0.0666573579361080525984562, 0.0587666392926673580854313, -0.0523674852303482457616113,
        0.0466667150077840625632675, -0.0407629191276836500001934, 0.0337852580001353069993897,
        -0.0254517624932312641616861, 0.016599329773529201970117, -0.00889896195887655491740809,
        0.00370026744188713119232403, -0.00110611831486672482563471, 0.000209850076645816976906797,
        -1.88796008463073496563746e-5];
    const SINCOS: [f64; 9] = [-0.166666666666666657414808, 0.00833333333333332974823815, -0.000198412698412696162806809,
        2.75573192239198747630416e-6, -2.50521083763502045810755e-8, 1.60590430605664501629054e-10,
        -7.64712219118158833288484e-13, 2.81009972710863200091251e-15, -7.97255955009037868891952e-18];
    #[inline(always)]
    fn log(x: f64) -> f64 {
        let (x2, x4) = (x * x, x * x * (x * x));
        let a = fma(fma(fma(0.1479819860511658591, x4, 0.1818357216161805012), x4, 0.2857142874366239149), x4, 0.6666666666666735130);
        let b = fma(fma(0.1531383769920937332, x4, 0.2222219843214978396), x4, 0.3999999999940941908) * x4;
        fma(a, x2, b)
    }
}

hf_impl!(f32, u32, 23, 127, 32, [3.140625, 0.0009670257568359375, 6.2771141529083251953e-7, 1.2154201256553420762e-10],
    |x: f32| x.round_ties_even() as i64 as i32);
hf_impl!(f64, u64, 52, 1023, 64, [3.1415926218032836914, 3.1786509424591713469e-8, 1.2246467864107188502e-16, 1.2736634327021899816e-24],
    |x: f64| x.round_ties_even() as i64 as i32);

/// Branchless float select (bit mask): keeps LLVM from turning selects whose
/// arms contain a division back into branches, which blocks vectorization.
#[inline(always)]
pub fn fsel<T: Hf>(m: bool, a: T, b: T) -> T {
    let k = T::U::u(0).wadd(T::U::i(-(m as i32)));
    T::from_bits((a.bits() & k) | (b.bits() & !k))
}
#[inline(always)]
fn sign<T: Hf>(x: T) -> T::U { x.bits() & (T::U::u(1) << (T::BITS - 1)) }
#[inline(always)]
fn xor<T: Hf>(x: T, s: T::U) -> T { T::from_bits(x.bits() ^ s) }
#[inline(always)]
fn abs<T: Hf>(x: T) -> T { xor(x, sign(x)) }

/// 2^e (e small), Highway `Pow2I`.
#[inline(always)]
fn pow2i<T: Hf>(e: i32) -> T { T::from_bits(T::U::i(e + T::BIAS as i32) << T::MANT) }
#[inline(always)]
fn load_exp<T: Hf>(x: T, e: i32) -> T { let y = e >> 1; x * pow2i::<T>(y) * pow2i::<T>(e - y) }

#[inline(always)]
fn exp_q<T: Hf + MulAdd>(x: T) -> (i32, T) {
    let q = fma(x, T::c(1.442695040888963407359924681), xor(T::c(0.5), sign(x))).to_i32();
    let qf = T::from_i32(q);
    let r = fma(qf, T::pick(-0.693145751953125, -0.6931471805596629565116018), x);
    (q, fma(qf, T::pick(-1.428606765330187045e-6, -0.28235290563031577122588448175e-12), r))
}

#[inline(always)]
pub fn exp<T: Hf + MulAdd>(x: T) -> T {
    let (q, r) = exp_q(x);
    let y = load_exp(T::exp_poly(r) + T::c(1.0), q);
    fsel(x >= T::pick(-104.0, -1000.0), y, T::c(0.0))
}

#[inline(always)]
pub fn exp2<T: Hf + MulAdd>(x: T) -> T {
    let q = x.nearest_i32();
    let f = x - T::from_i32(q);
    let r = fma(f, T::pick(0.193147182464599609375, 0.1931471805599453139823396), f * T::c(0.5));
    let y = load_exp(T::exp_poly(r) + T::c(1.0), q);
    fsel(x >= T::pick(-150.0, -1075.0), y, T::c(0.0))
}

#[inline(always)]
pub fn expm1<T: Hf + MulAdd>(x: T) -> T {
    let (q, r) = exp_q(x);
    let y = T::exp_poly(r);
    let z = fsel(abs(x) < T::c(0.346573590279972654708616), y, load_exp(y + T::c(1.0), q) - T::c(1.0));
    fsel(x < T::pick(-104.0, -1000.0), T::c(-1.0), z)
}

#[inline(always)]
pub fn log_impl<T: Hf + MulAdd, const SUB: bool>(x: T) -> T {
    let f32_ = T::BITS == 32;
    let u = T::U::u;
    let magic = u(if f32_ { 0x3F3504F3 } else { 0x3FE6A09E00000000 });
    let expmask = u(if f32_ { 0x3F800000 } else { 0x3FF0000000000000 });
    let lower = u(if f32_ { 0 } else { 0xFFFFFFFF });
    let manmask = u(if f32_ { 0x7FFFFF } else { 0xFFFFF00000000 });
    let mut x = x;
    let mut scale = u(0);
    if SUB {
        let den = x < T::pick(1.175494351e-38, 2.2250738585072014e-308);
        x = fsel(den, x * T::pick(3.355443200e+7, 1.8014398509481984e+16), x);
        scale = if den { T::U::i(if f32_ { -25 } else { -54 }) } else { u(0) };
    }
    let eb = x.bits().wadd(expmask.wadd(!magic).wadd(u(1)));
    let e: T = (eb >> T::MANT).wadd(scale).wadd(T::U::i(-(T::BIAS as i32))).sf();
    let y = T::from_bits((x.bits() & lower) | (eb & manmask).wadd(magic));
    let ym1 = y - T::c(1.0);
    let z = ym1 / (y + T::c(1.0));
    let lo = e * T::pick(9.0580006145e-6, 1.90821492927058770002e-10);
    let t = fma(z, ym1 - T::log_poly(z), -lo) - ym1;
    fma(e, T::pick(0.69313812256, 0.693147180369123816490), -t)
}

#[inline(always)]
pub fn log1p<T: Hf + MulAdd>(x: T) -> T {
    let y = x + T::c(1.0);
    let pole = y == T::c(1.0);
    let div = fsel(pole, T::c(0.0), y) - T::c(1.0);
    fsel(pole, x, log_impl::<T, false>(y) * (x / div))
}

#[inline(always)]
fn asin_core<T: Hf + MulAdd>(x: T) -> (bool, T::U, T, T) {
    let s = sign(x);
    let ax = xor(x, s);
    let m = ax < T::c(0.5);
    let yy = fsel(m, ax * ax, fma(-ax, T::c(0.5), T::c(0.5)));
    let y = fsel(m, ax, yy.sqrt());
    (m, s, yy, y)
}

#[inline(always)]
pub fn acos<T: Hf + MulAdd>(x: T) -> T {
    let (m, s, yy, y) = asin_core(x);
    let t = T::asin_poly(yy) * (y * yy);
    let tpy = t + y;
    let z = fsel(m, T::c(1.57079632679489661923132169) - (xor(y, s) + xor(t, s)), tpy + tpy);
    fsel(m || x >= T::c(0.0), z, T::c(core::f64::consts::PI) - z)
}

#[inline(always)]
pub fn asin<T: Hf + MulAdd>(x: T) -> T {
    let (m, s, yy, y) = asin_core(x);
    let z0 = fma(T::asin_poly(yy), yy * y, y);
    let z1 = fma(-z0, T::c(2.0), T::c(1.57079632679489661923132169));
    T::from_bits(fsel(m, z0, z1).bits() | s)
}

/// Shared tail of Highway's Acosh/Asinh: log(y2) with the log1p-style pole fix.
#[inline(always)]
fn ash_tail<T: Hf + MulAdd>(y1: T, y2: T) -> (T, T) {
    let z = log_impl::<T, false>(y2);
    let pole = y2 == T::c(1.0);
    let div = fsel(pole, T::c(0.0), y2) - T::c(1.0);
    (z, fsel(pole, y1, z * y1 / div))
}

#[inline(always)]
pub fn acosh<T: Hf + MulAdd>(x: T) -> T {
    let (one, two) = (T::c(1.0), T::c(2.0));
    let large = x > T::c(268435456.0);
    let gt2 = x > two;
    let xm1 = x - one;
    let y0 = fma(two, x, -(one / (fma(x, x, -one).sqrt() + x)));
    let y1 = fma(xm1, two, xm1 * xm1).sqrt() + xm1;
    let y2 = fsel(gt2, fsel(large, x, y0), y1 + one);
    let (z, small) = ash_tail(y1, y2);
    fsel(gt2, z, small) + fsel(large, T::c(0.693147180559945286227), T::c(0.0))
}

#[inline(always)]
pub fn asinh<T: Hf + MulAdd>(x: T) -> T {
    let (one, two) = (T::c(1.0), T::c(2.0));
    let s = sign(x);
    let ax = xor(x, s);
    let large = ax > T::c(268435456.0);
    let lt2 = ax < two;
    let x2 = x * x;
    let sq = (x2 + one).sqrt();
    let y0 = fma(ax, two, one / (sq + ax));
    let y1 = x2 / (sq + one) + ax;
    let y2 = fsel(lt2, y1 + one, fsel(large, ax, y0));
    let (z, l) = ash_tail(y1, y2);
    let y = fsel(ax < T::c(1.0 / 268435456.0), x, l);
    let r = fsel(lt2, y, z) + fsel(large, T::c(0.693147180559945286227), T::c(0.0));
    T::from_bits(r.bits() | s)
}

#[inline(always)]
pub fn atan<T: Hf + MulAdd>(x: T) -> T {
    let s = sign(x);
    let ax = xor(x, s);
    let m = ax > T::c(1.0);
    let y = T::atan_poly(fsel(m, T::c(1.0) / fsel(m, ax, T::c(1.0)), ax));
    T::from_bits(fsel(m, T::c(1.57079632679489661923132169) - y, y).bits() | s)
}

#[inline(always)]
pub fn atanh<T: Hf + MulAdd>(x: T) -> T {
    let s = sign(x);
    let ax = xor(x, s);
    log1p((ax + ax) / (T::c(1.0) - ax)) * xor(T::c(0.5), s)
}

#[inline(always)]
pub fn atan2<T: Hf + MulAdd>(y: T, x: T) -> T {
    let (zero, pi) = (T::c(0.0), T::c(core::f64::consts::PI));
    let half = T::c(0.5);
    let pi2 = pi * half;
    let inf = |v: T| abs(v) == T::c(f64::INFINITY);
    let (xinf, xneg) = (inf(x), x < zero);
    let xneg_pi = fsel(xneg, pi, zero);
    let yinf_v = half * fsel(xinf, pi2 + xneg_pi, pi);
    let mut t = atan(y / x) + xor(xneg_pi, sign(y));
    t = fsel(xinf, xneg_pi, t);
    t = fsel(x == zero, pi2, t);
    t = fsel(inf(y), yinf_v, t);
    t = fsel(y == zero, xneg_pi, t);
    let r = fsel(y.is_nan() || x.is_nan(), T::c(f64::NAN), xor(abs(t), sign(y)));
    // Codon: atan2(-0, -0) = -pi, atan2(+0, -0) = +pi.
    fsel(x == zero && sign(x) != T::U::u(0) && y == zero, xor(pi, sign(y)), r)
}

/// Highway `Sin` (`COS = false`) / `Cos`, valid for |x| below Codon's limits.
#[inline(always)]
pub fn sincos<T: Hf + MulAdd, const COS: bool>(x: T) -> T {
    let ax = abs(x);
    let one_pi = T::c(0.31830988618379067153);
    let (q, h) = if COS {
        ((ax * one_pi).to_i32().wrapping_shl(1).wrapping_add(1), T::c(-0.5))
    } else {
        (fma(ax, one_pi, T::c(0.5)).to_i32(), T::c(-1.0))
    };
    let qf = T::from_i32(q);
    let mut r = ax;
    for p in T::PI_PARTS {
        r = fma(qf, h * T::c(p), r);
    }
    let s = if COS {
        T::U::i(!q & 2) << (T::BITS - 2)
    } else {
        (T::U::i(q & 1) << (T::BITS - 1)) ^ sign(x)
    };
    T::sincos_poly(xor(r, s))
}

#[inline(always)]
pub fn sinh<T: Hf + MulAdd>(x: T) -> T {
    let s = sign(x);
    let y = expm1(xor(x, s));
    xor((y + T::c(2.0)) / (y + T::c(1.0)) * (y * T::c(0.5)), s)
}

#[inline(always)]
pub fn tanh<T: Hf + MulAdd>(x: T) -> T {
    let s = sign(x);
    let ax = xor(x, s);
    let y = expm1(ax * T::c(2.0));
    xor(fsel(ax > T::c(18.714973875), T::c(1.0), y / (y + T::c(2.0))), s)
}

/// Highway `Hypot` (exponent-rescaled sqrt(a^2 + b^2)).
#[inline(always)]
pub fn hypot<T: Hf + MulAdd>(a: T, b: T) -> T {
    let (m, bias) = (T::MANT, T::BIAS as i32);
    let (kmin, kmax) = (bias / 2 + m as i32, bias + bias / 2 - 1);
    let (aa, ab) = (abs(a), abs(b));
    let inf = T::c(f64::INFINITY);
    let (ea, eb) = ((aa.bits() >> m).to_i32(), (ab.bits() >> m).to_i32());
    let mx = ea.max(eb);
    let zero = T::c(0.0);
    let mn = if aa == zero || ab == zero { mx } else { ea.min(eb) };
    let scl = (kmin - mn).max(0).min(kmax - mx);
    let f = T::from_bits(T::U::i(bias + scl) << m);
    let g = T::from_bits(T::U::i(bias - scl) << m);
    let (sa, sb) = (aa * f, ab * f);
    let h = fma(sa, sa, sb * sb).sqrt() * g;
    fsel(aa == inf || ab == inf, inf, h)
}

/// Log/log2/log10 with Codon's special-case wrapper.
#[inline(always)]
pub fn log_codon<T: Hf + MulAdd>(x: T, k: T) -> T {
    let (zero, inf) = (T::c(0.0), T::c(f64::INFINITY));
    let l = log_impl::<T, true>(x);
    let l = if k == T::c(1.0) { l } else { l * k };
    fsel(x == zero, -inf, fsel(x < zero, T::c(f64::NAN), fsel(x == inf || x.is_nan(), x, l)))
}
