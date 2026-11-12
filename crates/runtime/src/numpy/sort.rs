//! numpy sort kernels.
//!
//! Integers and floats use `radsort` (stable LSD radix sort, 2-4x faster than
//! comparison sorts at 10M elements, faster than Codon's vqsort); `uint128` uses
//! `glidesort` by key. Stable is a strict superset of the unstable contract. Ordering semantics: `cnp_sort_float*` places NaNs last and
//! keeps `-0.0` before `+0.0`; `*_nan_last` places NaNs last.

/// Compact non-NaN values to the front and fill the tail with canonical NaN.
/// Returns the number of non-NaN values.
fn partition_nan_last<T: Copy + PartialEq>(d: &mut [T], nan: T) -> usize {
    let mut w = 0;
    for r in 0..d.len() {
        let v = d[r];
        #[allow(clippy::eq_op)]
        if v == v {
            d[w] = v;
            w += 1;
        }
    }
    for x in &mut d[w..] {
        *x = nan;
    }
    w
}

unsafe fn slice<'a, T>(data: *mut T, n: i64) -> &'a mut [T] {
    if n <= 0 || data.is_null() { &mut [] } else { unsafe { core::slice::from_raw_parts_mut(data, n as usize) } }
}

macro_rules! int_sort {
    ($($name:ident: $t:ty),*) => {$(
        /// # Safety
        /// `data` valid for `n` elements.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(data: *mut $t, n: i64) {
            radsort::sort(unsafe { slice(data, n) });
        }
    )*};
}

int_sort!(cnp_sort_int16: i16, cnp_sort_uint16: u16, cnp_sort_int32: i32, cnp_sort_uint32: u32,
          cnp_sort_int64: i64, cnp_sort_uint64: u64);

macro_rules! float_sort {
    ($t:ty, $data:expr) => {{
        let d: &mut [$t] = $data;
        let end = partition_nan_last(d, <$t>::NAN);
        // radsort uses the IEEE total order: -0.0 before +0.0, matching C++
        // `sortPreservingZeros` (NaNs were moved out above).
        radsort::sort(&mut d[..end]);
    }};
}

/// # Safety
/// `data` valid for `n` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnp_sort_float32(data: *mut f32, n: i64) {
    float_sort!(f32, unsafe { slice(data, n) });
}

/// # Safety
/// `data` valid for `n` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnp_sort_float64(data: *mut f64, n: i64) {
    float_sort!(f64, unsafe { slice(data, n) });
}

/// # Safety
/// `data` valid for `n` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnp_sort_float32_nan_last(data: *mut f32, n: i64) {
    float_sort!(f32, unsafe { slice(data, n) });
}

/// # Safety
/// `data` valid for `n` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnp_sort_float64_nan_last(data: *mut f64, n: i64) {
    float_sort!(f64, unsafe { slice(data, n) });
}

/// `hwy::uint128_t`: {lo, hi} little-endian halves; ordered by (hi, lo).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U128 {
    pub lo: u64,
    pub hi: u64,
}

/// # Safety
/// `data` valid for `n` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cnp_sort_uint128(data: *mut U128, n: i64) {
    glidesort::sort_by_key(unsafe { slice(data, n) }, |x: &U128| (x.hi, x.lo));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ints() {
        let mut v: Vec<i32> = vec![5, -1, 3, 3, 0, i32::MIN, i32::MAX];
        unsafe { cnp_sort_int32(v.as_mut_ptr(), v.len() as i64) };
        assert_eq!(v, vec![i32::MIN, -1, 0, 3, 3, 5, i32::MAX]);
        let mut u: Vec<u16> = (0..1000u16).rev().collect();
        unsafe { cnp_sort_uint16(u.as_mut_ptr(), u.len() as i64) };
        assert!(u.windows(2).all(|w| w[0] <= w[1]));
        unsafe { cnp_sort_int64(core::ptr::null_mut(), 0) };
    }

    #[test]
    fn floats() {
        let mut v = vec![1.0f64, f64::NAN, -0.0, 0.0, -0.0, -3.0, f64::INFINITY, 0.0];
        unsafe { cnp_sort_float64(v.as_mut_ptr(), v.len() as i64) };
        assert_eq!(&v[..7].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            &[-3.0f64, -0.0, -0.0, 0.0, 0.0, 1.0, f64::INFINITY].iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        assert!(v[7].is_nan());
        let mut f = vec![f32::NAN, 2.0, 1.0, f32::NAN];
        unsafe { cnp_sort_float32_nan_last(f.as_mut_ptr(), 4) };
        assert_eq!(&f[..2], &[1.0, 2.0]);
        assert!(f[2].is_nan() && f[3].is_nan());
    }

    #[test]
    fn u128s() {
        let mut v = vec![U128 { lo: 5, hi: 1 }, U128 { lo: 9, hi: 0 }, U128 { lo: 1, hi: 1 }];
        unsafe { cnp_sort_uint128(v.as_mut_ptr(), 3) };
        assert_eq!(v, vec![U128 { lo: 9, hi: 0 }, U128 { lo: 1, hi: 1 }, U128 { lo: 5, hi: 1 }]);
    }
}
