//! Runtime type-information layout shared with
//! compiler-emitted code.

use core::ffi::c_void;

use crate::types::{SeqInt, SeqStr};

/// Field indices of [`TypeInfo`] as used by the code generator.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeInfoField {
    TypeId,
    MroCount,
    MroIds,
    MroOffsets,
    SortedMroIds,
    SlotCount,
    Slots,
    MethodCount,
    Methods,
    Polymorphic,
    RawName,
    NiceName,
    BaseName,
    ParameterCount,
    Parameters,
    ValueSize,
    Reference,
    InlineValue,
    ExceptionOffset,
    SystemExitOffset,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TypeSlot {
    pub name: SeqStr,
    pub ty: SeqInt,
    pub offset: SeqInt,
}

pub const METHOD_HASH_MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;
pub const METHOD_HASH_SHIFT: u32 = 32;
pub const MAX_LINEAR_MRO: SeqInt = 8;
pub const ANY_INLINE_BYTES: u32 = 8;
pub const CATCH_ALL_SELECTOR: i32 = 0;
pub const BASE_EXCEPTION_SELECTOR: i32 = -1;

pub const fn method_hash(id: u64) -> u64 {
    let mixed = id.wrapping_mul(METHOD_HASH_MULTIPLIER);
    mixed ^ (mixed >> METHOD_HASH_SHIFT)
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MethodSlot {
    pub id: SeqInt,
    pub target: *mut c_void,
}

/// Runtime type information record; layout must match the code generator.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TypeInfo {
    pub id: SeqInt,
    pub n_mro: SeqInt,
    pub mro: *const SeqInt,
    /// The first `n_mro` entries are segment offsets in MRO order. For RTTI objects an
    /// immutable keyed index follows: mask, then mask+1 {owner ID, offset} pairs.
    pub offsets: *const SeqInt,
    pub sorted_mro: *const SeqInt,
    pub n_slots: SeqInt,
    pub slots: *const TypeSlot,
    pub n_methods: SeqInt,
    pub methods: *const MethodSlot,
    pub rtti: bool,
    pub raw_name: SeqStr,
    pub nice_name: SeqStr,
    pub base_name: SeqStr,
    pub n_params: SeqInt,
    pub params: *const SeqInt,
    pub size: SeqInt,
    pub reference: bool,
    pub inline_value: bool,
    pub exception_offset: SeqInt,
    pub system_exit_offset: SeqInt,
}

/// `isinstance` check against a type id, as used by the exception personality.
///
/// # Safety
/// `obj` must point to an object whose first word is a `*const TypeInfo`.
pub unsafe fn isinstance(obj: *mut c_void, ty: SeqInt) -> bool {
    unsafe {
        let info = &**(obj as *const *const TypeInfo);
        if info.id == ty {
            return true;
        }
        if info.n_mro > MAX_LINEAR_MRO {
            let s = core::slice::from_raw_parts(info.sorted_mro, info.n_mro as usize);
            return s.binary_search(&ty).is_ok();
        }
        (1..info.n_mro).any(|i| *info.mro.add(i as usize) == ty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_and_isinstance() {
        assert_eq!(method_hash(0), 0);
        assert_ne!(method_hash(1), 1);
        let mro = [5i64, 3, 1];
        let sorted = [1i64, 3, 5];
        let info = TypeInfo {
            id: 5,
            n_mro: 3,
            mro: mro.as_ptr(),
            offsets: core::ptr::null(),
            sorted_mro: sorted.as_ptr(),
            n_slots: 0,
            slots: core::ptr::null(),
            n_methods: 0,
            methods: core::ptr::null(),
            rtti: true,
            raw_name: SeqStr::EMPTY,
            nice_name: SeqStr::EMPTY,
            base_name: SeqStr::EMPTY,
            n_params: 0,
            params: core::ptr::null(),
            size: 0,
            reference: true,
            inline_value: false,
            exception_offset: -1,
            system_exit_offset: -1,
        };
        let mut obj: *const TypeInfo = &info;
        let p = &mut obj as *mut _ as *mut c_void;
        unsafe {
            assert!(isinstance(p, 5));
            assert!(isinstance(p, 1));
            assert!(!isinstance(p, 2));
        }
    }
}
