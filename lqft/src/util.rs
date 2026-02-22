use std::simd::Simd;

use atomic_float::{AtomicF32, AtomicF64};

pub type AtomicFType = <FType as AtomicFTypeSelector>::AtomicTy;
pub type FType = f64;
pub type SimdUType = Simd<usize, SIMD_LANES>;
pub type SimdFType = Simd<FType, SIMD_LANES>;

pub const SIMD_LANES: usize = 4;

/// Selects the correct precision atomic floating point type.
pub trait AtomicFTypeSelector {
    type AtomicTy;
}

impl AtomicFTypeSelector for f32 {
    type AtomicTy = AtomicF32;
}

impl AtomicFTypeSelector for f64 {
    type AtomicTy = AtomicF64;
}
