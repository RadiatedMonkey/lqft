#![feature(portable_simd)]

use std::cell::UnsafeCell;
use std::hint::black_box;
use std::simd::{Select, Simd, StdFloat, ToBytes, f64x4};
use std::simd::cmp::SimdPartialOrd;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::RngExt;
use rand::rngs::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::{Rng, SeedableRng};
use rayon::prelude::*;

type SimdFType = f64x4;

struct SyncWrapper(pub Vec<UnsafeCell<f64>>);

unsafe impl Send for SyncWrapper {}
unsafe impl Sync for SyncWrapper {}

struct GlobalLattice {
    sites: SyncWrapper,
    red_indices: Vec<usize>,
    black_indices: Vec<usize>
}

struct ColoredLattice {
    black: Vec<f64>,
    red: Vec<f64>,
}

fn old_flip(lattice: &mut GlobalLattice) {
    let sites = &lattice.sites;
    lattice.red_indices.par_iter().for_each_init(
        || {
            rand::rng()
        },
        |rng, &index| {
            let val = unsafe { *sites.0[index].get() };
            let new_val = rng.random_range(val - 0.5..val + 0.5);

            let prob = (val - new_val).exp();
            let realised = rng.random_range(0.0..1.0);

            if realised > prob {
                unsafe { *sites.0[index].get() = new_val }
            }
        });

    lattice.black_indices.par_iter().for_each_init(
        || {
            rand::rng()
        },
        |rng, &index| {
            let val = unsafe { *sites.0[index].get() };
            let new_val = rng.random_range(val - 0.5..val + 0.5);

            let prob = (val - new_val).exp();
            let realised = rng.random_range(0.0..1.0);

            if realised > prob {
                unsafe { *sites.0[index].get() = new_val }
            }
        });
}

fn reg_flip(lattice: &mut GlobalLattice) {
    let sites = &lattice.sites;
    lattice.red_indices.par_iter().for_each_init(
        || {
            let mut rng = rand::rng();
            Xoshiro256PlusPlus::from_rng(&mut rng)
        },
        |rng, &index| {
        let val = unsafe { *sites.0[index].get() };
        let new_val = rng.random_range(val - 0.5..val + 0.5);

        let prob = (val - new_val).exp();
        let realised = rng.random_range(0.0..1.0);

        if realised > prob {
            unsafe { *sites.0[index].get() = new_val }
        }
    });

    lattice.black_indices.par_iter().for_each_init(
        || {
            let mut rng = rand::rng();
            Xoshiro256PlusPlus::from_rng(&mut rng)
        },
        |rng, &index| {
        let val = unsafe { *sites.0[index].get() };
        let new_val = rng.random_range(val - 0.5..val + 0.5);

        let prob = (val - new_val).exp();
        let realised = rng.random_range(0.0..1.0);

        if realised > prob {
            unsafe { *sites.0[index].get() = new_val }
        }
    });
}

fn contig_flip(lattice: &mut ColoredLattice) {
    let red = &mut lattice.red;
    red.par_iter_mut().for_each_init(
        || {
            let mut rng = rand::rng();
            Xoshiro256PlusPlus::from_rng(&mut rng)
        },
        |rng, site| {
        let curr = *site;
        let new_val = rng.random_range(curr - 0.5..curr + 0.5);

        let prob = (curr - new_val).exp();
        let realised = rng.random_range(0.0..1.0);

        let keep_new = (realised > prob) as i64 as f64;
        *site = new_val * keep_new + curr * (1.0 - keep_new);
    });

    let black = &mut lattice.black;
    black.par_iter_mut().for_each_init(
        || {
            let mut rng = rand::rng();
            Xoshiro256PlusPlus::from_rng(&mut rng)
        },
        |rng, site| {
            let curr = *site;
            let new_val = rng.random_range(curr - 0.5..curr + 0.5);

            let prob = (curr - new_val).exp();
            let realised = rng.random_range(0.0..1.0);

            let keep_new = (realised > prob) as i64 as f64;
            *site = new_val * keep_new + curr * (1.0 - keep_new);

            // if realised > prob {
            //     *site = new_val;
            // }
        });
}

fn simd_flip(lattice: &mut ColoredLattice) {
    let red = &mut lattice.red;
    red.par_chunks_mut(LANES).for_each_init(
        || {
            let mut rng = rand::rng();
            Xoshiro256PlusPlus::from_rng(&mut rng)
        },
        |rng, chunk| {
            let curr = SimdFType::from_slice(chunk);

            let mut rand_vals = [0.0f64; LANES];
            let mut probs = [0.0f64; LANES];

            for i in 0..LANES {
                rand_vals[i] = rng.random_range(-0.5..0.5);
                probs[i] = rng.random_range(0.0..1.0);
            }

            let offset = SimdFType::from_array(rand_vals);
            let realized = SimdFType::from_array(probs);
            let new_val = curr + offset;

            let prob_threshold = (curr - new_val).exp();
            let mask = realized.simd_gt(prob_threshold);
            let result = mask.select(new_val, curr);

            result.copy_to_slice(chunk);
        }
    );

    let black = &mut lattice.black;
    black.par_chunks_mut(LANES).for_each_init(
        || {
            let mut rng = rand::rng();
            Xoshiro256PlusPlus::from_rng(&mut rng)
        },
        |rng, chunk| {
            let curr = SimdFType::from_slice(chunk);

            let mut probs = [0.0f64; LANES];
            let mut rand_vals = [0.0f64; LANES];

            for i in 0..LANES {
                rand_vals[i] = rng.random_range(-0.5..0.5);
                probs[i] = rng.random_range(0.0..1.0);
            }

            let offset = SimdFType::from_array(rand_vals);
            let realized = SimdFType::from_array(probs);
            let new_val = curr + offset;

            let prob_threshold = (curr - new_val).exp();
            let mask = realized.simd_gt(prob_threshold);
            let result = mask.select(new_val, curr);

            result.copy_to_slice(chunk);
        }
    );
}

fn simd_flip2(lattice: &mut ColoredLattice) {
    let red = &mut lattice.red;
    red.par_chunks_mut(LANES).for_each_init(
        || {
            let mut rng = rand::rng();
            Xoshiro256PlusPlus::from_rng(&mut rng)
        },
        |rng, chunk| {
            let curr = SimdFType::from_slice(chunk);

            let mut offset = SimdFType::splat(0.0);
            let mut realized = SimdFType::splat(0.0);

            for i in 0..LANES {
                realized[i] = rng.random_range(-0.5..0.5);
                offset[i] = rng.random_range(0.0..1.0);
            }

            let new_val = curr + offset;

            let prob_threshold = (curr - new_val).exp();
            let mask = realized.simd_gt(prob_threshold);
            let result = mask.select(new_val, curr);

            result.copy_to_slice(chunk);
        }
    );

    let black = &mut lattice.black;
    black.par_chunks_mut(LANES).for_each_init(
        || {
            let mut rng = rand::rng();
            Xoshiro256PlusPlus::from_rng(&mut rng)
        },
        |rng, chunk| {
            let curr = SimdFType::from_slice(chunk);

            let mut offset = SimdFType::splat(0.0);
            let mut realized = SimdFType::splat(0.0);

            for i in 0..LANES {
                realized[i] = rng.random_range(-0.5..0.5);
                offset[i] = rng.random_range(0.0..1.0);
            }

            let new_val = curr + offset;

            let prob_threshold = (curr - new_val).exp();
            let mask = realized.simd_gt(prob_threshold);
            let result = mask.select(new_val, curr);

            result.copy_to_slice(chunk);
        }
    );
}

fn gen_random_vec(len: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    (0..len).map(|_| rng.random_range(-1.0..1.0)).collect()
}

fn gen_random_interior_vec(len: usize) -> Vec<UnsafeCell<f64>> {
    let mut rng = rand::rng();
    (0..len).map(|_| UnsafeCell::new(rng.random_range(-1.0..1.0))).collect()
}

const DIMS: [usize; 4] = [80, 40, 40, 40];
const TOTAL: usize = DIMS[0] * DIMS[1] * DIMS[2] * DIMS[3];
const LANES: usize = 4;

fn to_index([t, x, y, z]: [usize; 4]) -> usize {
    (t * DIMS[1] * DIMS[2] * DIMS[3]) + (x * DIMS[2] * DIMS[3]) + (y * DIMS[3]) + z
}

fn flip_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("contig_flip");

    let mut colored = ColoredLattice {
        red: gen_random_vec(TOTAL.div_ceil(2)),
        black: gen_random_vec(TOTAL.div_ceil(2))
    };

    let mut red_indices = Vec::with_capacity(TOTAL.div_ceil(2));
    let mut black_indices = Vec::with_capacity(TOTAL.div_ceil(2));

    for t in 0..DIMS[0] {
        for x in 0..DIMS[1] {
            for y in 0..DIMS[2] {
                for z in 0..DIMS[3] {
                    let idx = to_index([t, x, y, z]);
                    if (t + x + y + z) % 2 == 0 {
                        red_indices.push(idx);
                    } else {
                        black_indices.push(idx);
                    }
                }
            }
        }
    }

    let mut global = GlobalLattice {
        sites: SyncWrapper(gen_random_interior_vec(TOTAL)),
        red_indices, black_indices
    };

    group.bench_function("old_flip", |b| b.iter(|| old_flip(black_box(&mut global))));
    // group.bench_function("global_flip", |b| b.iter(|| reg_flip(&mut global)));
    // group.bench_function("contig_flip", |b| b.iter(|| contig_flip(&mut colored)));
    group.bench_function("simd_flip", |b| b.iter(|| simd_flip(black_box(&mut colored))));
    group.bench_function("simd_rand_flip", |b| b.iter(|| simd_flip2(black_box(&mut colored))));

    group.finish();
}

criterion_group!(benches, flip_benchmark);
criterion_main!(benches);