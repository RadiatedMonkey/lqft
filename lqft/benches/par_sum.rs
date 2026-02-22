use criterion::{Bencher, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use rand::prelude::*;
use rayon::prelude::*;

type FType = f32;

#[inline]
fn par_sum(size: usize) -> FType {
    let mut rng = rand::rng();

    let rand_vec = (0..size)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect::<Vec<_>>();

    rand_vec.par_iter().sum::<FType>()
}

#[inline]
fn single_sum(size: usize) -> FType {
    let mut rng = rand::rng();

    let rand_vec = (0..size)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect::<Vec<_>>();

    rand_vec.iter().sum::<FType>()
}

fn par_sum_benchmark(c: &mut Criterion) {
    rayon::ThreadPoolBuilder::new().build_global().unwrap();

    const COUNT: usize = 40 * 20 * 20 * 20;

    let mut group = c.benchmark_group("sum");

    group.bench_function("single_sum", |b| b.iter(|| single_sum(black_box(COUNT))));
    group.bench_function("par_sum", |b| b.iter(|| par_sum(black_box(COUNT))));

    group.finish();
}

criterion_group!(benches, par_sum_benchmark);
criterion_main!(benches);
