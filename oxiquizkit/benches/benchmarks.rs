use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use oxiquizkit::fibonacci::{fibonacci_slow, fibonacci_fast};

pub fn benchmark_fibonacci(c: &mut Criterion) {
    c.bench_function("fib(20)", |b| b.iter(|| fibonacci_fast(black_box(20))));
}

criterion_group!(benches, benchmark_fibonacci);
criterion_main!(benches);
