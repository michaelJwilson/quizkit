use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use oxiquizkit::fibonacci::{fibonacci_fast, fibonacci_slow};
use oxiquizkit::holography::HolographyEngine;
use std::hint::black_box;

fn benchmark_fibonacci(c: &mut Criterion) {
    c.bench_function("fib(20)", |b| b.iter(|| fibonacci_fast(black_box(20))));
}

fn benchmark_gerchberg_saxton(c: &mut Criterion) {
    let size: usize = 512;
    let iterations = 25;

    let mut group = c.benchmark_group("Gerchberg-Saxton");

    // TODO
    group.sample_size(10);

    // TODO
    let mut target_amplitude = Array2::<f64>::zeros((size, size));

    let center = size / 2;
    let spacing = 20;

    target_amplitude[[center - spacing, center - spacing]] = 1.0;
    target_amplitude[[center + spacing, center - spacing]] = 1.0;
    target_amplitude[[center - spacing, center + spacing]] = 1.0;
    target_amplitude[[center + spacing, center + spacing]] = 1.0;

    let slm_illumination = Array2::<f64>::from_elem((size, size), 1.0);
    let mut engine = HolographyEngine::new(size, size);

    group.bench_with_input(
        BenchmarkId::new("GS_512x512", iterations),
        &iterations,
        |b, &iters| {
            b.iter(|| {
                engine.gerchberg_saxton_slm_phase(&target_amplitude, &slm_illumination, iters)
            })
        },
    );

    group.finish();
}

// NB cargo bench -- --fail-fast
criterion_group!(benches, benchmark_gerchberg_saxton);
criterion_main!(benches);
