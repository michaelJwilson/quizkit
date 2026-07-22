use libm::lgamma;
use ndarray::{ArrayViewD, Zip};
use rayon::prelude::*;
use std::sync::OnceLock;

const CACHE_SIZE: usize = 10_000;
static LN_FACT_CACHE: OnceLock<Vec<f64>> = OnceLock::new();

#[inline(always)]
fn ln_factorial(x: u32, cache: &[f64]) -> f64 {
    let idx = x as usize;
    if idx < CACHE_SIZE {
        cache[idx]
    } else {
        lgamma(x as f64 + 1.0)
    }
}

fn init_lgamma_cache() -> &'static [f64] {
    LN_FACT_CACHE.get_or_init(|| (0..CACHE_SIZE).map(|i| lgamma(i as f64 + 1.0)).collect())
}

pub fn nb(
    r: &mut ndarray::ArrayD<f64>,
    k: &ArrayViewD<'_, f64>,
    n: &ArrayViewD<'_, f64>,
    p: &ArrayViewD<'_, f64>,
) {
    let cache = init_lgamma_cache();

    Zip::from(r)
        .and(k)
        .and(n)
        .and(p)
        .par_for_each(|r, &k, &n, &p| {
            let n32 = n as u32;
            let k32 = k as u32;

            *r = ln_factorial(k32 + n32 - 1, cache)
                - ln_factorial(n32 - 1, cache)
                - ln_factorial(k32, cache)
                + k * (1.0 - p).ln()
                + n * p.ln();
        });
}
