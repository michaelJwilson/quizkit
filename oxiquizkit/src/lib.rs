use pyo3::prelude::*;

// 1. Tell Rust compiler to look for the other files
pub mod core;
pub mod python;

// 2. Define the main Python module entry point here
#[pymodule]
fn oxiquizkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 3. Register the function from the python module
    m.add_function(wrap_pyfunction!(python::nb, m)?)?;
    Ok(())
}



/*
use ndarray::ArrayD;
use numpy::{
    IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn,
};
use pyo3::prelude::*;

#[pyfunction]
fn nb<'py>(
    py: Python<'py>,
    ink: PyReadonlyArrayDyn<'py, f64>,
    inn: PyReadonlyArrayDyn<'py, f64>,
    inp: PyReadonlyArrayDyn<'py, f64>,
) -> Bound<'py, PyArrayDyn<f64>> {    
    let k = ink.as_array();
    let n = inn.as_array();
    let p = inp.as_array();

    let mut result = ArrayD::<f64>::zeros(k.raw_dim());

    core::nb(&mut result, &k, &n, &p);

    result.into_pyarray(py)
}

#[pymodule]
fn oxiquizkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nb, m)?)?;
    Ok(())
}

pub mod core {
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

    // NB x3.5 speedup for caching hits.
    fn init_cache() -> &'static [f64] {
        LN_FACT_CACHE.get_or_init(|| {
            (0..CACHE_SIZE).map(|i| lgamma(i as f64 + 1.0)).collect()
        })
    }

    pub fn nb(
        r: &mut ndarray::ArrayD<f64>,
        k: &ArrayViewD<'_, f64>,
        n: &ArrayViewD<'_, f64>,
        p: &ArrayViewD<'_, f64>,
    ) {
        let cache = init_cache();

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
}*/