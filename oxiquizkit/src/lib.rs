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

    #[inline(always)]
    fn ln_factorial(n: u32) -> f64 {
        lgamma(n as f64 + 1.0)
    }

    pub fn nb(
        r: &mut ndarray::ArrayD<f64>,
        k: &ArrayViewD<'_, f64>,
        n: &ArrayViewD<'_, f64>,
        p: &ArrayViewD<'_, f64>,
    ) {
        // NB rayon threaded; exclude for small arrays.
        Zip::from(r)
            .and(k)
            .and(n)
            .and(p)
            .par_for_each(|r, &k, &n, &p| {
                let n32 = n as u32;
                let k32 = k as u32;

                *r = ln_factorial(k32 + n32 - 1)
                    - ln_factorial(n32 - 1)
                    - ln_factorial(k32)
                    + k * (1.0 - p).ln()
                    + n * p.ln();
            });
    }
}