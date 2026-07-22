use ndarray::ArrayD;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

use crate::core; // Import your pure Rust logic

#[pyfunction]
pub fn nb<'py>(
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