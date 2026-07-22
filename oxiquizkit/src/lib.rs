use pyo3::prelude::*;

pub mod readers;
pub mod	holography;
pub mod core;
pub mod python;

#[pymodule]
fn oxiquizkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python::nb, m)?)?;
    Ok(())
}
