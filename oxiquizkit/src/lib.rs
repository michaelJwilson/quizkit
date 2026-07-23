use pyo3::prelude::*;

pub mod readers;
pub mod writers;
pub mod	holography;
pub mod core;
pub mod python;
pub mod fibonacci;

#[pymodule]
fn oxiquizkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python::nb, m)?)?;
    Ok(())
}
