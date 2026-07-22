use pyo3::prelude::*;

mod readers;

pub mod core;
pub mod python;

#[pymodule]
fn oxiquizkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python::nb, m)?)?;
    Ok(())
}
