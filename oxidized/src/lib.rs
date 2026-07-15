use pyo3::prelude::*;

#[pyfunction]
fn hello_world() -> PyResult<()> {
    println!("hello world");
    Ok(())
}

#[pymodule]
fn oxidized(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_world, m)?)?;
    Ok(())
}