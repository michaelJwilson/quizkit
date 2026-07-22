use ndarray::ArrayD;
use numpy::{
    IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn,
};
use pyo3::prelude::*;

#[pymodule]
fn oxiquizkit(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn nb<'py>(
        py: Python<'py>,
        ink: PyReadonlyArrayDyn<'py, f64>,
        inn: PyReadonlyArrayDyn<'py, f64>,
        inp: PyReadonlyArrayDyn<'py, f64>,
    ) -> &'py PyArrayDyn<f64> {
        let k = ink.as_array();
        let n = inn.as_array();
        let p = inp.as_array();

        // NB k.raw_dim() returns the .shape of the array.
        let mut result = ArrayD::<f64>::zeros(k.raw_dim());

        __oxiquizkit::nb(&mut result, &k, &n, &p);

        result.into_pyarray(py)
    }

    Ok(())
}

mod __oxiquizkit {
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

#[cfg(test)]
mod tests {
    use super::__oxiquizkit;
    use approx::assert_relative_eq; 
    use ndarray::{array, ArrayD};

    #[test]
    fn test_nb_distribution_nd() {
        // NB convert fixed dimension rust arrays to dynamically sized.
        let k = array![[2.0, 0.0], [2.0, 0.0]].into_dyn();
        let n = array![[3.0, 3.0], [3.0, 3.0]].into_dyn();
        let p = array![[0.4, 0.4], [0.4, 0.4]].into_dyn();
        
        let mut result = ArrayD::<f64>::zeros(k.raw_dim());
        
        __oxiquizkit::nb(&mut result, &k.view(), &n.view(), &p.view());
        
        let expected_log_pmf_0 = 0.13824_f64.ln();
        let expected_log_pmf_1 = 0.064_f64.ln();

        assert_relative_eq!(result[[0, 0]], expected_log_pmf_0, epsilon = 1e-6);
        assert_relative_eq!(result[[0, 1]], expected_log_pmf_1, epsilon = 1e-6);
        assert_relative_eq!(result[[1, 0]], expected_log_pmf_0, epsilon = 1e-6);
        assert_relative_eq!(result[[1, 1]], expected_log_pmf_1, epsilon = 1e-6);
    }
}