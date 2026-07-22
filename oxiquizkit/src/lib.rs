use ndarray;
use ndarray::{Array, Array1, Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArrayDyn,
};
use rayon::prelude::*;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

// See https://itnext.io/how-to-bind-python-numpy-with-rust-ndarray-2efa5717ed21
#[pymodule]
fn oxiquizkit(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn nb<'py>(
        py: Python<'py>,
        ink: PyReadonlyArray1<f64>,
        inn: PyReadonlyArray1<f64>,
        inp: PyReadonlyArray1<f64>,
    ) -> &'py PyArray1<f64> {
        let k = ink.as_array();
        let n = inn.as_array();
        let p = inp.as_array();

        let shape = k.shape();
        let mut result = Array1::<f64>::zeros(shape[0]);
        let mut vresult = result.view_mut();

        __core::nb(&mut vresult, &k, &n, &p);
        
        result.into_pyarray(py)
    }

    Ok(())
}

mod __core {
    // extern crate rayon;
    // use rayon::prelude::*; // move to top.

    use libm::lgamma;
    use ndarray::{arr1, Array1, Array2, Array3, Zip}; // redeclaration?
    use numpy::ndarray::{
        ArrayView1, ArrayView2, ArrayView3, ArrayViewD, ArrayViewMut1, ArrayViewMut2,
        ArrayViewMut3, ArrayViewMutD,
    };

    fn ln_factorial(n: u32) -> f64 {
       lgamma(n as f64 + 1.0)
    }

    pub fn nb(
        r: &mut ArrayViewMut1<'_, f64>,
        k: &ArrayView1<'_, f64>,
        n: &ArrayView1<'_, f64>,
        p: &ArrayView1<'_, f64>,
    ) {
        /*
        See:
            https://docs.rs/GSL/latest/rgsl/gamma_beta/factorials/fn.lnchoose.html
            https://mathworld.wolfram.com/NegativeBinomialDistribution.html
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html
        */

        Zip::from(r)
            .and(k)
            .and(n)
            .and(p)
            .par_for_each(|r, &k, &n, &p| {
                /*
                See:
                  https://docs.rs/GSL/latest/rgsl/gamma_beta/factorials/fn.lnchoose.html
                  https://mathworld.wolfram.com/NegativeBinomialDistribution.html
                  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html
                */

                let n32 = n as u32;
                let k32 = k as u32;

                // NB in-place
                *r = ln_factorial(k32 + n32 - 1) - ln_factorial(n32 - 1) - ln_factorial(k32)
                    + k * (1. - p).ln()
                    + n * p.ln();
            });
    }
}