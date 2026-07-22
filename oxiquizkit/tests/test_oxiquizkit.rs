use approx::assert_relative_eq;
use ndarray::{array, ArrayD};
use oxiquizkit;

#[test]
fn test_nb_distribution_nd() {
    // NB convert fixed dimension rust arrays to dynamically sized.
    let k = array![[2.0, 0.0], [2.0, 0.0]].into_dyn();
    let n = array![[3.0, 3.0], [3.0, 3.0]].into_dyn();
    let p = array![[0.4, 0.4], [0.4, 0.4]].into_dyn();
    
    let mut result = ArrayD::<f64>::zeros(k.raw_dim());
    
    oxiquizkit::core::nb(&mut result, &k.view(), &n.view(), &p.view());
    
    let expected_log_pmf_0 = 0.13824_f64.ln();
    let expected_log_pmf_1 = 0.064_f64.ln();

    assert_relative_eq!(result[[0, 0]], expected_log_pmf_0, epsilon = 1e-6);
    assert_relative_eq!(result[[0, 1]], expected_log_pmf_1, epsilon = 1e-6);
    assert_relative_eq!(result[[1, 0]], expected_log_pmf_0, epsilon = 1e-6);
    assert_relative_eq!(result[[1, 1]], expected_log_pmf_1, epsilon = 1e-6);
}