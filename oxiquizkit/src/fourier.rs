use fftw::plan::*;
use fftw::types::*;
use fftw::array::AlignedVec;
use num_complex::Complex64;
use ndarray::ArrayD;

pub fn fft_image_r2c(image: &ArrayD<f64>) -> ArrayD<f64> {
    let shape = image.shape();

    let rows = shape[0];
    let cols = shape[1];
    let n = rows * cols;

    assert_eq!(shape.len(), 2, "FFT requires a 2D image matrix");

    // NB pre-allocation of fftw aligned memory before, real-to-complex @ 2x 
    let mut in_buf = AlignedVec::<f64>::new(n);
    let mut out_buf = AlignedVec::<Complex64>::new(rows * (cols / 2 + 1));

    // TODO?  assumes standard C-contiguouos memory (as for numpy)
    in_buf.copy_from_slice(image.as_slice().expect("TODO"));

    // NB create a (platform-specific) optimization plan and dimensions.
    let mut plan: R2CPlan64 = R2CPlan::new(
        &[rows, cols],
        &mut in_buf,
        &mut out_buf,
        Flag::MEASURE, // optimize the plan for the problem
    ).unwrap();

    // in is consumed in favour of out.
    plan.r2c(&mut in_buf, &mut out_buf).expect("FFT execution failed");

    let out_cols = cols / 2 + 1;
    let mut magnitude_array = ndarray::Array2::<f64>::zeros((rows, out_cols));

    for i in 0..rows {
        for j in 0..out_cols {
            let idx = i * out_cols + j;
            magnitude_array[[i, j]] = out_buf[idx].norm();
        }
    }

    println!("Successful r2c.");

    magnitude_array.into_dyn()
}