use fftw::plan::*;
use fftw::types::*;
use fftw::array::AlignedVec;
use num_complex::Complex64;
use ndarray::Array2;

use ndarray::Zip;
use rand::Rng;

pub struct HolographyEngine {
    rows: usize,
    cols: usize,
    buf_a: AlignedVec<Complex64>,
    buf_b: AlignedVec<Complex64>,
    forward_plan: C2CPlan64,
    backward_plan: C2CPlan64,
}

impl HolographyEngine {
    pub fn new(rows: usize, cols: usize) -> Self {
        let n_elements = rows * cols;
        
        let mut buf_a = AlignedVec::<Complex64>::new(n_elements);
        let mut buf_b = AlignedVec::<Complex64>::new(n_elements);

        for i in 0..n_elements {
            buf_a[i] = Complex64::default();
            buf_b[i] = Complex64::default();
        }

        let forward_plan = C2CPlan::new(
            &[rows, cols], &mut buf_a, &mut buf_b, Sign::Forward, Flag::MEASURE
        ).unwrap();

        let backward_plan = C2CPlan::new(
            &[rows, cols], &mut buf_b, &mut buf_a, Sign::Backward, Flag::MEASURE
        ).unwrap();

        HolographyEngine { rows, cols, buf_a, buf_b, forward_plan, backward_plan }
    }

    fn initialize_in_place_random(buf: &mut [Complex64], illumination: &Array2<f64>) {
        let mut rng = rand::thread_rng();
        let ill_slice = illumination.as_slice().unwrap();

        for (c, &amp) in buf.iter_mut().zip(ill_slice.iter()) {
            let phase = rng.gen_range(0.0..(2.0 * std::f64::consts::PI));
            *c = Complex64::from_polar(amp, phase);
        }
    }

    #[inline(always)]
    fn constrain_amplitude(complex_buf: &mut [Complex64], target_amp_slice: &[f64]) {
        for (c, &target_amp) in complex_buf.iter_mut().zip(target_amp_slice.iter()) {
            let norm = c.norm();
            if norm > 1e-15 {
                let scale = target_amp / norm;
                c.re *= scale;
                c.im *= scale;
            } else {
                *c = Complex64::new(target_amp, 0.0);
            }
        }
    }

    pub fn gerchberg_saxton_slm_phase(
        &mut self, 
        target_amplitude: &Array2<f64>, 
        slm_illumination: &Array2<f64>, 
        iterations: usize
    ) -> Array2<f64> {
        let target_amp_slice = target_amplitude.as_slice().unwrap();
        let slm_ill_slice = slm_illumination.as_slice().unwrap();

        Self::initialize_in_place_random(&mut self.buf_a, slm_illumination);

        for _ in 0..iterations {
            self.forward_plan.c2c(&mut self.buf_a, &mut self.buf_b).unwrap();

            Self::constrain_amplitude(&mut self.buf_b, target_amp_slice);

            self.backward_plan.c2c(&mut self.buf_b, &mut self.buf_a).unwrap();

            Self::constrain_amplitude(&mut self.buf_a, slm_ill_slice);
        }

        let mut phase_map = Array2::<f64>::zeros((self.rows, self.cols));
        let phase_slice = phase_map.as_slice_mut().unwrap();

        for (p, c) in phase_slice.iter_mut().zip(self.buf_a.iter()) {
            *p = c.arg() + std::f64::consts::PI;
        }

        phase_map
    }
}