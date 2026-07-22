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

        // NB forward: slm -> trap
        let forward_plan = C2CPlan::new(
            &[rows, cols], &mut buf_a, &mut buf_b, Sign::Forward, Flag::MEASURE
        ).unwrap();

        // NB backward: trap -> slm
        let backward_plan = C2CPlan::new(
            &[rows, cols], &mut buf_b, &mut buf_a, Sign::Backward, Flag::MEASURE
        ).unwrap();

        println!("Allocated memory buffers and plans for (fftw) holography.");

        HolographyEngine { rows, cols, buf_a, buf_b, forward_plan, backward_plan }
    }

    fn initialize_random_phase(amplitude: &Array2<f64>) -> Array2<Complex64> {
        let mut rng = rand::thread_rng();
        let mut field = Array2::<Complex64>::zeros(amplitude.raw_dim());

        Zip::from(&mut field)
            .and(amplitude)
            .for_each(|f, &amp| {
                let phase = rng.gen_range(0.0..(2.0 * std::f64::consts::PI));
                *f = Complex64::from_polar(amp, phase);
            });

        field
    }

    fn constrain_amplitude(complex_field: &mut AlignedVec<Complex64>, target_amplitude: &Array2<f64>) {
        let amplitude_slice = target_amplitude.as_slice().unwrap();

        for (c, &target_amp) in complex_field.iter_mut().zip(amplitude_slice.iter()) {
            let phase = c.arg(); 
            *c = Complex64::from_polar(target_amp, phase);
        }
    }

    fn extract_phase(complex_field: &Array2<Complex64>) -> Array2<f64> {
        let mut phase_map = Array2::<f64>::zeros(complex_field.raw_dim());

        Zip::from(&mut phase_map)
            .and(complex_field)
            .for_each(|p, c| {
                // NB (0 to 2π)
                *p = c.arg() + std::f64::consts::PI;
        });

        phase_map
    }

    pub fn gerchberg_saxton_slm_phase(
        &mut self, 
        target_amplitude: &Array2<f64>, 
        slm_illumination: &Array2<f64>, 
        iterations: usize
    ) -> Array2<f64> {
        let mut current_slm_field = Self::initialize_random_phase(slm_illumination);

        for _ in 0..iterations {
            self.buf_a.copy_from_slice(current_slm_field.as_slice().unwrap());

            self.forward_plan.c2c(&mut self.buf_a, &mut self.buf_b).unwrap();

            Self::constrain_amplitude(&mut self.buf_b, target_amplitude);

            self.backward_plan.c2c(&mut self.buf_b, &mut self.buf_a).unwrap();

            Self::constrain_amplitude(&mut self.buf_a, slm_illumination);
            
            current_slm_field.as_slice_mut().unwrap().copy_from_slice(&self.buf_a);
        }

        println!("Completed Gerchberg-Saxton");

        Self::extract_phase(&current_slm_field)
    }
}