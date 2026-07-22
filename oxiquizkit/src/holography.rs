use fftw::plan::*;
use fftw::types::*;
use fftw::array::AlignedVec;
use num_complex::Complex64;
use ndarray::Array2;
use rand::Rng;

pub struct HolographyEngine {
    rows: usize,
    cols: usize,
    inb: AlignedVec<Complex64>,
    outb: AlignedVec<Complex64>,
    forward_plan: C2CPlan64,
    backward_plan: C2CPlan64,
}

impl HolographyEngine {
    pub fn new(rows: usize, cols: usize) -> Self {
        let n_elements = rows * cols;
        
        let mut inb = AlignedVec::<Complex64>::new(n_elements);
        let mut outb = AlignedVec::<Complex64>::new(n_elements);

        for i in 0..n_elements {
            inb[i] = Complex64::default();
            outb[i] = Complex64::default();
        }

        let forward_plan = C2CPlan::new(
            &[rows, cols], &mut inb, &mut outb, Sign::Forward, Flag::MEASURE
        ).unwrap();

        let backward_plan = C2CPlan::new(
            &[rows, cols], &mut outb, &mut inb, Sign::Backward, Flag::MEASURE
        ).unwrap();

        HolographyEngine { rows, cols, inb, outb, forward_plan, backward_plan }
    }

    fn initialize_random_phase(buf: &mut [Complex64], illumination: &Array2<f64>) {
        let mut rng = rand::thread_rng();
        let ill_slice = illumination.as_slice().unwrap();

        const TO_RAD: f64 = (2.0 * std::f64::consts::PI) / (u32::MAX as f64 + 1.0);

        for (c, &amp) in buf.iter_mut().zip(ill_slice.iter()) {
            let raw_u32 = rng.gen::<u32>(); 
            let phase = (raw_u32 as f64) * TO_RAD;
    
            *c = Complex64::from_polar(amp, phase);
        }
    }
    
    fn constrain_amplitude(
        complex_buf: &mut [Complex64], 
        target_amp_slice: &[f64], 
        iter: usize, 
        max_iters: usize
    ) {
        const EPSILON: f64 = 1e-15;
        let progress = iter as f64 / max_iters as f64;
        let mut rng = rand::thread_rng();

        for (c, &target_amp) in complex_buf.iter_mut().zip(target_amp_slice.iter()) {
            let norm = c.norm();
            
            let safe_norm = if norm < EPSILON { EPSILON } else { norm };
            let safe_target = if target_amp < EPSILON { EPSILON } else { target_amp };

            let rand_val: f64 = rng.gen();
            let t = progress + (1.0 - progress) * rand_val;

            let log_norm = safe_norm.ln();
            let log_target = safe_target.ln();
            let new_log_amplitude = log_norm + t * (log_target - log_norm);

            let scale = new_log_amplitude.exp() / safe_norm;

            c.re *= scale;
            c.im *= scale;
        }
    }

    pub fn gerchberg_saxton_slm_phase(
        &mut self, 
        target_amplitude: &Array2<f64>, 
        slm_illumination: &Array2<f64>, 
        max_iters: usize
    ) {
        let target_amp_slice = target_amplitude.as_slice().unwrap();
        let slm_ill_slice = slm_illumination.as_slice().unwrap();

        Self::initialize_random_phase(&mut self.inb, slm_illumination);

        for iter in 0..max_iters {
            self.forward_plan.c2c(&mut self.inb, &mut self.outb).unwrap();

            Self::constrain_amplitude(&mut self.outb, target_amp_slice, iter, max_iters);

            self.backward_plan.c2c(&mut self.outb, &mut self.inb).unwrap();

            Self::constrain_amplitude(&mut self.inb, slm_ill_slice, iter, max_iters);
        }

        for c in self.inb.iter_mut() {
            *c = (c.arg() + std::f64::consts::PI).into();
        }
    }
}