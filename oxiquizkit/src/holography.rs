use fftw::array::AlignedVec;
use fftw::plan::*;
use fftw::types::*;
use ndarray::Array2;
use num_complex::Complex64;
use rand::Rng;
use rayon::prelude::*;

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
            &[rows, cols],
            &mut inb,
            &mut outb,
            Sign::Forward,
            Flag::MEASURE,
        )
        .unwrap();

        let backward_plan = C2CPlan::new(
            &[rows, cols],
            &mut outb,
            &mut inb,
            Sign::Backward,
            Flag::MEASURE,
        )
        .unwrap();

        HolographyEngine {
            rows,
            cols,
            inb,
            outb,
            forward_plan,
            backward_plan,
        }
    }

    fn generate_uniforms(n: usize) -> AlignedVec<f64> {
        let mut rng = rand::thread_rng();
        let mut phases = AlignedVec::<f64>::new(n);

        for p in phases.iter_mut() {
            let raw_u32 = rng.gen::<u32>();
            *p = raw_u32 as f64;
        }

        phases
    }

    pub fn initialize_random_phase(
        buf: &mut [Complex64],
        illumination: &Array2<f64>,
        illumination_precision: f64,
    ) {
        let ill_slice = illumination.as_slice().unwrap();
        let uniforms = Self::generate_uniforms(buf.len());

        const TO_RAD: f64 = (2.0 * std::f64::consts::PI) / (u32::MAX as f64 + 1.0);

        for ((c, &amp), &uniform) in buf.iter_mut().zip(ill_slice.iter()).zip(uniforms.iter()) {
            *c = Complex64::from_polar(
                amp + illumination_precision * (uniform - 0.5),
                TO_RAD * uniform,
            );
        }
    }

    fn constrain_amplitude(
        complex_buf: &mut [Complex64],
        target_amplitude: &[f64],
        iter: usize,
        max_iters: usize,
    ) {
        const EPSILON: f64 = 1e-15;
        let progress = iter as f64 / max_iters as f64;
        let progress_complement = 1.0 - progress;

        let uniforms = Self::generate_uniforms(complex_buf.len());

        complex_buf
            .par_iter_mut()
            .zip(target_amplitude.par_iter())
            .for_each_init(
                || rand::thread_rng(),
                |rng, (c, &target_amp)| {
                    let rand_val: f64 = rng.gen();

                    let norm = c.norm().max(EPSILON);
                    let safe_target = target_amp.max(EPSILON);

                    let log_norm = norm.ln();
                    let log_target = safe_target.ln();

                    let t = progress + rand_val * progress_complement;
                    let target_amplitude = (log_norm + t * (log_target - log_norm)).exp();

                    let scale = target_amplitude / norm;
                    c.re *= scale;
                    c.im *= scale;
                },
            );
    }

    pub fn gerchberg_saxton_slm_phase(
        &mut self,
        target_amplitude: &Array2<f64>,
        slm_illumination: &Array2<f64>,
        max_iters: usize,
    ) -> Array2<f64> {
        let target_amp_slice = target_amplitude.as_slice().unwrap();
        let slm_ill_slice = slm_illumination.as_slice().unwrap();

        Self::initialize_random_phase(&mut self.inb, slm_illumination, 0.1);

        for iter in 0..max_iters {
            self.forward_plan
                .c2c(&mut self.inb, &mut self.outb)
                .unwrap();

            Self::constrain_amplitude(&mut self.outb, target_amp_slice, iter, max_iters);

            self.backward_plan
                .c2c(&mut self.outb, &mut self.inb)
                .unwrap();

            Self::constrain_amplitude(&mut self.inb, slm_ill_slice, iter, max_iters);
        }

        // NB calculate final image plane
        self.forward_plan
            .c2c(&mut self.inb, &mut self.outb)
            .unwrap();

        let phase_vec: Vec<f64> = self
            .outb
            .iter()
            .map(|c| c.arg() + std::f64::consts::PI)
            .collect();

        println!("Successful phase extraction with Gerchberg-Saxton.");

        // Reshape into an ndarray and return
        Array2::from_shape_vec((self.rows, self.cols), phase_vec)
            .expect("Shape mismatch during phase extraction")
    }
}
