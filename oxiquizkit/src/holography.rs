use fftw::plan::*;
use fftw::types::*;
use fftw::array::AlignedVec;
use num_complex::Complex64;
use ndarray::Array2;

use ndarray::Zip;
use rand::Rng;
use num_complex::ComplexFloat;

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

        HolographyEngine { rows, cols, buf_a, buf_b, forward_plan, backward_plan }
    }

    /// Creates the initial complex field combining a known amplitude with a pure random phase.
    fn create_random_phase_field(amplitude: &Array2<f64>) -> Array2<Complex64> {
        let mut rng = rand::thread_rng();
        let mut field = Array2::<Complex64>::zeros(amplitude.raw_dim());

        Zip::from(&mut field)
            .and(amplitude)
            .for_each(|f, &amp| {
                // Generate a random phase between 0 and 2π
                let phase = rng.gen_range(0.0..(2.0 * std::f64::consts::PI));
            
                // Reconstruct the complex number using polar notation: Amplitude * e^(i * Phase)
                *f = Complex64::from_polar(amp, phase);
            });

        field
    }

    /// Computes the phase from a complex field, replaces its amplitude
    /// with the 'target_amplitude', and modifies the complex field in-place.
    fn constrain_amplitude(complex_field: &mut AlignedVec<Complex64>, target_amplitude: &Array2<f64>) {
        // Both structures represent the exact same 2D grid in memory.
        // We iterate over them simultaneously.
        let amplitude_slice = target_amplitude.as_slice().unwrap();

        for (c, &target_amp) in complex_field.iter_mut().zip(amplitude_slice.iter()) {
            // Extract strictly the phase from the mutated complex field
            let phase = c.arg(); 
        
            // Rebuild the complex number combining the target amplitude and the calculated phase
            *c = Complex64::from_polar(target_amp, phase);
        }
    }

    /// Extracts only the phase angle map from a complex field. 
    /// Useful as the final output that tells the SLM hardware what phase delay to apply to each pixel.
    fn extract_phase(complex_field: &Array2<Complex64>) -> Array2<f64> {
        let mut phase_map = Array2::<f64>::zeros(complex_field.raw_dim());

        Zip::from(&mut phase_map)
            .and(complex_field)
            .for_each(|p, c| {
                // .arg() returns the angle in radians (-π to +π). 
                // Often SLMs prefer 0 to 2π, so you can easily shift it here if needed:
                // *p = c.arg() + std::f64::consts::PI;
                *p = c.arg();
        });

        phase_map
    }

    pub fn gerchberg_saxton_slm_phase(
        &mut self, 
        target_amplitude: &Array2<f64>, 
        slm_illumination: &Array2<f64>, 
        iterations: usize
    ) -> Array2<f64> {
        /// Runs `iterations` of the GS algorithm.
        /// `target_amplitude` is the optical trap lattice you want.
        /// `slm_illumination` is the laser beam profile hitting your SLM.
        /// Returns the computed SLM Phase map (0 to 2π).

        let mut current_slm_field = create_random_phase_field(slm_illumination);

        for _ in 0..iterations {
            self.buf_a.copy_from_slice(current_slm_field.as_slice().unwrap());

            // B. FFT Forward -> Trap Plane (buf_a to buf_b)
            self.forward_plan.c2c(&mut self.buf_a, &mut self.buf_b).unwrap();

            // C. Apply Trap constraints (Replace amplitude, keep phase)
            // Field = Target_Amplitude * exp(i * Phase)
            constrain_amplitude(&mut self.buf_b, target_amplitude);

            // D. IFFT Backward -> SLM Plane (buf_b to buf_a)
            self.backward_plan.c2c(&mut self.buf_b, &mut self.buf_a).unwrap();

            // E. Apply SLM constraints (Replace amplitude, keep phase)
            constrain_amplitude(&mut self.buf_a, slm_illumination);
            
            // F. Extract the constrained field to loop it again
            current_slm_field.as_slice_mut().unwrap().copy_from_slice(&self.buf_a);
        }

        // Return just the pure phase angle of the final SLM field (what the hardware needs)
        extract_phase(&current_slm_field)
    }
}