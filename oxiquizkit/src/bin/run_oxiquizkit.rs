use ndarray::Array2;
use oxiquizkit::holography::HolographyEngine;
use oxiquizkit::readers::read_hdf5;
use oxiquizkit::writers::write_hologram;
use std::env;
use std::process;

fn process_image(hdf5_str: &str) -> ndarray::ArrayD<f64> {
    // TODO
    let internal_path = "test_group/test_data";
    let data_array = read_hdf5(hdf5_str, internal_path).expect("Failed to read HDF5 file");

    println!("Successful load of {}", hdf5_str);

    data_array
}

// NB cargo run --bin run_oxiquizkit --release -- ../results/data/test_image.hdf5
fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: cargo run --bin run_oxiquizkit -- <path_to_hdf5_file>");
        process::exit(1);
    }

    let hdf5_path = &args[1];
    let image = process_image(hdf5_path);

    let target_amplitude: Array2<f64> = image
        .into_dimensionality::<ndarray::Ix2>()
        .expect("Must be able to convert image to 2D for holography.");

    let shape = target_amplitude.shape();
    let rows = shape[0];
    let cols = shape[1];

    let slm_illumination = Array2::<f64>::from_elem((rows, cols), 1.0);

    let mut holography_engine = HolographyEngine::new(rows, cols);

    let iterations = 50;

    let slm_phase = holography_engine.gerchberg_saxton_slm_phase(
        &target_amplitude,
        &slm_illumination,
        iterations,
    );

    write_hologram(
        "../results/data/hologram.h5",
        slm_phase.view(),
        slm_illumination.view(),
        target_amplitude.view(),
    )
    .expect("Failed to write HDF5 data");
}
