use oxiquizkit::readers::read_hdf5;
use std::env;
use std::path::PathBuf;
use std::process;

fn process_image(hdf5_str: &str) -> ndarray::ArrayD<f64> {
    // TODO
    let internal_path = "test_group/test_data";
    let data_array = read_hdf5(hdf5_str, internal_path)
        .expect("Failed to read HDF5 file");
        
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

    _ = process_image(hdf5_path);
}