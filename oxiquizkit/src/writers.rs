use hdf5::{File, Result};
use ndarray::ArrayView2;

pub fn write_hologram(
    filepath: &str,
    slm_phases: ArrayView2<f64>,
    slm_illumination: ArrayView2<f64>,
    target_image: ArrayView2<f64>,
) -> Result<()> {
    let file = File::create(filepath)?;

    file.new_dataset::<f64>()
        .shape(slm_phases.shape())
        .create("slm_phases")?
        .write(&slm_phases)?;

    file.new_dataset::<f64>()
        .shape(slm_illumination.shape())
        .create("slm_illumination")?
        .write(&slm_illumination)?;

    file.new_dataset::<f64>()
        .shape(target_image.shape())
        .create("target_image")?
        .write(&target_image)?;

    Ok(())
}
