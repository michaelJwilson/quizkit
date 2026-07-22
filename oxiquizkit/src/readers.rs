use fitsio::FitsFile;
use ndarray::ArrayD;

pub fn read_hdf5(filepath: &str, dataset_name: &str) -> hdf5::Result<ArrayD<f64>> {
    let file = hdf5::File::open(filepath)?;
    let dataset = file.dataset(dataset_name)?;
    let array = dataset.read_dyn::<f64>()?;

    Ok(array)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::fs;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_dummy_h5() -> hdf5::Result<()> {
        let dataset_name = "test_image";

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let filepath = temp_file.path().to_str().unwrap();

        {
            let file = hdf5::File::create(filepath)?;
            let group = file.create_group("test_group")?;
            let data = array![[1.0, 1.5], [2.0, 2.5]].into_dyn();

            let builder = group.new_dataset::<f64>();
            let dataset = builder.shape(data.shape()).create(dataset_name)?;

            dataset.write(&data)?;
        }

        let full_path = format!("test_group/{}", dataset_name);
        let read_data = read_hdf5(filepath, &full_path)?;

        assert_eq!(read_data.shape(), &[2, 2]);
        assert_eq!(read_data[[1, 0]], 2.0);

        Ok(())
    }
}
