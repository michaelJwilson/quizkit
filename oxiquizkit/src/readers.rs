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

    #[test]
    fn test_read_dummy_h5() -> hdf5::Result<()> {
        let filepath = "test_dataset.h5";
        let dataset_name = "test_image";

        {
            let file = hdf5::File::create(filepath)?;
            let group = file.create_group("test_group")?;
            let data = array![[1.0, 1.5], [2.0, 2.5]].into_dyn();

            let builder = group.new_dataset::<f64>();
            let dataset = builder.create(dataset_name, data.shape())?;
            dataset.write(&data)?;
        }

        let full_path = format!("test_group/{}", dataset_name);
        let read_data = read_hdf5(filepath, &full_path)?;

        assert_eq!(read_data.shape(), &[2, 2]);
        assert_eq!(read_data[[1, 0]], 2.0);

        let _ = fs::remove_file(filepath);

        Ok(())
    }
}
