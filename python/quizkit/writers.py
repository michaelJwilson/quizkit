import h5py
import logging

logger = logging.getLogger(__name__)


# TODO
def write_hdf5(
    filepath, data, group_name, dataset_name, compression="gzip", **metadata
):
    try:
        with h5py.File(filepath, "a") as f:
            if group_name not in f:
                h5_group = f.create_group(group_name)
            else:
                h5_group = f[group_name]

            dataset = h5_group.create_dataset(
                name=dataset_name, data=data, compression=compression
            )

            for key, value in metadata.items():
                dataset.attrs[key] = value

        logger.info(
            f"Successfully written {data.shape} dataset to {filepath} at {group_name}/{dataset_name}"
        )

    except Exception as e:
        logger.error(f"Failed to write HDF5 file {filepath}: {e}")
        raise
