import h5py
import logging

logger = logging.getLogger(__name__)


# TODO
def write_hdf5(
    filepath, 
    data, 
    group_name, 
    dataset_name, 
    compression="gzip", 
    overwrite=False, 
    **metadata
):
    try:
        with h5py.File(filepath, "a") as f:
            if group_name not in f:
                h5_group = f.create_group(group_name)
            else:
                h5_group = f[group_name]

            if dataset_name in h5_group:
                if overwrite:
                    logger.info(f"Dataset '{dataset_name}' already exists. Overwriting...")
                    del h5_group[dataset_name]
                else:
                    logger.warning(
                        f"Dataset '{dataset_name}' already exists in group '{group_name}'. "
                        "Set override=True to overwrite. Skipping write."
                    )
                    return

            dataset = h5_group.create_dataset(
                name=dataset_name, 
                data=data, 
                compression=compression
            )

            for key, value in metadata.items():
                dataset.attrs[key] = value

        logger.info(
            f"Successfully written {data.shape} dataset to {filepath} at {group_name}/{dataset_name}"
        )

    except Exception as e:
        logger.error(f"Failed to write HDF5 file {filepath}: {e}")
        raise
