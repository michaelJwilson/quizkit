import h5py
import numpy as np
import logging

logger = logging.getLogger(__name__)

def read_hdf5(filepath, group_name, dataset_name):
    try:
        with h5py.File(filepath, "r") as f:
            if group_name not in f:
                raise KeyError(f"Group '{group_name}' not found in {filepath}")
            h5_group = f[group_name]

            if dataset_name not in h5_group:
                raise KeyError(f"Dataset '{dataset_name}' not found in group '{group_name}'")
            
            dataset = h5_group[dataset_name]
            
            # Load into memory. If you skip np.array(), the dataset becomes 
            # inaccessible as soon as the 'with' block closes the file.
            data = np.array(dataset)
            
            # Extract metadata safely
            metadata = {key: value for key, value in dataset.attrs.items()}
            
        return data, metadata

    except Exception as e:
        logger.error(f"Failed to read HDF5 file {filepath} [{group_name}/{dataset_name}]: {e}")
        raise
