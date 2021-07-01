"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import os
import logging
import h5py
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Metrics:
    """Data class for storing training errors
    """
    file_name:str = "log_metrics.hdf5"

    def push(self, **kwargs) -> None:
        """Pushes elements in kwargs into the attributes of this class

        Args:
            **kwargs: Attributes to save
        """
        for key, value in kwargs.items():
            # Warn if value is not float or integer, there could be issues with it being stored in the hdf5 file
            if not isinstance(value, (int, float)):
                logger.warn('Some values in metrics are not floats or integers.')
            if(hasattr(self, key)):
                attrib = getattr(self, key)
                setattr(self, key, attrib+[value])
            else:
                setattr(self, key, [value])

    def writeToHDF5(self, file_name: str) -> None:
        """Write the classes attributes to HDF% file

        Args:
            file_name (str): File name to write to
        """
        with h5py.File(file_name, "a") as f:
            for attr, value in self.__dict__.items():
                if attr == 'file_name':
                    continue
                # Check if data-set is already in h5file
                if f.__contains__(attr):
                    data0 = list(f[attr])
                    del f[attr]
                    value = data0 + value
                f.create_dataset(attr, data=np.array(value))

    def delHDF5(self) -> None:
        """Deletes hdf5 file if it exists
        """
        if os.path.exists(self.file_name):
            os.remove(self.file_name)