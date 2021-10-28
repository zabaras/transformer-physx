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

    Args:
        file_path (str, optional): Path to write logging files
        file_name (str, optional): Log file name
    """
    file_path: str = "."
    file_name: str = "log_metrics.h5"

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

    def writeToHDF5(self, file_name: str = None) -> None:
        """Write the classes attributes to HDF5 file

        Args:
            file_name (str, optional): File name to write to
        """
        self.__dumpToHDF5(file_name, mode='w')

    def appendToHDF5(self, file_name: str = None) -> None:
        """Appends the classes attributes to HDF5 file

        Args:
            file_name (str, optional): File name to write to
        """
        self.__dumpToHDF5(file_name, mode='a')

    def __dumpToHDF5(self, file_name: str = None, mode: str ='a') -> None:
        """Dump the classes attributes to HDF5 file

        Args:
            file_name (str, optional): File name to write to
            mode (str, optional): Edit mode (w, a)
        """
        if file_name is None:
            file_name = os.path.join(self.file_path, self.file_name)
        else:
            file_name = os.path.join(self.file_path, file_name)

        with h5py.File(file_name, "w") as f:
            for attr, value in self.__dict__.items():
                if attr == 'file_name':
                    continue
                # Check if data-set is already in h5file
                if f.__contains__(attr):
                    if mode == 'a': # Append mode
                        data0 = list(f[attr])
                        del f[attr]
                        value = data0 + value
                    else: # Write
                        del f[attr]

                f.create_dataset(attr, data=np.array(value))

    def delHDF5(self, file_name: str = None) -> None:
        """Deletes hdf5 file if it exists

        Args:
            file_name (str, optional): File name
        """
        if file_name is None:
            file_name = os.path.join(self.file_path, self.file_name)
        else:
            file_name = os.path.join(self.file_path, file_name)

        if os.path.exists(file_name):
            os.remove(file_name)