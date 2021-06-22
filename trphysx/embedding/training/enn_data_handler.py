"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import numpy as np
import os, time
import h5py
import torch
import logging
from typing import Tuple, List
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class EmbeddingDataHandler(object):
    """Base class for embedding data handlers.
    Data handlers are used to create the training and
    testing datasets.
    """
    mu = None
    std = None

    @property
    def norm_params(self) -> Tuple:
        """Get normalization parameters

        Raises:
            ValueError: If normalization parameters have not been initialized

        Returns:
            (Tuple): mean and standard deviation
        """
        if self.mu is None or self.std is None:
            raise ValueError("Normalization constants set yet!")
        return self.mu, self.std

    @abstractmethod
    def createTrainingLoader(self, *args, **kwargs):
        pass

    @abstractmethod
    def createTestingLoader(self, *args, **kwargs):
        pass

class LorenzDataHandler(EmbeddingDataHandler):
    """Built in embedding data handler for Lorenz system
    """
    class LorenzDataset(Dataset):
        """Dataset for training Lorenz embedding model.

        Args:
            examples (List): list of training/testing examples
        """
        def __init__(self, examples: List):
            """Constructor
            """
            self.examples = examples

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {"states": self.examples[i]}

    @dataclass
    class LorenzDataCollator:
        """Data collator for lorenz embedding problem
        """
        # Default collator
        def __call__(self, examples:List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            # Stack examples in mini-batch
            x_data_tensor =  torch.stack([example['states'] for example in examples])

            return {"states": x_data_tensor}

    def createTrainingLoader(self, 
        file_path: str,  #hdf5 file
        block_size: int, # Length of time-series
        stride: int = 1,
        ndata: int = -1,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader:
        """Creating training data loader for Lorenz system.
        For a single training simulation, the total time-series is sub-chunked into
        smaller blocks for training.

        Args:
            file_path (str): Path to HDF5 file with training data
            block_size (int): The length of time-series blocks
            stride (int): Stride of each time-series block
            ndata (int, optional): Number of training time-series. If negative, all of the provided 
            data will be used. Defaults to -1.
            batch_size (int, optional): Training batch size. Defaults to 32.
            shuffle (bool, optional): Turn on mini-batch shuffling in dataloader. Defaults to True.

        Returns:
            (DataLoader): Training loader
        """
        logger.info('Creating training loader.')
        assert os.path.isfile(file_path)
        examples = []
        with h5py.File(file_path, "r") as f:
            # Iterate through stored time-series
            samples = 0
            for key in f.keys():
                data_series = torch.Tensor(f[key])
                # Stride over time-series
                for i in range(0,  data_series.size(0) - block_size + 1, stride):  # Truncate in block of block_size
                    examples.append(data_series[i : i + block_size].unsqueeze(0))

                samples = samples + 1
                if(ndata > 0 and samples > ndata): #If we have enough time-series samples break loop
                    break

        # Calculate normalization constants
        data = torch.cat(examples, dim=0)
        self.mu = torch.tensor([torch.mean(data[:,:,0]), torch.mean(data[:,:,1]), torch.mean(data[:,:,2])])
        self.std = torch.tensor([torch.std(data[:,:,0]), torch.std(data[:,:,1]), torch.std(data[:,:,2])])

        # Needs to min-max normalization due to the reservoir matrix, needing to have a spectral density below 1
        if(data.size(0) < batch_size):
            logger.warning('Lower batch-size to {:d}'.format(data.size(0)))
            batch_size = data.size(0)

        # Create dataset, collator, and dataloader
        dataset = self.LorenzDataset(data)
        data_collator = self.LorenzDataCollator()
        training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator, drop_last=True)
        
        return training_loader

    def createTestingLoader(self, 
        file_path: str,
        block_size: int,
        ndata: int = -1,
        batch_size: int =32,
        shuffle: bool =False
    ) -> DataLoader:
        """Creating testing/validation data loader for Lorenz system.
        For a data case with time-steps [0,T], this method extract a smaller
        time-series to be used for testing [0, S], s.t. S < T.

        Args:
            file_path (str): Path to HDF5 file with testing data
            block_size (int): The length of testing time-series
            ndata (int, optional): Number of testing time-series. If negative, all of the provided 
            data will be used. Defaults to -1.
            batch_size (int, optional): Testing batch size. Defaults to 32.
            shuffle (bool, optional): Turn on mini-batch shuffling in dataloader. Defaults to False.

        Returns:
            (DataLoader): Testing/validation data loader
        """
        logger.info('Creating testing loader')
        assert os.path.isfile(file_path)
        examples = []
        with h5py.File(file_path, "r") as f:
            # Iterate through stored time-series
            samples = 0
            for key in f.keys():
                data_series = torch.Tensor(f[key])
                # Stride over time-series
                for i in range(0,  data_series.size(0) - block_size + 1, block_size):  # Truncate in block of block_size
                    examples.append(data_series[i : i + block_size].unsqueeze(0))
                    break

                samples = samples + 1
                if(ndata > 0 and samples >= ndata): #If we have enough time-series samples break loop
                    break

        # Combine data-series
        data = torch.cat(examples, dim=0)
        if(data.size(0) < batch_size):
            logger.warning('Lower batch-size to {:d}'.format(data.size(0)))
            batch_size = data.size(0)

        dataset = self.LorenzDataset(data)
        data_collator = self.LorenzDataCollator()
        testing_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator, drop_last=False)

        return testing_loader


class CylinderDataHandler(EmbeddingDataHandler):
    """Built in embedding data handler for flow around a cylinder system
    """
    class CylinderDataset(Dataset):
        """Dataset for training flow around a cylinder embedding model

        Args:
            examples (List): list of training/testing example flow fields
            visc (List): list of training/testing example viscosities
        """
        def __init__(self, examples: List, visc: List) -> None:
            """Constructor
            """
            self.examples = examples
            self.visc = visc

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {"states": self.examples[i], "viscosity": self.visc[i]}

    @dataclass
    class CylinderDataCollator:
        """Data collator for flow around a cylinder embedding problem
        """
        # Default collator
        def __call__(self, examples:List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            # Stack examples in mini-batch
            x_data_tensor =  torch.stack([example["states"] for example in examples])
            visc_tensor =  torch.stack([example["viscosity"] for example in examples])

            return {"states": x_data_tensor, "viscosity": visc_tensor}

    def createTrainingLoader(self, 
        file_path: str,
        block_size: int,
        stride: int = 1,
        ndata: int = -1,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader:
        """Creating training data loader for the flow around a cylinder system.
        For a single training simulation, the total time-series is sub-chunked into
        smaller blocks for training.

        Args:
            file_path (str): Path to HDF5 file with training data
            block_size (int): The length of time-series blocks
            stride (int): Stride of each time-series block
            ndata (int, optional): Number of training time-series. If negative, all of the provided 
            data will be used. Defaults to -1.
            batch_size (int, optional): Training batch size. Defaults to 32.
            shuffle (bool, optional): Turn on mini-batch shuffling in dataloader. Defaults to True.

        Returns:
            (DataLoader): Training loader
        """
        logging.info('Creating training loader')
        assert os.path.isfile(file_path)

        examples = []
        visc = []
        with h5py.File(file_path, "r") as f:
            # Iterate through stored time-series
            samples = 0

            for key in f.keys():
                visc0 = (2.0/float(key))
                ux = torch.Tensor(f[key+'/ux'])
                uy = torch.Tensor(f[key + '/uy'])
                p = torch.Tensor(f[key + '/p'])
                data_series = torch.stack([ux, uy, p], dim=1)

                # Stride over time-series
                for i in range(0, data_series.size(0) - block_size + 1, stride):  # Truncate in block of block_size
                    examples.append(data_series[i: i + block_size])
                    visc.append(torch.tensor([visc0]))

                samples = samples + 1
                if (ndata > 0 and samples > ndata):  # If we have enough time-series samples break loop
                    break

        data = torch.stack(examples, dim=0)
        self.mu = torch.tensor([torch.mean(data[:,:,0]), torch.mean(data[:,:,1]), torch.mean(data[:,:,2]), torch.mean(torch.tensor(visc))])
        self.std = torch.tensor([torch.std(data[:,:,0]), torch.std(data[:,:,1]), torch.std(data[:,:,2]), torch.std(torch.tensor(visc))])
        # Needs to min-max normalization due to the reservoir matrix, needing to have a spectral density below 1
        if (data.size(0) < batch_size):
            logging.warn('Lower batch-size to {:d}'.format(data.size(0)))
            batch_size = data.size(0)

        dataset = self.CylinderDataset(data, torch.stack(visc, dim=0))
        data_collator = self.CylinderDataCollator()
        training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator, drop_last=True)
        return training_loader

    def createTestingLoader(self, 
            file_path: str,
            block_size: int,
            ndata: int = -1,
            batch_size: int = 32,
            shuffle: bool =False,
        ) -> DataLoader:
        """Creating testing/validation data loader for the flow around a cylinder system.
        For a data case with time-steps [0,T], this method extract a smaller
        time-series to be used for testing [0, S], s.t. S < T.

        Args:
            file_path (str): Path to HDF5 file with testing data
            block_size (int): The length of testing time-series
            ndata (int, optional): Number of testing time-series. If negative, all of the provided 
            data will be used. Defaults to -1.
            batch_size (int, optional): Testing batch size. Defaults to 32.
            shuffle (bool, optional): Turn on mini-batch shuffling in dataloader. Defaults to False.

        Returns:
            (DataLoader): Testing/validation data loader
        """
        logging.info('Creating testing loader')
        assert os.path.isfile(file_path)

        examples = []
        visc = []
        with h5py.File(file_path, "r") as f:
            # Iterate through stored time-series
            samples = 0
            for key in f.keys():
                visc0 = (2.0/float(key))
                ux = torch.Tensor(f[key + '/ux'])
                uy = torch.Tensor(f[key + '/uy'])
                p = torch.Tensor(f[key + '/p'])
                data_series = torch.stack([ux, uy, p], dim=1)
                # Stride over time-series data_series.size(0)
                for i in range(0, data_series.size(0) - block_size + 1, block_size):  # Truncate in block of block_size
                    examples.append(data_series[i: i + block_size])
                    visc.append(torch.tensor([visc0]))
                    break

                samples = samples + 1
                if (ndata > 0 and samples >= ndata):  # If we have enough time-series samples break loop
                    break

        # Combine data-series
        data = torch.stack(examples, dim=0)
        if (data.size(0) < batch_size):
            logging.warning('Lower batch-size to {:d}'.format(data.size(0)))
            batch_size = data.size(0)

        dataset = self.CylinderDataset(data, torch.stack(visc, dim=0))
        data_collator = self.CylinderDataCollator()
        testing_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator, drop_last=False)

        return testing_loader


class GrayScottDataHandler(EmbeddingDataHandler):
    """Built in embedding data handler for the Gray-Scott system
    """
    class GrayScottDataset(Dataset):
        """Dataset for Gray-Scott system. Dynamically loads data from file each
        mini-batch since loading an entire data-set would be way too large. This data-set
        support the loading of sub-chunked time-series.

        Args:
            h5_file (str): Path to hdf5 file with raw data
            keys (List): List of keys corresponding to each example
            indices (List): List of start indices for each time-series block
            block_size (int, optional): List to time-series block sizes for each example. Defaults to 1.
        """
        def __init__(self, 
            h5_file: str, 
            keys: List, 
            indices: List, 
            block_size: int = 1
        ) -> None:
            """Constructor
            """
            self.h5_file = h5_file
            self.keys = keys 
            self.idx = indices
            self.block_size = block_size

        def __len__(self):
            return len(self.keys)

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            idx0 = self.idx[i] # Time-step start index
            key = self.keys[i] # HDF5 dataset key
            # Read from file and extract time-series of given block size
            with h5py.File(self.h5_file, "r") as h5_file:
                u = h5_file['/'.join((key, 'u'))][idx0: idx0 + self.block_size, :, :, :]
                v = h5_file['/'.join((key, 'v'))][idx0: idx0 + self.block_size, :, :, :]

            data = torch.stack([torch.Tensor(u), torch.Tensor(v)], dim=1)

            return {'states': data}

    @dataclass
    class GrayScottDataCollator:
        """Data collator for the Gray-scott embedding problem
        """
        # Default collator
        def __call__(self, examples:List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            
            x_data_tensor =  torch.stack([example['states'] for example in examples])

            return {"states": x_data_tensor}

    def createTrainingLoader(self,
        file_path: str,  # hdf5 file
        block_size: int,  # Length of time-series
        stride: int = 1,
        ndata: int = -1,
        batch_size: int = 32,
        shuffle: bool = True,
        mpi_rank: int = -1,
        mpi_size: int = 1
    ) -> DataLoader:
        """Creating training data loader for the Gray-Scott system.
        For a single training simulation, the total time-series is sub-chunked into
        smaller blocks for training. This particular dataloader support splitting the
        dataset between GPU processes for parallel training if needed. 

        Args:
            file_path (str): Path to HDF5 file with training data
            block_size (int): The length of time-series blocks
            stride (int): Stride of each time-series block
            ndata (int, optional): Number of training time-series. If negative, all of the provided 
            data will be used. Defaults to -1.
            batch_size (int, optional): Training batch size. Defaults to 32.
            shuffle (bool, optional): Turn on mini-batch shuffling in dataloader. Defaults to True.
            mpi_rank (int, optional): Rank of current MPI process. Defaults to -1.
            mpi_size (int, optional): Number of training processes. Set to 1 for serial training. Defaults to 1. 

        Returns:
            (DataLoader): Training loader
        """
        logger.info('Creating training loader')
        assert os.path.isfile(file_path)

        idx = []
        key = []
        with h5py.File(file_path, "r") as h5_file:
            # Iterate through stored time-series
            nsamp = 0
            # Loop through kill rates
            for key0 in h5_file.keys():
                
                t = torch.Tensor(h5_file[key0 + '/t'])
                # Create smaller time-series blocks
                for i in range(0, t.size(0) - block_size + 1, stride):  # Truncate in block of block_size
                    idx.append(i)
                    key.append(key0)

                nsamp = nsamp + 1
                if ndata > 0 and nsamp > ndata:  # If we have enough time-series samples break loop
                    break

        # Normalization for u, v
        # These values are pre-processed
        self.mu = torch.tensor([0.64169478, 0.11408507])
        self.std = torch.tensor([0.25380379, 0.11043673])
        
        if len(idx) < batch_size:
            logger.info('Lowering batch-size to {:d}'.format(len(key)))
            batch_size = len(idx)
        logger.info('Number of training blocks: {}'.format(len(idx)))

        if mpi_size > 1:
            logger.info('Splitting data-set between MPI processes')
            rng = np.random.default_rng(seed=12356) # Seed needs to be consistent between MPI processes
            data_len = len(idx)
            indexes = [x for x in range(0, data_len)]
            rng.shuffle(indexes)

            chunks = int(data_len//mpi_size)
            indexes = indexes[chunks*mpi_rank : chunks*(mpi_rank+1)]

            dataset = self.GrayScottDataset(file_path, [key[i] for i in indexes], [idx[i] for i in indexes], block_size)
        else:
            dataset = self.GrayScottDataset(file_path, key, idx, block_size)

        data_collator = self.GrayScottDataCollator()
        training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, collate_fn=data_collator, num_workers=1)
        
        return training_loader

    def createTestingLoader(self, 
        file_path: str,  # hdf5 file
        block_size: int,
        ndata: int = -1,
        batch_size: int = 32,
        shuffle=False
    ) -> DataLoader:
        """Creating testing/validation data loader for the Gray-Scott system.
        For a data case with time-steps [0,T], this method extract a smaller
        time-series to be used for testing [0, S], s.t. S < T.

        Args:
            file_path (str): Path to HDF5 file with testing data
            block_size (int): The length of testing time-series
            ndata (int, optional): Number of testing time-series. If negative, all of the provided 
            data will be used. Defaults to -1.
            batch_size (int, optional): Testing batch size. Defaults to 32.
            shuffle (bool, optional): Turn on mini-batch shuffling in dataloader. Defaults to False.

        Returns:
            (DataLoader): Testing/validation data loader
        """
        logger.info('Creating testing loader')
        assert os.path.isfile(file_path)

        idx = []
        key = []
        with h5py.File(file_path, "r") as h5_file:
            # Iterate through stored time-series
            nsamp = 0
            # Loop through kill rates
            for key0 in h5_file.keys():
                
                t = torch.Tensor(h5_file[key0 + '/t'])
                # Create smaller time-series blocks
                for i in range(0, t.size(0) - block_size + 1, block_size):  # Truncate in block of block_size
                    idx.append(i)
                    key.append(key0)

                nsamp = nsamp + 1
                if ndata > 0 and nsamp > ndata:  # If we have enough time-series samples break loop
                    break

        if len(idx) < batch_size:
            logger.warning('Lowering batch-size to {:d}'.format(len(idx)))
            batch_size = len(idx)

        dataset = self.GrayScottDataset(file_path, key, idx, block_size)
        data_collator = self.GrayScottDataCollator()
        testing_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator, drop_last=False)

        return testing_loader

LOADER_MAPPING = OrderedDict(
    [
        ("lorenz", LorenzDataHandler),
        ("cylinder", CylinderDataHandler),
        ("grayscott", GrayScottDataHandler)
    ]
)
class AutoDataHandler():
    """Helper class for intializing different built in data-handlers for embedding training
    """
    @classmethod
    def load_data_handler(cls, model_name: str, **kwargs) -> EmbeddingDataHandler:
        """Gets built-in data handler.
        Currently supports: "lorenz", "cylinder", "grayscott"

        Args:
            model_name (str): Model name

        Raises:
            ValueError: If model_name is not a supported model type

        Returns:
            (EmbeddingDataHandler): Embedding data handler
        """
        # First check if the model name is a pre-defined config
        if (model_name in LOADER_MAPPING.keys()):
            loader_class = LOADER_MAPPING[model_name]
            # Init config class
            loader = loader_class(**kwargs)
        else:
            err_str = "Provided model name: {}, not present in built=int data handlers".format(model_name)
            raise ValueError(err_str)

        return loader