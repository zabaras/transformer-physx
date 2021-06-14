import numpy as np
import os, time
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Dataset
from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class EmbeddingDataHandler(object):

    mu = None
    std = None
    @property
    def norm_constants(self):
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

    class LorenzDataset(Dataset):
        def __init__(self, examples):
            self.examples = examples

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {"input_states": self.examples[i]}

    @dataclass
    class LorenzDataCollator:
        """
        Data collator for lorenz embedding problem
        """
        # Default collator
        def __call__(self, examples:List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            
            x_data_tensor =  torch.stack([example['input_states'] for example in examples])
            return {"input_states": x_data_tensor}

    def createTrainingLoader(self, file_path: str,  #hdf5 file
            block_size: int, # Length of time-series
            stride:int = 1,
            ndata:int = -1,
            batch_size:int=32,
            shuffle=True,
        ):
        '''
        Loads time-series data and creates training/testing loaders
        '''
        print('Creating training loader')
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

        data = torch.cat(examples, dim=0)
        self.mu = torch.mean(data)
        self.std = torch.std(data)
        self.mu = torch.tensor([torch.mean(data[:,:,0]), torch.mean(data[:,:,1]), torch.mean(data[:,:,2])])
        self.std = torch.tensor([torch.std(data[:,:,0]), torch.std(data[:,:,1]), torch.std(data[:,:,2])])

        # Needs to min-max normalization due to the reservoir matrix, needing to have a spectral density below 1
        if(data.size(0) < batch_size):
            print('Lower batch-size to {:d}'.format(data.size(0)))
            batch_size = data.size(0)

        dataset = self.LorenzDataset(data)
        data_collator = self.LorenzDataCollator()
        training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator, drop_last=True)
        return training_loader

    def createTestingLoader(self, file_path: str, #hdf5 file
            block_size: int,
            ndata:int = -1,
            batch_size:int=32,
            shuffle=False,
            mu:float=0.0,
            std:float=1.0
        ):
        '''
        Loads time-series data and creates training/testing loaders
        '''
        print('Creating testing loader')
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
            print('Lower batch-size to {:d}'.format(data.size(0)))
            batch_size = data.size(0)

        dataset = self.LorenzDataset((data-mu)/std)
        data_collator = self.LorenzDataCollator()
        testing_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator, drop_last=False)

        return testing_loader


class CylinderDataHandler(EmbeddingDataHandler):

    class CylinderDataset(Dataset):
        def __init__(self, examples, visc):
            self.examples = examples
            self.visc = visc

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {'input_states': self.examples[i], "viscosity": self.visc[i]}

    @dataclass
    class CylinderDataCollator:
        """
        Data collator for cylinder embedding problem
        """
        # Default collator
        def __call__(self, examples:List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            
            x_data_tensor =  torch.stack([example['input_states'] for example in examples])
            visc_tensor =  torch.stack([example['viscosity'] for example in examples])
            return {"input_states": x_data_tensor, "viscosity": visc_tensor}

    def createTrainingLoader(self, file_path: str,  # hdf5 file
                             block_size: int,  # Length of time-series
                             stride: int = 1,
                             ndata: int = -1,
                             batch_size: int = 32,
                             shuffle=True,
                             ):
        '''
        Loads time-series data and creates training/testing loaders
        '''
        print('Creating training loader')
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
            print('Lower batch-size to {:d}'.format(data.size(0)))
            batch_size = data.size(0)

        dataset = self.CylinderDataset(data, torch.stack(visc, dim=0))
        data_collator = self.CylinderDataCollator()
        training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator, drop_last=True)
        return training_loader

    def createTestingLoader(self, file_path: str,  # hdf5 file
                            block_size: int,
                            ndata: int = -1,
                            batch_size: int = 32,
                            shuffle=False,
                            mu: float = 0.0,
                            std: float = 1.0
                            ):
        '''
        Loads time-series data and creates training/testing loaders
        '''
        print('Creating testing loader')
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
            print('Lower batch-size to {:d}'.format(data.size(0)))
            batch_size = data.size(0)

        dataset = self.CylinderDataset(data, torch.stack(visc, dim=0))
        data_collator = self.CylinderDataCollator()
        testing_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator, drop_last=False)

        return testing_loader


class GrayScottDataHandler(EmbeddingDataHandler):
    """Class for creating training and testing loaders for the Gray-Scott embedding model.
    """
    class GrayScottDataset(Dataset):
        """PyTorch dataset for Gray-Scott system, dynamically loads data from file each
        mini-batch since loading an entire data-set would be way too large

        :param h5_file: path to hdf5 file with raw data
        :type h5_file: str
        :param f: list of string Gray-scott feed rates
        :type f: list
        :param k: list of string Gray-scott kill rates
        :type k: list
        :param indices: list of start indexes for each time-series block
        :type f: list
        :param block_size: stride interval to sample blocks from, defaults to 1
        :type block_size: int, optional
        """
        def __init__(self, h5_file, keys, indices, block_size=1, permutes=2):
            self.h5_file = h5_file
            self.keys = keys 
            self.idx = indices
            self.block_size = block_size

            self.permute_idxs = np.random.randint(0, 32, size=(permutes*len(self.keys), 3))
            print(self.permute_idxs.shape)

        def __len__(self):
            return len(self.keys)

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            idx0 = self.idx[i]  # start index
            key = self.keys[i]
            with h5py.File(self.h5_file, "r") as h5_file:
                u = h5_file['/'.join((key, 'u'))][idx0: idx0 + self.block_size, :, :, :]
                v = h5_file['/'.join((key, 'v'))][idx0: idx0 + self.block_size, :, :, :]


            data = torch.stack([torch.Tensor(u), torch.Tensor(v)], dim=1)

            # data = torch.roll(data, shifts=tuple(self.permute_idxs[i]), dims=(-3, -2, -1))

            return {'input_states': data}

    @dataclass
    class GrayScottDataCollator:
        """
        Data collator for the Gray-scott embedding problem
        """
        # Default collator
        def __call__(self, examples:List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            
            x_data_tensor =  torch.stack([example['x_data'] for example in examples])
            return {"input_states": x_data_tensor}

    def createTrainingLoader(self,
         file_path: str,  # hdf5 file
         block_size: int,  # Length of time-series
         stride: int = 1,
         ndata: int = -1,
         batch_size: int = 32,
         shuffle = True,
         mpi_rank=-1,
         mpi_size=1
         ):
        """Create training data loader from raw simulation data. A single simulation
        time series is broken down into smaller blocks for training.

        :file_path file_path: path to hdf5 with simulation data
        :type file_path: str
        :param block_size: length of time-series blocks to make for training Koopman dynamics
        :type block_size: int
        :param stride: stride interval to sample blocks from, defaults to 1
        :type stride: int, optional
        :param ndata: number of training data samples to use, uses all data if negative, defaults to -1
        :type ndata: int, optional
        :param batch_size: training batch size, defaults to 32
        :type batch_size: int, optional
        :param shuffle: shuffle training data, default to True
        :type shuffle: bool, optional
        :returns:
            - training_loader: PyTorch data loader
        :rtype: torch.training_utils.data.DataLoader
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

                if ndata > 0 and nsamp > ndata:  # If we have enough time-series samples break loop
                    break
        # Normalization for u, v, f, k
        # These values are pre-processed
        self.mu = torch.tensor([0.6823825, 0.09115206, 0.02448318, 0.0525])
        self.std = torch.tensor([0.20818108, 0.10649372, 0.01630819, 0.00790569])

        self.mu = torch.tensor([0.])
        self.std = torch.tensor([1.0])

        self.mu = torch.tensor([0.64169478, 0.11408507])
        self.std = torch.tensor([0.25380379, 0.11043673])
        
        if len(idx) < batch_size:
            logger.info('Lowering batch-size to {:d}'.format(len(key)))
            batch_size = len(idx)
        logger.info('Number of training blocks: {}'.format(len(idx)))

        if not mpi_rank == -1:
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

    def createTestingLoader(self, file_path: str,  # hdf5 file
                            block_size: int,
                            ndata: int = -1,
                            batch_size: int = 32,
                            shuffle=False,
                            mu: float = 0.0,
                            std: float = 1.0
                            ):
        '''
        Loads time-series data and creates training/testing loaders
        '''
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

                if ndata > 0 and nsamp > ndata:  # If we have enough time-series samples break loop
                    break

        if len(idx) < batch_size:
            logger.info('Lowering batch-size to {:d}'.format(len(idx)))
            batch_size = len(idx)

        dataset = self.GrayScottDataset(file_path, key, idx, block_size)
        data_collator = self.GrayScottDataCollator()
        testing_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator, drop_last=False)

        return testing_loader

LOADER_MAPPING = OrderedDict(
    [
        ("lorenz", LorenzDataHandler),
        ("cylinder", CylinderDataHandler),
        ("cylinder-auto", CylinderDataHandler),
        ("grayscott", GrayScottDataHandler)
    ]
)
class AutoDataHandler():
    @classmethod
    def load_data_handler(cls, model_name, **kwargs):
        # First check if the model name is a pre-defined config
        if (model_name in LOADER_MAPPING.keys()):
            loader_class = LOADER_MAPPING[model_name]
            # Init config class
            loader = loader_class(**kwargs)
        else:
            err_str = "Provided model name: {}, not present in built=int data handlers".format(model_name)
            raise ValueError(err_str)
        return loader