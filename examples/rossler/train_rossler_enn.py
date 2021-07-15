"""
=====
Training embedding model for the Rossler numerical example.
This is example illustrates how to use the Transformer-Physx
package to learn physics of a custom problem.

Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import sys
import os
import logging
import h5py
from typing import Dict, List

import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from rossler_module.configuration_rossler import RosslerConfig
from rossler_module.embedding_rossler import RosslerEmbeddingTrainer
from trphysx.embedding.training import EmbeddingParser, EmbeddingDataHandler, EmbeddingTrainer

logger = logging.getLogger(__name__)

class RosslerDataHandler(EmbeddingDataHandler):
    """Embedding data handler for Rossler system.
    Contains methods for creating training and testing loaders,
    dataset class and data collator.
    """
    class RosslerDataset(Dataset):
        def __init__(self, examples):
            self.examples = examples

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {"input_states": self.examples[i]}

    class RosslerDataCollator:
        """
        Data collator for rossler embedding problem
        """
        # Default collator
        def __call__(self, examples:List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            
            x_data_tensor =  torch.stack([example['input_states'] for example in examples])
            return {"input_states": x_data_tensor}

    def createTrainingLoader(
        self,
        file_path: str,
        block_size: int,
        stride:int = 1,
        ndata:int = -1,
        batch_size:int = 32,
        shuffle=True,
    ) -> DataLoader:
        """Creating embedding training data loader for Rossler system.
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
        logger.info('Creating training loader')
        assert os.path.isfile(file_path), "Training HDF5 file {} not found".format(file_path)

        examples = []
        with h5py.File(file_path, "r") as f:
            # Iterate through stored time-series
            samples = 0
            for key in f.keys():
                data_series = torch.Tensor(f[key])
                # Stride over time-series by specified block size
                for i in range(0,  data_series.size(0) - block_size + 1, stride): 
                    examples.append(data_series[i : i + block_size].unsqueeze(0))

                samples = samples + 1
                if(ndata > 0 and samples > ndata): #If we have enough time-series samples break loop
                    break

        data = torch.cat(examples, dim=0)
        logger.info("Training data-set size: {}".format(data.size()))

        # Normalize training data
        # Normalize x and y with Gaussian, normalize z with max/min
        self.mu = torch.tensor([torch.mean(data[:,:,0]), torch.mean(data[:,:,1]), torch.min(data[:,:,2])])
        self.std = torch.tensor([torch.std(data[:,:,0]), torch.std(data[:,:,1]), torch.max(data[:,:,2])-torch.min(data[:,:,2])])

        # Needs to min-max normalization due to the reservoir matrix, needing to have a spectral density below 1
        if(data.size(0) < batch_size):
            logger.warn('Lower batch-size to {:d}'.format(data.size(0)))
            batch_size = data.size(0)

        dataset = self.RosslerDataset(data)
        data_collator = self.RosslerDataCollator()
        training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator, drop_last=True)
        return training_loader

    def createTestingLoader(self, 
        file_path: str,
        block_size: int,
        ndata:int = -1,
        batch_size:int=32,
        shuffle=False
    ) -> DataLoader:
        """Creating testing/validation data loader for Rossler system.
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
        assert os.path.isfile(file_path), "Testing HDF5 file {} not found".format(file_path)
        
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
        logger.info("Testing data-set size: {}".format(data.size()))

        if(data.size(0) < batch_size):
            logger.warn('Lower batch-size to {:d}'.format(data.size(0)))
            batch_size = data.size(0)

        data = (data - self.mu.unsqueeze(0).unsqueeze(0)) / self.std.unsqueeze(0).unsqueeze(0)
        dataset = self.RosslerDataset(data)
        data_collator = self.RosslerDataCollator()
        testing_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator, drop_last=False)

        return testing_loader

if __name__ == '__main__':

    sys.argv = sys.argv + ["--exp-name", "rossler"]
    sys.argv = sys.argv + ["--training_h5_file", "./data/rossler_training.hdf5"]
    sys.argv = sys.argv + ["--eval_h5_file", "./data/rossler_valid.hdf5"]
    sys.argv = sys.argv + ["--stride", "16"]
    sys.argv = sys.argv + ["--batch-size", "256"]
    sys.argv = sys.argv + ["--block-size", "16"]
    sys.argv = sys.argv + ["--ntrain", "1024"]
    sys.argv = sys.argv + ["--ntest", "8"]

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    args = EmbeddingParser().parse()    
    if(torch.cuda.is_available()):
        use_cuda = "cuda"
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Torch device: {}".format(args.device))

    data_handler =RosslerDataHandler()

    # Set up data-loaders
    training_loader = data_handler.createTrainingLoader(
        args.training_h5_file, 
        block_size=args.block_size, 
        stride=args.stride, 
        ndata=args.ntrain, 
        batch_size=args.batch_size)

    testing_loader = data_handler.createTestingLoader(
        args.eval_h5_file, 
        block_size=32, 
        ndata=args.ntest, 
        batch_size=8)

    # Load configuration file then init model
    config = RosslerConfig()
    model = RosslerEmbeddingTrainer(config)
    mu, std = data_handler.norm_params
    model.embedding_model.mu = mu.to(args.device)
    model.embedding_model.std = std.to(args.device)
    if args.epoch_start > 1:
        model.load_model(args.ckpt_dir, args.epoch_start)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr*0.995**(args.epoch_start), weight_decay=1e-8)
    scheduler = ExponentialLR(optimizer, gamma=0.995)

    trainer = EmbeddingTrainer(model, args, (optimizer, scheduler))

    trainer.train(training_loader, testing_loader)

