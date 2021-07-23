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
import random
import argparse
import os, errno, json
import torch
from typing import List

HOME = os.getcwd()

class EmbeddingParser(argparse.ArgumentParser):
    """Arguments for training embedding models
    """
    def __init__(self):
        super().__init__(description='Arguments for training the embedding models for transformers of physical systems')
        self.add_argument('--exp_dir', type=str, default="./outputs", help='directory to save experiments')
        self.add_argument('--exp_name', type=str, default="lorenz", help='experiment name')

        # data
        self.add_argument('--training_h5_file', type=str, default=None, help='file path to the training data hdf5 file')
        self.add_argument('--eval_h5_file', type=str, default=None, help='file path to the evaluation data hdf5 file')
        self.add_argument('--ntrain', type=int, default=2048, help='number of training data')
        self.add_argument('--ntest', type=int, default=16, help='number of testing data')
        self.add_argument('--stride', type=int, default=16, help='number of time-steps as encoder input')
        self.add_argument('--block_size', type=int, default=64, help='number of time-steps as encoder input')
        self.add_argument('--batch_size', type=int, default=64, help='batch size for training')

        # training
        self.add_argument('--epoch_start', type=int, default=0, help='epoch to start at, will load pre-trained network')
        self.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=0.001, help='ADAM learning rate')
        self.add_argument('--seed', type=int, default=12345, help='manual seed used in PyTorch and Numpy')
        self.add_argument('--n_gpu', type=int, default=1, help='Max number of GPUs to use')
        
        # logging
        self.add_argument('--plot_freq', type=int, default=25, help='how many epochs to wait before plotting test output')
        self.add_argument('--test_freq', type=int, default=5, help='how many epochs to test the model')
        self.add_argument('--save_steps', type=int, default=25, help='how many epochs to wait before saving model')
        self.add_argument('--notes', type=str, default='')

    def mkdirs(self, *directories: str) -> None:
        """Makes a directory if it does not exist

        Args:
           directories (str...): a sequence of directories to create

        Raises:
            OSError: if directory cannot be created
        """
        for directory in list(directories):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def parse(self, args:List = None, dirs: bool = True) -> None:
        """Parse program arguments

        Args:
            args (List, optional): Explicit list of arguments. Defaults to None.
            dirs (bool, optional): Make experiment directories. Defaults to True.
        """
        if args:
            args = self.parse_args(args=args)
        else:
            args = self.parse_args()

        if len(args.notes) > 0:
            args.run_dir = os.path.join(HOME, args.exp_dir, "embedding_{}".format(args.exp_name), 
                    "ntrain{}_epochs{:d}_batch{:d}_{:s}".format(args.ntrain, args.epochs, args.batch_size, args.notes))
        else:
            args.run_dir = os.path.join(HOME, args.exp_dir, "embedding_{}".format(args.exp_name), 
                    "ntrain{}_epochs{:d}_batch{:d}".format(args.ntrain, args.epochs, args.batch_size))
        args.ckpt_dir = os.path.join(args.run_dir,"checkpoints")
        args.plot_dir = os.path.join(args.run_dir, "predictions")

        if(dirs):
            self.mkdirs(args.run_dir, args.ckpt_dir, args.plot_dir)

        # Set random seed
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)

        # Dump args into JSON for reference
        if dirs:
            with open(os.path.join(args.run_dir, "args.json"), 'w') as args_file:
                json.dump(vars(args), args_file, indent=4)

        return args