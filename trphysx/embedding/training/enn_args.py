import numpy as np
import random
import argparse
import os, errno, copy, json
import torch

class EmbeddingParser(argparse.ArgumentParser):
    def __init__(self):
        super(EmbeddingParser, self).__init__(description='Arguments for training the embedding models for transformers of physical systems')
        self.add_argument('--exp-dir', type=str, default="./embedding", help='directory to save experiments')
        self.add_argument('--exp-name', type=str, default="lorenz", help='experiment name')

        # data
        self.add_argument('--training_h5_file', type=str, default=None, help='file path to the training data hdf5 file')
        self.add_argument('--eval_h5_file', type=str, default=None, help='file path to the evaluation data hdf5 file')
        self.add_argument('--ntrain', type=int, default=2048, help="number of training data")
        self.add_argument('--ntest', type=int, default=16, help="number of training data")
        self.add_argument('--stride', type=int, default=16, help="number of time-steps as encoder input")
        self.add_argument('--block-size', type=int, default=64, help="number of time-steps as encoder input")
        self.add_argument('--batch-size', type=int, default=64, help='batch size for training')

        # training
        self.add_argument('--epoch-start', type=int, default=0, help='epoch to start at, will load pre-trained network')
        self.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=0.001, help='ADAM learning rate')
        self.add_argument('--seed', type=int, default=12345, help='manual seed used in PyTorch and Numpy')
        self.add_argument('--n_gpu', type=int, default=1, help='Max number of GPUs to use')
        
        # logging
        self.add_argument('--plot-freq', type=int, default=25, help='how many epochs to wait before plotting test output')
        self.add_argument('--test-freq', type=int, default=5, help='how many epochs to test the model')
        self.add_argument('--save_steps', type=int, default=25, help='how many epochs to wait before saving model')
        self.add_argument('--notes', type=str, default='')

    def mkdirs(self, *directories):
        '''
        Makes a directory if it does not exist
        '''
        for directory in list(directories):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def parse(self, dirs=True):
        '''
        Parse program arguements
        Args:
            dirs (boolean): True to make file directories for predictions and models
        '''
        args = self.parse_args()
        args.run_dir = args.exp_dir + '/{}_ntrain{}'.format(args.exp_name, args.ntrain)

        args.ckpt_dir = args.run_dir + '/checkpoints'
        args.pred_dir = args.run_dir + "/predictions"
        if(dirs):
            self.mkdirs(args.run_dir, args.ckpt_dir, args.pred_dir)

        # Set random seed
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)

        if dirs:
            with open(args.run_dir + "/args.json", 'w') as args_file:
                json.dump(vars(args), args_file, indent=4)

        return args