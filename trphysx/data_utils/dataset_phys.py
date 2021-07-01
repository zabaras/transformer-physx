"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import logging
import os
import pickle
import time
import h5py

import torch
from typing import Dict
from abc import abstractmethod
from filelock import FileLock
from torch.utils.data.dataset import Dataset
from ..embedding.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)

class PhysicalDataset(Dataset):
    """Parent class for training and evaluation datasets for physical transformers.
    The caching of the dataset is based on the Hugging Face implementation.

    Args:
        embedder (EmbeddingModel): Embedding neural network
        file_path (str): Path to hdf5 raw data file
        block_size (int): Length of time-series blocks for training
        stride (int, optional): Stride interval to sample blocks from the raw time-series. Defaults to 1.
        ndata (int, optional): Number of time-series from the HDF5 file to block. Will use all if negative. Defaults to -1.
        eval (bool, optional): If this is a eval data-set, which will provide target states. Defaults to False.
        overwrite_cache (bool, optional): Overwrite cache file if it exists, i.e. embed the raw data from file. Defaults to False.
        cache_path (str, optional): Path to save the cached embeddings at. Defaults to None.
    """
    def __init__(
        self,
        embedder: EmbeddingModel,
        file_path: str,
        block_size: int,
        stride: int = 1,
        ndata: int = -1,
        eval: bool = False,
        overwrite_cache: bool = False,
        cache_path: str = None,
        **kwargs
    ):
        """Constructor method
        """
        assert os.path.isfile(file_path), 'Provided data file path does not exist!'

        self.block_size = block_size
        self.stride = stride
        self.ndata = ndata
        self.eval = eval

        directory, filename = os.path.split(file_path)
        if cache_path is None or not os.path.isdir(cache_path):
            cache_path = directory
        cached_features_file = os.path.join(
            cache_path, "cached{}_{}_{}_{}".format(ndata, embedder.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples, self.states = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start)

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                self.states = []
                # Read file and embed data using embedding model
                with h5py.File(file_path, "r") as f:
                    self.embed_data(f, embedder, **kwargs)

                start = time.time()
                os.makedirs(cache_path, exist_ok=True)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump((self.examples, self.states), handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    @abstractmethod
    def embed_data(self, h5_file: h5py.File, embedder: EmbeddingModel):
        """Embeds raw physical data into a 1D vector representation for the transformer.
        This is problem specific and thus must be overridden.

        Args:
            h5_file (h5py.File): HDF5 file object to read raw data from
            embedder (EmbeddingModel): Embedding neural network

        Raises:
            NotImplementedError: If function has not been overridden by a child dataset class.
        """
        raise NotImplementedError("embed_data function has not been properly overridden by a child class")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Eval dataset need to return states
        if self.eval:
            return {'inputs_embeds': self.examples[i][:1], 'labels_embeds': self.examples[i], 'states': self.states[i]}
        else:
            return {'inputs_embeds': self.examples[i][:-1], 'labels_embeds': self.examples[i][1:]}
            