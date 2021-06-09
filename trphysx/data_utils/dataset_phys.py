'''
=====
- Associated publication:
url: 
doi: 
github: 
=====
'''
import logging
import os
import pickle
import time
import h5py

import torch
from typing import Dict, List, NewType, Tuple, Optional
from dataclasses import dataclass
from abc import abstractmethod
from collections import OrderedDict
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from ..embedding.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)

class PhysicalDataset(Dataset):
    """Parent class for various datasets.
    The caching of the dataset is based on the Hugging Face implementation.

    Args:
        embedder (:class:`trphysx.embedding.embedding_model.EmbeddingModel`): Embedding neural network
        file_path (str): Path to hdf5 raw data file
        block_size (int): Length of time-series blocks for training
        stride (int, optional): Stride interval to sample blocks from the raw time-series. Defaults to 1.
        ndata (int, optional): Number of time-series from the HDF5 file to use (not necessary equal to the number of time-series blocks). Defaults to -1.
        save_states (bool, optional): To save the physical states or not, should be True for validation and testing. Defaults to False.
        overwrite_cache (bool, optional): Overwrite cache file if it exists, i.e. embed the raw data from file. Defaults to False.
        cache_path (str, optional): Path to save the cached embeddings at. Defaults to None.
    """
    def __init__(
        self,
        embedder: EmbeddingModel,
        file_path: str,
        block_size: int,
        stride:int = 1,
        ndata:int = -1,
        patch_size:int = 1, # Patch size, used for the vizgpt2 model
        save_states:bool = False, # Save physical states as well (used for testing)
        overwrite_cache:bool = False,
        cache_path:Optional[str] = None
    ):
        """Constructor method
        """
        self.block_size = block_size + patch_size # Add patch_size because initial state is not predicted
        self.stride = stride
        self.ndata = ndata
        self.patch_size = patch_size
        assert os.path.isfile(file_path), 'Provided data file path does not exist!'

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
                # TODO: Speed up the embedding process if possible
                with h5py.File(file_path, "r") as f:
                    self.embed_data(f, embedder, save_states)

                if not save_states:
                    self.states = None
                start = time.time()
                os.makedirs(cache_path, exist_ok=True)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump((self.examples, self.states), handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    @abstractmethod
    def embed_data(self, h5_file: h5py.File, embedder: EmbeddingModel, save_states: bool = False):
        """Embeds raw physical data into a 1D vector representation for the transformer.
        This is problem specific and thus must be overridden.

        TODO: Remove redundant arguments

        Args:
            h5_file (h5py.File): HDF5 file object to read raw data from
            embedder (:class:`trphysx.embedding.embedding_model.EmbeddingModel`): Embedding neural network
            save_states (bool, optional): To save the physical states or not, should be True for validation and testing. Defaults to False.

        Raises:
            NotImplementedError: If function has not been overridden by a child dataset class.
        """
        raise NotImplementedError("embed_data function has not been properly overridden by a child class")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {'inputs_embeds': self.examples[i][:-1], 'labels_embeds': self.examples[i][1:]}