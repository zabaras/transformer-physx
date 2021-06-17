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
import json
import logging
from typing import Optional
from collections import OrderedDict
from .embedding_lorenz import LorenzEmbedding, LorenzEmbeddingTrainer
from .embedding_cylinder import CylinderEmbedding, CylinderEmbeddingTrainer
from .embedding_grayscott import GrayScottEmbedding, GrayScottEmbeddingTrainer

MODEL_MAPPING = OrderedDict(
    [
        ("lorenz", LorenzEmbedding),
        ("cylinder", CylinderEmbedding),
        ("grayscott", GrayScottEmbedding)
    ]
)

TRAINING_MAPPING = OrderedDict(
    [
        ("lorenz", LorenzEmbeddingTrainer),
        ("cylinder", CylinderEmbeddingTrainer),
        ("grayscott", GrayScottEmbeddingTrainer)
    ]
)

logger = logging.getLogger(__name__)

class AutoEmbeddingModel:
    """Helper class for initializing of loading various embedding models.

    Raises:
        EnvironmentError: If direct initialization of this class is attempted.
    """
    def __init__(self):
        raise EnvironmentError(
            "AutoEmbeddingModel should not be initiated directly. The class methods should be used instead."
        )

    @classmethod
    def init_model(cls, model_name:str, config):
        """Initialize embedding model
        Currently supports: "lorenz", "cylinder", "grayscott"

        Args:
            model_name (str): Keyword/name of embedding model
            config (PhysConfig): Transformer configuration class

        Raises:
            AssertionError: If model_name is not a supported model type

        Returns:
            (EmbeddingModel): Initialized embedding model
        """
        # First check if the model name is a pre-defined config
        if(model_name in MODEL_MAPPING.keys()):
            model_class = MODEL_MAPPING[model_name]
            # Init config class
            model = model_class(config)
        else:
            err_str = "Provided model name, {:s}, not found in existing embedding models.".format(model_name)
            raise AssertionError(err_str)

        return model
    
    @classmethod
    def init_trainer(cls, model_name:str, config):
        """Initialize embedding model with a training head
        Currently supports: "lorenz", "cylinder", "grayscott"

        Args:
            model_name (str): Keyword/name of embedding model
            config (PhysConfig): Transformer configuration class

        Raises:
            AssertionError: If model_name is not a supported trainer model types

        Returns:
            (EmbeddingModel): Initialized embedding model trainer
        """
        # First check if the model name is a pre-defined config
        if(model_name in TRAINING_MAPPING.keys()):
            model_class = TRAINING_MAPPING[model_name]
            # Init config class
            model = model_class(config)
        else:
            err_str = "Provided model name, {:s}, not found in existing training models.".format(model_name)
            raise AssertionError(err_str)

        return model

    @classmethod
    def load_model(cls, model_name, config, file_or_path_directory:Optional[str]=None, epoch:int=0):
        """Initialize and load embedding model from memory
        Currently supports: "lorenz", "cylinder", "grayscott"

        Args:
            model_name (str): Keyword/name of embedding model
            config (PhysConfig): Transformer configuration class
            file_or_path_directory (str, optional): embedding model file or directory path
            epoch (int, optional): Epoch to load model from, only used if function is provided a directory

        Raises:
            AssertionError: If model_name is not a supported model type

        Returns:
            (EmbeddingModel): Initialized embedding model with loaded weights
        """
        # First check if the model name is a pre-defined config
        if(model_name in MODEL_MAPPING.keys()):
            model_class = MODEL_MAPPING[model_name]
            # Init config class
            model = model_class(config)
        else:
            err_str = "Provided model name, {:s}, not found in existing models.".format(model_name)
            raise AssertionError(err_str)

        # Attempt to load model from file.
        if(not file_or_path_directory is None):
            model.load_model(file_or_path_directory, epoch)

        return model
