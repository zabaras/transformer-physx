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
import copy
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

CONFIG_NAME = "config_trphysx.json"

class PhysConfig(object):
    """Parent class for physical transformer configuration.
    This is a slimmed version of the pretrainedconfig from the Hugging Face
    repository.

    Args:
        n_ctx (int): Context window of transformer model.
        n_embd (int): Dimensionality of the embeddings and hidden states.
        n_layer (int): Number of hidden layers in the transformer.
        n_head (int): Number of self-attention heads in each layer.
        state_dims (List): List of physical state dimensionality. Used in embedding models.
        activation_function (str, optional): Activation function. Defaults to "gelu_new".
        resid_pdrop (float, optional):
            The dropout probability for all fully connected layers in the transformer.
            Defaults to 0.0.
        embd_pdrop (float, optional):
            The dropout ratio for the embeddings. Defaults to 0.0.
        attn_pdrop (float, optional):
            The dropout ratio for the multi-head attention. Defaults to 0.0.
        layer_norm_epsilon (float, optional):
            The epsilon to use in the layer normalization layers. Defaults to 1e-5.
        initializer_range (float, optional):
            The standard deviation for initializing all weight matrices. Defaults to 0.02.
        output_hidden_states (bool, optional): Output embeddeding states from transformer. Defaults to False.
        output_attentions (bool, optional): Output attention values from transformer. Defaults to False.
        use_cache (bool, optional): Store transformers internal state for rapid predictions. Defaults to True.

    Raises:
        AssertionError: If provided parameter is not a config parameter
    """
    model_type: str = ""

    def __init__(self, **kwargs) -> None:
        # Transformer architecture parameters
        self.n_ctx = kwargs.pop("n_ctx")
        self.n_embd = kwargs.pop("n_embd")
        self.n_layer = kwargs.pop("n_layer")
        self.n_head = kwargs.pop("n_head")
        # Embedding model parameters
        self.state_dims = kwargs.pop("state_dims")

        self.activation_function = kwargs.pop("activation_function", "gelu_new")

        # Dropout regularization
        self.resid_pdrop = kwargs.pop("resid_pdrop", 0.0)
        self.embd_pdrop = kwargs.pop("embd_pdrop", 0.0)
        self.attn_pdrop = kwargs.pop("attn_pdrop", 0.0)

        self.layer_norm_epsilon = kwargs.pop("layer_norm_epsilon", 1e-5)
        self.initializer_range = kwargs.pop("initializer_range", 0.01)

        # Output/Prediction related attributes
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.use_cache = kwargs.pop("use_cache", True)  # Not used by all models

        # Parameters for sequence generation
        self.max_length = kwargs.pop("max_length", 8)
        self.min_length = kwargs.pop("min_length", 0)

        # Special parameters for different transformer models
        self.k_size = kwargs.pop("k_size", 1)

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save a configuration object to JSON file.

        Args:
            save_directory (str): Directory where the configuration JSON file will be saved.

        Raises:
            AssertionError: If provided directory does not exist.
        """
        if os.path.isfile(save_directory):
            raise AssertionError("Provided path ({}) should be a directory, not a file".format(save_directory))
        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file, use_diff=True)
        logger.info("Configuration saved in {}".format(output_config_file))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, any], **kwargs) -> "PhysConfig":
        """
        Constructs a config from a Python dictionary of parameters.

        Args:
            config_dict (Dict[str, any]): Dictionary of parameters.
            kwargs (Dict[str, any]): Additional parameters from which to initialize the configuration object.

        Returns:
            (PhysConfig): An instance of a configuration object
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model config %s", str(config))
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    def to_dict(self) -> Dict[str, any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            (Dict[str, any]): Dictionary of config attributes
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            (str): String of configuration instance in JSON format.
        """
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: str) -> None:
        """
        Save config instance to JSON file.

        Args:
            json_file_path (str): Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def update(self, config_dict: Dict) -> None:
        """
        Updates attributes of this class with attributes from provided dictionary.

        Args:
            config_dict (Dict): Dictionary of attributes that shall be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)