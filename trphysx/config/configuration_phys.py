import os
import json
import logging
import copy
from collections import OrderedDict
from typing import Dict, Tuple

logger = logging.getLogger(__name__)
CONFIG_NAME = "config-phys.json"

class PhysConfig(object):
    '''
    Slimmed version of pretrainedconfig
    '''
    model_type: str = ""

    def __init__(self, **kwargs):
        # Attributes with defaults
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.use_cache = kwargs.pop("use_cache", True)  # Not used by all models

        # Parameters for sequence generation
        self.max_length = kwargs.pop("max_length", 8)
        self.min_length = kwargs.pop("min_length", 0)

        # Special parameters for different tformer models
        self.k_size = kwargs.pop("k_size", 1)

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    def save_pretrained(self, save_directory):
        """
        Save a configuration object to the directory `save_directory`, so that it
        can be re-loaded using the :func:`~transformers.PretrainedConfig.from_pretrained` class method.

        Args:
            save_directory (:obj:`string`):
                Directory where the configuration JSON file will be saved.
        """
        if os.path.isfile(save_directory):
            raise AssertionError("Provided path ({}) should be a directory, not a file".format(save_directory))
        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file, use_diff=True)
        logger.info("Configuration saved in {}".format(output_config_file))

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs) -> "PretrainedConfig":
        """
        Constructs a `Config` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be retrieved
                from a pre-trained checkpoint by leveraging the :func:`~transformers.PretrainedConfig.get_config_dict`
                method.
            kwargs (:obj:`Dict[str, any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            :class:`PretrainedConfig`: An instance of a configuration object
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

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output

    def to_json_string(self, use_diff=True):
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (:obj:`bool`):
                If set to True, only the difference between the config instance and the default PretrainedConfig() is serialized to JSON string.
        Returns:
            :obj:`string`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path, use_diff=True):
        """
        Save this instance to a json file.

        Args:
            json_file_path (:obj:`string`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (:obj:`bool`):
                If set to True, only the difference between the config instance and the default PretrainedConfig() is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def update(self, config_dict: Dict):
        """
        Updates attributes of this class with attributes from `config_dict`.

        Args:
            :obj:`Dict[str, any]`: Dictionary of attributes that shall be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)