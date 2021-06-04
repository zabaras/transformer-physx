import os
import json
import logging
from collections import OrderedDict
from .configuration_lorenz import LorenzConfig
from .configuration_cylinder import CylinderConfig
from .configuration_grayscott import GrayScottConfig
from .configuration_reservoir import ReservoirConfig
from .configuration_phys import PhysConfig

CONFIG_MAPPING = OrderedDict(
    [
        ("lorenz", LorenzConfig),
        ("cylinder", CylinderConfig),
        ("grayscott", GrayScottConfig),
        ("reservoir", ReservoirConfig),
    ]
)
CONFIG_NAME = "config-phys.json"

logger = logging.getLogger(__name__)

class AutoPhysConfig:

    @classmethod
    def load_config(cls, model_name_or_path, **kwargs):

        # Check if file is a folder path
        if os.path.isdir(model_name_or_path):
            config_file = os.path.join(model_name_or_path, CONFIG_NAME)
            config_dict = cls.from_json_file(config_file)
        elif os.path.isfile(model_name_or_path):
            config_file = model_name_or_path
            config_dict = cls.from_json_file(config_file)
        else:
            config_dict = {"model_type": model_name_or_path}

        # First check if the model name is a pre-defined config
        if(model_name_or_path in CONFIG_MAPPING.keys()):
            config_class = CONFIG_MAPPING[config_dict["model_type"]]
            # Init config class
            config = config_class(config_dict, **kwargs)
        else:
            config = PhysConfig.from_dict(config_dict, **kwargs)

        return config

    @classmethod
    def from_json_file(cls, json_file: str) -> "dict":
        """
        Constructs a `Config` from the path to a json file of parameters.
        Args:
            json_file (:obj:`string`):
                Path to the JSON file containing the parameters.
        Returns:
            :class:`PretrainedConfig`: An instance of a configuration object
        """
        config_dict = cls._dict_from_json_file(json_file)
        return config_dict

    @classmethod
    def _dict_from_json_file(cls, json_file: str):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)