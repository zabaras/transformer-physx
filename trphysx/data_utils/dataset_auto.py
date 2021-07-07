"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
from collections import OrderedDict
from .dataset_lorenz import LorenzDataset
from .dataset_cylinder import CylinderDataset
from .dataset_grayscott import GrayscottDataset
from .dataset_phys import PhysicalDataset

DATASET_MAPPING = OrderedDict(
        [
            ("lorenz", LorenzDataset),
            ("cylinder", CylinderDataset),
            ("grayscott", GrayscottDataset),
        ]
    )

class AutoDataset():
    """Helper class for creating training data-sets for different numerical examples

    Raises:
        EnvironmentError: If direct initialization of this class is attempted.
    """
    def __init__(self):
        raise EnvironmentError(
            "AutoDataset should not be initiated directly. The class methods should be used instead."
        )

    @classmethod
    def create_dataset(cls, dataset_name: str, *args, **kwargs) -> PhysicalDataset:
        """Creates a data-set for testing or validation
        Currently supports: "lorenz", "cylinder", "grayscott"

        Args:
            dataset_name (str): Keyword/name of the data-set needed
            
        Raises:
            KeyError: If dataset_name is not a supported model type

        Returns:
            (PhysicalDataset): Initialized data-set
        """
        # First check if the model name is a pre-defined config
        if (dataset_name in DATASET_MAPPING.keys()):
            dataset_class = DATASET_MAPPING[dataset_name]
            # Init config class
            dataset = dataset_class(*args, **kwargs)
        else:
            raise KeyError("Provided model name: {}, not present in supported data-sets".format(dataset_name))
        return dataset