'''
=====
- Associated publication:
url: 
doi: 
github: 
=====
'''
from collections import OrderedDict
from .dataset_lorenz import LorenzDataset, LorenzPredictDataset
from .dataset_cylinder import CylinderDataset, CylinderPredictDataset
from .dataset_grayscott import GrayscottDataset, GrayscottPredictDataset

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
    def create_dataset(cls, dataset_name: str, *args, **kwargs):
        """Creates a data-set for testing or validation
        Currently supports: "lorenz", "cylinder", "grayscott"

        Args:
            dataset_name (str): Keyword/name of the data-set needed
            *args: Arguments to be passed to dataset class constructor
            **kwargs: Keyword arguments to be passed to dataset class constructor

        Raises:
            AssertionError: If dataset_name is not a supported model type

        Returns:
            :class:`trphysx.data_utils.dataset_phys.PhysicalDataset`: Initialized data-set
        """
        # First check if the model name is a pre-defined config
        if (dataset_name in DATASET_MAPPING.keys()):
            dataset_class = DATASET_MAPPING[dataset_name]
            # Init config class
            dataset = dataset_class(*args, **kwargs)
        else:
            raise ValueError("Provided model name: {}, not present in supported data-sets".format(dataset_name))
        return dataset


PREDSET_MAPPING = OrderedDict(
        [
            ("lorenz", LorenzPredictDataset),
            ("cylinder", CylinderPredictDataset),
            ("grayscott", GrayscottPredictDataset),
        ]
    )

class AutoPredictionDataset():
    """Helper class for creating testing/validation data-sets for different numerical examples

    Raises:
        EnvironmentError: If direct initialization of this class is attempted.
    """
    def __init__(self):
        raise EnvironmentError(
            "AutoPredictionDataset should not be initiated directly. The class methods should be used instead."
        )

    @classmethod
    def create_dataset(cls, dataset_name: str, *args, **kwargs):
        """Creates a data-set for testing or validation
        Currently supports: "lorenz", "cylinder", "grayscott"

        Args:
            dataset_name (str): Keyword/name of the data-set needed
            *args: Arguments to be passed to dataset class constructor
            **kwargs: Keyword arguments to be passed to dataset class constructor

        Raises:
            AssertionError: If dataset_name is not a supported model type

        Returns:
            :class:`trphysx.data_utils.dataset_phys.PhysicalDataset`: Initialized data-set
        """
        # First check if the model name is a pre-defined config
        if (dataset_name in PREDSET_MAPPING.keys()):
            dataset_class = PREDSET_MAPPING[dataset_name]
            # Init config class
            dataset = dataset_class(*args, **kwargs)
        else:
            raise ValueError("Provided model name: {}, not present in supported prediction data-sets".format(dataset_name))
        return dataset