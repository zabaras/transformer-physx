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
from .viz_lorenz import LorenzViz
from .viz_cylinder import CylinderViz
from .viz_grayscott import GrayScottViz
from .viz_model import Viz

VIZ_MAPPING = OrderedDict(
    [
        ("lorenz", LorenzViz),
        ("cylinder", CylinderViz),
        ("grayscott", GrayScottViz),
    ]
)

class AutoViz():
    """Helper class for initializing visualization classes.

    Raises:
        EnvironmentError: If direct initialization of this class is attempted.
    """
    def __init__(self):
        raise EnvironmentError(
            "AutoEmbeddingModel should not be initiated directly. The class methods should be used instead."
        )

    @classmethod
    def init_viz(cls, viz_name: str) -> Viz:
        """Initializes visualization class.
        Currently supports: "lorenz", "cylinder", "grayscott"

        Args:
            viz_name (str): Keyword/name of visualization class

        Raises:
            ValueError: If viz_name is not a supported visualization type

        Returns:
            (Viz): Initialized viz class
        """
        # First check if the model name is a pre-defined config
        if(viz_name in VIZ_MAPPING.keys()):
            return VIZ_MAPPING[viz_name]
        else:
            err_str = "Provided viz name, {:s}, not found in existing visualization classes.".format(viz_name)
            raise ValueError(err_str)