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
from .configuration_phys import PhysConfig

logger = logging.getLogger(__name__)

class CylinderConfig(PhysConfig):
    """ This is the configuration class for the modeling of the flow around a cylinder system.
    """

    model_type = "cylinder"

    def __init__(
        self,
        n_ctx=16,
        n_embd=128,
        n_layer=3,
        n_head=8,
        state_dims=[3,64,128],
        activation_function="gelu_new",
        **kwargs
    ):
        super().__init__(
            n_ctx=n_ctx,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            state_dims=state_dims,
            activation_function=activation_function,
            **kwargs
        )

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer