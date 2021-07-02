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

class GrayScottConfig(PhysConfig):
    """ This is the configuration class for the modeling of the Gray-scott system.
    """

    model_type = "cylinder"

    def __init__(
        self,
        n_ctx=128,
        n_embd=512,
        n_layer=2,
        n_head=32, # n_head must be a factor of n_embd
        state_dims=[2, 32, 32, 32],
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
