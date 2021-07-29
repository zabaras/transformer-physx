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
from trphysx.config import PhysConfig

logger = logging.getLogger(__name__)

class RosslerConfig(PhysConfig):
    """
    This is the configuration class for the modeling of the Rossler system.
    """

    model_type = "rossler"

    def __init__(
        self,
        n_ctx=32,
        n_embd=32,
        n_layer=4,
        n_head=4, # n_head must be a factor of n_embd
        state_dims=[3],
        activation_function="gelu_new",
        initializer_range=0.02,
        **kwargs
    ):
        super().__init__(
            n_ctx=n_ctx,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            state_dims=state_dims,
            activation_function=activation_function,
            initializer_range=initializer_range,
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
