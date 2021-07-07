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
from typing import Tuple, Dict, Union, List
import torch
from torch import LongTensor, nn
from .phys_transformer_base import PhysformerBase
from ..embedding.embedding_model import EmbeddingModel
from ..config.configuration_phys import PhysConfig

logger = logging.getLogger(__name__)

Tensor = torch.Tensor

class PhysformerTrain(PhysformerBase):
    """Model head for training the physics transformer base.

    Args:
        config (PhysConfig): Phys-transformer config object
        transformer_model (PhysformerBase): Initialized transformer model
    """
    def __init__(self, config: PhysConfig, transformer_model: PhysformerBase = None) -> None:
        """Constructor
        """
        super().__init__(config)
        self.transformer = transformer_model
        self.transformer.apply(self._init_weights)

    def forward(
        self,
        inputs_embeds: Tensor,
        labels_embeds: Tensor,
        **kwargs
    ) -> Tuple[Union[float, Tensor]]:
        """Forward method for this head calculates the MSE between the predicted time-series and target embeddings
        This head allows for easy distribution to multiple GPUs and CPUs. See transformer 

        Args:
            inputs_embeds (Tensor): [B, T, n_embed] Input features
            labels_embeds (Tensor): [B, T, n_embed] Target output features
            **kwargs (optional): Additional tensformer forward pass arguments

        Returns:
            Tuple[Union[float, Tensor]]: mse loss, last hidden state, (present attention state), 
            (all hidden_states), (attention scores)
        """
        outputs = self.transformer.forward(
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        # If label embeddings are provided, compute loss
        if labels_embeds is not None:
            hidden_states = outputs[0]

            # Flatten the tokens
            loss_fct = nn.MSELoss()
            loss = loss_fct(hidden_states, labels_embeds)

            # loss = loss+ loss_fct(shift_hidden[:,:3], shift_labels[:,:3])
            outputs = (loss,) + (hidden_states, labels_embeds,) + outputs[1:]

        return outputs # (loss), last hidden state, (presents), (all hidden_states), (attentions)

    def evaluate(
        self,
        inputs_embeds: Tensor,
        labels_embeds: Tensor,
        **kwargs
    ) -> Tuple[Union[float, Tensor]]:
        """Generate a time-series prediction using the transformer and calc MSE error.

        Args:
            inputs_embeds (Tensor): [B, 1, n_embed] Starting input feature(s)
            labels_embeds (Tensor): [B, T, n_embed] Target output features
            **kwargs (optional): Additional tensformer forward pass arguments

        Returns:
            Tuple[Union[float, Tensor]]: mse loss, last hidden state, (present attention state), 
            (all hidden_states), (attention scores)
        """

        max_length = labels_embeds.size(1)

        outputs = self.transformer.generate(
            inputs_embeds=inputs_embeds,
            max_length = max_length,
            **kwargs
        )

        pred_embeds = outputs[0]

        # Flatten the tokens
        err_fct = nn.MSELoss()
        error = err_fct(pred_embeds, labels_embeds)

        outputs = (error,) + (pred_embeds, labels_embeds,) + outputs[1:]

        return outputs

    def generate(self, *args, **kwargs):
        """
        Generate call is just the forward call of the transformer
        """
        return self.transformer.generate(*args, **kwargs)

    def save_model(self, *args, **kwargs):
        """
        Saves physformer model
        """
        self.transformer.save_model(*args, **kwargs)


    def load_model(self, *args, **kwargs):
        """
        Load a physformer model
        """
        self.transformer.load_model(*args, **kwargs)