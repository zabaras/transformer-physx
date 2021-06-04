import logging
import os, sys
sys.path.append("../")
from typing import Optional, Dict
import torch
from torch import nn
from .phys_transformer_base import PhysformerBase
from ..embedding.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)

class PhysformerTrain(PhysformerBase):
    """Model head for training the physics transformer base.
    """
    def __init__(self, config, transformer_model :Optional[PhysformerBase] = None):
        super().__init__(config)
        self.transformer = transformer_model
        self.transformer.apply(self._init_weights)

    def forward(
            self,
            inputs_embeds=None,
            labels_embeds=None,
            prop_embeds=None,
            past=None,
            attention_mask=None,
            position_ids=None,
            head_mask=None,
            use_cache=True,
            output_attentions=None,
    ):
        """
        Forward method for this head calculates the MSE between the predicted time-series and target embeddings
        This head allows for easy distribution to multiple GPUs and CPUs.
        """

        outputs = self.transformer.forward(
            inputs_embeds=inputs_embeds,
            past=past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            prop_embeds=prop_embeds,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
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

    def generate(self, *args, **kwargs):
        '''
        Generate call is just the forward call of the transformer
        '''
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


class PhysformerPredict(PhysformerBase):
    """Model head for prediction/evaluation the physics transformer. Used with plotting.
    """

    def __init__(self, config, embedding_model: Optional[EmbeddingModel] = None, transformer_model :Optional[PhysformerBase] = None):
        super().__init__(config)
        self.transformer = transformer_model
        self.transformer.apply(self._init_weights)

        self.embedding_model = embedding_model

    @torch.no_grad()
    def forward(self, input_states: torch.Tensor, nsteps: int = 100) -> Dict[
        str, torch.Tensor]:
        """
        Predicts time-series for nsteps given the input_states
        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """
        input_states = input_states.to(self.embedding_model.devices[0])
        inputs_embeds = self.embedding_model.embed(input_states)
        self.transformer.eval()

        output_embeds = self.transformer.generate(inputs_embeds, max_length=nsteps)
        yPred = self.embedding_model.recover(output_embeds)  # Undo embedding

        return {'yPred': yPred}

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