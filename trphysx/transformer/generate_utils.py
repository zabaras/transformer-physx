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
import torch
from typing import Tuple, Dict
from torch import Tensor
from torch.nn import functional as F

logger = logging.getLogger(__name__)

Tensor = torch.Tensor
LongTensor = torch.LongTensor

class GenerationMixin:
    """Class containing generative functions for transformers
    """
    def prepare_inputs_for_generation(
        self, 
        inputs_embeds: Tensor,
        position_ids: Tensor = None,
        prop_embeds: Tensor = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """Prepares input features for prediction

        Args:
            inputs_features (Dict[str, Tensor]): Input feature tensors
            that are being generated.

        Returns:
            Dict[str, Tensor]: Dictionary of model inputs
        """
        inputs_features = {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "prop_embeds": prop_embeds }
        inputs = {}

        for k, v in inputs_features.items():
            if isinstance(v, torch.Tensor):
                # Make sure all embeddings are of equal and proper length
                inputs[k] = v[:, -self.config.n_ctx:]

        if "past" in kwargs.keys():
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v[:, -1].unsqueeze(1)

        return {**inputs, **kwargs}

    def _use_cache(self, outputs, use_cache):
        """During generation, decide whether to pass the `past` variable to the next forward pass."""
        if len(outputs) <= 1 or use_cache is False:
            return False
        if hasattr(self.config, "mem_len") and self.config.mem_len == 0:
            return False
        return True

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: Tensor,
        position_ids: Tensor = None,
        prop_embeds: Tensor = None,
        max_length: int = None,
        attention_mask: LongTensor = None,
        use_cache: bool = False,
        **model_specific_kwargs
    ) -> Tensor:
        """Generated a predicted sequence of features

        Args:
            inputs_embeds (Tensor): [batch, seq, n_embed] Input feature tensor
            position_ids (Tensor, optional): [seq, n_embed] Position tensor. Defaults to None.
            prop_embeds (Tensor, optional): [batch, seq, n_embed] Property tensor. Defaults to None.
            max_length (int, optional): Length of time series to predict. Defaults to None.
            attention_mask (LongTensor, optional): Manual attention mask. Defaults to None.
            use_cache (bool, optional): Cache past transformer states for faster generation. Defaults to False.

        Returns:
            Tensor: [batch, max_length, n_embed] Predicted feature tensor
        """
        max_length = max_length if max_length is not None else self.config.max_length
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."

        # create attention mask if necessary
        # if attention_mask is None:
        #     attention_mask = torch.ones(inputs_embeds.shape).to(inputs_embeds.device)

        output = self._generate_time_series(
            inputs_embeds,
            position_ids,
            prop_embeds,
            max_length=max_length,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **model_specific_kwargs,
        )

        return output

    def _generate_time_series(
        self,
        inputs_embeds: Tensor,
        position_ids: Tensor,
        prop_embeds: Tensor,
        max_length: int,
        use_cache: bool = None,
        **model_specific_kwargs
    ) -> Tensor:
        """Function that calls model forward to predict 

        Args:
            inputs_embeds (Tensor): [batch, seq, n_embed] Input feature tensor
            position_ids (Tensor, optional): [seq, n_embed] Position tensor. Defaults to None.
            prop_embeds (Tensor, optional): [batch, seq, n_embed] Property tensor. Defaults to None.
            max_length (int, optional): Length of time series to predict. Defaults to None.
            attention_mask (LongTensor, optional): Manual attention mask. Defaults to None.
            use_cache (bool, optional): [description]. Defaults to None.

        Returns:
            Tensor: [batch, max_length, n_embed] Predicted feature tensor
        """
        past = None

        cur_len = inputs_embeds.shape[1]
        assert (
            cur_len < max_length
        ), f"The input context is {cur_len}, but `max_length` is only {max_length}. Please make sure that `max_length` larger than the input"

        while cur_len < max_length:
            # Prepare inputs for transformer
            model_inputs = self.prepare_inputs_for_generation(
                inputs_embeds, 
                position_ids, 
                prop_embeds, 
                use_cache=use_cache, 
                past = past,
                **model_specific_kwargs,
            )

            outputs = self.forward(**model_inputs)

            next_output = outputs[0][:,-1:]

            if self._use_cache(outputs, use_cache):
                past = [output[:, :, :, -(self.config.n_ctx-1):] for output in outputs[1]]

            # add past output embedding and increase length by one
            inputs_embeds = torch.cat([inputs_embeds, next_output], dim=1)
            cur_len = cur_len + 1

            # If number of time-steps has surpassed model capacity, start dropping
            # the earliest time-step from the past states
            # if(cur_len > self.config.n_ctx):
                # Dim [keys/query, batch, heads, tsteps, n_embed]
                # past = tuple(attention_state[:,:,:,1:] for attention_state in past)

        return inputs_embeds

    @staticmethod
    def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:
        return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)