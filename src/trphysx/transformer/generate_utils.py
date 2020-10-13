import logging
from typing import Iterable, Optional, Tuple

import torch
from .function_wrappers import AutoGenerationWrapper
from torch import Tensor
from torch.nn import functional as F


logger = logging.getLogger(__name__)


class GenerationMixin:
    """
    A class contraining all of the functions supporting generation, to be used as a mixin in PreTrainedModel.
    """

    def prepare_inputs_for_generation(self, inputs_embeds, past, use_cache=True, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        # If we provide the past attention states we only need to provide the current input
        if past:
            inputs_embeds = inputs_embeds[:, -1].unsqueeze(1)

        return {"inputs_embeds": inputs_embeds, "past": past, "use_cache": use_cache}

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
        inputs_embeds: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **model_specific_kwargs
    ) -> torch.LongTensor:
        r"""
        """

        max_length = max_length if max_length is not None else self.config.max_length
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        batch_size = inputs_embeds.shape[0]  # overriden by the input batch_size

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."

        # create attention mask if necessary
        if attention_mask is None:
            attention_mask = inputs_embeds.new_ones(inputs_embeds.shape)

        cur_len = inputs_embeds.shape[1]

        assert (
            cur_len < max_length
        ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

        output = self._generate_time_series(
            inputs_embeds,
            cur_len=cur_len,
            max_length=max_length,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **model_specific_kwargs,
        )

        return output

    def _generate_time_series(
        self,
        inputs_embeds,
        cur_len,
        max_length,
        attention_mask,
        use_cache,
        **model_specific_kwargs,
    ):
        past = None

        forward = AutoGenerationWrapper.forward_wrapper(self.config.model_type, self.forward)
        while cur_len < max_length:
            if(inputs_embeds.size(1) <= self.config.n_ctx-1):
                model_inputs = self.prepare_inputs_for_generation(
                     inputs_embeds, past=None, attention_mask=attention_mask, use_cache=use_cache,
                )
            else:
                model_inputs = self.prepare_inputs_for_generation(
                    inputs_embeds[:,-(self.config.n_ctx):], past=None, attention_mask=attention_mask, use_cache=use_cache,
                )

            position_ids = torch.arange(0, inputs_embeds.size(1), dtype=torch.long, device=inputs_embeds.device)[-(self.config.n_ctx):].unsqueeze(0).repeat(inputs_embeds.size(0),1)
            # model_args = {**model_inputs, **model_specific_kwargs}
            outputs = forward(**model_inputs, position_ids=None)
            # print(outputs[0].size())
            next_output = outputs[0][:,-1:]

            # if model has past, then set the past variable to speed up decoding
            # if inputs_embeds.size(1) == self.config.n_ctx:
            #     print(outputs[1][0].size(), past[0].size())
            #     print(torch.sum(past[0][:,:,:,:] - outputs[1][0][:,:,:,:-1]))


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