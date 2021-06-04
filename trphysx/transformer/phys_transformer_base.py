import logging
import math
import os, sys
from typing import Optional
from abc import abstractmethod

import torch
from torch import nn
from torch.nn import MSELoss
from .utils import  Conv1D

logger = logging.getLogger(__name__)

class PhysformerBase(nn.Module):

    model_name: str = "transformer_model"

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        # Save config in model
        self.config = config

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def generate(self):
        pass

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            # print(name, param.numel())
            count += param.numel()
        return count
    
    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.
        """
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None:
            self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """ Tie or clone module weights depending of whether we are using TorchScript or not
        """
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = torch.nn.functional.pad(
                output_embeddings.bias.data,
                (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def save_model(self, save_directory, epoch=0):
        """
        Saves embedding model to the specified directory.
        """
        if os.path.isfile(save_directory):
            raise AssertionError("Provided path ({}) should be a directory, not a file".format(save_directory))

        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "{}{:d}.pth".format(self.model_name, epoch))
        # Save pytorch model to file
        torch.save(self.state_dict(), output_model_file)


    def load_model(self, file_or_path_directory, epoch=0):
        """
        Load a embedding model from the specified file or path
        """
        if os.path.isfile(file_or_path_directory):
            logger.info('Loading transformer model from file: {}'.format(file_or_path_directory))
            self.load_state_dict(torch.load(file_or_path_directory, map_location=lambda storage, loc: storage))
        elif os.path.isdir(file_or_path_directory):
            file_path = os.path.join(file_or_path_directory, "{}{:d}.pth".format(self.model_name, epoch))
            logger.info('Loading transformer model from file: {}'.format(file_path))
            self.load_state_dict(torch.load(file_path, map_location=lambda storage, loc: storage))
        else:
            logger.warn("Provided path or file ({}) does not exist".format(file_or_path_directory))