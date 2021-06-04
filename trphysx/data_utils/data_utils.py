'''
=====
- Associated publication:
url: 
doi: 
github: 
=====
'''
import logging
import torch
from typing import Dict, List, NewType, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DataCollator:
    """
    Data collator used for training datasets.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    mlm: bool = False # Masked language modeling, not supported

    # Default collator
    def __call__(self, examples:List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # inputs = self._tensorize_batch([example['input'] for example in examples])
        # props = self._tensorize_batch([example['positions'] for example in examples])
        # if self.mlm:
            # inputs, labels = self.mask_tokens(batch)
            # return {"input_ids": inputs, "labels": labels}
        # else:
        # labels = inputs[:, 1:].clone().detach()
        # inputs = inputs[:, :-1]
        # props = props[:, :-1]

        training_data = {}
        for key in examples[0].keys():
            training_data[key] = self._tensorize_batch([example[key] for example in examples])

        return training_data

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            raise ValueError("Padding not currently supported for physics transformers")
            return

@dataclass
class EvalDataCollator:
    """
    Data collator used for evaluation/testing datasets.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    # Default collator
    def __call__(self, examples:List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        eval_data = {}
        for key in examples[0].keys():
            eval_data[key] = self._tensorize_batch([example[key] for example in examples])
        # inputs = self._tensorize_batch([example['input'] for example in examples])
        # props = self._tensorize_batch([example['positions'] for example in examples])
        # targets = self._tensorize_batch([example['targets'] for example in examples])

        return eval_data

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)

        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            raise ValueError("Padding not currently supported for physics transformers")
            return