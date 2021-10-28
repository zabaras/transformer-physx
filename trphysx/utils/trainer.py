"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import sys
import os
import logging
import torch
import torch.nn as nn
import numpy as np

from typing import Any, Dict, Tuple
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from ..transformer.phys_transformer_helpers import PhysformerTrain
from ..config.args import  TrainingArguments
from ..data_utils.data_utils import DataCollator
from ..viz.viz_model import Viz
from ..embedding.embedding_model import EmbeddingModel
from .metrics import Metrics

logger = logging.getLogger(__name__)

Optimizer = torch.optim.Optimizer
Scheduler = torch.optim.lr_scheduler._LRScheduler
Tensor = torch.Tensor

def set_seed(seed: int) -> None:
    """Set random seed

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Trainer:
    """Generalized trainer for physics transformer models

    Args:
        model (PhysformerTrain): Transformer with training head
        args (TrainingArguments): Training arguments
        optimizers (Tuple[Optimizer, Scheduler], optional): Tuple of Pytorch optimizer and lr scheduler.
        train_dataset (Dataset, optional): Training dataset. Defaults to None.
        eval_dataset (Dataset, optional): Eval/Validation dataset. Defaults to None.
        embedding_model (EmbeddingModel, optional): Embedding model. Used for recovering states during
            state evaluation of the model. Defaults to None.
        viz (Viz, optional): Visualization class. Defaults to None.
    """
    def __init__(self,
        model: PhysformerTrain,
        args: TrainingArguments,
        optimizers: Tuple[Optimizer, Scheduler],
        train_dataset: Dataset = None,
        eval_dataset: Dataset = None,
        embedding_model: EmbeddingModel = None,
        viz: Viz = None
    ) -> None:
        
        self.model = model.to(args.src_device)
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizers = optimizers
        self.log_metrics = Metrics(file_path = self.args.exp_dir, file_name = "log_metrics.h5")
        self.embedding_model = embedding_model
        self.viz = viz

        # Load pre-trained state dictionaries if necessary
        if self.args.epoch_start > 0:
            logger.info('Attempting to load optimizer, model and scheduler from epoch: {:d}'.format(self.args.epoch_start))

            optimizer_path =  os.path.join(self.args.ckpt_dir, "optimizer{:d}.pt".format(self.args.epoch_start))
            if os.path.isfile(optimizer_path):
                optimizer_dict = torch.load(optimizer_path, map_location=lambda storage, loc: storage)
                self.optimizers[0].load_state_dict(optimizer_dict)

            schedular_path =  os.path.join(self.args.ckpt_dir, "scheduler{:d}.pt".format(self.args.epoch_start))
            if os.path.isfile(schedular_path):
                schedular_dict = torch.load(schedular_path, map_location=lambda storage, loc: storage)
                self.optimizers[1].load_state_dict(schedular_dict)

            self.model.load_model(self.args.ckpt_dir, epoch=self.args.epoch_start)

        set_seed(self.args.seed)

    
    def get_train_dataloader(self, train_dataset: Dataset = None) -> DataLoader:
        """Creates a training dataloader. Overload for unusual training cases.

        Args:
            train_dataset (Dataset, optional): Optional training dataset. If none is provided,
            the class training data will be used. Defaults to None.

        Raises:
            ValueError: If both the dataset parameter and class dataset have not been provided

        Returns:
            DataLoader: Training dataloader
        """
        train_dataset = train_dataset if train_dataset is not None else self.train_dataset
        if train_dataset is None:
            raise ValueError("Training dataset not provided.")

        train_batch_size = len(train_dataset) if self.args.train_batch_size > len(
            train_dataset) else self.args.train_batch_size

        train_sampler = RandomSampler(train_dataset)

        data_collator = DataCollator()

        data_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        return data_loader

    def get_eval_dataloader(self, eval_dataset: Dataset = None) -> DataLoader:
        """Creates a evaluation dataloader used for validation or testing of model.

        Args:
            eval_dataset (Dataset, optional): Optional eval dataset. If none is provided,
            the class eval data will be used. Defaults to None.

        Raises:
            ValueError: If both the dataset parameter and class dataset have not been provided

        Returns:
            DataLoader: Evaluation dataloader
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Evaluation dataset not provided.")

        eval_batch_size = len(eval_dataset) if self.args.eval_batch_size > len(
            eval_dataset) else self.args.eval_batch_size

        sampler = SequentialSampler(eval_dataset)

        data_collator = DataCollator()

        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        return data_loader

    def train(self) -> None:
        """Trains the transformer model
        """
        optimizer = self.optimizers[0]
        lr_scheduler = self.optimizers[1]

        model = self.model

        # Set up model parallelize if available
        # multi-gpu training
        if self.args.n_gpu > 1:
            logger.info('Using {:d} GPUs to train.'.format(self.args.n_gpu))
            model = torch.nn.DataParallel(model)

        # Distributed training
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        # Loop over epochs
        training_loader = self.get_train_dataloader()
        for epoch in range(self.args.epoch_start+1, self.args.epochs + 1):
            
            self.args.gradient_accumulation_steps = min([self.args.gradient_accumulation_steps, len(training_loader)])
            
            loss_total = 0.0
            model.zero_grad()
            # Loop over mini-batched
            for mbidx, inputs in enumerate(training_loader):
                
                loss0, _, _ =  self.training_step(model, inputs)

                loss_total = loss_total + loss0/len(training_loader)

                # Optimize model
                if (mbidx + 1) % self.args.gradient_accumulation_steps == 0 or mbidx == len(training_loader)-1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step(epoch + float(mbidx) / len(training_loader))
                    model.zero_grad()
                    
                    self.epoch = epoch + (mbidx + 1.) / len(training_loader)

            for param_group in optimizer.param_groups:
                cur_lr = param_group['lr']
                break

            logger.info("Current Learning rate: {:.05f}".format(cur_lr))
            logger.info("Epoch {:d}: Training loss {:.05f}".format(epoch, loss_total))
            self.log_metrics.push(epoch=epoch, loss=loss_total)

            # Evaluate model
            if(epoch % self.args.eval_steps == 0 or epoch == 1):
                for param_group in optimizer.param_groups:
                    cur_lr = param_group['lr']
                    break
                logger.info("Current Learning rate: {:.05f}".format(cur_lr))
                logger.info('Evaluating...')
                self.evaluate(epoch=epoch)

            # Checkpointing model
            if epoch % self.args.save_steps == 0 or epoch == 1:
                # In all cases (even distributed/parallel), self.model is always a reference
                # to the model we want to save.
                if hasattr(model, "module"):
                    assert model.module is self.model
                else:
                    assert model is self.model
                logger.info("Checkpointing model, optimizer and scheduler.")
                # Save model checkpoint
                self.model.save_model(self.args.ckpt_dir, epoch=epoch)
                torch.save(optimizer.state_dict(), os.path.join(self.args.ckpt_dir, "optimizer{:d}.pt".format(epoch)))
                torch.save(lr_scheduler.state_dict(), os.path.join(self.args.ckpt_dir, "scheduler{:d}.pt".format(epoch)))
                # Save log file
                self.log_metrics.writeToHDF5()



    def training_step(
        self, 
        model: PhysformerTrain, 
        inputs: Dict[str, Any]
    ) -> Tuple[float, Tensor, Tensor]:
        """Calls a forward pass of the training model and backprops 
        for a single time-step

        Args:
            model (PhysformerTrain): Transformer model with training head, could be 
            inputs (Dict[str, Any]): Dictionary of model inputs for forward pass

        Returns:
            Tuple[float, Tensor, Tensor]: Tuple containing: loss value, hidden states
                of transformer, attention states of the transformer.
        """
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.src_device)

        # Training head forward
        outputs = model(**inputs)
        loss = outputs[0] # Loss value is always the first output

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        # Backward
        loss.backward()

        return loss.item(), outputs[1], outputs[2]

    @torch.no_grad()
    def evaluate(
        self, 
        epoch: int = None
    ) -> Dict[str, float]:
        """Run evaluation and return metrics.

        Args:
            epoch (int, optional): Current epoch, used for naming figures. Defaults to None.

        Returns:
            Dict[str, float]: Dictionary of prediction metrics
        """

        eval_dataloader = self.get_eval_dataloader()

        eval_error = 0
        state_error = 0
        timestep_error = None

        for mbidx, inputs in enumerate(eval_dataloader):

            states = inputs['states']
            del inputs['states']

            if mbidx == 0:
                timestep_error = torch.zeros(inputs['labels_embeds'].size(1))            

            pred_error0, timestep_error0, pred_embeds = self.eval_step(self.model, inputs)

            eval_error += pred_error0/len(eval_dataloader)
            timestep_error += timestep_error0/len(eval_dataloader)
            
            plot_id = mbidx*self.args.eval_batch_size # Plotting id used to index figures
            state_error0 = self.eval_states(pred_embeds, states, epoch, plot_id=plot_id)
            state_error += state_error0/len(eval_dataloader)

        logger.info('Eval embedding error: {:.02f}, State error: {:.02f}'.format(eval_error, state_error))
        self.log_metrics.push(eval_epoch=epoch, eval_error=float(eval_error), state_error=float(state_error))
        self.log_metrics.time_error = timestep_error.cpu().numpy()

        return {'eval_error': eval_error}

    @torch.no_grad()
    def eval_step(
        self, 
        model: PhysformerTrain, 
        inputs: Dict[str, Any]
    ) -> Tuple[float, Tensor, Tensor]:
        """Calls a eval pass of the training model.

        Args:
            model (PhysformerTrain): Transformer model with training head
            inputs (Dict[str, Any]): Dictionary of model inputs for forward pass

        Returns:
            Tuple[float, Tensor, Tensor]: Tuple containing: prediction error value, 
                time-step error, predicted embeddings.
        """
        model.eval()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.src_device)

        # Training head forward
        outputs = model.evaluate(**inputs)
        pred_error = outputs[0] # Loss value is always the first output

        # Compute loss at each time-step
        mseLoss = nn.MSELoss(reduction='none') # Manual summing
        timestep_error = mseLoss(outputs[1], outputs[2]).mean(dim=(0,2)).cpu()

        return pred_error, timestep_error, outputs[1]

    @torch.no_grad()
    def eval_states(
        self, 
        pred_embeds: Tensor, 
        states: Any, 
        epoch: int = None, 
        plot_id: int = 0, 
        plot: bool = True
    ) -> float:
        """Evaluates the predicted states by recovering the state space from
        the predicted embedding vectors. Can be overloaded for cases with
        special methods for recovering the state field.

        Args:
            pred_embeds (Tensor): [B, T, n_embed] Predicted embedded vectors
            states (Any): Target states / data for recovery 
            epoch (int, optional): Current epoch, used for naming figures. Defaults to None.
            plot_id (int, optional): Secondary plotting id to distinguish between numerical cases. Defaults to 0.
            plot (bool, optional): Plot models states. Defaults to True.

        Returns:
            float: Predicted state MSE error
        """
        if self.embedding_model is None:
            logger.warning('No embedding model provided, cannot recover state predictions.')
            return 0

        bsize = pred_embeds.size(0)
        tsize = pred_embeds.size(1)
        device = self.embedding_model.devices[0]
        
        states = states.to(device)
        x_in = pred_embeds.contiguous().view(-1, pred_embeds.size(-1)).to(device)
        out = self.embedding_model.recover(x_in)
        out = out.view([bsize, tsize] + self.embedding_model.input_dims)

        mse = nn.MSELoss()
        state_error = mse(out, states)

        if self.viz and plot:
            # Loop through mini batch and plot eval cases
            for i in range(bsize):
                plot_id += i
                # Dont plot if exceed max plot limit
                if plot_id < self.args.plot_max:
                    self.viz.plotPrediction(
                        out[i], 
                        states[i],
                        self.args.plot_dir, 
                        epoch=epoch, 
                        pid=plot_id )

        return state_error