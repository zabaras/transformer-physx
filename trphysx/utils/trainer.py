'''
=====
- Associated publication:
url: 
doi: 
github: 
=====
'''
import sys
import os
import logging
import h5py
import torch
import torch.nn as nn
import numpy as np

from typing import Any, Union, Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from ..transformer.phys_transformer_helpers import PhysformerTrain
from ..config.args import  TrainingArguments
from ..data_utils.data_utils import DataCollator, EvalDataCollator
from ..viz.viz_model import Viz
from .metrics import Metrics

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Trainer:
    """Trainer for physics transformer model

    Args:
        model (PhysformerTrain): Transformer
        args (TrainingArguments): Training arguements
        optimizers (Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR], optional): Tuple of pytorch optimizer and lr scheduler. Defaults to None.
        train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
        eval_dataset (Optional[Dataset], optional): Eval/Validation dataset. Defaults to None.
        viz (Optional[Viz], optional): Visualization class. Defaults to None.
    """
    def __init__(self,
            model: PhysformerTrain,
            args: TrainingArguments,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            viz: Optional[Viz] = None,
        ):
        
        self.model = model.to(args.src_device)
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizers = optimizers
        self.log_metrics = Metrics(file_name = os.path.join(self.args.exp_dir, "log_metrics.h5"))
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

    # TODO: Think about moving these to data_utils file....
    def get_train_dataloader(self, train_dataset: Optional[Dataset] = None) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = train_dataset if train_dataset is not None else self.train_dataset

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

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        eval_batch_size = len(eval_dataset) if self.args.eval_batch_size > len(
            eval_dataset) else self.args.eval_batch_size

        sampler = SequentialSampler(eval_dataset)

        data_collator = EvalDataCollator()

        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        return data_loader

    def train(self):
        """
        Trains the transformer model
        TODO: Add loading of optimizer and scheduler
        """
        optimizer = self.optimizers[0]
        lr_scheduler = self.optimizers[1]

        model = self.model

        # Set up model parellize if available
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
        tr_loss = 0.0
        for epoch in range(self.args.epoch_start+1, self.args.epochs + 1):
            
            training_loader = self.get_train_dataloader()
            self.args.gradient_accumulation_steps = min([self.args.gradient_accumulation_steps, len(training_loader)])
            
            loss_total = 0.0
            model.zero_grad()
            for mbidx, inputs in enumerate(training_loader):
                
                loss0, _, _ =  self._training_step(model, inputs)

                tr_loss = tr_loss + loss0
                loss_total = loss_total + loss0/len(training_loader)

                if (mbidx + 1) % self.args.gradient_accumulation_steps == 0 or mbidx == len(training_loader)-1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step(epoch + float(mbidx) / len(training_loader))
                    model.zero_grad()
                    tr_loss = 0
                    # self.global_step += 1
                    self.epoch = epoch + (mbidx + 1.) / len(training_loader)

            for param_group in optimizer.param_groups:
                cur_lr = param_group['lr']
                break
            logger.info("Current Learning rate: {:.05f}".format(cur_lr))
            logger.info("Epoch {:d}: Training loss {:.05f}".format(epoch, loss_total))
            self.log_metrics.push(epoch=epoch, loss=loss_total)

            if(epoch % self.args.eval_steps == 0 or epoch == 1):
                for param_group in optimizer.param_groups:
                    cur_lr = param_group['lr']
                    break
                logger.info("Current Learning rate: {:.05f}".format(cur_lr))
                logger.info('Evaluating...')
                self.evaluate(epoch=epoch)

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

        # If starting from beginning delete log file
        if self.args.epoch_start == 0:
            self.log_metrics.delHDF5()
        self.log_metrics.writeToHDF5(os.path.join(self.args.exp_dir, "log_metrics.h5"))


    def _training_step(self, model: PhysformerTrain, inputs: Dict[str, Union[torch.Tensor, Any]]) -> float:
        """Trains a single time-step

        Args:
            model (PhysformerTrain): Transformer model
            inputs (Dict[str, Union[torch.Tensor, Any]]): Dictionary of model keyword arguments

        Returns:
            (tuple): tuple containing:
                hidden_states), (attentions)
                | (float): Training loss
                | (torch.Tensor): Hidden states from transformer
                | (torch.Tensor): Attention states from transformer
        """
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.src_device)

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        loss.backward()

        return loss.item(), outputs[1], outputs[2]

    @torch.no_grad()
    def evaluate(self, eval_dataset: Optional[Dataset] = None, epoch:Optional[int] = None) -> Dict[str, float]:
        """Run evaluation and return metrics.

        Args:
            eval_dataset (Optional[Dataset], optional): Pass a dataset if you wish to override the 
                one on the instance. Defaults to None.
            epoch (Optional[int], optional): Current epoch, used for naming figures. Defaults to None.

        Returns:
            Dict[str, float]: Dictionary of prediction metrics
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        pred_error = 0
        timestep_error = None
        mseLoss = nn.MSELoss(reduction='none') # Manual summing
        for mbidx, inputs in enumerate(eval_dataloader):

            target_states = inputs['targets'].to(self.args.src_device)
            del inputs['targets']
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.src_device)

            if timestep_error is None:
                timestep_error = torch.zeros(inputs['inputs_embeds'].size(1))

            output_embeds = self.model.generate(inputs, max_length=target_states.size(1), position_ids=None)
            # Recover features, note we have to move the time-dim into the batch before feeding it
            # into the recovery model.
            output = eval_dataloader.dataset.recover(output_embeds.reshape(-1, output_embeds.size(-1)))
            output = output.view([inputs['inputs_embeds'].size(0), -1] + list(output.shape[1:]))

            # For generation there is no shift!
            # The outputs includes the first step
            if mbidx == 0 and self.viz:
                self.viz.plotPrediction(output[0], target_states[0], self.args.plot_dir, epoch=epoch, pid=0)
                self.viz.plotPrediction(output[-1], target_states[-1], self.args.plot_dir, epoch=epoch, pid=1)
                # self.viz.plotPrediction(output[2,:512,:3], targets[2,:512,:3], self.args.plot_dir, epoch=epoch, pid=2)
                # self.viz.plotPrediction(output[3,:512,:3], targets[3,:512,:3], self.args.plot_dir, epoch=epoch, pid=3)

            endIdx = min([output.size(1), target_states.size(1)])
            pred_error = pred_error + mseLoss(output[:, :endIdx], target_states[:, :endIdx]).mean().item()/len(eval_dataloader)
            # Compute error as a function of time-steps
            dims = np.delete(np.arange(0, len(output.shape), 1 , dtype=np.uint8), 1)
            # timestep_error = timestep_error + mseLoss(output[:,:endIdx], target_states[:,:endIdx]).mean(dim=tuple(dims)).cpu()/len(eval_dataloader)
            

        logger.info('Test loss: {:.02f}'.format(pred_error))
        self.log_metrics.push(eval_epoch=epoch, eval_loss=pred_error)
        self.log_metrics.time_error = timestep_error.cpu().numpy()

        return {'pred_error': pred_error}

    @torch.no_grad()
    def evaluate_error(
            self, eval_dataset: Optional[Dataset] = None, epoch: Optional[int] = None) -> Dict:
        """Run evaluation and return metrics.
        TODO: Make sure not used ans remove
        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataset (Optional[Dataset], optional): Pass a dataset if you wish to override the 
                one on the instance. Defaults to None.
            epoch (Optional[int], optional): Current epoch, used for naming figures. Defaults to None.

        Returns:
            Dict[str, float]: Dictionary of prediction metrics
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        pred_error = 0
        timestep_error = None
        mseLoss = nn.MSELoss(reduction='none')  # Manual summing
        for mbidx, inputs in enumerate(eval_dataloader):

            inputs_embeds = inputs['inputs_embeds'][:, :1].to(self.args.src_device)
            targets = eval_dataloader.dataset.recover(inputs['inputs_embeds']).to(self.args.src_device)

            if timestep_error is None:
                timestep_error = torch.zeros(inputs['inputs_embeds'].size(1)).to(self.args.src_device)

            output_embeds = self.model.generate(inputs_embeds, max_length=targets.size(1))

            output = eval_dataloader.dataset.recover(output_embeds)

            endIdx = min([output.size(1), targets.size(1)])
            pred_error = pred_error + mseLoss(output[:, :endIdx, :3], targets[:, :endIdx, :3]).mean().item() / len(eval_dataloader)
            timestep_error = timestep_error + mseLoss(output[:, :endIdx, :3], targets[:, :endIdx, :3]).mean(dim=(0, 2)) / len(eval_dataloader)

        return {'pred_error': pred_error, 'time_error': timestep_error}