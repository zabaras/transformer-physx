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
import argparse

from typing import Any, Union, Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from ..embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class EmbeddingTrainer:
    """Trainer for Koopman embedding model

    Args:
        model (EmbeddingModel): Embedding model
        args (TrainingArguments): Training arguments
        optimizers (Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR], optional): Tuple of pytorch optimizer and lr scheduler. Defaults to None.
        train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
        eval_dataset (Optional[Dataset], optional): Eval/Validation dataset. Defaults to None.
        viz (Optional[Viz], optional): Visualization class. Defaults to None.
    """
    def __init__(self,
            model: EmbeddingModel,
            args: argparse.ArgumentParser,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
            viz = None
        ):
        
        self.model = model.to(args.device)
        self.args = args
        self.optimizers = optimizers
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

    def trainKoopman(self, training_loader:DataLoader, eval_dataloader:DataLoader):
        """Trains the transformer model
        TODO: Add loading of optimizer and scheduler
        """
        optimizer = self.optimizers[0]
        lr_scheduler = self.optimizers[1]

        model = self.model.train()

        # Loop over epochs
        for epoch in range(self.args.epoch_start+1, self.args.epochs + 1):
              
            loss_total = 0.0
            loss_reconstruct = 0.0
            model.zero_grad()
            for mbidx, inputs in enumerate(training_loader):

                loss0, loss_reconstruct0 = model(**inputs)
                loss0 = loss0.sum()

                loss_reconstruct = loss_reconstruct + loss_reconstruct0.sum()
                loss_total = loss_total + loss0.detach()
                # Backwards!
                loss0.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()

                if mbidx+1 % 10 == 0:
                    logger.info('Epoch {:d}: Completed mini-batch {}/{}.'.format(epoch, mbidx+1, len(training_loader)))

            lr_scheduler.step()
            for param_group in optimizer.param_groups:
                cur_lr = param_group['lr']
                break
            logger.info("Epoch {:d}: Training loss {:.03f}, Lr {:.05f}".format(epoch, loss_total, cur_lr))

            if(epoch%5 == 0 or epoch == 1):
                output = self.evaluate(eval_dataloader, epoch=epoch)
                logger.info('Epoch {:d}: Test loss: {:.02f}'.format(epoch, output['test_error']))

            if epoch % self.args.save_steps == 0:
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


    @torch.no_grad()
    def evaluate(self, eval_dataloader:DataLoader, epoch:int = 0) -> Dict[str, float]:
        """Run evaluation and return metrics.

        Args:
            eval_dataset (Optional[Dataset], optional): Pass a dataset if you wish to override the 
                one on the instance. Defaults to None.
            epoch (Optional[int], optional): Current epoch, used for naming figures. Defaults to None.

        Returns:
            Dict[str, float]: Dictionary of prediction metrics
        """
        model = self.model.embedding_model.eval()

        test_loss = 0
        mseLoss = nn.MSELoss()
        for mbidx, inputs in enumerate(eval_dataloader):
            
            # Pull out targets from prediction dataset
            yTarget = inputs['input_states'][:,1:].to(self.args.device)

            xInput = inputs['input_states'][:,:-1].to(self.args.device)
            yPred = torch.zeros(yTarget.size()).to(self.args.device)
            
            del inputs['input_states']
            # Keep extra arguements that are provided by the collocator
            for key in inputs.keys():
                inputs[key] = inputs[key].to(self.args.device)

            # Test accuracy of one time-step
            for i in range(xInput.size(1)):
                xInput0 = xInput[:,i].to(self.args.device)
                g0 = model.embed(xInput0, **inputs)
                yPred0 = model.recover(g0)
                yPred[:,i] = yPred0.squeeze().detach()

            test_loss = test_loss + mseLoss(yTarget, yPred)

            if not self.viz is None and mbidx == 0:
                self.viz.plotEmbeddingPrediction(yPred,yTarget, epoch=epoch)

            return {'test_error': test_loss/len(eval_dataloader)}