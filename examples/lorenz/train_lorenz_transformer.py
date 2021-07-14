"""
=====
Training transformer model for the Lorenz numerical example.
This is a built-in model from the paper.

Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import sys
import logging
import torch
from trphysx.config import HfArgumentParser
from trphysx.config.args import ModelArguments, TrainingArguments, DataArguments, ArgUtils
from trphysx.config import AutoPhysConfig
from trphysx.transformer import PhysformerTrain, PhysformerGPT2
from trphysx.embedding import AutoEmbeddingModel
from trphysx.viz import AutoViz
from trphysx.data_utils import AutoDataset
from trphysx.utils.trainer import Trainer

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    sys.argv = sys.argv + ["--init_name", "lorenz"]
    sys.argv = sys.argv + ["--embedding_file_or_path", "./embedding_lorenz300.pth"]
    sys.argv = sys.argv + ["--training_h5_file","./data/lorenz_training_rk.hdf5"]
    sys.argv = sys.argv + ["--eval_h5_file","./data/lorenz_valid_rk.hdf5"]
    sys.argv = sys.argv + ["--train_batch_size", "32"]
    sys.argv = sys.argv + ["--stride", "64"]
    sys.argv = sys.argv + ["--n_train", "2048"]
    sys.argv = sys.argv + ["--save_steps", "25"]
    sys.argv = sys.argv + ["--n_eval", "16"]

    # Parse arguments using the hugging face argument parser
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN)
    # Configure arguments after intialization 
    model_args, data_args, training_args = ArgUtils.config(model_args, data_args, training_args)

    # Load model configuration
    config = AutoPhysConfig.load_config(model_args.config_name)

    # Load embedding model
    embedding_model = AutoEmbeddingModel.load_model(
        model_args.embedding_name, 
        config, 
        model_args.embedding_file_or_path).to(training_args.src_device)

    # Load visualization utility class
    viz = AutoViz.init_viz(model_args.viz_name)(training_args.plot_dir)
    
    # Init transformer model
    transformer = PhysformerGPT2(config, model_args.model_name)
    model  = PhysformerTrain(config, transformer)
    if(training_args.epoch_start > 0):
        model.load_model(training_args.ckpt_dir, epoch=training_args.epoch_start)
    if(model_args.transformer_file_or_path):
        model.load_model(model_args.transformer_file_or_path)
    
    # Initialize training and validation datasets
    training_data = AutoDataset.create_dataset(
        model_args.model_name,
        embedding_model, 
        data_args.training_h5_file, 
        block_size=config.n_ctx, 
        stride=data_args.stride,
        ndata=data_args.n_train, 
        overwrite_cache=data_args.overwrite_cache)

    eval_data = AutoDataset.create_dataset(
        model_args.model_name,
        embedding_model, 
        data_args.eval_h5_file, 
        block_size=256,
        stride=1024,
        ndata=data_args.n_eval, 
        eval = True,
        overwrite_cache=data_args.overwrite_cache)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.lr, weight_decay=1e-10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 14, 2, eta_min=1e-9)
    trainer = Trainer(
        model, 
        training_args, 
        (optimizer, scheduler), 
        train_dataset = training_data, 
        eval_dataset = eval_data, 
        embedding_model = embedding_model,
        viz=viz )
    
    trainer.train()