import sys
import logging
import torch
from trphysx.config import HfArgumentParser
from trphysx.config.args import ModelArguments, TrainingArguments, DataArguments, ArgUtils
from rossler_module.configuration_rossler import RosslerConfig
from rossler_module.embedding_rossler import RosslerEmbedding
from rossler_module.viz_rossler import RosslerViz
from rossler_module.dataset_rossler import RosslerDataset, RosslerPredictDataset
from trphysx.transformer import PhysformerTrain, PhysformerGPT2
from trphysx.utils.trainer import Trainer

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    sys.argv = sys.argv + ["--init_name", "rossler"]
    sys.argv = sys.argv + ["--embedding_file_or_path", "./embedding/rossler_ntrain256/checkpoints/embedding_rossler300.pth"]
    sys.argv = sys.argv + ["--training_h5_file", "./data/rossler_training.hdf5"]
    sys.argv = sys.argv + ["--eval_h5_file", "./data/rossler_valid.hdf5"]
    sys.argv = sys.argv + ["--n_train", "256"]
    sys.argv = sys.argv + ["--stride", "64"]
    sys.argv = sys.argv + ["--train_batch_size", "64"]
    sys.argv = sys.argv + ["--max_grad_norm", "1.0"]

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

    # Rossler configuration
    config = RosslerConfig()
    # Load embedding model
    embedding_model = RosslerEmbedding(config).to(training_args.src_device)
    embedding_model.load_model(model_args.embedding_file_or_path)

    # Load visualization utility class
    viz = RosslerViz(training_args.plot_dir)
    
    # Init transformer model
    transformer = PhysformerGPT2(config, model_args.model_name)
    model  = PhysformerTrain(config, transformer)
    if(training_args.epoch_start > 0):
        model.load_model(training_args.ckpt_dir, epoch=training_args.epoch_start)
    if(model_args.transformer_file_or_path):
        model.load_model(model_args.transformer_file_or_path)
    
    # Initialize 
    training_data = RosslerDataset(
        embedding_model, 
        data_args.training_h5_file, 
        block_size=config.n_ctx, 
        stride=data_args.stride,
        ndata=data_args.n_train, 
        overwrite_cache=data_args.overwrite_cache)

    eval_data = RosslerPredictDataset(
        embedding_model, 
        data_args.eval_h5_file, 
        block_size=256,
        neval=data_args.n_eval, 
        overwrite_cache=data_args.overwrite_cache)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 14, 2, eta_min=1e-8)
    trainer = Trainer(model, training_args, (optimizer, scheduler), train_dataset=training_data, eval_dataset=eval_data, viz=viz)
    
    trainer.train()
