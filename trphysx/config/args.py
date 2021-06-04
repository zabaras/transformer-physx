import os
import logging
import torch
from dataclasses import dataclass, field
from typing import Optional #Needs python 3.8 for literal

HOME = os.getcwd()
INITS = ['lorenz', 'cylinder', 'grayscott']
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    init_name: Optional[str] = field(
        default='lorenz', metadata={"help": "Used as a global default initialization token for different experiments."}
    )
    model_name: Optional[str] = field(
        default=None, metadata={"help": "The name model of the transformer model"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    embedding_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained embedding model name"}
    )
    embedding_file_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained embedding model path"}
    )
    transformer_file_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained transformer model path"}
    )
    viz_name: Optional[str] = field(
        default=None, metadata={"help": "Visualization class name"}
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to training and evaluation data.
    """
    n_train: Optional[int] = field(
        default=2048, metadata={"help": "Number of training time-series to use"}
    )
    n_eval: Optional[int] = field(
        default=256, metadata={"help": "Number of evaluation time-series to use"}
    )
    stride: Optional[int] = field(
        default=32, metadata={"help": " Stride to segment the training data at"}
    )
    training_h5_file: Optional[str] = field(
        default=None, metadata={"help": "File path to the training data hdf5 file"}
    )
    eval_h5_file: Optional[str] = field(
        default=None, metadata={"help": "File path to the evaluation data hdf5 file"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    cache_path:Optional[str] = field(
        default=None, metadata={"help": "File directory to write cache file to"}
    )

@dataclass
class TrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    # Training paths for logging, checkpoints etc.
    exp_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory to store data related to the experiment"}
    )
    ckpt_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory to save model checkpoints during training"}
    )
    plot_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory to save plots during training"}
    )
    save_steps: Optional[int] = field(
        default=25, metadata={"help": "Epoch stride to save checkpoints"}
    )
    eval_steps: Optional[int] = field(
        default=25, metadata={"help": "Epoch stride to evaluate validation data-set"}
    )

    epoch_start: Optional[int] = field(
        default=0, metadata={"help": "Epoch to start training at"}
    )
    epochs: Optional[int] = field(
        default=200, metadata={"help": "Number of epochs to train"}
    )

    # ===== Optimization parameters =====
    lr: Optional[float] = field(
        default=0.001, metadata={"help": "Learning rate"}
    )
    max_grad_norm: Optional[float] = field(
        default=0.1, metadata={"help": "Norm limit for clipping gradients"}
    )
    dataloader_drop_last: bool = field(
        default=True, metadata={"help": "Drop training cases no in a full mini-batch"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=int(1), metadata={"help": "How many mini-batches to compute before updating weights"}
    )

    # ===== Data loader parameters =====
    train_batch_size: Optional[int] = field(
        default=256, metadata={"help": "Number of training cases in mini-batch"}
    )
    eval_batch_size: Optional[int] = field(
        default=16, metadata={"help": "Number of evaluation cases in mini-batch"}
    )

    # ===== Parallel parameters =====
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Local rank of the CPU process, -1 means just use a single CPU"}
    )
    n_gpu: Optional[int] = field(
        default=1, metadata={"help": "Number of GPUs per CPU"}
    )
    seed: Optional[int] = field(
        default=12345, metadata={"help": "Random seed for reproducibility"}
    )
    notes: Optional[str] = field(
        default=None, metadata={"help": "Notes that will be appended to experiment folder"}
    )


class ArgUtils:
    '''
    Argument utility class for modifying particular arguments after initialization
    '''
    @classmethod
    def config(cls, modelArgs:ModelArguments, dataArgs:DataArguments, trainingArgs:TrainingArguments, create_paths=True):
        modelArgs = cls.configModelNames(modelArgs)
        if create_paths:
            modelArgs, dataArgs, trainingArgs = cls.configPaths(modelArgs, dataArgs, trainingArgs)
        trainingArgs = cls.configTorchDevices(trainingArgs)
        return modelArgs, dataArgs, trainingArgs

    @classmethod
    def configModelNames(cls, modelArgs:ModelArguments):
        # Set up model, config, viz and embedding names
        if not modelArgs.init_name in INITS:
            logger.warn('Selected init name not in built-in models. Be careful.')

        attribs = ["model_name", "config_name", "embedding_name", "viz_name"]
        for attrib in attribs:
            if getattr(modelArgs, attrib) is None:
                setattr(modelArgs, attrib, modelArgs.init_name)

        return modelArgs

    @classmethod
    def configPaths(cls, modelArgs:ModelArguments, dataArgs:DataArguments, trainingArgs:TrainingArguments):
        # Set up training paths
        if(trainingArgs.exp_dir is None):
            trainingArgs.exp_dir = os.path.join(HOME, 'outputs', '{:s}'.format(modelArgs.config_name), \
                    'ntrain{:d}_epochs{:d}_batch{:d}'.format(dataArgs.n_train, trainingArgs.epochs, trainingArgs.train_batch_size))
            if trainingArgs.notes: # If notes add them to experiment folder name
                trainingArgs.exp_dir = os.path.join(os.path.dirname(trainingArgs.exp_dir), os.path.basename(trainingArgs.exp_dir)+'_{:s}'.format(trainingArgs.notes))
            
        if(trainingArgs.ckpt_dir is None):
            trainingArgs.ckpt_dir = os.path.join(trainingArgs.exp_dir, 'checkpoints')
        
        if(trainingArgs.plot_dir is None):
            trainingArgs.plot_dir = os.path.join(trainingArgs.exp_dir, 'viz')

        # Create directories if they don't exist already
        os.makedirs(trainingArgs.exp_dir, exist_ok=True)
        os.makedirs(trainingArgs.ckpt_dir, exist_ok=True)
        os.makedirs(trainingArgs.plot_dir, exist_ok=True)

        return modelArgs, dataArgs, trainingArgs

    @classmethod
    def configTorchDevices(cls, args:TrainingArguments):
        # Set up PyTorch device(s)
        if(torch.cuda.device_count() > 1 and args.n_gpu > 1):
            if(torch.cuda.device_count() < args.n_gpu):
                args.n_gpu = torch.cuda.device_count()
            if(args.n_gpu < 1):
                args.n_gpu = torch.cuda.device_count()
            logging.info("Looks like we have {:d} GPUs to use. Going parallel.".format(args.n_gpu))
            args.device_ids = [i for i in range(0,args.n_gpu)]
            args.src_device = "cuda:{}".format(args.device_ids[0])
        elif(torch.cuda.is_available()):
            logging.info("Using a single GPU for training.")
            args.device_ids = [0]
            args.src_device = "cuda:{}".format(args.device_ids[0])
            args.n_gpu = 1
        else:
            logging.info("No GPUs found, will be training on CPU.")
            args.src_device = "cpu"
        return args