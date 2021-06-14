import sys, os
sys.path.append('../..')
import logging
import torch
from torch.utils.data import DataLoader, SequentialSampler

from trphysx.config import HfArgumentParser
from trphysx.config.args import ModelArguments, TrainingArguments, DataArguments, ArgUtils
from trphysx.config import AutoPhysConfig
from trphysx.transformer import PhysformerGPT2
from trphysx.embedding import AutoEmbeddingModel
from trphysx.viz import AutoViz
from trphysx.data_utils import AutoDataset, AutoPredictionDataset
from trphysx.data_utils.data_utils import EvalDataCollator
from trphysx.data_utils.dataset_lorenz import LorenzPredictDataset

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # Change this to different
    sys.argv = sys.argv + ["--init_name", "lorenz"]
    sys.argv = sys.argv + ["--embedding_file_or_path", "./embedding_lorenz300.pth"]
    sys.argv = sys.argv + ["--transformer_file_or_path", "./transformer_lorenz200.pth"]
    sys.argv = sys.argv + ["--eval_h5_file", "../../data/lorenz/lorenz_test.hdf5"]
    sys.argv = sys.argv + ["--n_eval", "16"]
    sys.argv = sys.argv + ["--overwrite_cache"]

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
        model_args.embedding_file_or_path).eval().to(training_args.src_device)

    # Init transformer model
    # config.n_embd = 12
    model = PhysformerGPT2(config, model_args.model_name).to(training_args.src_device)
    if (training_args.epoch_start > 0):
        model.load_model(training_args.ckpt_dir, epoch=training_args.epoch_start)
    if (model_args.transformer_file_or_path):
        model.load_model(model_args.transformer_file_or_path)

    viz = AutoViz.init_viz(model_args.viz_name)()

    eval_dataset = LorenzPredictDataset(
        embedding_model,
        data_args.eval_h5_file,
        block_size=1024,
        neval=data_args.n_eval,
        overwrite_cache=data_args.overwrite_cache,
        cache_path='./cache')

    sampler = SequentialSampler(eval_dataset)
    data_collator = EvalDataCollator()
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=sampler,
        batch_size=4,
        collate_fn=data_collator,
        drop_last=True
    )

    for mbidx, inputs in enumerate(eval_dataloader):

        inputs_embeds = inputs['inputs_embeds'][:, :1].to(training_args.src_device)
        # position_ids = inputs['position_ids'].to(training_args.src_device)
        targets = inputs['target_states'][:,:320].to(training_args.src_device)

        output_embeds = model.generate(inputs_embeds, max_length=320)
        # Recover features, note we have to move the time-dim into the batch before feeding it
        # into the recovery model.
        output = eval_dataloader.dataset.recover(output_embeds.reshape(-1, output_embeds.size(-1)))
        output = output.view([-1, output_embeds.size(1)] + list(output.shape[1:]))

        plot_dir = './imgs'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        logger.info('Plotting predictions for mini-batch {:d}'.format(mbidx))
        viz.plotMultiPrediction(output, targets, nplots=4, plot_dir=plot_dir, pid=mbidx)