"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import logging
import h5py
import torch
from trphysx.data_utils import PhysicalDataset
from trphysx.embedding import EmbeddingModel

logger = logging.getLogger(__name__)

class RosslerDataset(PhysicalDataset):
    """Dataset for the Rossler numerical example
    """
    def embed_data(self, h5_file: h5py.File, embedder: EmbeddingModel) -> None:
        """Embeds rossler data into a 1D vector representation for the transformer.

        Args:
            h5_file (h5py.File): HDF5 file object of raw data
            embedder (EmbeddingModel): Embedding neural network
        """
        # Iterate through stored time-series
        samples = 0
        for key in h5_file.keys():
            data_series = torch.Tensor(h5_file[key]).to(embedder.devices[0]).view([-1] + embedder.input_dims)
            with torch.no_grad():
                embedded_series = embedder.embed(data_series).cpu()
            # Stride over time-series
            for i in range(0, data_series.size(0) - self.block_size + 1,
                           self.stride):  # Truncate in block of block_size
                data_series0 = embedded_series[i: i + self.block_size]
                self.examples.append(data_series0)

                if self.eval:
                    self.states.append(data_series[i: i + self.block_size].cpu())

            samples = samples + 1
            if (self.ndata > 0 and samples >= self.ndata):  # If we have enough time-series samples break loop
                break
        
        logger.info(
            'Collected {:d} time-series from hdf5 file. Total of {:d} time-series.'.format(samples, len(self.examples))
            )