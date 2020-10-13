'''
=====
- Associated publication:
url: 
doi: 
github: 
=====
'''
import logging
import h5py
import torch
from .dataset_phys import PhysicalDataset
from ..embedding.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)

class GrayscottDataset(PhysicalDataset):
    """Dataset class for the Gray-scott numerical example.
    """
    def embed_data(self, h5_file: h5py.File, embedder: EmbeddingModel, save_states: bool = False):
        """Embeds cylinder flow data into a 1D vector representation for the transformer.

        TODO: Remove redundant arguments, add minibatch option for the encoding

        Args:
            h5_file (h5py.File): HDF5 file object of Lorenz raw data
            embedder (EmbeddingModel): Embedding neural network
            save_states (bool, optional): To save the physical states or not, should be True for validation and testing. Defaults to False.
        """
        # Iterate through stored time-series
        samples = 0
        embedder.eval()

        logger.info('Parsing hdf5 file and embedding data, this could take a bit...')
        # Loop simulations
        for key in h5_file.keys():

            u = torch.Tensor(h5_file[key + '/u'])
            v = torch.Tensor(h5_file[key + '/v'])
            data_series = torch.stack([u, v], dim=1).to(embedder.devices[0])
            # data_series = torch.nn.functional.interpolate(data_series, (32, 32, 32), mode='trilinear', align_corners=True)

            embedded_series = torch.zeros([data_series.size(0)]+[embedder.embedding_dims])
            with torch.no_grad():
                # Mini-batch embedding due to model size
                for i in range(0, data_series.size(0), 96):
                    embedded_series[i: i+96] = embedder.embed(data_series[i: i+96]).cpu()

            # Stride over time-series
            for i in range(0, data_series.size(0) - self.block_size + 1, self.stride):  # Truncate in block of block_size
                data_series0 = embedded_series[i: i + self.block_size]
                self.examples.append(data_series0)
                self.position_ids.append(torch.arange(0, self.block_size, dtype=torch.long)+i)
                if save_states:
                    self.states.append(data_series[i: i + self.block_size].cpu())

            samples = samples + 1
            if self.ndata > 0 and samples >= self.ndata:  # If we have enough time-series samples break loop
                break

        logger.info('Collected {:d} time-series from hdf5 file for a total of {:d} time-series.'.format(samples, len(self.examples)))


class GrayscottPredictDataset(GrayscottDataset):
    """Prediction data-set for the flow around a cylinder numerical example. Used during testing/validation
    since this data-set will store the embedding model and target states.
    TODO: Use mix-in for recover and get item methods?

    Args:
        embedder (EmbeddingModel): Embedding neural network
        file_path (str): Path to hdf5 raw data file
        block_size (int): Length of time-series blocks for training
        stride (int, optional): Stride interval to sample blocks from the raw time-series. Defaults to 1.
        neval (int, optional): Number of time-series from the HDF5 file to use for testing. Defaults to 16.
        overwrite_cache (bool, optional): Overwrite cache file if it exists, i.e. embeded the raw data from file. Defaults to False.
        cache_path (str, optional): Path to save the cached embeddings at. Defaults to None.
    """
    def __init__(self, embedder: EmbeddingModel, file_path: str, block_size: int, neval: int = 16,
                 overwrite_cache: bool =False, cache_path: str =None):
        """Constructor method
        """  
        super().__init__(embedder, file_path, block_size, stride=block_size, ndata=neval, save_states=True,
                         overwrite_cache=overwrite_cache, cache_path=cache_path)
        self.embedder = embedder

    @torch.no_grad()
    def recover(self, x0, mb_size:int = 96):
        """Recovers the physical state variables from an embedded vector

        Args:
            x0 (torch.Tensor): [B, config.n_embd] Time-series of embedded vectors
            mb_size (int, optional): Mini-batch size for recovering the state variables

        Returns:
            (torch.Tensor): [B, 2, H, W, D] physical state variable tensor
        """
        x = x0.contiguous().view(-1, self.embedder.embedding_dims).to(self.embedder.devices[0])
        out = torch.zeros([x.size(0)] + self.embedder.input_dims)
        # Mini-batch 
        for i in range(0, x.size(0), mb_size):
            out[i: i+mb_size] = self.embedder.recover(x[i: i+mb_size]).cpu()

        # out = self.embedder.recover(x)
        return out.view([-1] + self.embedder.input_dims)

    def __getitem__(self, i) -> torch.Tensor:
        return {'input': self.examples[i], 'targets': self.states[i], 'positions': self.position_ids[i]}