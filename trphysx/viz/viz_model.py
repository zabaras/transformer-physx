'''
=====
- Associated publication:
url: 
doi: 
github: 
=====
'''
import os
import torch
from abc import abstractmethod
from typing import Optional
import matplotlib.pyplot as plt

class Viz(object):
    """Parent class for visualization

    Args:
        plot_dir (Optional[str], optional): Directory to save visualizations in. Defaults to None.
    """
    def __init__(self, plot_dir:Optional[str] = None):
        """Constructor method
        """
        super().__init__()
        self.plot_dir = plot_dir

    @abstractmethod
    def plotPrediction(self, y_pred:torch.Tensor, y_target:torch.Tensor, plot_dir:Optional[str] = None, **kwargs):
        """Plots model prediction and target values

        Args:
            y_pred (torch.Tensor): prediction tensor
            y_target (torch.Tensor): target tensor
            plot_dir (Optional[str], optional): Directory to save plot at. Defaults to None.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: If function has not been overridden by a child dataset class.
        """
        raise NotImplementedError("plotPrediction not initialized by child class.")

    @abstractmethod
    def plotEmbeddingPrediction(self, y_pred:torch.Tensor, y_target:torch.Tensor, plot_dir:Optional[str] = None, **kwargs):
        """Plots model prediction and target values during the embedding training

        Args:
            y_pred (torch.Tensor): mini-batch of prediction tensor
            y_target (torch.Tensor): mini-batch target tensor
            plot_dir (Optional[str], optional): Directory to save plot at. Defaults to None.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: If function has not been overridden by a child dataset class.
        """
        raise NotImplementedError("plotEmbeddingPrediction not initialized by child class.")

    def saveFigure(self, plot_dir:Optional[str] = None, file_name:str='plot', savepng=True, savepdf=False):
        """Saves active matplot lib figure to file

        Args:
            plot_dir (Optional[str], optional): Directory to save plot at, will use class plot_dir if none provided . Defaults to None.
            file_name (str, optional): File name of the saved figure. Defaults to 'plot'.
            savepng (bool, optional): Save figure in png format. Defaults to True.
            savepdf (bool, optional): Save figure in pdf format. Defaults to False.
        """
        if plot_dir is None:
            plot_dir = self.plot_dir

        assert os.path.isdir(plot_dir), 'Provided directory string is not a valid directory: {:s}'.format(plot_dir)
        # Create plotting path if it does not exist
        os.makedirs(plot_dir, exist_ok=True)
        
        if savepng:
            plt.savefig(os.path.join(plot_dir, file_name)+".png", bbox_inches='tight')
        if savepdf:
            plt.savefig(os.path.join(plot_dir, file_name)+".pdf", bbox_inches='tight')

