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
import numpy as np
from typing import Optional
import torch.nn.functional as F

import matplotlib as mpl

mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase

from .viz_model import Viz

class GrayScottViz(Viz):
    """Visualization class for the 3D Gray-scott system.

    Args:
        plot_dir (Optional[str], optional): Directory to save visualizations in. Defaults to None.
    """
    def __init__(self, plot_dir: Optional[str] = None):
        """Constructor method
        """
        super().__init__(plot_dir=plot_dir)

    def _createColorBarVertical(self, fig, ax0, c_min, c_max, label_format="{:02.2f}", cmap='viridis'):

        p0 = ax0[0].get_position().get_points().flatten()
        p1 = ax0[-1].get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p1[2] + 0.01, p1[1], 0.0075, p0[3] - p1[1]])
        # ax_cbar = fig.add_axes([p0[0], p0[1]-0.075, p0[2]-p0[0], 0.02])
        ticks = np.linspace(0, 1, 5)
        tickLabels = np.linspace(c_min, c_max, 5)
        tickLabels = [label_format.format(t0) for t0 in tickLabels]
        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
        cbar.set_ticklabels(tickLabels)

    def plotPrediction(self,
            y_pred: torch.Tensor,
            y_target: torch.Tensor,
            plot_dir: Optional[str] = None,
            epoch: Optional[int] = None,  # Epoch for file_name
            pid: Optional[int] = 0,  # Secondary plot ID
            nsteps: int = 10,
            stride: int = 5
        ):
        """Plots z-slice of Gray-Scott prediction along the z-axis and saves to file

        Args:
            y_pred (torch.Tensor): [T, 2, H, W, D] prediction time-series of states
            y_target (torch.Tensor): [T, 2, H, W, D] target time-series of states
            plot_dir (str, optional): Directory to save plots to. If none is provided the class plot_dir is used,
                defaults to None
            epoch (int, optional): Current epoch for file name, defaults to None
            pid (int, optional): Secondary plotting index value for filename, defaults to 0
            nsteps (int, optional): Number of time-steps to plot, defaults to 5
            stride (int, optional): Number of timesteps in between plots, defaults to 1
        """
        if plot_dir is None:
            plot_dir = self.plot_dir
        # Convert to numpy array
        y_pred = y_pred.detach().cpu().numpy()
        y_target = y_target.detach().cpu().numpy()

        plt.close('all')
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rcParams['figure.dpi'] = 200
        # mpl.rcParams['xtick.labelsize'] = 2
        # mpl.rcParams['ytick.labelsize'] = 2
        # rc('text', usetex=True)

        nz = y_pred.shape[-1]
        # Set up figure
        cmap0 = 'PuOr'

        fig, ax = plt.subplots(4, nsteps, figsize=(2.5*nsteps, 8))
        fig.subplots_adjust(wspace=0.2, hspace=0.2)

        # Plot each time-step
        for t0 in range(nsteps):
            # Species U
            ax[0, t0].imshow(y_target[t0 * stride, 0, :, :, nz // 2], vmin=0, vmax=1, cmap=cmap0, origin='lower')
            ax[1, t0].imshow(y_pred[t0 * stride, 0, :, :, nz // 2], vmin=0, vmax=1, cmap=cmap0, origin='lower')
            # Species V
            ax[2, t0].imshow(y_target[t0 * stride, 1, :, :, nz // 2], vmin=0, vmax=1, cmap=cmap0, origin='lower')
            ax[3, t0].imshow(y_pred[t0 * stride, 1, :, :, nz // 2], vmin=0, vmax=1, cmap=cmap0, origin='lower')

            ax[0, t0].set_title('t={:d}'.format(t0*stride))

            for ax0 in ax[:,t0]:
                ax0.set_xticks(np.linspace(0, 32, 5))
                ax0.set_yticks(np.linspace(0, 32, 5))
                for tick in ax0.xaxis.get_major_ticks():
                    tick.label.set_fontsize(10)
                for tick in ax0.yaxis.get_major_ticks():
                    tick.label.set_fontsize(10)

        ax[0, 0].set_ylabel('u Target')
        ax[1, 0].set_ylabel('u Prediction')
        ax[2, 0].set_ylabel('v Target')
        ax[3, 0].set_ylabel('v Prediction')

        self._createColorBarVertical(fig, ax[:,-1], c_min=0.0, c_max=1.0, cmap=cmap0)

        if not epoch is None:
            file_name = 'grayScottPred{:d}_{:d}'.format(pid, epoch)
        else:
            file_name = 'grayScottPred{:d}'.format(pid)
        self.saveFigure(plot_dir, file_name)