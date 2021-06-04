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

class CylinderViz(Viz):
    """Visualization class for flow around a cylinder

    Args:
        plot_dir (Optional[str], optional): Directory to save visualizations in. Defaults to None.
    """
    def __init__(self, plot_dir: Optional[str] = None):
        """Constructor method
        """
        super().__init__(plot_dir=plot_dir)

    def _createColorBarVertical(self, fig, ax0, c_min, c_max, label_format="{:02.2f}", cmap='viridis'):

        p0 = ax0[0, -1].get_position().get_points().flatten()
        p1 = ax0[1, -1].get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p1[2] + 0.005, p1[1], 0.0075, p0[3] - p1[1]])
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
            epoch: Optional[int] = None,
            pid: Optional[int] = 0,
            nsteps: int = 10,
            stride: int = 10
        ):
        """Plots the predicted x-velocity, y-velocity and pressure field contours

        Args:
            y_pred (torch.Tensor): [T, 3, H, W] Prediction tensor.
            y_target (torch.Tensor): [T, 3, H, W] Target tensor.
            plot_dir (Optional[str], optional): Directory to save figure, overrides plot_dir one if provided. Defaults to None.
            epoch (Optional[int], optional): Current epoch, used for file name. Defaults to None.
            pid (int, Optional): Optional plotting id for indexing file name manually, Defaults to 0.
            nsteps (int, Optional): Number of timesteps to plot, Defaults to 10.
            stride (int, optional): Number of timesteps in between plots. Defaults to 20.
        """
        if plot_dir is None:
            plot_dir = self.plot_dir
        # Convert to numpy array
        y_pred = y_pred.detach().cpu().numpy()
        y_target = y_target.detach().cpu().numpy()

        plt.close('all')
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rcParams['figure.dpi'] = 300
        mpl.rcParams['xtick.labelsize'] = 2
        mpl.rcParams['ytick.labelsize'] = 2
        # rc('text', usetex=True)

        # Set up figure
        cmap0 = 'inferno'
        for i, field in enumerate(['ux', 'uy', 'p']):
            plt.close("all")

            # fig, ax = plt.subplots(2+yPred0.size(0), yPred0.size(2), figsize=(2*yPred0.size(1), 3+3*yPred0.size(0)))
            fig, ax = plt.subplots(2, nsteps, figsize=(2.1*nsteps, 2.25))
            fig.subplots_adjust(wspace=0.25)

            c_max = max([np.amax(y_target[:, i, :, :])])
            c_min = min([np.amin(y_target[:, i, :, :])])
            for t0 in range(nsteps):
                # Plot target
                ax[0, t0].imshow(y_target[t0 * stride, i, :, :], extent=[-2, 14, -4, 4], cmap=cmap0, origin='lower',
                                 vmax=c_max, vmin=c_min)
                # Plot sampled predictions
                pcm = ax[1, t0].imshow(y_pred[t0 * stride, i, :, :], extent=[-2, 14, -4, 4], cmap=cmap0, origin='lower',
                                 vmax=c_max, vmin=c_min)
                # fig.colorbar(pcm, ax=ax[1, t0], shrink=0.6)

                ax[0, t0].set_xticks(np.linspace(-2, 14, 9))
                ax[0, t0].set_yticks(np.linspace(-4, 4, 5))
                ax[1, t0].set_xticks(np.linspace(-2, 14, 9))
                ax[1, t0].set_yticks(np.linspace(-4, 4, 5))

                for tick in ax[0, t0].xaxis.get_major_ticks():
                    tick.label.set_fontsize(5)
                for tick in ax[0, t0].yaxis.get_major_ticks():
                    tick.label.set_fontsize(5)
                for tick in ax[1, t0].xaxis.get_major_ticks():
                    tick.label.set_fontsize(5)
                for tick in ax[1, t0].yaxis.get_major_ticks():
                    tick.label.set_fontsize(5)

                ax[0, t0].set_title('t={:d}'.format(t0 * stride), fontsize=8)

            self._createColorBarVertical(fig, ax, c_min, c_max, cmap=cmap0)
            ax[0, 0].set_ylabel('Target', fontsize=8)
            ax[1, 0].set_ylabel('Prediction', fontsize=8)

            if (not epoch is None):
                file_name = 'cylinder{:s}Pred{:d}_{:d}'.format(field, pid, epoch)
            else:
                file_name = 'cylinder{:s}Pred{:d}'.format(field, pid)
            self.saveFigure(plot_dir, file_name)

    def plotPredictionVorticity(self,
            y_pred: torch.Tensor, #[t, 3, nx, ny]
            y_target: torch.Tensor, #[t, 3, nx, ny]
            plot_dir: Optional[str] = None,
            epoch: Optional[int] = None,  # Epoch for file_name
            pid: Optional[int] = 0,  # Secondary plot ID
            nsteps: int = 5,
            stride: int = 1
        ):
        """Plots vorticity contours of flow around a cylinder at several time-steps

        Args:
            y_pred (torch.Tensor): [T, 3, H, W] Prediction tensor.
            y_target (torch.Tensor): [T, 3, H, W] Target tensor.
            plot_dir (Optional[str], optional): Directory to save figure, overrides class plot_dir if provided. Defaults to None.
            epoch (Optional[int], optional): Current epoch, used for file name. Defaults to None.
            pid (int, Optional): Optional plotting id for indexing file name manually, Defaults to 0.
            nsteps (int, Optional): Number of timesteps to plot, Defaults to 10.
            stride (int, optional): Number of timesteps in between plots. Defaults to 20.
        """
        @torch.no_grad()
        def xGrad(u, dx=1, padding=(1, 1, 1, 1)):
            WEIGHT_H = torch.FloatTensor([[[[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]]]]) / 8.
            ux = F.conv2d(F.pad(u, padding, mode='replicate'), WEIGHT_H.to(u.device),
                     stride=1, padding=0, bias=None) / (dx)
            return ux

        @torch.no_grad()
        def yGrad(u, dy=1, padding=(1, 1, 1, 1)):
            WEIGHT_V = torch.FloatTensor([[[[-1, 0, 1],
                                                [-2, 0, 2],
                                                [-1, 0, 1]]]]).transpose(-1, -2) / 8.
            uy = F.conv2d(F.pad(u, padding, mode='replicate'), WEIGHT_V.to(u.device),
                     stride=1, padding=0, bias=None) / (dy)
            return uy

        if plot_dir is None:
            plot_dir = self.plot_dir
        # Convert to numpy array
        dx = 6. / 64
        dy = 6. / 64

        vortPred = xGrad(y_pred[:,1].unsqueeze(1), dx=dx) - yGrad(y_pred[:,0].unsqueeze(1), dy=dy)
        vortTarget = xGrad(y_target[:,1].unsqueeze(1), dx=dx) - yGrad(y_target[:,0].unsqueeze(1), dy=dy)

        vortPred = vortPred.view(y_pred[:,0].shape).detach().cpu().numpy()
        vortTarget = vortTarget.view(y_target[:,0].shape).detach().cpu().numpy()

        plt.close('all')
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rcParams['figure.dpi'] = 200
        mpl.rcParams['xtick.labelsize'] = 2
        mpl.rcParams['ytick.labelsize'] = 2
        # rc('text', usetex=True)

        # Set up figure
        cmap0 = 'seismic'
        # fig, ax = plt.subplots(2+yPred0.size(0), yPred0.size(2), figsize=(2*yPred0.size(1), 3+3*yPred0.size(0)))
        fig, ax = plt.subplots(2, nsteps, figsize=(2*nsteps, 2.25))
        fig.subplots_adjust(wspace=0.25)

        c_max = max([np.amax(vortTarget[:, :, :])-4])
        c_min = min([np.amin(vortTarget[:, :, :])+4])
        c_max = 7
        c_min = -7
        print(vortPred.shape)
        for t0 in range(nsteps):
            # Plot target
            ax[0, t0].imshow(vortTarget[t0 * stride, :, :], extent=[-2, 14, -4, 4], cmap=cmap0, origin='lower',
                             vmax=c_max, vmin=c_min)
            # Plot sampled predictions
            pcm = ax[1, t0].imshow(vortPred[t0 * stride, :, :], extent=[-2, 14, -4, 4], cmap=cmap0, origin='lower',
                             vmax=c_max, vmin=c_min)
            # fig.colorbar(pcm, ax=ax[1, t0], shrink=0.6)
            ax[0, t0].set_xticks(np.linspace(-2, 14, 9))
            ax[0, t0].set_yticks(np.linspace(-4, 4, 5))
            ax[1, t0].set_xticks(np.linspace(-2, 14, 9))
            ax[1, t0].set_yticks(np.linspace(-4, 4, 5))

            for tick in ax[0, t0].xaxis.get_major_ticks():
                tick.label.set_fontsize(5)
            for tick in ax[0, t0].yaxis.get_major_ticks():
                tick.label.set_fontsize(5)
            for tick in ax[1, t0].xaxis.get_major_ticks():
                tick.label.set_fontsize(5)
            for tick in ax[1, t0].yaxis.get_major_ticks():
                tick.label.set_fontsize(5)

            ax[0, t0].set_title('t={:d}'.format(t0 * stride), fontsize=8)

        ax[0, 0].set_ylabel('Target', fontsize=8)
        ax[1, 0].set_ylabel('Prediction', fontsize=8)

        if not epoch is None:
            file_name = 'cylinderVortPred{:d}_{:d}'.format(pid, epoch)
        else:
            file_name = 'cylinderVortPred{:d}'.format(pid)

        self.saveFigure(plot_dir, file_name)