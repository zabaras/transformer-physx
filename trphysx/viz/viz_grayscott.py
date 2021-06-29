"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import torch
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from .viz_model import Viz

Tensor = torch.Tensor

class GrayScottViz(Viz):
    """Visualization class for the 3D Gray-scott system.

    Args:
        plot_dir (str, optional): Directory to save visualizations in. Defaults to None.
    """
    def __init__(self, plot_dir: str = None) -> None:
        """Constructor method
        """
        super().__init__(plot_dir=plot_dir)

    def _createColorBarVertical(self, fig, ax0, c_min, c_max, label_format="{:02.2f}", cmap='viridis'):
        """Util method for plotting a colorbar next to a subplot
        """
        p0 = ax0[0].get_position().get_points().flatten()
        p1 = ax0[-1].get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p1[2] + 0.01, p1[1], 0.0075, p0[3] - p1[1]])

        ticks = np.linspace(0, 1, 5)
        tickLabels = np.linspace(c_min, c_max, 5)
        tickLabels = [label_format.format(t0) for t0 in tickLabels]
        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
        cbar.set_ticklabels(tickLabels)

    def plotPrediction(self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
        plot_dir: str = None,
        epoch: int = None,
        pid: int = 0,
        nsteps: int = 10,
        stride: int = 5
    ) -> None:
        """Plots z-slice of Gray-Scott prediction along the z-axis and saves to file

        Args:
            y_pred (torch.Tensor): [T, 2, H, W, D] prediction time-series of states
            y_target (torch.Tensor): [T, 2, H, W, D] target time-series of states
            plot_dir (str, optional): Directory to save figure, overrides class plot_dir if provided. Defaults to None.
            epoch (int, optional): Current epoch, used for file name. Defaults to None.
            pid (int, optional): Optional plotting id for indexing file name manually. Defaults to 0.
            nsteps (int, optional): Number of timesteps to plot. Defaults to 10.
            stride (int, optional): Number of timesteps in between plots. Defaults to 5.
        """
        if plot_dir is None:
            plot_dir = self.plot_dir
        # Convert to numpy array
        y_pred = y_pred.detach().cpu().numpy()
        y_target = y_target.detach().cpu().numpy()

        plt.close('all')
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rcParams['figure.dpi'] = 200

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