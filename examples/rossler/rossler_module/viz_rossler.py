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
from typing import Optional

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
from trphysx.viz import Viz

# Interface to LineCollection:
def _colorline3d(x, y, z, t=None, cmap=plt.get_cmap('viridis'), linewidth=1, alpha=1.0, ax=None):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    https://stackoverflow.com/questions/52884221/how-to-plot-a-matplotlib-line-plot-using-colormap
    '''
    # Default colors equally spaced on [0,1]:
    if t is None:
        t = np.linspace(0.25, 1.0, len(x))
    if ax is None:
        ax = plt.gca()

    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    colors = np.array([cmap(i) for i in t])
    lc = Line3DCollection(segments, colors=colors, linewidth=linewidth,  alpha=alpha)
    ax.add_collection(lc)
    ax.scatter(x, y, z, c=colors, marker='*', alpha=alpha) #Adding line markers

class HandlerColormap(HandlerBase):
    def __init__(self, cmap, num_stripes=8, **kw):
        HandlerBase.__init__(self, **kw)
        self.cmap = cmap
        self.num_stripes = num_stripes
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        stripes = []
        for i in range(self.num_stripes):
            s = Rectangle([xdescent + i * width / self.num_stripes, ydescent],
                          width / self.num_stripes,
                          height,
                          fc=self.cmap((2 * i + 1) / (2 * self.num_stripes)),
                          transform=trans)
            stripes.append(s)
        return stripes

class RosslerViz(Viz):
    """Visualization class for Lorenz ODE

    Args:
        plot_dir (Optional[str], optional): Directory to save visualizations in. Defaults to None.
    """
    def __init__(self, plot_dir:Optional[str] = None):
        super().__init__(plot_dir=plot_dir)

    def plotPrediction(self,
            y_pred:torch.Tensor,
            y_target:torch.Tensor,
            plot_dir:Optional[str] = None,
            epoch:Optional[int] = None,
            pid:Optional[int] = 0,
        ):
        """Plots a 3D line of a single Lorenz prediction

        Args:
            y_pred (torch.Tensor): [T, 3] Prediction tensor.
            y_target (torch.Tensor): [T, 3] Target tensor.
            plot_dir (Optional[str], optional): Directory to save figure, overrides plot_dir one if provided. Defaults to None.
            epoch (Optional[int], optional): Current epoch, used for file name. Defaults to None.
            pid (int, Optional): Optional plotting id for indexing file name manually, Defaults to 0.
        """
        # Convert to numpy array
        y_pred = y_pred.detach().cpu().numpy()
        y_target = y_target.detach().cpu().numpy()

        plt.close('all')
        mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
        mpl.rcParams['figure.dpi'] = 300
        rc('text', usetex=True)
        # Set up figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        cmaps = [plt.get_cmap("Reds"), plt.get_cmap("Blues")]
        _colorline3d(y_pred[:,0], y_pred[:,1], y_pred[:,2], cmap=cmaps[0], ax=ax)
        _colorline3d(y_target[:,0], y_target[:,1], y_target[:,2], cmap=cmaps[1], ax=ax)

        ax.set_xlim([-20,20])
        ax.set_ylim([-20,20])
        ax.set_zlim([10,50])

        cmap_handles = [Rectangle((0, 0), 1, 1) for _ in cmaps]
        handler_map = dict(zip(cmap_handles,
                            [HandlerColormap(cm, num_stripes=8) for cm in cmaps]))
        # Create custom legend with color map rectangels
        ax.legend(handles=cmap_handles, labels=['Prediction','Target'], handler_map=handler_map, loc='upper right', framealpha=0.95)

        if(not epoch is None):
            file_name = 'rosslerPred{:d}_{:d}'.format(pid, epoch)
        else:
            file_name = 'rosslerPred{:d}'.format(pid)

        self.saveFigure(plot_dir, file_name)

    def plotMultiPrediction(self,
            y_pred: torch.Tensor, # [mb, t, 3]
            y_target: torch.Tensor, # [mb, t, 3]
            plot_dir: Optional[str] = None,
            epoch: Optional[int] = None,  # Epoch for file_name
            pid: Optional[int] = 0,  # Secondary plot ID
            nplots:int = 2,
        ):
        """Plots the 3D lines of multiple Lorenz predictions

        Args:
            y_pred (torch.Tensor): [T, 3] Prediction tensor.
            y_target (torch.Tensor): [T, 3] Target tensor.
            plot_dir (Optional[str], optional): Directory to save figure, overrides plot_dir one if provided. Defaults to None.
            epoch (Optional[int], optional): Current epoch, used for file name. Defaults to None.
            pid (int, Optional): Optional plotting id for indexing file name manually, Defaults to 0.
            nplots (int, Optional): Number of cases to plot, Defaults to 2.
        """
        assert y_pred.size(0) >= nplots, 'Number of provided predictions is less than the requested number of subplots'
        assert y_target.size(0) >= nplots, 'Number of provided targets is less than the requested number of subplots'
        # Convert to numpy array
        y_pred = y_pred.detach().cpu().numpy()
        y_target = y_target.detach().cpu().numpy()

        plt.close('all')
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rcParams['figure.dpi'] = 300
        rc('text', usetex=True)
        # Set up figure
        fig, ax = plt.subplots(1, nplots, figsize=(6*nplots, 6), subplot_kw={'projection': '3d'})
        plt.subplots_adjust(wspace=0.025)

        cmaps = [plt.get_cmap("Reds"), plt.get_cmap("Blues")]
        for i in range(nplots):
            _colorline3d(y_pred[i, :, 0], y_pred[i, :, 1], y_pred[i, :, 2], cmap=cmaps[0], ax=ax[i], alpha=0.6)
            _colorline3d(y_target[i, :, 0], y_target[i, :, 1], y_target[i, :, 2], cmap=cmaps[1], ax=ax[i], alpha=0.6)

            ax[i].set_xlim([-20, 20])
            ax[i].set_ylim([-20, 20])
            ax[i].set_zlim([10, 50])

            ax[i].set_xlabel('x', fontsize=14)
            ax[i].set_ylabel('y', fontsize=14)
            ax[i].set_zlabel('z', fontsize=14)
        cmap_handles = [Rectangle((0, 0), 1, 1) for _ in cmaps]
        handler_map = dict(zip(cmap_handles,
                               [HandlerColormap(cm, num_stripes=10) for cm in cmaps]))
        # Create custom legend with color map rectangels
        ax[-1].legend(handles=cmap_handles, labels=['Prediction', 'Target'], handler_map=handler_map, loc='upper right',
                  framealpha=0.95)

        if epoch is not None:
            file_name = 'rosslerMultiPred{:d}_{:d}'.format(pid, epoch)
        else:
            file_name = 'rosslerMultiPred{:d}'.format(pid)

        self.saveFigure(plot_dir, file_name)

    def plotPredictionScatter(self,
            y_pred:torch.Tensor,
            plot_dir:Optional[str] = None,
            epoch:Optional[int] = None,
            pid:Optional[int] = 0
        ):
        """Plots scatter plots of a Lorenz prediction contoured based on distance from the basins

        Args:
            y_pred (torch.Tensor): [T, 3] Prediction tensor.
            plot_dir (Optional[str], optional): Directory to save figure, overrides plot_dir one if provided. Defaults to None.
            epoch (Optional[int], optional): Current epoch, used for file name. Defaults to None.
            pid (int, Optional): Optional plotting id for indexing file name manually, Defaults to 0.
        """
        # Convert to numpy array
        y_pred = y_pred.detach().cpu().numpy()

        plt.close('all')
        mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
        mpl.rcParams['figure.dpi'] = 300
        rc('text', usetex=True)
        # Set up figure
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        cmap = plt.get_cmap("plasma")
        # Lorenz attraction centers
        s=10
        r=28
        b=2.667
        cp0 = np.array([np.sqrt(b*(r-1)),np.sqrt(b*(r-1)),r-1])
        cp1 = np.array([-np.sqrt(b*(r-1)),-np.sqrt(b*(r-1)),r-1])

        c = np.minimum(np.sqrt((y_pred[:,0]-cp0[0])**2 + (y_pred[:,1]-cp0[1])**2 + (y_pred[:,2]-cp0[2])**2),
        np.sqrt((y_pred[:,0]-cp1[0])**2 + (y_pred[:,1]-cp1[1])**2 + (y_pred[:,2]-cp1[2])**2))
        c = np.maximum(0, 1 - c/25)

        ax.set_xlim([-20,20])
        ax.set_ylim([-20,20])
        ax.set_zlim([10,50])

        ax.scatter(y_pred[:,0], y_pred[:,1], y_pred[:,2], c=c)

        if(not epoch is None):
            file_name = 'rosslerScatter{:d}_{:d}'.format(pid, epoch)
        else:
            file_name = 'rosslerScatter{:d}'.format(pid)

        self.saveFigure(plot_dir, file_name)