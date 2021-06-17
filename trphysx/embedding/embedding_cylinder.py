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
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from .embedding_model import EmbeddingModel, EmbeddingTrainingHead
from trphysx.config.configuration_phys import PhysConfig
from torch.autograd import Variable

logger = logging.getLogger(__name__)
# Custom types
Tensor = torch.Tensor
TensorTuple = Tuple[torch.Tensor]
FloatTuple = Tuple[float]

class CylinderEmbedding(EmbeddingModel):
    """Embedding Koopman model for the 2D flow around a cylinder system

    Args:
        config (PhysConfig): Configuration class with transformer/embedding parameters
    """
    model_name = "embedding_cylinder"

    def __init__(self, config: PhysConfig) -> None:
        """Constructor method
        """
        super().__init__(config)

        X, Y = np.meshgrid(np.linspace(-2, 14, 128), np.linspace(-4, 4, 64))
        self.mask = torch.tensor(np.sqrt(X**2 + Y**2) < 1, dtype=torch.bool)

        # Encoder conv. net
        self.observableNet = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3, 3), stride=2, padding=1, padding_mode='replicate'),
            # nn.BatchNorm2d(16),
            nn.ReLU(True),
            # 8, 32, 64
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1, padding_mode='replicate'),
            # nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 16, 16, 32
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1, padding_mode='replicate'),
            # nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 16, 8, 16
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1, padding_mode='replicate'),
            # nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 16, 4, 8
            nn.Conv2d(128, config.n_embd // 32, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),
        )

        self.observableNetFC = nn.Sequential(
            # nn.Linear(config.n_embd // 32 * 4 * 8, config.n_embd-1),
            nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon),
            # nn.BatchNorm1d(config.n_embd, eps=config.layer_norm_epsilon),
            nn.Dropout(config.embd_pdrop)
        )

        # Decoder conv. net
        self.recoveryNet = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(config.n_embd // 32, 128, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            # 16, 8, 16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            # 16, 16, 32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            # 8, 32, 64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            # 16, 64, 128
            nn.Conv2d(16, 3, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),
        )
        # Learned Koopman operator parameters
        self.obsdim = config.n_embd
        # We parameterize the Koopman operator as a function of the viscosity
        self.kMatrixDiagNet = nn.Sequential(nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, self.obsdim))

        # Off-diagonal indices
        xidx = []
        yidx = []
        for i in range(1, 3):
            yidx.append(np.arange(i, self.obsdim))
            xidx.append(np.arange(0, self.obsdim-i))
        self.xidx = torch.LongTensor(np.concatenate(xidx))
        self.yidx = torch.LongTensor(np.concatenate(yidx))

        # The matrix here is a small NN since we need to make it dependent on the viscosity
        self.kMatrixUT = nn.Sequential(nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, self.xidx.size(0)))
        # Normalization occurs inside the model
        self.register_buffer('mu', torch.tensor([0., 0., 0., 0.]))
        self.register_buffer('std', torch.tensor([1., 1., 1., 1.]))
        logger.info('Number of embedding parameters: {}'.format( super().num_parameters ))

    def forward(self, x: Tensor, visc: Tensor) -> TensorTuple:
        """Forward pass

        Args:
            x (Tensor): [B, 3, H, W] Input feature tensor
            visc (Tensor): [B] Viscosities of the fluid in the mini-batch

        Returns:
            (TensorTuple): Tuple containing:

                | (Tensor): [B, config.n_embd] Koopman observables
                | (Tensor): [B, 3, H, W] Recovered feature tensor
        """
        # Concat viscosities as a feature map
        x = torch.cat([x, visc.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(x[:,:1])], dim=1)
        x = self._normalize(x)
        g0 = self.observableNet(x)
        g = self.observableNetFC(g0.view(g0.size(0),-1))
        # Decode
        out = self.recoveryNet(g.view(g0.shape))
        xhat = self._unnormalize(out)
        # Apply cylinder mask
        mask0 = self.mask.repeat(xhat.size(0), xhat.size(1), 1, 1) is True
        xhat[mask0] = 0

        return g, xhat

    def embed(self, x: Tensor, visc: Tensor) -> Tensor:
        """Embeds tensor of state variables to Koopman observables

        Args:
            x (Tensor): [B, 3, H, W] Input feature tensor
            visc (Tensor): [B] Viscosities of the fluid in the mini-batch

        Returns:
            (Tensor): [B, config.n_embd] Koopman observables
        """
        # Concat viscosities as a feature map
        x = torch.cat([x, visc.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(x[:,:1])], dim=1)
        x = self._normalize(x)

        g = self.observableNet(x)
        g = self.observableNetFC(g.view(x.size(0), -1))
        return g

    def recover(self, g: Tensor) -> Tensor:
        """Recovers feature tensor from Koopman observables

        Args:
            g (Tensor): [B, config.n_embd] Koopman observables

        Returns:
            (Tensor): [B, 3, H, W] Physical feature tensor
        """
        x = self.recoveryNet(g.view(-1, self.obsdim//32, 4, 8))
        x = self._unnormalize(x)
        # Apply cylinder mask
        mask0 = self.mask.repeat(x.size(0), x.size(1), 1, 1) == True
        x[mask0] = 0
        return x

    def koopmanOperation(self, g: Tensor, visc: Tensor) -> Tensor:
        """Applies the learned Koopman operator on the given observables

        Args:
            g (Tensor): [B, config.n_embd] Koopman observables
            visc (Tensor): [B] Viscosities of the fluid in the mini-batch

        Returns:
            (Tensor): [B, config.n_embd] Koopman observables at the next time-step
        """
        # Koopman operator
        kMatrix = Variable(torch.zeros(g.size(0), self.obsdim, self.obsdim)).to(self.devices[0])
        # Populate the off diagonal terms
        kMatrix[:,self.xidx, self.yidx] = self.kMatrixUT(100*visc)
        kMatrix[:,self.yidx, self.xidx] = -self.kMatrixUT(100*visc)

        # Populate the diagonal
        ind = np.diag_indices(kMatrix.shape[1])
        self.kMatrixDiag = self.kMatrixDiagNet(100*visc)
        kMatrix[:, ind[0], ind[1]] = self.kMatrixDiag

        # Apply Koopman operation
        gnext = torch.bmm(kMatrix, g.unsqueeze(-1))
        self.kMatrix = kMatrix
        return gnext.squeeze(-1) # Squeeze empty dim from bmm

    @property
    def koopmanOperator(self, requires_grad: bool =True) -> Tensor:
        """Current Koopman operator

        Args:
            requires_grad (bool, optional): If to return with gradient storage. Defaults to True

        Returns:
            (Tensor): Full Koopman operator tensor
        """
        if not requires_grad:
            return self.kMatrix.detach()
        else:
            return self.kMatrix

    def _normalize(self, x: Tensor) -> Tensor:
        x = (x - self.mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / self.std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return x

    def _unnormalize(self, x: Tensor) -> Tensor:
        return self.std[:3].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)*x + self.mu[:3].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    @property
    def koopmanDiag(self):
        return self.kMatrixDiag

class CylinderEmbeddingTrainer(EmbeddingTrainingHead):
    """Training head for the Lorenz embedding model

    Args:
        config (PhysConfig): Configuration class with transformer/embedding parameters
    """
    def __init__(self, config: PhysConfig) -> None:
        """Constructor method
        """
        super().__init__()
        self.embedding_model = CylinderEmbedding(config)

    def forward(self, states: Tensor, viscosity: Tensor) -> FloatTuple:
        """Trains model for a single epoch

        Args:
            states (Tensor): [B, T, 3, H, W] Time-series feature tensor
            viscosity (Tensor): [B] Viscosities of the fluid in the mini-batch

        Returns:
            FloatTuple: Tuple containing:
            
                | (float): Koopman based loss of current epoch
                | (float): Reconstruction loss
        """
        assert states.size(0) == viscosity.size(0), 'State variable and viscosity tensor should have the same batch dimensions.'

        self.embedding_model.train()
        device = self.embedding_model.devices[0]

        loss_reconstruct = 0
        mseLoss = nn.MSELoss()

        xin0 = states[:,0].to(device) # Time-step
        viscosity = viscosity.to(device)

        # Model forward for initial time-step
        g0, xRec0 = self.embedding_model(xin0, viscosity)
        loss = (1e3)*mseLoss(xin0, xRec0)
        loss_reconstruct = loss_reconstruct + mseLoss(xin0, xRec0).detach()

        g1_old = g0
        # Loop through time-series
        for t0 in range(1, states.shape[1]):
            xin0 = states[:,t0,:].to(device) # Next time-step
            _, xRec1 = self.embedding_model(xin0, viscosity)
            # Apply Koopman transform
            g1Pred = self.embedding_model.koopmanOperation(g1_old, viscosity)
            xgRec1 = self.embedding_model.recover(g1Pred)

            # Loss function
            loss = loss + mseLoss(xgRec1, xin0) + (1e3)*mseLoss(xRec1, xin0) \
                + (1e-1)*torch.sum(torch.pow(self.embedding_model.koopmanOperator, 2))
                
            loss_reconstruct = loss_reconstruct + mseLoss(xRec1, xin0).detach()
            g1_old = g1Pred

        return loss, loss_reconstruct