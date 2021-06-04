'''
=====
- Associated publication:
url: 
doi: 
github: 
=====
'''
import torch
import torch.nn as nn
import logging
import numpy as np
from .embedding_model import EmbeddingModel
from torch.autograd import Variable

class LorenzEmbedding(EmbeddingModel):
    """Embedding Koopman model for the Lorenz ODE system

    Args:
        config (:class:`config.configuration_phys.PhysConfig`) Configuration class with transformer/embedding parameters
    """
    model_name = "embedding_lorenz"

    def __init__(self, config):
        """Constructor method
        """
        super().__init__(config)

        hidden_states = int(abs(config.state_dims[0] - config.n_embd)/2) + 1
        hidden_states = 500

        self.observableNet = nn.Sequential(
            nn.Linear(config.state_dims[0], hidden_states),
            nn.ReLU(),
            nn.Linear(hidden_states, config.n_embd),
            nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon),
            nn.Dropout(config.embd_pdrop)
        )

        self.recoveryNet = nn.Sequential(
            nn.Linear(config.n_embd, hidden_states),
            nn.ReLU(),
            nn.Linear(hidden_states, config.state_dims[0])
        )
        # Learned koopman operator
        # Learns skew-symmetric matrix with a diagonal
        self.obsdim = config.n_embd
        self.kMatrixDiag = nn.Parameter(torch.linspace(1, 0, config.n_embd))

        xidx = []
        yidx = []
        for i in range(1, 3):
            yidx.append(np.arange(i, config.n_embd))
            xidx.append(np.arange(0, config.n_embd-i))

        self.xidx = torch.LongTensor(np.concatenate(xidx))
        self.yidx = torch.LongTensor(np.concatenate(yidx))
        self.kMatrixUT = nn.Parameter(0.1*torch.rand(self.xidx.size(0)))
        # Normalization occurs inside the model
        self.register_buffer('mu', torch.tensor([0., 0., 0.]))
        self.register_buffer('std', torch.tensor([1., 1., 1.]))
        print('Number of embedding parameters: {}'.format( super().num_parameters ))

    def forward(self, x):
        """Forward pass

        Args:
            x (torch.Tensor): [B, 3] Input feature tensor

        Returns:
            (tuple): tuple containing:

                | (torch.Tensor): [B, config.n_embd] Koopman observables
                | (torch.Tensor): [B, 3] Recovered feature tensor
        """
        # Encode
        x = self._normalize(x)
        g = self.observableNet(x)
        # Decode
        out = self.recoveryNet(g)
        xhat = self._unnormalize(out)
        return g, xhat

    def embed(self, x):
        """Embeds tensor of state variables to Koopman observables

        Args:
            x (torch.Tensor): [B, 3] input feature tensor

        Returns:
            (torch.Tensor): [B, config.n_embd] Koopman observables
        """
        x = self._normalize(x)
        g = self.observableNet(x)
        return g

    def recover(self, g):
        """Recovers feature tensor from Koopman observables

        Args:
            g (torch.Tensor): [B, config.n_embd] Koopman observables

        Returns:
            (torch.Tensor): [B, 3] Physical feature tensor
        """
        out = self.recoveryNet(g)
        x = self._unnormalize(out)
        return x

    def koopmanOperation(self, g):
        """Applies the learned koopman operator on the given observables.

        Args:
            (torch.Tensor): [B, config.n_embd] Koopman observables

        Returns:
            (torch.Tensor): [B, config.n_embd] Koopman observables at the next time-step
        """
        # Koopman operator
        kMatrix = Variable(torch.zeros(self.obsdim, self.obsdim)).to(self.kMatrixUT.device)
        # Populate the off diagonal terms
        kMatrix[self.xidx, self.yidx] = self.kMatrixUT
        kMatrix[self.yidx, self.xidx] = -self.kMatrixUT

        # Populate the diagonal
        ind = np.diag_indices(kMatrix.shape[0])
        kMatrix[ind[0], ind[1]] = self.kMatrixDiag

        # Apply Koopman operation
        gnext = torch.bmm(kMatrix.expand(g.size(0), kMatrix.size(0), kMatrix.size(0)), g.unsqueeze(-1))
        self.kMatrix = kMatrix
        return gnext.squeeze(-1) # Squeeze empty dim from bmm

    @property
    def koopmanOperator(self, requires_grad=True):
        """Current Koopman operator

        Args:
            requires_grad (bool, optional): if to return with gradient storage, defaults to True
        """
        if not requires_grad:
            return self.kMatrix.detach()
        else:
            return self.kMatrix

    def _normalize(self, x):
        return (x - self.mu.unsqueeze(0))/self.std.unsqueeze(0)

    def _unnormalize(self, x):
        return self.std.unsqueeze(0)*x + self.mu.unsqueeze(0)

    @property
    def koopmanDiag(self):
        return self.kMatrixDiag

class LorenzEmbeddingTrainer(nn.Module):
    """Training head for the Lorenz embedding model for parallel training

    Args:
        config (:class:`config.configuration_phys.PhysConfig`) Configuration class with transformer/embedding parameters
    """
    def __init__(self, config):
        """Constructor method
        """
        super().__init__()
        self.embedding_model = LorenzEmbedding(config)

    def forward(self, input_states=None):
        '''
        Trains model for a single epoch
        '''
        self.embedding_model.train()
        device = self.embedding_model.devices[0]

        loss_reconstruct = 0
        mseLoss = nn.MSELoss()

        xin0 = input_states[:,0].to(device) # Time-step

        # Model forward for both time-steps
        g0, xRec0 = self.embedding_model(xin0)
        loss = (1e4)*mseLoss(xin0, xRec0)
        loss_reconstruct = loss_reconstruct + mseLoss(xin0, xRec0).detach()

        g1_old = g0
        # Koopman transform
        for t0 in range(1, input_states.shape[1]):
            xin0 = input_states[:,t0,:].to(device) # Next time-step
            _, xRec1 = self.embedding_model(xin0)

            g1Pred = self.embedding_model.koopmanOperation(g1_old)
            xgRec1 = self.embedding_model.recover(g1Pred)

            loss = loss + mseLoss(xgRec1, xin0) + (1e4)*mseLoss(xRec1, xin0) \
                + (1e-1)*torch.sum(torch.pow(self.embedding_model.koopmanOperator, 2))

            loss_reconstruct = loss_reconstruct + mseLoss(xRec1, xin0).detach()
            g1_old = g1Pred

        return loss, loss_reconstruct

    def save_model(self, *args, **kwargs):
        """
        Saves the embedding model
        """
        self.embedding_model.save_model(*args, **kwargs)


    def load_model(self, *args, **kwargs):
        """
        Load the embedding model
        """
        self.embedding_model.load_model(*args, **kwargs)
