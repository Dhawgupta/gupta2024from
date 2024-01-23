import numpy as np
import torch
import torch.nn as nn
from src.utils.utils_nn import initialize_network

from src.optimizers.registry import get_optimizer
from src.agents.MultiBiSARSA import MultiBiSARSA

class MultiBiSARSA3(MultiBiSARSA):
    def __init__(self, params):
        super().__init__(params)


    def qf(self, x):
        # Forward value function
        # print(self.model(x).reshape((-1,2)) [:,0])
        
        return self.model(x).reshape((-1,2 * self.actions))[:, self.actions:].reshape(-1)

    def qr(self, x):
        # Reverse value function
        return self.qb(x) - self.qf(x)

    def qb(self, x):
        # Bi value function
        return self.model(x).reshape((-1, 2 * self.actions))[:, :self.actions].reshape(-1)

    def update(self, x, a, xp, ap, r, gamma, terminal, terminated):
        loss = 0
        # forward V loss
        loss_vf = self.computeQFLoss(x, a, xp, ap, r, gamma, terminal)
        # backward V loss
        loss_vr = torch.tensor(0.0).float()
        if self.update_head:
            loss_vr = self.computeQRLoss(x, a, gamma)
        # Bi V loss
        loss_vb = torch.tensor(0.0).float()
        if self.update_head:
            loss_vb = self.computeQBLoss(x, a, xp, ap, r, gamma, terminal)

        self.xh = x
        self.rh = r
        self.ah = a
        if terminal or terminated:
            # Also take care to update before termiantion of episode
            loss += self.computeQRLoss(xp, ap, gamma)
            self.resetEpisode()

        loss += loss_vf + loss_vr + loss_vb
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.use_wandb:
            import wandb
            wandb.log({'loss': loss.item(),
                        'loss_qf': loss_vf.item(),
                        'loss_qr': loss_vr.item(),
                        'loss_qb': loss_vb.item(),
                        })

        return loss.item()

   