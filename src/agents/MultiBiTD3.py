import numpy as np
import torch
import torch.nn as nn
from src.utils.utils_nn import initialize_network

from src.optimizers.registry import get_optimizer
from src.agents.MultiBiTD import MultiBiTD
# This version parametereizes the value function as the different of \BV and \RV
class MultiBiTD3(MultiBiTD):
    def __init__(self, params):
        super().__init__(params)

    # Just redefine the value functions
    def vf(self, x):
        # Forward value function
        # print(self.model(x).reshape((-1,2)) [:,0])
        # return self.model(x).reshape((-1,2))[:, 0]
        return self.model(x).reshape((-1,2))[:, 1]
        

    def vr(self, x):
        # Reverse value function
        # return self.model(x).reshape((-1,2))[:, 1]
        return self.vb(x) - self.vf(x)

    def vb(self, x):
        # Bi value function
        # return self.vf(x) + self.vr(x)
        return self.model(x).reshape((-1,2))[:, 0]

    def update(self, x, a, xp, r, gamma, terminal):
        loss = 0
        # forward V loss
        loss_vf = self.computeVFLoss(x, a, xp, r, gamma, terminal)
        # backward V loss
        loss_vr = torch.tensor(0.0).float()
        if self.update_head:
            loss_vr = self.computeVRLoss(x, gamma)
        # Bi V loss
        loss_vb = self.computeVBLoss(x, a, xp, r, gamma, terminal)

        self.xh = x
        self.rh = r
        if terminal:
            # Also take care to update before termiantion of episode
            loss += self.computeVRLoss(xp, gamma)
            self.resetEpisode()
        loss += loss_vf + loss_vr + loss_vb
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.use_wandb:
            import wandb
            wandb.log({'loss': loss.item(),
                        'loss_vf': loss_vf.item(),
                        'loss_vr': loss_vr.item(),
                        'loss_vb': loss_vb.item(),
                        })

        return loss.item()
