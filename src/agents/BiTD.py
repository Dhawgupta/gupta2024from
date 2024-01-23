import numpy as np
import torch
import torch.nn as nn
from src.utils.utils_nn import initialize_network

from src.optimizers.registry import get_optimizer

class BiTD:
    def __init__(self, params):
        self.features = params['features']
        self.actions = params['actions']
        self.hidden_units = params['hidden_units']
        self.optimizer_type = params['optimizer']
        self.lambda_ = params['lambda']
        self.use_wandb = params['use_wandb']
        if self.hidden_units == 0:
            # make a linear model
            self.model = nn.Sequential(nn.Linear(self.features, 2)).float()
        else: # make a 1 layer neural network
            # create a 1 layer neural network
            self.model = nn.Sequential(nn.Linear(self.features, self.hidden_units), nn.ReLU(), nn.Linear(self.hidden_units, 2)).float()
            
        self.optimizer = get_optimizer(self.optimizer_type)(self.model.parameters(), lr = params['alpha'])
            
        self.xh = None
        self.rh = None

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=params['alpha'])

    def resetAgent(self):
        initialize_network(self.model)
    
    def computeVFLoss(self, x, a, xp, r, gamma, terminal):
        with torch.no_grad():
            target = r + (1 - int(terminal)) * self.vf(xp) * gamma
        # return target
        v = self.vf(x)
        loss = 0.5 * nn.MSELoss()(v, target)
        return loss

    def computeVRLoss(self, x, gamma):
        with torch.no_grad():
            target = torch.tensor([0.0]).float()
            if self.xh is not None:
                target = self.lambda_ * gamma * (self.rh +  self.vr(self.xh))
        vr = self.vr(x)
        loss = 0.5 * nn.MSELoss()(vr, target)
        return loss

    def computeVBLoss(self, x, a, xp, r, gamma, terminal):
        with torch.no_grad():
            target = torch.tensor([0.0]).float()

            target += (r * (1 - (gamma * gamma * self.lambda_)))
            if not terminal:
                target += gamma * self.vb(xp)
            if self.xh is not None:
                target += gamma * self.lambda_ * self.vb(self.xh)

            target /= (1 + (gamma * gamma * self.lambda_))

        vb = self.vb(x)
        loss = 0.5 * nn.MSELoss()(vb, target)
        return loss
        # return target


    def resetEpisode(self):
        self.xh = None
        self.rh = None

    def vf(self, x):
        # Forward value function
        # print(self.model(x).reshape((-1,2)) [:,0])
        return self.model(x).reshape((-1,2))[:, 0]

    def vr(self, x):
        # Reverse value function
        return self.model(x).reshape((-1,2))[:, 1]

    def vb(self, x):
        # Bi value function
        return self.vf(x) + self.vr(x)

    def update(self, x, a, xp, r, gamma, terminal):
        loss = 0
        # forward V loss
        loss_vf = self.computeVFLoss(x, a, xp, r, gamma, terminal)
        # backward V loss
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

    def getW(self):
        raise NotImplementedError
        
    
    def getValues(self, states_features):
        '''
        Returns the value of the states
        states : np.array of shape (n, features)
        '''
        #FIXME
        # Only use the first head for value function
        # return self.model(states_features).detach().numpy()[:, 0]
        return self.vf(states_features).detach().numpy()
        # return self.model(states_features).detach().numpy()


    
