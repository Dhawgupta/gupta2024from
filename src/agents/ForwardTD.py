import numpy as np
import torch
import torch.nn as nn
from src.utils.utils_nn import initialize_network

from src.optimizers.registry import get_optimizer


class ForwardTD:
    def __init__(self,params):
        self.features = params['features']
        self.actions = params['actions']
        self.hidden_units = params['hidden_units']
        self.optimizer_type = params['optimizer']
        self.use_wandb = params.get('use_wandb', False)

        # create a 1 layer neural network
        self.model = nn.Sequential(nn.Linear(self.features, self.hidden_units), nn.ReLU(), nn.Linear(self.hidden_units, 1)).float()
        
        self.optimizer = get_optimizer(self.optimizer_type)(self.model.parameters(), lr = params['alpha'])

        
    def resetAgent(self):
        initialize_network(self.model)
    
    def computeLoss(self, x, a, xp, r, gamma, terminal):
        with torch.no_grad():
            target = r + (1 - int(terminal)) * self.model(xp) * gamma
        v = self.model(x)
        loss = 0.5 * nn.MSELoss()(v, target)
        return loss

    
  

    def update(self, x, a, xp, r, gamma, terminal):
        loss = self.computeLoss(x, a, xp, r, gamma, terminal)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.use_wandb:
            import wandb
            wandb.log({'loss_vf': loss.item()})
            wandb.log({'loss': loss.item()})

        return loss.item()

    def getW(self):
        raise NotImplementedError
        
    
    def getValues(self, states_features):
        '''
        Returns the value of the states
        states : np.array of shape (n, features)
        '''
        return self.model(states_features).detach().numpy()


    
