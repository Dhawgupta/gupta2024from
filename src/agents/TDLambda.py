import numpy as np
import torch
import torch.nn as nn
from src.utils.utils_nn import initialize_network

from src.optimizers.registry import get_optimizer


class TDLambda:
    def __init__(self, params):
        self.features = params['features']
        self.actions = params['actions']
        self.hidden_units = params['hidden_units']
        self.optimizer_type = params['optimizer']
        self.lambda_ = params['lambda']
        self.use_wandb = params['use_wandb']
        if self.hidden_units == 0:
            # make a linear model
            self.model = nn.Sequential(nn.Linear(self.features, 1)).float()
        else:
            # create a 1 layer neural network
            self.model = nn.Sequential(nn.Linear(self.features, self.hidden_units), nn.ReLU(), nn.Linear(self.hidden_units, 1)).float()
        
        self.optimizer = get_optimizer(self.optimizer_type)(self.model.parameters(), lr = params['alpha'])
        
        self.resetTrace()


        
    
    def resetAgent(self):
        initialize_network(self.model)
        self.resetTrace()

    
    def resetTrace(self):
        self.z = [torch.zeros_like(m) for m in self.model.parameters()]
    
    def computeDelta(self, x, a, xp, r, gamma, terminal):
        with torch.no_grad():
            target = r + (1 - int(terminal)) * self.model(xp) * gamma
            v = self.model(x)
            delta = target - v
        return delta
  
    def computeLoss(self, x, a, xp, r, gamma, terminal):
        with torch.no_grad():
            target = r + (1 - int(terminal)) * self.model(xp) * gamma
        v = self.model(x)
        loss = nn.MSELoss()(v, target)
        return loss

    def resetEpisode(self):
        self.resetTrace()
        
    def update(self, x, a, xp, r, gamma, terminal):

        v = self.model(x)
        vgrad = torch.autograd.grad(v, self.model.parameters(), create_graph=True)
        delta = self.computeDelta(x, a, xp, r, gamma, terminal)
        for i in range(len(vgrad)):
            self.z[i] = (gamma * self.lambda_ * self.z[i]) + vgrad[i]

        z1 = self.z
        if terminal:
            self.resetEpisode()
            # self.resetTrace()


        self.optimizer.zero_grad()

        #update the gradients fo the model
        for p, z_ in zip(self.model.parameters(), z1):
            p.grad = - delta * z_
        self.optimizer.step()
       
        if self.use_wandb:
            import wandb
            wandb.log({'loss_vf': delta.item()**2})
            wandb.log({'loss': delta.item()**2})


        return delta.item()**2


    def getW(self):
        return self.theta

    def getTrace(self):
        return self.z
    
    def getValues(self, states_features):
        '''
        Returns the value of the states
        states : np.array of shape (n, features)
        '''
        return self.model(states_features).detach().numpy()


if __name__ == '__main__':
    agent = TDLambda(3, 2, {'alpha': 0.1, 'lambda': 0.5})
    print("Done")