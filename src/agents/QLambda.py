import numpy as np
import torch
import torch.nn as nn
from src.utils.utils_nn import initialize_network

from src.optimizers.registry import get_optimizer


class QLambda:
    def __init__(self, params):
        self.features = params['features']
        self.actions = params['actions']
        self.hidden_units = params['hidden_units']
        self.optimizer_type = params['optimizer']
        self.lambda_ = params['lambda']
        self.use_wandb = params['use_wandb']
        self.epsilon = params['epsilon']
        if self.hidden_units == 0:
            # make a linear model
            self.model = nn.Sequential(nn.Linear(self.features, self.actions)).float()
        else:
            # create a 1 layer neural network
            self.model = nn.Sequential(nn.Linear(self.features, self.hidden_units), nn.ReLU(), nn.Linear(self.hidden_units, self.actions)).float()
        
        self.optimizer = get_optimizer(self.optimizer_type)(self.model.parameters(), lr = params['alpha'])
        
        self.resetTrace()

    def qf(self, x):
        return self.model(x)

    def selectAction(self, x):
        p = np.random.rand()
        if p < self.epsilon:
            # Choose a random action
            return np.random.choice(self.actions)

        return np.argmax(self.qf(x).detach().numpy())
        
    
    def resetAgent(self):
        initialize_network(self.model)
        self.resetTrace()

    
    def resetTrace(self):
        self.z = [torch.zeros_like(m) for m in self.model.parameters()]
    
    def computeDelta(self, x, a, xp, ap, r, gamma, terminal):
        with torch.no_grad():
            # ap = self.selectAction(xp)
            qsap = self.qf(xp)[ap]
            target = r + (1 - int(terminal)) * qsap * gamma
            # v = self.model(x)
            qsa = self.qf(x)[a]
            delta = target - qsa
        return delta

    # Not using, only usable with one step methods
    # FIXME  : Remove the below section
    def computeLoss(self, x, a, xp, r, gamma, terminal):
        with torch.no_grad():
            target = r + (1 - int(terminal)) * self.model(xp) * gamma
        v = self.model(x)
        loss = nn.MSELoss()(v, target)
        return loss

    def resetEpisode(self):
        self.resetTrace()
        
    def update(self, x, a, xp, ap, r, gamma, terminal, terminated):
        with torch.no_grad():
            a_max = torch.argmax(self.qf(x))
            ap_max = torch.argmax(self.qf(xp))    
        qsa = self.qf(x)
        qgrad = torch.autograd.grad(qsa[a], self.model.parameters(), create_graph=True)
        # vgrad = torch.autograd.grad(v, self.model.parameters(), create_graph=True)
        delta = self.computeDelta(x, a_max, xp, ap_max, r, gamma, terminal)
        for i in range(len(qgrad)):
            self.z[i] = (gamma * self.lambda_ * self.z[i]) + qgrad[i]
        z1 = self.z
        if terminal or terminated:
            self.resetEpisode()
            # self.resetTrace()
        self.optimizer.zero_grad()

        #update the gradients fo the model
        for pgrad,p, z_ in zip(qgrad,self.model.parameters(), z1):
            p.grad = - (delta * z_ + (qsa[a_max].detach() - qsa[a].detach()) * pgrad)
        self.optimizer.step()
       
        if self.use_wandb:
            import wandb
            wandb.log({'loss_qf': delta.item()**2})
            wandb.log({'loss': delta.item()**2})


        return delta.item()**2


    # def getW(self):
    #     return self.theta

    def getTrace(self):
        return self.z
    
    # def getValues(self, states_features):
    #     '''
    #     Returns the value of the states
    #     states : np.array of shape (n, features)
    #     '''
    #     return self.model(states_features).detach().numpy()


if __name__ == '__main__':
   pass