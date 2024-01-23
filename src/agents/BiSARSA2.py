import numpy as np
import torch
import torch.nn as nn
from src.utils.utils_nn import initialize_network

from src.optimizers.registry import get_optimizer

class BiSARSA2:
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
            self.model = nn.Sequential(nn.Linear(self.features, 2 * self.actions)).float()
        else: # make a 1 layer neural network
            # create a 1 layer neural network
            self.model = nn.Sequential(nn.Linear(self.features, self.hidden_units), nn.ReLU(), nn.Linear(self.hidden_units, 2 * self.actions)).float()
            
        self.optimizer = get_optimizer(self.optimizer_type)(self.model.parameters(), lr = params['alpha'])
            
        self.xh = None # Previous state
        self.rh = None # Previous reward
        self.ah = None # Previous action

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=params['alpha'])

    def resetAgent(self):
        initialize_network(self.model)

    def selectAction(self, x):
        p = np.random.rand()
        if p < self.epsilon:
            # Choose a random action
            return np.random.choice(self.actions)

        return np.argmax(self.qf(x).detach().numpy())

    def computeQFLoss(self, x, a, xp, ap, r, gamma, terminal):
        with torch.no_grad():
            target = r + (1 - int(terminal)) * self.qf(xp)[ap] * gamma
        # return target
        q = self.qf(x)[a]
        loss = 0.5 * nn.MSELoss()(q, target)
        return loss

    def computeQRLoss(self, x, a, gamma):
        with torch.no_grad():
            target = torch.tensor(0.0).float()
            if self.xh is not None:
                target = self.lambda_ * gamma * (self.rh +  self.qr(self.xh)[self.ah])
        qr = self.qr(x)[a]
        loss = 0.5 * nn.MSELoss()(qr, target)
        return loss

    def computeQBLoss(self, x, a, xp, ap, r, gamma, terminal):
        with torch.no_grad():
            target = torch.tensor([0.0]).float()

            target += (r * (1 - (gamma * gamma * self.lambda_)))
            if not terminal:
                target += gamma * self.qb(xp)[ap]
            if self.xh is not None:
                target += gamma * self.lambda_ * self.qb(self.xh)[self.ah]

            target /= (1 + (gamma * gamma * self.lambda_))

        qb = self.qb(x)[a]
        loss = 0.5 * nn.MSELoss()(qb, target)
        return loss
        # return target


    def resetEpisode(self):
        self.xh = None
        self.rh = None
        self.ah = None

    def qf(self, x):
        # Forward value function
        # print(self.model(x).reshape((-1,2)) [:,0])
        return self.qb(x) - self.qr(x)

    def qr(self, x):
        # Reverse value function
        return self.model(x).reshape((-1,2 * self.actions))[:, self.actions:].reshape(-1)

    def qb(self, x):
        # Bi value function
        return self.model(x).reshape((-1, 2 * self.actions))[:, :self.actions].reshape(-1)


    def update(self, x, a, xp, ap, r, gamma, terminal, terminated):
        loss = 0
        # forward V loss
        loss_vf = self.computeQFLoss(x, a, xp, ap, r, gamma, terminal)
        # backward V loss
        loss_vr = self.computeQRLoss(x, a, gamma)
        # Bi V loss
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

    def getW(self):
        raise NotImplementedError
        
    
    # def getValues(self, states_features):
    #     '''
    #     Returns the value of the states
    #     states : np.array of shape (n, features)
    #     '''
    #     #FIXME
    #     # Only use the first head for value function
    #     # return self.model(states_features).detach().numpy()[:, 0]
    #     return self.qf(states_features).detach().numpy()
    #     # return self.model(states_features).detach().numpy()
    #

    
