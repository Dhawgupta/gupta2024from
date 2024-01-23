import numpy as np
import torch
import gym



class Acrobot():
    def __init__(self, params = None):
        self.env = gym.make('Acrobot-v1')
        self._seed = params.get('seed',0)
        # self.env.seed(self._seed)
        self.actions = self.env.action_space.n
        self.states = self.env.observation_space.shape[0]
        self.time = 0


    # same as reset, manifestation of old code
    def start(self):
        return self.env.reset(seed = self._seed)[0]


    def reset(self):
        self.time = 0
        return self.env.reset(seed = self._seed)[0]



    def step(self, a):
        terminate = False
        sp, r, t, _, _ = self.env.step(a)
        self.time += 1
        if self.time >= 500:
            terminate = True
        return (r, sp, t, terminate)
    

class IdentityRepresentation():
    def __init__(self, params = None):
        # should not need any params by default
        pass

    def features(self):
        return 6
    def encode(self, s):
        # convert into a tensor and stuff
        return torch.tensor(s).float()
        # return s

    def decode(self, s):
        return s

