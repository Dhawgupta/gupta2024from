'''
This problem is a random walk with abrupt reward function changes.'''
import numpy as np

import os, sys, time
sys.path.append(os.getcwd())

from src.utils.utils import getSteadyStateDist, OptimalWeights

LEFT = 0
RIGHT = 1

class RandomWalkAbrupt():
    def __init__(self, params): # the reward scale is the
        self.states = params['states']
        self.reward_scale = params.get('reward_scale', 5)
        # Random State initialization
        # self.state = self.states // 2
        self.state = np.random.randint(0, self.states)
        self.rewardFun = self.buildAverageReward()
        self.actions = 2
        # self.state = 1
        self.eps_len = 0
    
    def reset(self):
        self.state = np.random.randint(0, self.states)
        # start from middle
        # self.state = self.states // 2
        # self.state = 1
        self.eps_len = 0
        return self.state
    
    def step(self, a):
        if self.state == -1 or self.state == self.states:
            raise Exception("Episode is over, Reset the environment")
            
        self.eps_len += 1
        reward = self.rewardFun[self.state]
        terminal = False
        sp = self.state + 2 * a - 1

        if sp == -1 :
            sp = self.states
            # reward = -1 # give a negative reward on left
            terminal = True
        elif sp == self.states:
            sp = self.states
            # reward = 1 # give a positive reward on right
            terminal = True
        self.state = sp
        return (reward, sp, terminal)

    def buildTransitionMatrix(self, policy):
        P = np.zeros((self.states + 1, self.states + 1))
        pl, pr = policy(0)
        P[0,1] = pr
        P[0, self.states] = pl
        for i in range(1, self.states):
            pl, pr = policy(i)
            P[i,i-1] = pl
            P[i,i+1] = pr
        # P[self.states, self.states//2] = 1

        return P
    
    def buildAverageReward(self, policy = None):
        R = np.zeros(self.states + 1)

        # R[0] =0
        # return R
        # positive reward on
        # R[self.states - 1] = pr * 1
        # to make the reward change abruptly
        for i in range(0, self.states):
            if i % 2 == 1:
                R[i] = -self.reward_scale
            else:
                R[i] = self.reward_scale
            
            # pl, pr = policy(i)
            # R[i] = pl * -1 + pr * 1
        # print(R)
        return R
    
    
    def getSteadyStateDist(self, policy):
        # Transition matrix _without_ terminal absorbing state
        P = np.zeros((self.states, self.states))

        pl, pr = policy(0)
        P[0, : ] = pl/self.states
        P[0, 1] += pr

        pl, pr = policy(self.states - 1)
        P[self.states - 1, self.states - 2] = pl
        P[self.states - 1, :] += pr/self.states

        for i in range(1, self.states - 1):
            pl, pr = policy(i)
            P[i, i - 1] += pl
            P[i, i + 1] += pr

        # now include the terminal state so dimensions are consistent
        db = np.zeros(self.states + 1)
        db[:self.states] = getSteadyStateDist(P)
        return db
    

    def computeValueFunction(self, policy, gamma= 0.99):
        '''
        Returns the True value function for the given policy
        '''
        P = np.zeros((self.states, self.states))

        pl, pr = policy(0)
        P[0, : ] = pl/self.states
        P[0, 1] += pr

        pl, pr = policy(self.states - 1)
        P[self.states - 1, self.states - 2] = pl
        P[self.states - 1, :] += pr/self.states

        for i in range(1, self.states - 1):
            pl, pr = policy(i)
            P[i, i - 1] += pl
            P[i, i + 1] += pr
        P = self.buildTransitionMatrix(policy)
        I = np.eye(P.shape[0])
        A = I - gamma * P
        Ainv = np.linalg.inv(A)
        r = self.buildAverageReward(policy)
        v = np.dot(Ainv, r)
        return v


class Representation():
    def __init__(self, states):
        self.states = states 
        # assert (states-1) % 4 == 0, ("States should be a multiple of 4")
        self.feature_size = int(self.states/4) + 1
        self.map = np.zeros((self.states +  1, self.feature_size))
        for i in range(self.states):
            self.map[i] = self.state_representation(i)
    
    def encode(self, s):
        return self.map[s]
    
    def features(self):
        return self.feature_size
    
    def __len__(self):
        return self.states

    def feature_map(self):
        return self.map
    

    def state_representation(self, position):
        # returns the appropriate state representation for position
        # need to invert the postition
        position = self.states - position
        assert (position <= self.states), "Position out of bounds"
        self.representation = np.zeros( int(self.states/4)  + 1 ) 
        if position == 0:
            pass
        elif (position-1)%4 == 0 :# i.e. a multiple of 4
            self.representation[int( int(self.states/4)   - ((position-1)/4))] = 1
        else:
            x = int((position-1)/4) # dont know but for 1, 1//4 was giving -1
            y =(position-1)%4
            self.representation[ int(self.states/4)  - (x+1) ] = float(y)/4.0
            self.representation[int(self.states/4)  - x] = float(4-y)/4.0

        return self.representation



if __name__ == '__main__':
    from src.utils.utils import random_policy
    env = RandomWalkAbrupt(states = 8)
    rep = Representation(env.states)
    v = env.computeValueFunction(random_policy,gamma = 0.99)
    weights = OptimalWeights(env, rep, random_policy)
    print("WEights", weights)
    estimated_values = rep.feature_map().dot(weights)
    # distance between values functions
    dist_v = np.linalg.norm(v - estimated_values)
    print("Distnace to V", dist_v)
    # print(rep.feature_map().dot(weights))
    feature_map = rep.feature_map()
    
    for f, v, in zip(feature_map, v):
        print("Feature/ Value,", f, v)
    # print(rep.feature_map())
    # print(v)

    print("Pass")