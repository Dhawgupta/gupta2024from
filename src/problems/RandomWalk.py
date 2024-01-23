import numpy as np
from src.utils.utils import getSteadyStateDist, OptimalWeights

LEFT = 0
RIGHT = 1

class RandomWalk():
    def __init__(self, params):
        self.states = params['states']
    
        # Random State initialization
        # self.state = self.states // 2
        self.state = np.random.randint(0, self.states)
        # self.state = 1
        self.eps_len = 0
        self.actions = 2
    
    def reset(self):
        self.state = np.random.randint(0, self.states)
        # start from middle
        # self.state = self.states // 2
        # self.state = 1
        self.eps_len = 0
        return self.state
    
    def step(self, a):
        self.eps_len += 1
        reward = 0
        terminal = False
        sp = self.state + 2 * a - 1

        if sp == -1 :
            sp = self.states
            reward = -1 # give a negative reward on left
            terminal = True
        elif sp == self.states:
            sp = self.states
            reward = 1 # give a positive reward on right
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
    
    def buildAverageReward(self, policy):
        R = np.zeros(self.states + 1)

        # probability of transition times the reward received.
        pl, _ = policy(0)
        # negative reward on left
        R[0] = pl * -1

        _, pr = policy(self.states - 1)
        # positive reward on
        R[self.states - 1] = pr * 1

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
    def __init__(self, states = 5):
        self.states = states
        self.map = np.zeros([states+1, states])
        self.map[:self.states] = np.eye(states)
    
    def encode(self, s):
        return self.map[s]
    
    def decode(self, v):
        return np.argmax(v)

    def features(self):
        return self.map.shape[1]
    
    def __len__(self):
        return self.states
    
    def feature_map(self):
        return self.map


if __name__ == '__main__':
    from utils import random_policy
    env = RandomWalk(states = 11)
    rep = Representation(states = 11)
    v = env.computeValueFunction(random_policy)
    weights = OptimalWeights(env, rep, random_policy)

    print(v)
    print(weights)



    print("Pass")