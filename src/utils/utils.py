from ssl import OP_NO_TLSv1_2
import numpy as np

from typing import Any, Callable, List, Sequence, Union, Iterator, Optional

def matrix_power(X, n):
    A = X
    for _ in range(n):
        A = np.dot(A, X)

    return A
    
def random_policy(state):
    return [0.5, 0.5]
    # if state == 0:
    #     return [0.5, 0.5]
    # elif state == 1:
    #     return [0.5, 0.5]

def trap_policy(state):
    if state == 0:
        return [0.001, 0.999]
    elif state == 1:
        return [0.999, 0.001]

def bad_policy(state):
    if state == 0:
        return [0.0, 1.0]
    elif state == 1:
        return [0.5, 0.5]

    
# def getSteadyStateDist(p):
#     # Transition matrix _without_ terminal absorbing state
#     p1,p2 = p
    
#     P = np.zeros((2,2))
#     P[0,1] = 1.0 # resets in the right state
#     P[1,0] = p2
#     P[1,1] = (1-p2) # reset to current state
#     # print(P)

#     # now include the terminal state so dimensions are consistent
#     db = np.zeros(2)
#     db = getSteadyStateDist_(P)
#     return db

def getSteadyStateDist(P):
    A = matrix_power(P, 1000)
    d = np.mean(A, axis=0)
    return d    

def OptimalWeights(env, rep, policy = random_policy, gamma = 0.99):
    feature_map = rep.feature_map() # size states x features
    V = env.computeValueFunction(policy, gamma=gamma)
    db = env.getSteadyStateDist(policy)
    # print(db)
    # print(db.sum())
    XDX = np.dot(np.dot(feature_map.T, np.diag(db)), feature_map)
    XDXinv = np.linalg.inv(XDX)
    weights = XDXinv.dot(feature_map.T).dot(np.diag(db)).dot(V)
    return weights

def getMSVE(val, valhat, db):
    return np.sqrt(np.sum(np.square(val - valhat) * db))

def getMSTDE(valhat, db, P, R, gamma):
    return np.sqrt(np.sum(np.square(valhat - (np.dot(P, valhat) * gamma + R)) * db))


def sample(arr):
    r = np.random.rand()
    s = 0
    for i, p in enumerate(arr):
        s += p
        if s > r or s == 1:
            return i

    # worst case if we run into floating point error, just return the last element
    # we should never get here
    print("Got here")
    return len(arr) - 1

# 2 state random specific analytic solution
def value1(p , r, gamma, lambda_, x):
  r1, r2, r3, r4 = r
  p1, p2 = p
  num = (1 - p1)*r1 + p1*r2 + gamma * p1 * (1-p2) * r4 + gamma * p1 * p2 * r3
  den = 1 - gamma * gamma * p1 * p2
  return num / den

def value2(p, r, gamma, lambda_, x):
  r1, r2, r3, r4 = r
  p1, p2 = p
  num = (1 - p2)*r4 + p2*r3 + gamma * p2 * (1-p1) * r1 + gamma * p1 * p2 * r2
  den = 1 - gamma * gamma * p1 * p2
  return num / den

def optimal_weight(p, r, gamma, lambda_, x):
  x1, x2 = x
  db = getSteadyStateDist(p)
  db1, db2 = db
#   print(db)
  num = x1 * value1(p, r, gamma, lambda_, x) * db1 + x2 * value2(p, r, gamma, lambda_, x) * db2
  den = x1**2 * db1 + x2**2 * db2
  return num/den

def argsmax(arr: np.ndarray):
    ties: List[int] = [0 for _ in range(0)]  # <-- trick njit into knowing the type of this empty list
    top: float = arr[0]

    for i in range(len(arr)):
        if arr[i] > top:
            ties = [i]
            top = arr[i]

        elif arr[i] == top:
            ties.append(i)

    if len(ties) == 0:
        ties = list(range(len(arr)))

    return ties

def choice(arr, rng):
    idxs = rng.permutation(len(arr))
    return arr[idxs[0]]

# argmax that breaks ties randomly
def argmax(vals, rng):
    ties = argsmax(np.asarray(vals))
    return choice(ties, rng)