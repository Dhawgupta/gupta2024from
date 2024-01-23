import src.problems.RandomWalk as RandomWalk
import src.problems.RandomWalkAbrupt as RandomWalkAbrupt
import src.problems.MountainCar as MountainCar
import src.problems.Acrobot as Acrobot
import src.problems.CartPole as CartPole
import src.problems.LunarLander as LunarLander

def get_problem(name):
    if name == 'RandomWalk':
        return RandomWalk.RandomWalk, RandomWalk.Representation 
    elif name == 'RandomWalkAbrupt':
        return RandomWalkAbrupt.RandomWalkAbrupt, RandomWalkAbrupt.Representation
    elif name == 'MountainCar':
        return MountainCar.MountainCar, MountainCar.IdentityRepresentation
    elif name == 'Acrobot':
        return Acrobot.Acrobot, Acrobot.IdentityRepresentation
    elif name == 'CartPole':
        return CartPole.CartPole, CartPole.IdentityRepresentation
    elif name == 'LunarLander':
        return LunarLander.LunarLander, LunarLander.IdentityRepresentation
    
    else:
        raise ValueError('Unknown problem: ' + name)