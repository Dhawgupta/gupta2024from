import torch

def get_optimizer(name):
    if name == 'Adam':
        return torch.optim.Adam
    if name == 'RMSprop':
        return torch.optim.RMSprop
    if name == 'SGD':
        return torch.optim.SGD
    raise Exception("Optimizer not found")
    