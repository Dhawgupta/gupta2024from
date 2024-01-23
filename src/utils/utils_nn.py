import torch 
import torch.nn as nn

# intialize a network with random weights
def initialize_network(model, mean = 0, std = 0.1):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean,std)
            m.bias.data.zero_()
    


    
