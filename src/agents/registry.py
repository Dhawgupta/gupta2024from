from src.agents.ForwardTD import ForwardTD
from src.agents.TDLambda import TDLambda
from src.agents.BiTD import BiTD
from src.agents.TDLambdaOnline import TDLambdaOnline
from src.agents.SARSALambda import SARSALambda
from src.agents.BiTD2 import BiTD2
from src.agents.BiSARSA import BiSARSA
from src.agents.BiSARSA2 import BiSARSA2

from src.agents.MultiBiTD import MultiBiTD
from src.agents.MultiBiTD2 import MultiBiTD2
from src.agents.MultiBiTD3 import MultiBiTD3

from src.agents.MultiBiSARSA import MultiBiSARSA
from src.agents.MultiBiSARSA2 import MultiBiSARSA2
from src.agents.MultiBiSARSA3 import MultiBiSARSA3
from src.agents.QLambda import QLambda

def get_agent(name):
    # Prediction Methods
    if name == 'ForwardTD':
        return ForwardTD
    if name == 'TDLambda':
        return TDLambda
    if name == 'BiTD':
        return BiTD
    if name  == 'TDLambdaOnline':
        return TDLambdaOnline
    if name == 'BiTD2':
        return BiTD2
    if name == 'MultiBiTD':
        return MultiBiTD
    if name == 'MultiBiTD2':
        return MultiBiTD2
    if name == 'MultiBiTD3':
        return MultiBiTD3
    
    # Control methods
    if name == 'SARSALambda':
        return SARSALambda
    if name == 'BiSARSA':
        return BiSARSA
    if name == 'BiSARSA2':
        return BiSARSA2
    
    if name == 'MultiBiSARSA':
        return MultiBiSARSA
    if name == 'MultiBiSARSA2':
        return MultiBiSARSA2
    if name == 'MultiBiSARSA3':
        return MultiBiSARSA3
    
    if name == 'QLambda':
        return QLambda

    raise Exception("Agent not found")


