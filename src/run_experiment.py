# make this into a file which uses argparse to run experiments
import os, sys, time
sys.path.append(os.getcwd())

import torch
import numpy as np
import matplotlib.pyplot as plt
from analysis.utils import pkl_saver
from src.agents.registry import get_agent
from src.problems.registry import get_problem
from src.utils.formatting import create_file_name
from src.utils.utils import random_policy, getMSTDE, getMSVE



def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

def encode(rep, s):
    return torch.tensor(rep.encode(s)).float()






def get_output_filename(ex):
    folder, filename = create_file_name(ex)
    if not os.path.exists(folder):
        time.sleep(2)
        try:
            os.makedirs(folder)
        except:
            pass
    output_file_name = folder + filename
    return output_file_name
    

def run_experiment(params = None, force_run = False):
    output_file_name = get_output_filename(params)
    
    if not force_run:
        if os.path.exists(output_file_name + '.dw'):
            print("Already Done")
            exit()
    
    steps = params['steps']
    gamma = params['gamma']
    seed = params['seed']
    set_seeds(seed)
    problem = params['problem']
    
    # Get Problem
    env_type, rep_type = get_problem(problem)
    env = env_type(params)
    rep = rep_type(env.states)
    
    # Get Problem Specifics
    vf = env.computeValueFunction(random_policy, gamma = gamma)
    db = env.getSteadyStateDist(random_policy)
    P = env.buildTransitionMatrix(random_policy)
    R = env.buildAverageReward(random_policy)
    fixedFeatures = torch.tensor(rep.feature_map()).float()

    


    params['features'] = rep.features()
    params['actions'] = env.actions
    
    
    # Get Agent
    agent_name = params['agent']

    use_wandb = params.get('use_wandb', False)
    params['use_wandb'] = use_wandb
    
    agent = get_agent(agent_name)(params)



    if use_wandb:
        import wandb
        wandb.init(project="Etraces", config=params)
        wandb.watch_called = False
        wandb.watch(agent.model, log='all', log_freq=100)
    
    
    # Metrics to Measure
    msve_losses = np.zeros(steps)
    mstde = np.zeros(steps)
    losses = np.zeros(steps)

    # initialize experiment
    s = env.reset()
    x = encode(rep, s)
    eps = 0
    start_time = time.time()

    # Run Experiment
    for step in range(steps):
        probs = random_policy(s)
        a = np.random.choice(env.actions, p = probs)
        r, sp, terminal = env.step(a)
        xp = encode(rep, sp)
        loss = agent.update(x, a, xp, r, gamma, terminal)
        s = sp
        x = xp
        if terminal:
            eps += 1
            s = env.reset()
            x = encode(rep, s)
        
        # get vhat
        vhat = agent.getValues(fixedFeatures)

        # measure stats
        losses[step] = loss
        msve_losses[step] = getMSVE(vf, vhat.reshape(-1), db)
        mstde[step] = getMSTDE(vhat.reshape(-1), db, P, R, gamma)
        if use_wandb:
            wandb.log({'MSVE': msve_losses[step], 'MSTDE': mstde[step], 'Loss': loss})

    
    finish_time = time.time()
    
    # plt.plot(msve_losses, label='MSVE')
    # plt.plot(mstde, label='MSTDE')
    # # plt.plot(losses, label='Loss')
    # plt.legend()
    # plt.show()

    save_stats = {
        'losses': losses,
        'msve': msve_losses,
        'mstde': mstde,
    }

    pkl_saver(save_stats, output_file_name + '.dw')

    print(f"Finished {output_file_name} in {finish_time - start_time} seconds")



    

    


