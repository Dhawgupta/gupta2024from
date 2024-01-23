# make this into a file which uses argparse to run experiments
import os, sys, time
sys.path.append(os.getcwd())

import argparse
from run_experiment import run_experiment



if __name__ == '__main__':
    # get the parameters
    args = argparse.ArgumentParser()
    args.add_argument('--problem', type = str, default = 'RandomWalk')
    args.add_argument('--states', type = int, default = 5)

    args.add_argument('--agent', type = str, default = 'TDLambdaOnline')
    args.add_argument('--gamma', type = float, default = 0.99)
    args.add_argument('--lambda', type = float, default = 0.99)
    args.add_argument('--steps', type = int, default = 10000)

    args.add_argument('--optimizer', type = str, default = 'SGD')
    args.add_argument('--alpha', type = float, default = 0.01)
    args.add_argument('--hidden_units', type = int, default = 8)

    args.add_argument('--seed', type = int, default = 0)


    
    args.add_argument('--use_wandb', type = bool, default = False)
    args.add_argument('--force_run', type = bool, default = False)
    
    args = args.parse_args()


    
    # Convert aargs to dict
    params = vars(args)

    run_experiment(params, args.force_run)
    
