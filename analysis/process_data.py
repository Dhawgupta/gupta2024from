'''
This file will take in json files and process the data across different runs to store the summary
'''
import os, sys, time, copy
sys.path.append(os.getcwd())
import numpy as np

from src.utils.json_handling import get_sorted_dict , get_param_iterable_runs
from src.utils.formatting import create_file_name, create_folder
from analysis.utils import load_different_runs, pkl_saver, pkl_loader

# read the arguments etc
if len(sys.argv) < 2:
    print("usage : python analysis/process_data.py <list of json files")
    exit()

json_files = sys.argv[1:] # all the json files

# convert all json files to dict
json_handles = [get_sorted_dict(j) for j in json_files]

def process_runs(runs):
    # get mean and std
    mean = np.mean(runs, axis = 0)
    stderr = np.std(runs , axis = 0) / np.sqrt(runs.shape[0])
    return mean , stderr



# currentl doesnt not handle frames
def process_data_interface(json_handles):
    for js in json_handles:
        runs = []
        iterables = get_param_iterable_runs(js)
        for i in iterables:
            folder, file = create_file_name(i, 'processed')
            create_folder(folder) # make the folder before saving the file
            filename = folder + file + '.pcsd'
            # check if file exists
            print(filename)
            if os.path.exists(filename):
                print("Processed")

            else:
                losses, msve, mstde = load_different_runs(i)

                # train, test, validation, loss = load_different_runs(i)
                mean_losses, stderr_losses = process_runs(losses)
                mean_msve, stderr_msve = process_runs(msve)
                mean_mstde, stderr_mstde = process_runs(mstde)

                # losses
                loss_data = {
                    'mean' : mean_losses,
                    'stderr' : stderr_losses
                }
                # msve
                msve_data = {
                    'mean' : mean_msve,
                    'stderr' : stderr_msve
                }
                # mstde
                mstde_data = {
                    'mean' : mean_mstde,
                    'stderr' : stderr_mstde
                }

                pkl_saver({
                    'losses' : loss_data,
                    'msve' : msve_data,
                    'mstde' : mstde_data
                }, filename)
                print("Saved")

    # print(iterables)

if __name__ == '__main__':
    process_data_interface(json_handles)