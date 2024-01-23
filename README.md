# Code for Paper: From Past to Future: Rethinking Eligibility Traces

The code for different agents in present in `src/agents`, where `BiTD` agent is defined in `MultiBiTD`, similarly for other methods. There are multiple other agents also present, so reader can choose which ever code they want to run. 

The configurations of a experiment is specified using `json` files, which can be found in the `experiments` folder of each experiment. To run a configuration(lets for example in Chain).

Running and Experiment from configuration
```bash
cd code;
python src/mainjson.py experiments/aaai2024/BiTD.json 0
```

To run the 0th configuration in the `experiments/aaai2024/BiTD.json` file. The results are stored in the `results` folder. 

```bash
python run/pending.py -j experiments/aaai2024/BiTD.json 
```
To find all the pending experiments to be run for the `BiTD.json` file. 

```bash
python run/local.py -p src/mainjson.py -j experiments/aaai2024/BiTD.json -c 8
```
To run all configurations in the `BiTD.json` file using 8 threads in parallel. 

After completely running a configuration file, first have to process the data in order to average over all the seeds

```bash
python analysis/process_data.py  experiments/aaai2024/BiTD.json
```

Then to plot the results

```bash
python analysis/learning_curve.py y mstde auc experiments/aaai2024/BiTD.json
```



