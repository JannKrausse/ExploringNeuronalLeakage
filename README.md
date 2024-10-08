# ExploringNeuronalLeakage
This is the repository with the code and supplementary information of our Paper "Exploring Neuronal Leakage for Spiking Neural Networks on Event-Driven Hardware" that was presented at IEEE/ACM ICONS'24.

In the Supplementaries, you will find the parameters defining the different leakage types found during their optimization using hyperparameter searches. 

The code provides custom Tensorflow RNN cell classes for each neuron of the different leakage types as well as a basic training script and an optuna hyperparameter search script. Using this code, it is possible to reproduce all results presented in the paper.

## Installation and Running Experiments
The requirements should be installed in a Python 3.9 environment. 
```bash
pip install -r requirements.txt
```
For execution, run either 
```bash
python3 main.py
```
for a single training or
```bash
python3 optuna_hps.py
```
for a whole metric-centered optimization run. Be sure to set the respective desired flags.

## Citation
Please consider citing our work if our results are interesing to you or when using our code:
-- citation to be added after publication of proceedings
