# Deep Learning for Frequency Response Prediction of a  Multi-Mass Oscillator

TODO Add abstract


TODO add Link to Journal article

## Setup 
We recommend Anaconda to set up the python environment. If Anaconda is installed and activated, run the following to setup the environment acoustics_mmo:

''
source setup.sh
''


## Train

The scripts to run a training are given in  acousticnn/mmo/run.sh

to run a single training:


''
cd acousticnn/mmo
python run.py --config configs/implicit_mlp.yaml --dir path/to/save_directory --encoding None --seed 0
''


## Evaluate a trained model