# Deep Learning for Frequency Response Prediction of a  Multi-Mass Oscillator

TODO Add abstract


TODO add Link to Journal article

## Setup 

To perform the training on gpus please make sure to have a cuda-capable system (https://developer.nvidia.com/cuda-zone)

### Using Anaconda
We recommend Anaconda to set up the python environment. If Anaconda is installed and activated, run the following to setup the environment acoustics_mmo:

```
source setup.sh
```
### Using pip
To install pytorch and the additional dependencies run

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

And then to install the mmo package as an editable install run 
``` 
pip install -e .
```

## Train

The scripts to run a training are given in  acousticnn/mmo/run.sh. To 

to run a single training:

```
cd acousticnn/mmo
python run.py --config implicit_mlp.yaml --dir path/to/save_directory --encoding none --seed 0 --device cuda
```

the --wildcard arg specifies the number of samples. The --encoding arg specifies whether the input parameters are encoded with a sinosoidal embedding function. To use the scalar parameters, choose --encoding none.
To select the model architecture change the --config .yaml file arg. --device specifies whether to use cuda or cpu for training.


