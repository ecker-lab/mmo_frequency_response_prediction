# Deep Learning for Frequency Response Prediction of a  Multi-Mass Oscillator

"Noise prevention in product development is becoming more and more important due to health effects and comfort restrictions.
In the product development process, costly and time-consuming simulations are carried out over parameter spaces, e.g.,
via the finite element method, in order to find quiet product designs. The solution of dynamic systems is limited in their
maximum frequency and the size of the parameter space. Therefore, the substitution of high-fidelity models by machine
learning approaches is desirable. In this contribution, we consider an academic benchmark: Training a neural network to
predict the frequency response of a multi-mass oscillator. Neural network architectures based on multi-layer perceptrons
(MLP) and transformers are investigated and compared with respect to their accuracy in frequency response prediction. Our
investigations suggest that the transformer architecture is better suited, in terms of accuracy and in terms of capability to
handle multiple system configurations in a single model."


TODO add Link to Journal article

## Setup 

To perform the training on gpus please make sure to have a cuda-capable system (https://developer.nvidia.com/cuda-zone)

### Using Anaconda
We recommend Anaconda to set up the python environment. If Anaconda is installed and activated, run the following to setup the environment acoustics_mmo:

```
source setup.sh
```
### Using pip
Inside your python environment install pytorch and the additional dependencies by running:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

To install the mmo package as an editable install run: 
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


