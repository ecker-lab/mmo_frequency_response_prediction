
conda create -n acoustics_mmo pytorch torchvision torchaudio pytorch-cuda=11.7 timm -c pytorch -c nvidia -c conda-forge 

conda activate acoustics_mmo
pip install munch torchinfo matplotlib ipykernel jupyter transformers scipy wandb 
pip install h5py hdf5plugin
pip install -e .

python -m ipykernel install --user --name=acoustics_mmo