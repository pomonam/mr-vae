# Learning VAE with hypernetwork

## Setting up

### Weights and Biases logging
The repository supports `wandb` logging. To log the results in your `wandb` account, simply enable 
modify the `.env` file by providing the `WANDB_API_KEY`.

### Local
If you would like to run the code on a local machine, follow these steps:
```shell
conda create -n hvae python=3.9
conda activate hvae
pip install -e .
```

### Vector Vaughn Cluster
If you would like to run the code on a Vector cluster. Follow
```shell
mkdir $HOME/condaenvs
export PATH=/pkgs/anaconda3/bin:$PATH
conda create -p $HOME/condaenvs/hvae python=3.9
export PYTHONPATH=$HOME/condaenvs/hvae$PYTHONPATH
```

Then create environment file `envs/hvae_env`
```shell
export PATH=/pkgs/anaconda3/bin:$PATH
source activate $HOME/condaenvs/hvae
export PYTHONPATH=$HOME/condaenvs/hvae:$PYTHONPATH
export LD_LIBRARY_PATH=/pkgs/cuda-10.2/lib64:/pkgs/cudnn-10.2-v7.6.5.32/lib64:$LD_LIBRARY_PATH
```
Then
```shell
. envs/hvae_env
pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu102
mkdir codes
cd codes
git clone https://github.com/pomonam/hyper-vae
cd hyper-vae
pip install -e .
```
In order to prevent data corruption, we do:
```shell
python download_data.py
```
