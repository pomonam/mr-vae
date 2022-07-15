# Learning VAE with hypernetwork

## Setting up

### Weights and Biases logging
The repository supports `wandb` logging. To log the results in your `wandb` account, simply enable 
modify the `.env` file by providing the `WANDB_API_KEY`.

### Local
If you would like to run the code on a local machine, follow these steps:
```shell
conda create -n vae python=3.7
conda activate vae
pip install -e .
```

### Vector Vaughn Cluster
If you would like to run the code on a Vector cluster. Follow
```shell
mkdir $HOME/condaenvs
export PATH=/pkgs/anaconda3/bin:$PATH
conda create -p $HOME/condaenvs/vae python=3.7
export PYTHONPATH=$HOME/condaenvs/vae$PYTHONPATH
```

Then create environment file `envs/vae_env`
```shell
export PATH=/pkgs/anaconda3/bin:$PATH
source activate $HOME/condaenvs/vae
export PYTHONPATH=$HOME/condaenvs/vae:$PYTHONPATH
export LD_LIBRARY_PATH=/pkgs/cuda-10.2/lib64:/pkgs/cudnn-10.2-v7.6.5.32/lib64:$LD_LIBRARY_PATH
```
Then
```shell
. envs/vae_env
pip install torch==1.12.0 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu102
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
