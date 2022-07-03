# hyper-vae

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
If you would like to run the code on a Vector cluster.