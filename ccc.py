import torchvision.datasets as datasets
from pythae.trainers.training_callbacks import WandbCallback

mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=None)

train_dataset = mnist_trainset.data[:-10000].reshape(-1, 1, 28, 28) / 255.
eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.


from pythae.models import BetaVAE, BetaVAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST


config = BaseTrainerConfig(
    output_dir='logs/',
    learning_rate=1e-4,
    batch_size=100,
    num_epochs=10,
    # Change this to train the model a bit more
)


model_config = BetaVAEConfig(
    input_dim=(1, 28, 28),
    latent_dim=16,
    beta=2.

)

model = BetaVAE(
    model_config=model_config,
    encoder=Encoder_ResNet_VAE_MNIST(model_config),
    decoder=Decoder_ResNet_AE_MNIST(model_config)
)

pipeline = TrainingPipeline(
    training_config=config,
    model=model
)

callbacks = []
wandb_cb = WandbCallback()
wandb_cb.setup(
    training_config=config, # training config
    model_config=model_config, # model config
    project_name="your_wandb_project", # specify your wandb project
    entity_name="bae-group", # specify your wandb entity
)
callbacks.append(wandb_cb)

pipeline(
    train_data=train_dataset,
    eval_data=eval_dataset,
    callbacks=callbacks
)
