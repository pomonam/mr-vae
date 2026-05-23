# Multi-Rate VAE

Reference implementation of **Multi-Rate VAE: Train Once, Get the Full Rate-Distortion Curve** ([Bae et al., 2023](https://arxiv.org/abs/2212.03905)).

A Multi-Rate VAE (MR-VAE) trains a single hyper-network that produces a continuum of β-VAE models, so a single training run recovers the entire rate-distortion curve instead of training one model per β.

This repository is a pared-down, public-facing version of the original research code. It contains two runnable example pipelines:

- **Binarized image VAEs** on MNIST / Omniglot (28×28).
- **Continuous image VAEs** on CIFAR-10 / SVHN / CelebA.

Both have baseline β-VAE and hyper-network MR-VAE variants.

## Installation

Requires Python ≥ 3.8 and PyTorch ≥ 1.12.

```bash
git clone https://github.com/<your-fork>/mr-vae.git
cd mr-vae
pip install -e .
```

For development (running tests):

```bash
pip install -e .[dev]
```

## Logging with Weights & Biases

Training scripts log metrics and sample images to [W&B](https://wandb.ai). Two options:

1. **Log to your own account:** `wandb login` once, then run the scripts normally.
2. **Run offline (no account required):** set `WANDB_MODE=disabled` in your shell. All `wandb.log` calls become no-ops.

```bash
export WANDB_MODE=disabled
```

## Quickstart: rate-distortion curve in one command

This trains one MR-VAE on a 8k-example MNIST subset for a few epochs (CPU-friendly, ~3-5 min) and prints the rate-distortion sweep at the end:

```bash
WANDB_MODE=disabled python rd_curve_demo.py
```

You should see distortion fall as β decreases (the model spends more bits in the latent) and rise as β increases — i.e. a single training run traces out the entire RD curve. CSV of the points lands in `rd_curve.csv`.

## Quickstart: full MR-VAE on MNIST

```bash
cd experiments/binary_image
python hyper_train.py --data_name mnist --encoder_name resnet --decoder_name resnet --total_epochs 50
```

The β sampling range defaults to `[0.01, 10]` (paper Section 3.4). Override with `--sample_a` / `--sample_b`.

For comparison, you can train a single baseline β-VAE at a fixed β:

```bash
python baseline_train.py --data_name mnist --encoder_name resnet --decoder_name resnet --beta 1.0 --total_epochs 50
```

## Quickstart: train on CIFAR-10

```bash
cd experiments/image
python hyper_train.py --data_name cifar --arch_name resnet --total_epochs 100
```

For CelebA, pass `--data_name celeba`. PyTorch will download CelebA into `../../logs/data/` the first time you run; if you have it cached elsewhere, point `data_path` in `experiments/image/input_pipeline.py` at it.

## Repository layout

```
rd_curve_demo.py              Self-contained demo: train + sweep β (paper Fig. 1)

src/                          Core library
├── base_architecture.py        Encoder / Decoder abstract bases
├── base_model.py               VAE base class (Gaussian reparam)
├── config.py                   TrainConfig (argparse → fields)
├── evaluate.py                 Metric meters used during training
├── schedules.py                β annealing schedules (constant, monotonic, cyclic)
├── utils.py                    seed_everything, log_sum_exp
├── models/
│   ├── beta_vae.py             Standard β-VAE forward pass
│   └── resblock.py             ResBlock and HyperResBlock
└── hyper/                      Hyper-network MR-VAE machinery
    ├── base_model.py           HyperVAE: log-Uniform[a,b] β sampling
    ├── beta_vae.py             HyperBetaVAE: sample_forward / fixed_forward
    ├── base_architecture.py    BaseHyperEncoder / BaseHyperDecoder
    └── layers.py               Sigmoid-gate (encoder) and √(ReLU(1-exp)) gate (decoder), paper Eqn 7

experiments/
├── train_utils.py              Baseline β-VAE training loop
├── hyper_train_utils.py        MR-VAE training loop
├── wandb_utils.py              wandb initialization
├── binary_image/               MNIST / Omniglot pipeline
│   ├── input_pipeline.py
│   ├── models.py               Conv & ResNet encoders / decoders
│   ├── hyper_models.py         Hyper-network variants
│   ├── baseline_train.py       Train one β-VAE
│   └── hyper_train.py          Train an MR-VAE
└── image/                      CIFAR / SVHN / CelebA pipeline
    └── (same layout)

tests/                         Smoke tests for model construction and forward passes
```

## Running the tests

```bash
WANDB_MODE=disabled pytest tests/
```

These are model-construction smoke tests — they instantiate the convolutional and ResNet variants of both the baseline and hyper-network models for each dataset shape and run a forward pass. They do not train.

## Citation

```bibtex
@inproceedings{bae2023multirate,
  title={Multi-Rate VAE: Train Once, Get the Full Rate-Distortion Curve},
  author={Bae, Juhan and Zhang, Michael R. and Ruan, Michael and Wang, Eric and Hasegawa, So and Ba, Jimmy and Grosse, Roger B.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (see `setup.py`).
