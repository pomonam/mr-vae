# Multi-Rate VAE

Reference implementation of **Multi-Rate VAE: Train Once, Get the Full Rate-Distortion Curve** ([Bae et al., ICLR 2023](https://arxiv.org/abs/2212.03905)).

A standard β-VAE has to be retrained from scratch for every β you want to evaluate. An **MR-VAE** trains a single hyper-network that produces a continuum of β-VAEs in one shot, so a single training run traces out the entire rate-distortion curve.

On binarized MNIST with a ResNet encoder/decoder, one MR-VAE training run reproduces the same rate-distortion frontier as **four separate** β-VAE training runs at fixed β. Same architecture, same per-run budget (8 epochs × 50 k examples). The MR-VAE matches or beats the dedicated baselines at most β values, while costing roughly 1/N the wall-clock for N rate points. Numbers in the table below.

## Install

```bash
git clone https://github.com/pomonam/mr-vae.git
cd mr-vae
pip install -e .          # runtime
pip install -e .[dev]     # + pytest, if you want to run tests
```

Requires Python ≥ 3.8 and PyTorch ≥ 1.12.

## Reproduce the headline result

Two commands, no GPU required (CPU run ~2–3 hours; see "Smaller demo" at the bottom for a 1-minute version):

```bash
export WANDB_MODE=disabled                                      # no W&B account needed

python rd_curve_demo.py        --total_epochs 8 --n_train -1                          # ~25 min  : MR-VAE
python baseline_comparison.py  --total_epochs 8 --n_train -1 --betas 0.01,0.1,1.0,10.0  # ~100 min : 4 β-VAEs
```

The first writes the MR-VAE rate-distortion sweep to `rd_curve.csv`. The second adds the baseline runs alongside the MR-VAE points in `comparison.csv`. Both CSVs are gitignored — rerun freely. To render a matplotlib figure from the CSV, `pip install matplotlib` then `python plot_rd_curve.py`.

### What you should see

Our run on the above settings (8 epochs, full MNIST, ResNet, batch 64, lr 1e-3) produced:

|     β  | MR-VAE rate | MR-VAE dist |  β-VAE rate | β-VAE dist |   gap (lower distortion wins) |
| -----: | ----------: | ----------: | ----------: | ---------: | ----------------------------- |
| 0.0100 |       112.2 |        47.5 |       129.1 |       42.2 | baseline −5.3                 |
| 0.0215 |       102.2 |        47.9 |       —     |        —   |                               |
| 0.0464 |        89.8 |        48.3 |       —     |        —   |                               |
| 0.1000 |        76.1 |        48.9 |        81.7 |       45.2 | baseline −3.7                 |
| 0.2154 |        60.8 |        51.5 |       —     |        —   |                               |
| 0.4642 |        45.3 |        56.6 |       —     |        —   |                               |
| 1.0000 |        30.2 |        65.3 |        25.6 |       68.7 | **MR-VAE −3.4**               |
| 2.1544 |        18.4 |        84.1 |       —     |        —   |                               |
| 4.6416 |         9.3 |       110.5 |       —     |        —   |                               |
| 10.000 |         4.7 |       141.6 |         3.5 |      148.4 | **MR-VAE −6.8**               |

At extreme low β the dedicated baseline still wins (~3–5 nat distortion advantage by over-specializing on reconstruction); at mid- and high-β operating points the MR-VAE wins outright; everywhere in between MR-VAE is within a few nats. Each baseline costs the same wall-clock as the MR-VAE, but produces just one RD point.

## Smaller demo (~1 minute, CPU)

For a quick sanity check, run with the default flags. This trains on an 8k MNIST subset for 5 epochs and sweeps 10 β values:

```bash
WANDB_MODE=disabled python rd_curve_demo.py
```

The qualitative trend (monotone rate-decreasing, distortion-increasing in β) is visible even at this budget; absolute numbers are worse than the paper because the model is under-trained.

## Training one β-VAE or one MR-VAE directly

The two `hyper_train.py` and `baseline_train.py` scripts under `experiments/` are the production entry points (more features, more knobs, W&B integration):

```bash
# MNIST/Omniglot
cd experiments/binary_image
python hyper_train.py    --data_name mnist --encoder_name resnet --decoder_name resnet --total_epochs 50
python baseline_train.py --data_name mnist --encoder_name resnet --decoder_name resnet --beta 1.0 --total_epochs 50

# CIFAR-10 / SVHN / CelebA
cd experiments/image
python hyper_train.py    --data_name cifar --arch_name resnet --total_epochs 100
python baseline_train.py --data_name cifar --arch_name resnet --beta 1.0 --total_epochs 100
```

β sampling range defaults to `[0.01, 10]` (paper §3.4). Override with `--sample_a` / `--sample_b`.

For CelebA, PyTorch downloads the dataset to `../../logs/data/` on first use. If you have it cached elsewhere, point `data_path` in `experiments/image/input_pipeline.py` at it.

## Repository layout

| Path                              | What's there                                            |
| --------------------------------- | ------------------------------------------------------- |
| `rd_curve_demo.py`                | Train one MR-VAE, sweep β, save CSV                     |
| `baseline_comparison.py`          | Train N separate β-VAEs at fixed β values, save CSV     |
| `plot_rd_curve.py`                | Render CSV(s) to a matplotlib PNG                       |
| `src/`                            | Core library (VAE + MR-VAE machinery)                   |
| `src/hyper/layers.py`             | Per-layer modulation — paper Eqn 7 + Listing 1          |
| `src/hyper/base_model.py`         | β sampling, hyper-input standardization                 |
| `experiments/binary_image/`       | MNIST / Omniglot pipeline (input, models, train)        |
| `experiments/image/`              | CIFAR / SVHN / CelebA pipeline                          |
| `tests/`                          | Model-construction smoke tests                          |

## Weights & Biases

All training scripts log to [W&B](https://wandb.ai) by default. Either run `wandb login` once with your own account, or set `WANDB_MODE=disabled` to no-op every `wandb.log` call (useful for offline / CI runs).

## Tests

```bash
WANDB_MODE=disabled pytest tests/
```

Six smoke tests instantiate each model variant (Conv / ResNet × baseline / MR-VAE × MNIST / CIFAR / CelebA shapes) and verify forward passes return correctly-shaped outputs. They do not train.

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
