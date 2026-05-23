from setuptools import find_packages, setup

setup(
    name="mr-vae",
    version="0.1.0",
    description="Reference implementation of Multi-Rate VAE (Bae et al., 2023).",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://arxiv.org/abs/2212.03905",
    license="Creative Commons Attribution-NonCommercial-ShareAlike 4.0",
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12",
        "torchvision>=0.13",
        "numpy",
        "scipy",
        "tqdm",
        "wandb",
        "Pillow",
    ],
    extras_require={
        "dev": ["pytest"],
    },
    packages=find_packages(exclude=("tests", "tests.*")),
    test_suite="tests",
    zip_safe=False,
)
