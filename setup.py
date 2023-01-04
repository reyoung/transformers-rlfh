from setuptools import setup, find_packages

setup(
    name="transformers-rlfh",
    version="0.0.1.dev.3",
    packages=find_packages(exclude=["tests", "tests.*", "test"]),
    description="RLFH with transformers",
    long_description="RLFH with transformers",
    install_requires=[
        "transformers",
        "torch",
        "numpy",
        "wandb",
        "optuna",
        "datasets",
        "tqdm",
        "torchtyping",
    ]
)
