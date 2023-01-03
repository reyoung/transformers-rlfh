from setuptools import setup

setup(
    name="transformersrl",
    version="0.0.1.dev.0",
    install_requires=[
        "transformers",
        "torch",
        "numpy",
        "wandb",
        "optuna",
        "datasets",
        "tqdm",
    ]
)
