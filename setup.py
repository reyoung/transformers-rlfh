from setuptools import setup, find_packages

setup(
    name="transformers-rl",
    version="0.0.1.dev.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
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
    ]
)
