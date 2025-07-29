#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# Read the README.md file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Main dependencies required for the package
install_requires = [
    "jax[cuda12]==0.4.35",  # JAX with CUDA 12 support
    "flax==0.10.2",
    "chex==0.1.84",
    "optax==0.1.7",
    "dotmap==1.3.30",
    "evosax==0.1.5",
    "distrax==0.1.5",
    "brax==0.10.3",
    "orbax-checkpoint",
    "gymnax==0.0.6",
    "safetensors==0.4.2",
    "flashbax==0.1.0",
    "scipy==1.12.0",
]

# Additional dependencies that are less sensitive to version changes
extras_require = {
    "dev": [
        "pytest==8.3.3",
    ],
    "viz": [
        "wandb==0.18.7",
        "imageio==2.36.0",
        "pygame==2.6.1",
        "matplotlib>=3.8.3",
        "pillow>=10.2.0",
        "seaborn==0.13.2",
    ],
    "utils": [
        "tyro==0.9.2",
        "numpy>=1.26.1",
        "hydra-core>=1.3.2",
        "omegaconf>=2.3.0",
        "pettingzoo>=1.24.3",
        "tqdm>=4.66.0",
        "python-dotenv==1.0.1",
        "pandas==2.2.3",
    ],
}

# All extras combined
extras_require["all"] = [pkg for group in extras_require.values() for pkg in group]

setup(
    name="MEAL",
    version="0.1.0",
    url="https://github.com/TTomilin/MEAL",
    author="Tristan Tomilin",
    author_email='tristan.tomilin@hotmail.com',
    description="MEAL: A Benchmark for Continual Multi-Agent Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
)
