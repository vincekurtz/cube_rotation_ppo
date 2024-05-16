# PPO Cube Rotation

Reproducing the [OpenAI cube rotation
task](https://openai.com/index/learning-dexterity/) with Proximal Policy
Optimization and [Brax/MJX](https://github.com/google/brax).

Also similar to [this NVIDIA reproduction](https://arxiv.org/abs/2210.13702).

## Installation

Clone this repository and run the following commands in the repository root to
create and activate a conda environment with Cuda 12.3 support:

```bash
conda env create -n <env_name> -f environment.yml
conda activate <env_name>
```

Install the package and dependencies with pip:

```bash
pip install -e . --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Set up pre-commit hooks:

```bash
pre-commit autoupdate
pre-commit install
```
