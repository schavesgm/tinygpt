<h1 align="center">
    TinyGPT
</h1>

This repository contains a simple and easy-to-understand implementation of a GPT model using `jax`
and `flax`. The implementation is meant to be purely educational. As a result, the model size is
smaller than the one required to produce good results on real-world applications. The smaller model
size should allow users to test the implementation in their own personal environments. In addition,
no optimisations are implemented in the model, which means that inference would be relatively slow.

## Setup

In order to setup the repository, we need to perform the following actions: (1) (1) clone this
repository, (2) create a virtual environment in your preferred system
([conda](https://docs.conda.io/en/latest/), [venv](https://docs.python.org/3/library/venv.html)) and
(3) run the following command in the project root directory (where this file is located),

```bash
pip install -e ".[cpu]"
```

The `-e` flag (`--editable`) will allow you to make changes to the repository without having to
reinstall the package after each change. 

As `tlm` uses [`jax`](https://jax.readthedocs.io/en/latest/) as a linear algebra backend, two
versions of the package can be installed: `cpu` and `gpu`. Installing the `gpu` version will imply
that all `jax` arrays are allocated in the `gpu`. To install the `gpu` version, just run

```bash
pip install -e ".[gpu]"
```

After installation, the `tlm` package will be available in the environment. You can train a small
GPT model on the [emoji dataset](./dataset.txt) provided by running

```bash
python main.py
```

This will allow you to train a model and then perform predictions using the predicted parameters.
