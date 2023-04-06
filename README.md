# Discit: Deep learning tools

Discit (lat.: *it learns*) is a set of tools that were developed for various
learning tasks, from supervised to multi-agent reinforcement learning research.

It focuses on traceable tensor operations so that they can be significantly
accelerated with CUDA graphs.


---


## Installation

Start by downloading or cloning this repository.

Setup will check for the following packages:
- `numpy`: vectorised processing,
- `torch >= 2.0`: main processing and AI integration with CUDA graphs,
- `tensorboard`: tracking training progress.

It is strongly advised that PyTorch is installed separately beforehand.
See the [PyTorch instructions](https://pytorch.org/get-started/locally/)
and ensure that the installation targets your CUDA device.

Finally, install Discit in editable/development mode from the main directory with:
```
pip install -e .
```


---


## Development

Discit is being developed and used as part of the following works:
- [MazeBots](https://github.com/jernejpuc/mazebots) (WIP)
- [SiDeGame](https://github.com/jernejpuc/sidegame-py) (WIP)
