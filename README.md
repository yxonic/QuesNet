# DL Boilerplate

This project provides a good starting point for every deep learning experiment, and free myself from coding for features like configure/save/load a model, resume a training process, testing on each snapshots, etc.

The project is a complete deep hand-written digit recognition example with models written in [PyTorch](http://pytorch.org), but its design makes it easy to convert to other frameworks or applications, as models and training parts are seperate from the main setup-and-run logic.

## Design

There are two main design ideas behind this template: the workspace concept, and the configure-and-run workflow; both are inspired by popular building systems like [Automake](https://www.gnu.org/software/automake/), [CMake](https://cmake.org) and so on. The whole structure is like this:

<p align=center><img width="80%" src="docs/static/img/structure.png" /></p>

A workspace is where an experiment takes place, and where logs, results and snapshots are saved. In practice, we often want to try different setups and record them all, being able to resume training, do more tests, etc., on each setup we choose. With setups run under seperate workspaces, everything becomes straightforward.

Setups are saved as `<model>.json` file inside each workspace, with the `config` command. After that, we can run `train` or `test` command, which loads the configuration of that workspace, builds the model accordingly, and do training/evaluation stuff.

## Usage

## How to Extend