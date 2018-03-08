# DL Boilerplate

This project provides a good starting point for every deep learning experiment. It frees myself from coding for features like configure/save/load a model, resume a training process, testing on each snapshots, tracking progress, logging, etc.

## Design

There are two main design ideas behind this template: the workspace concept, and the configure-and-run workflow; both are inspired by popular building systems like [Automake](https://www.gnu.org/software/automake/), [CMake](https://cmake.org) and so on.

A workspace is where an experiment takes place, and where logs, results and snapshots are saved. In practice, we often want to try different setups and record them all, being able to resume training, do more tests, etc., on each setup we choose. With setups run under seperate workspaces, everything becomes straightforward.

Model configurations are saved as `config.toml` file inside each workspace, by the `config` command. After that, we can run `train` or `test` command, which loads the configuration in that workspace, builds the model accordingly, and does training/evaluation stuff.

## Usage

## How to Extend