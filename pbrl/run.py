#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import argparse
import warnings

from pbrl.agents import REINFORCEAgent
from pbrl.agents.models import LinearModel
from pbrl.environment import CatchEnvironment
from pbrl.utils import ParseWrapper, P, DotDict, bold

import torch
import wandb


def main():
    
    config, device = _initRun()

    # set flag variables
    D, S, T = config.flags.debug, config.flags.saveModel, config.flags.trackModel
    V = not config.flags.quiet
    W = not config.flags.offline

    # initialize environment
    env = CatchEnvironment(
        observation_type = config.env.obsType,
        rows = config.env.nRows,
        columns = config.env.nCols,
        speed = config.env.speed,
        max_steps = config.env.maxSteps,
        max_misses = config.env.maxMisses,
        seed = config.exp.seed,
    )

    # initialize model
    model = LinearModel(inputSize=env.stateSize, outputSize=env.actionSize)

    # initialize agent
    agent = REINFORCEAgent(
        # torch
        model = model,
        device = device,
        # hyperparameters
        alpha = config.agent.alpha,
        beta = config.agent.beta,
        gamma = config.agent.gamma,
        delta = config.agent.delta,
        batchSize = config.agent.batchSize,
    )

    # train agent
    agent.train(env, config.exp.nEpisodes, V, D, W, T)

    if S:
        _saveModel(config, agent)

    if W:
        # tell wandb wether or not the agent has converged
        wandb.run.summary['converged'] = agent.converged
        wandb.finish()
    
    return

def _initRun() -> tuple[DotDict, torch.device]:
    """ Sets up the experiment run.

    1. parses args
    2. sets random seed for torch
    3. creates config
    4. creates paths if they don't exist already
    5. initializes wandb
    6. checks for GPU availability

    ### Returns
    `DotDict` config: the parsed config containing all information about the experiment
    ```
    projectID: str
    runID: str
    nEpisodes: int
    nRuns: int
    seed: int
    hyperparams:
        alpha: float
        beta: float
        gamma: float
        delta: float
    flags:
        verbose: bool
        debug: bool
        offline: bool
        gpu: bool
        saveModel: bool
    env:
        obsType: Literal['pixel', 'vector']
        nRows: int
        nCols: int
        speed: float
        maxSteps: int
        maxMisses: int
    ```
    `torch.device` device: the device to run on
    """
    
    # parse args
    argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config = ParseWrapper(argParser)()

    # set random seed for torch
    torch.manual_seed(config.exp.seed)

    defaultEnv = CatchEnvironment()

    config.exp.projectID = f'pbrl-{config.exp.projectID}'
    config.env.maxSteps = defaultEnv.max_steps
    config.env.maxMisses = defaultEnv.max_misses

    # create gitignored paths
    for path in P.ignored:
        path.mkdir(exist_ok=True, parents=True)

    # initialize wandb
    if not config.flags.offline:
        wandb.init(
            dir = P.wandb,
            project = config.exp.projectID,
            name = config.exp.runID,
            config = config,
        )
        # a bit hacky, but we want to take the runID that wandb generates when we initialize it
        # TODO: find a better way to do this
        config.exp.runID = wandb.run.id
        wandb.config.update(config, allow_val_change=True)

    # set device
    if config.flags.gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            warnings.warn('CUDA not found, using CPU')
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    return config, device

def _saveModel(config: DotDict, agent: REINFORCEAgent) -> None:
    """ Handles model saving logic and stdout messages. """    
    # create normal config for yaml
    normalConfig = dict(
        exp = dict(config.exp),
        agent = dict(config.agent),
        env = dict(config.env),
        flags = dict(config.flags),
    )
    # create path
    path = P.models / f'{config.exp.runID}'
    path.mkdir(parents=True, exist_ok=True)
    # save config
    if (path / 'config.yaml').exists():
        warnings.warn(f'Overwriting existing config file for run {bold(config.exp.runID)}')
    with open(path / 'config.yaml', 'w') as f:
        yaml.dump(normalConfig, f)
    # save model
    if (path / 'model.pth').exists():
        warnings.warn(f'Overwriting existing model file for run {bold(config.exp.runID)}')
    agent.saveModel(path / 'model.pth')
    # stdout messages
    print(f'Saved model to {path.parent}/{bold(path.name)}')
    print(f'To render the model, run the following command:')
    print(f'pbrl-render {config.exp.runID}')
    return

if __name__ == '__main__':
    main()
