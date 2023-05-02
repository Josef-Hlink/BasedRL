#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import warnings

from pbrl.agents import REINFORCEAgent
from pbrl.agents.models import LinearModel
from pbrl.environment import CatchEnvironment
from pbrl.utils import ParseWrapper, P, DotDict, bold

import torch
import wandb


def main():
    
    config, device = setup()

    env = CatchEnvironment(
        observation_type = config.env.obsType,
        rows = config.env.nRows,
        columns = config.env.nCols,
        speed = config.env.speed,
        max_steps = config.env.maxSteps,
        max_misses = config.env.maxMisses,
        seed = config.seed,
    )

    # hardcode model for now
    model = LinearModel(inputSize=env.stateSize, outputSize=env.actionSize)

    agent = REINFORCEAgent(
        # torch
        model = model,
        device = device,
        
        # hyperparameters
        alpha = config.hyperparams.alpha,
        beta = config.hyperparams.beta,
        gamma = config.hyperparams.gamma,
        delta = config.hyperparams.delta,

        # flags
        V = config.flags.verbose,
        D = config.flags.debug,
        W = not config.flags.offline,
    )

    agent.train(env, config.nEpisodes)

    if config.flags.saveModel:
        path = P.models / f'{config.runID}'
        path.mkdir(parents=True, exist_ok=True)
        # save config
        if (path / 'config.json').exists():
            warnings.warn(f'Overwriting existing config file for run {bold(config.runID)}')
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        # save model
        if (path / 'model.pth').exists():
            warnings.warn(f'Overwriting existing model file for run {bold(config.runID)}')
        agent.saveModel(path / 'model.pth')

        print(f'Saved model to {bold(path)}\nTo render the model, run the following command:\nrenderrun {config.runID}')

    # tell wandb wether or not the agent has converged
    wandb.run.summary['converged'] = agent.converged
    wandb.finish()
    
    return

def setup() -> tuple[DotDict, torch.device]:
    """ Sets up the experiment.

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
    args = ParseWrapper(argParser)()

    # set random seed for torch
    torch.manual_seed(args.seed)

    defaultEnv = CatchEnvironment()

    # create config
    config = DotDict(dict(
        projectID = f'pbrl-{args.projectID}',
        runID = args.runID,
        nEpisodes = args.nEpisodes,
        nRuns = args.nRuns,
        seed = args.seed,
        hyperparams = DotDict(dict(
            alpha = args.alpha,
            beta = args.beta,
            gamma = args.gamma,
            delta = args.delta,
        )),
        flags = DotDict(dict(
            verbose = args.verbose,
            debug = args.debug,
            offline = args.offline,
            gpu = args.gpu,
            saveModel = args.saveModel,
        )),
        # TODO: add (some of the) env config options to args
        env = DotDict(dict(
            obsType = defaultEnv.observation_type,
            nRows = defaultEnv.rows,
            nCols = defaultEnv.columns,
            speed = defaultEnv.speed,
            maxSteps = defaultEnv.max_steps,
            maxMisses = defaultEnv.max_misses,
        )),
    ))

    # create paths
    for path in P.paths:
        path.mkdir(exist_ok=True, parents=True)

    # initialize wandb
    if not args.offline:
        wandb.init(
            dir = P.wandb,
            project = f'pbrl-{args.projectID}',
            name = args.runID,
            config = config,
        )
        # a bit hacky, but we want to take the runID that wandb generates when we initialize it
        config.update(runID = wandb.run.id)
        wandb.config.update(config, allow_val_change=True)

    # set device
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            warnings.warn('CUDA not found, using CPU')
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    return config, device


if __name__ == '__main__':
    main()
