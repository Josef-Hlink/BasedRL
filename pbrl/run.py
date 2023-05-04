#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import yaml
from typing import Type

from pbrl.agents import PBAgent, REINFORCEAgent, ActorCriticAgent
from pbrl.agents.models import ActorModel, CriticModel
from pbrl.environment import CatchEnvironment
from pbrl.utils import ParseWrapper, P, UC, DotDict, bold

import torch
import wandb


def main():
    
    config, device = _initRun()

    # set flag variables
    flags = config.flags
    Q, D, S, T, W = flags.quiet, flags.debug, flags.saveModels, flags.trackModels, flags.wandb

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

    # initialize actor model
    actor = ActorModel(inputSize=env.stateSize, outputSize=env.actionSize)
    
    # get correct agent type
    Agent = Type[PBAgent]

    if config.exp.agentType == 'RF':
        Agent = REINFORCEAgent
        critic = None
    elif config.exp.agentType == 'AC':
        Agent = ActorCriticAgent
        critic = CriticModel(inputSize=env.stateSize, outputSize=1)
    else:
        raise NotImplementedError(f'Agent type {config.exp.agentType} not implemented')

    # initialize agent
    agent = Agent(
        # hyperparameters
        alpha = config.agent.alpha,
        beta = config.agent.beta,
        gamma = config.agent.gamma,
        delta = config.agent.delta,
        batchSize = config.agent.batchSize,
        # torch
        device = device,
        actor = actor,
        critic = critic,
    )

    # train agent
    if not Q: print(UC.hd * 80)
    agent.train(env, config.exp.nTrainEps, Q, D, W, T)

    # evaluate agent
    evalReward = agent.evaluate(env, config.exp.nEvalEps)
    if not Q: print(f'avg. evaluation reward: {evalReward:.2f}')
    
    if S: _saveModels(config, agent)

    if not Q: print(UC.hd * 80)

    if W:
        wandb.run.summary['converged'] = agent.converged
        wandb.run.summary['evalReward'] = evalReward
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
    exp:
        projectID: str
        runID: str
        agentType: str
        nTrainEps: int
        nEvalEps: int
        seed: int
    agent:
        alpha: float
        beta: float
        gamma: float
        delta: float
        batchSize: int
    env:
        obsType: str
        nRows: int
        nCols: int
        speed: float
        maxSteps: int
        maxMisses: int
    flags:
        quiet: bool
        debug: bool
        wandb: bool
        saveModels: bool
        trackModels: bool
    ```
    `torch.device` device: the device to run on
    """
    
    # parse args
    argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config = ParseWrapper(argParser)()

    # set random seed for torch
    torch.manual_seed(config.exp.seed)

    config.exp.projectID = f'pbrl-{config.exp.projectID}'

    # create gitignored paths
    for path in P.ignored:
        path.mkdir(exist_ok=True, parents=True)

    # initialize wandb
    if config.flags.wandb:
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
            print(f'{bold("Warning")}: CUDA not found, using CPU')
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    return config, device

def _saveModels(config: DotDict, agent: REINFORCEAgent) -> None:
    """ Handles model saving logic and stdout messages. """    
    # create normal config for yaml
    # create path
    path = P.models / f'{config.exp.runID}'
    path.mkdir(parents=True, exist_ok=True)
    # save config
    if (path / 'config.yaml').exists():
        print(f'{bold("Warning")}: Overwriting existing config file for run {bold(config.exp.runID)}')
    with open(path / 'config.yaml', 'w') as f:
        yaml.dump(config.toDict(), f)
    # save models
    if (path / 'actor.pth').exists():
        print(f'{bold("Warning")}: Overwriting existing model file for run {bold(config.exp.runID)}')
    agent.saveModels(path)
    # stdout messages
    print(f'Saved model to {path.parent}/{bold(path.name)}')
    print(f'To render the model, run the following command:')
    print(f'pbrl-render {config.exp.runID}')
    return

if __name__ == '__main__':
    main()
