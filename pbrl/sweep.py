#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import yaml
from typing import Type
import random

from pbrl.agents.models import ActorModel, CriticModel
from pbrl.agents import PBAgent, REINFORCEAgent, ActorCriticAgent
from pbrl.environment import CatchEnvironment
from pbrl.utils import P, DotDict, generateID, generateSeed, bold

import torch
import wandb


def main():

    # create gitignored paths
    for path in P.ignored:
        path.mkdir(exist_ok=True, parents=True)

    # parse CLI arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('file', type=str, help='Name of the config file with hyperparameters')
    parser.add_argument('-PID', dest='projectID', type=str, default='glob', help='Project ID')
    parser.add_argument('-SID', dest='sweepID', type=str, default=None, help='Run ID')
    parser.add_argument('-c', '--count', dest='count', type=int, default=5, help='Number of runs in sweep')
    parser.add_argument('-G', '--gpu', dest='gpu', action='store_true', help='Try to use GPU')
    args = parser.parse_args()

    # initialize sweep config
    sweepConfig = dict(
        name = args.sweepID if args.sweepID is not None else generateID(),
        method = 'bayes',
        metric = dict(
            name = 'reward',
            goal = 'maximize',
        ),
    )

    # load default config from BasedRL/pbrl/defaults.yaml file
    with open(P.root / 'pbrl' / 'defaults.yaml', 'r') as f:
        defaultConfig = DotDict(yaml.load(f, Loader=yaml.FullLoader))

    # load sweep config from BasedRL/sweeps/<config>.yaml file
    with open(P.sweeps / f'{args.file}.yaml', 'r') as f:
        customConfig = DotDict(yaml.load(f, Loader=yaml.FullLoader))

    # merge default and custom configs
    config = merge(defaultConfig, customConfig)

    print(f'{bold("SWEEPING OVER")}:')
    print(config)

    # set seed
    # NOTE: this seed will appear to be the same for all runs in the sweep
    #       but it will actually be different for each run
    #       it is only used to set the RNG, from which we sample the actual seeds for individual runs
    if config.exp.parameters.seed.value is None:
        config.exp.parameters.seed.value = generateSeed()

    # set device
    global DEVICE
    if args.gpu:
        if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
        else:
            print(f'{bold("Warning")}: CUDA not found, using CPU')
            DEVICE = torch.device('cpu')
    else:
        DEVICE = torch.device('cpu')

    # set random number generator
    global RNG
    RNG = random.Random(config.exp.parameters.seed.value)

    # add parameters to sweep config
    sweepConfig['parameters'] = config.toDict()

    # initialize and run sweep
    wandb.agent(wandb.sweep(
        sweepConfig,
        project = f'pbrl-sweeps-{args.projectID}'),
        function = run,
        count = args.count
    )

def run():
    """ Perform a single run of the sweep """

    with wandb.init(dir=P.wandb):

        # load config from wandb
        config = DotDict(
            exp = DotDict(wandb.config.exp),
            agent = DotDict(wandb.config.agent),
            env = DotDict(wandb.config.env),
        )

        # fix seed
        seed = RNG.randint(0, 1000000)
        torch.manual_seed(seed)

        env = CatchEnvironment(
            observation_type = config.env.obsType,
            rows = config.env.nRows,
            columns = config.env.nCols,
            speed = config.env.speed,
            max_steps = config.env.maxSteps,
            max_misses = config.env.maxMisses,
            seed = seed,
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
            device = DEVICE,
            actor = actor,
            critic = critic,
        )

        # train agent
        agent.train(env, config.exp.nTrainEps, Q=True, D=False, W=True)

        # evaluate agent
        evalReward = agent.evaluate(env, config.exp.nEvalEps)

        # log results
        wandb.run.summary['converged'] = agent.converged
        wandb.run.summary['evalReward'] = evalReward
 
    return

def parseConfig(config: DotDict) -> DotDict:
    """ Make config compatible with wandb sweep format.
    Just put "parameters" before all level-1 values.
    Example:
    ```
    exp:
      t: 1
      n: 2
    agent:
      a: 0.1
      b: 0.2

    becomes

    exp:
      parameters:
        t: 1
        n: 2
    agent:
      parameters:
        a: 0.1
        b: 0.2
    ```
    """
    parsedConfig = DotDict()
    for key, value in config.items():
        parsedConfig[key] = DotDict(parameters=parseParameters(DotDict(value)))
    return parsedConfig

def parseParameters(parameters: DotDict) -> DotDict:
    """ Make parameters compatible with wandb sweep format.
    For all values that are lists, put "values" before it.
    For all values that are dicts, keep them as is.
    For all values that are not lists, put "value" before it.
    """
    parsedParameters = DotDict()
    for key, value in parameters.items():
        if isinstance(value, list):
            parsedParameters[key] = DotDict(values=value)
        elif isinstance(value, dict):
            parsedParameters[key] = DotDict(value)
        else:
            parsedParameters[key] = DotDict(value=value)
    return parsedParameters

def merge(defaultConfig: DotDict, customConfig: DotDict) -> DotDict:
    """ Merge default and custom configs.
    If a key is present in both, the value from customConfig is used.
    """
    
    config = DotDict()

    defaultConfig.exp.update(customConfig.exp) if customConfig.exp is not None else defaultConfig.exp
    defaultConfig.agent.update(customConfig.agent) if customConfig.agent is not None else defaultConfig.agent
    defaultConfig.env.update(customConfig.env) if customConfig.env is not None else defaultConfig.env
    
    config.exp = DotDict(defaultConfig.exp)
    config.agent = DotDict(defaultConfig.agent)
    config.env = DotDict(defaultConfig.env)
    
    return parseConfig(config)


if __name__ == '__main__':
    main()
