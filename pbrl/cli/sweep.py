#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import yaml
from typing import Type
from os import cpu_count

from pbrl.models import ActorModel, CriticModel
from pbrl.agents import PBAgent, REINFORCEAgent, ActorCriticAgent
from pbrl.environment import CatchEnvironment
from pbrl.utils import UC, P, DotDict, bold

import torch
import wandb


def main():

    # create gitignored paths
    for path in P.ignored:
        path.mkdir(exist_ok=True, parents=True)

    commandHelp = f"""
    | {bold("init")}: initialize a sweep from a config file
    and generate a bash command to run the sweep distributed in parallel
    over the number of CPU cores available minus two |
    | {bold("run")}: run a sweep from a sweep ID |
    """
    argHelp = f"""
    | {bold("init")}: name of the config file with hyperparameters |
    | {bold("run")}: sweep ID |
    """

    # parse CLI arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('command', type=str, choices=['init', 'run'], help=commandHelp)
    parser.add_argument('arg', type=str, help=argHelp)
    parser.add_argument('-c', '--count', dest='count',
        type=int, default=1, help='Number of runs for each wandb agent'
    )
    args = parser.parse_args()

    # initializing sweep
    if args.command == 'init':
        configFileName = args.arg
        initSweep(configFileName)
    
    # running sweep
    elif args.command == 'run':
        sweepID = args.arg
        runSweep(sweepID, args.count)

    return

def generateCommand(sweepID: str, totalRuns: int, count: int) -> str:
    """ Generate a bash command using `xargs` that runs multiple agents in parallel.
    
    Example for 10 runs per agent, total 200 runs, 16 available CPU cores (so 14 workers):
    ```bash
    seq 1 20 | xargs -I {} -P 14 sh -c 'pbrl-sweep run <sweepID> -c 10'
    ```
    Example for count = 1 (no -c parameter will be passed):
    ```bash
    seq 1 200 | xargs -I {} -P 14 sh -c 'pbrl-sweep run <sweepID>'
    ```
    """
    nWorkers = cpu_count() - 2
    command = ""
    command += f"seq 1 {totalRuns//count} | xargs -I {{}} -P {nWorkers} sh -c"
    command += f" 'pbrl-sweep run {sweepID}"
    if count > 1: command += f" -c {count}'"
    else: command += "'"
    return command

def initSweep(configFileName: str) -> None:
    """ Initializes a sweep from a config file.
    Will also call `generateCommand` to generate a bash command to run the sweep.
    """

    # load default config from BasedRL/pbrl/defaults.yaml
    with open(P.root / 'pbrl' / 'defaults.yaml', 'r') as f:
        defaultConfig = DotDict(yaml.load(f, Loader=yaml.FullLoader))

    # load sweep config from BasedRL/sweeps/<configFileName>.yaml
    with open(P.sweeps / f'{configFileName}.yaml', 'r') as f:
        customConfig = DotDict(yaml.load(f, Loader=yaml.FullLoader))

    # avoid having sweep name being None
    assert customConfig.sweep['name'] is not None, f'No sweep name found in {configFileName}.yaml'

    # merge default and custom configs
    config = merge(defaultConfig, customConfig)

    print(f'{bold("Initializing sweep with the following parameters")}:')
    print(config)

    # initialize sweep config
    sweepConfig = dict(
        name = config.sweep.name,
        method = config.sweep.method,
        metric = dict(
            name = config.sweep.metric,
            goal = 'maximize',
        ),
    )

    # add parameters to sweep config
    sweepConfig['parameters'] = parseConfig(config)

    sweepID = wandb.sweep(
        sweepConfig,
        project = f'pbrl-sweeps',
    )

    # generate and print bash command
    print(UC.hd * 80)
    print('To start multiple agents for this sweep, run the following command:')
    print(generateCommand(sweepID, config.sweep.totalRuns, config.sweep.count))
    print(UC.hd * 80)

    return

def runSweep(sweepID: str, count: int) -> None:
    """ Run a sweep from a sweep ID. """
    wandb.agent(
        sweep_id = sweepID,
        function = performSingleRun,
        count = count,
        project = f'pbrl-sweeps',
    )
    return

def performSingleRun():
    """ Perform a single run of the sweep. """

    with wandb.init(dir=P.wandb):

        # load config from wandb
        config = DotDict(
            exp = DotDict(wandb.config.exp),
            agent = DotDict(wandb.config.agent),
            env = DotDict(wandb.config.env),
        )

        # initialize environment
        env = CatchEnvironment(
            observation_type = config.env.obsType,
            rows = config.env.nRows,
            columns = config.env.nCols,
            speed = config.env.speed,
            max_steps = config.env.maxSteps,
            max_misses = config.env.maxMisses,
            seed = None,
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
            bootstrap = config.agent.bootstrap,
            baselineSubtraction = config.agent.baselineSubtraction,
            # torch
            device = torch.device('cpu'),
            actor = actor,
            critic = critic,
        )

        # train agent
        nEpisodes = agent.train(env, config.exp.budget, Q=True, D=False, W=True)

        # evaluate agent
        evalReward = agent.evaluate(env, config.exp.nEvalEps)

        # log results
        wandb.run.summary['convergedAt'] = nEpisodes if agent.converged else None
        wandb.run.summary['evalReward'] = evalReward
 
    return

def parseConfig(config: DotDict) -> dict:
    """ Make config compatible with wandb sweep format.
    1. Puts "parameters" before all level-1 values.
    2. Calls `parseParameters` on all level-2 values.

    Example:
    ```yaml
    exp:
      t: 1
    agent:
      a: 0.1
      b: 0.2
    env:
      x: [1, 2]
    sweep:
      m: hello
    ```
    becomes
    ```yaml
    exp:
      parameters:
        t:
          value: 1
    agent:
      parameters:
        a:
          value: 0.1
        b:
          value: 0.2
    env:
      parameters:
        x:
          values: [1, 2]
    sweep:
      parameters:
        m:
          value: hello
    ```
    """
    parsedConfig = DotDict()
    for key, value in config.items():
        parsedConfig[key] = DotDict(parameters=parseParameters(DotDict(value)))
    return parsedConfig.toDict()

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
    defaultConfig.sweep.update(customConfig.sweep) if customConfig.sweep is not None else defaultConfig.sweep
    
    config.exp = DotDict(defaultConfig.exp)
    config.agent = DotDict(defaultConfig.agent)
    config.env = DotDict(defaultConfig.env)
    config.sweep = DotDict(defaultConfig.sweep)
    
    return config


if __name__ == '__main__':
    main()
