#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import yaml
import random

from pbrl.agents.models import LinearModel
from pbrl.agents import REINFORCEAgent
from pbrl.environment import CatchEnvironment
from pbrl.utils import P, DotDict, generateID, generateSeed

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

    # load parameters from <config>.yaml file
    with open(P.sweeps / f'{args.file}.yaml', 'r') as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    # print parameters
    print(f'Sweeping over {args.file} {args.count} times:')
    for key, value in parameters.items():
        try: print(f'{key}: {value["value"]}')
        except KeyError: print(f'{key}: {value["values"]}')

    # set seed
    # NOTE: this seed will appear to be the same for all runs in the sweep
    #       but it will actually be different for each run
    #       it is only used to set the RNG, from which we sample the actual seeds for individual runs
    if parameters['seed']['value'] is None:
        parameters['seed']['value'] = generateSeed()

    # set random number generator
    global RNG
    RNG = random.Random(parameters['seed']['value'])

    # add parameters to sweep config
    sweepConfig['parameters'] = parameters

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
        config = DotDict(wandb.config)

        # fix seed
        seed = RNG.randint(0, 1000000)
        torch.manual_seed(seed)

        # instantiate environment
        env = CatchEnvironment(
            observation_type = config.obsType,
            rows = config.nRows,
            columns = config.nCols,
            speed = config.speed,
            max_steps = config.maxSteps,
            max_misses = config.maxMisses,
            seed = seed,
        )

        # instantiate model
        model = LinearModel(inputSize=env.stateSize, outputSize=env.actionSize)

        # initialize agent
        agent = REINFORCEAgent(
            # torch
            model = model,
            device = torch.device('cpu'),
            # hyperparameters
            alpha = config.alpha,
            beta = config.beta,
            gamma = config.gamma,
            delta = config.delta,
        )

        # train agent
        agent.train(env, config.nEpisodes, V=False, D=False, W=True)

        # tell wandb wether or not the agent has converged
        wandb.run.summary['converged'] = agent.converged
 
    return


if __name__ == '__main__':
    main()
