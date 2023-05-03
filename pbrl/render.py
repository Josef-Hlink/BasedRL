#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json

from pbrl.utils import P, DotDict
from pbrl.environment import CatchEnvironment
from pbrl.agents.models import LinearModel
from pbrl.agents import REINFORCEAgent

import torch


def main():
    """ Entry point for the program. """

    argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # basics
    argParser.add_argument('runID', help='ID of the run to render')
    argParser.add_argument('-n', dest='nEpisodes', default=1, help='Number of episodes to run for')
    # environment
    argParser.add_argument('-r', dest='nRows', type=int, default=None,
        help='Number of rows in the environment (overrides config)'
    )
    argParser.add_argument('-c', dest='nCols', type=int, default=None,
        help='Number of columns in the environment (overrides config)'
    )
    argParser.add_argument('-s', dest='speed', type=float, default=None,
        help='Speed of the ball in the environment (overrides config)'
    )

    args = DotDict(vars(argParser.parse_args()))

    renderTrainedAgent(args.runID, args.nEpisodes, args.nRows, args.nCols, args.speed)
    return

def renderTrainedAgent(
    runID: str, nEpisodes: int,
    nRows: int = None, nCols: int = None, speed: float = None
) -> None:
    """ Loads the agent's model from disk and renders it.
    
    Args:
        `str` runID: name of the run to load
        `int` nEpisodes: number of episodes to run for
    """
    
    # load config to see what env and agent to use
    with open(P.models / runID / 'config.json', 'r') as f:
        config = json.load(f)

    # override config where specified
    nRows = nRows or config['env']['nRows']
    nCols = nCols or config['env']['nCols']
    speed = speed or config['env']['speed']

    # check validity of overrides
    if config['env']['obsType'] == 'pixel':
        assert nRows == config['env']['nRows'], 'Cannot change number of rows in pixel observation'
        assert nCols == config['env']['nCols'], 'Cannot change number of columns in pixel observation'

    # load environment
    env = CatchEnvironment(
        rows=nRows,
        columns=nCols,
        speed=speed,
        max_steps=config['env']['maxSteps'],
        max_misses=config['env']['maxMisses'],
        observation_type=config['env']['obsType'],
        seed=config['seed'],
    )

    # create dummy model to be overwritten later
    model = LinearModel(inputSize=env.stateSize, outputSize=env.actionSize)

    # load agent
    # TODO: add other kinds of agents
    agent = REINFORCEAgent(
        model = model,
        device = torch.device('cpu'),
        alpha = config['hyperparams']['alpha'],
        beta = config['hyperparams']['beta'],
        gamma = config['hyperparams']['gamma'],
        delta = config['hyperparams']['delta']
    )

    # set the agent's model to the one loaded from disk
    agent.loadModel(P.models / runID / 'model.pth')

    _render(env, agent, nEpisodes)

    return

def _render(env: CatchEnvironment, agent: REINFORCEAgent, nEpisodes: int) -> None:
    """ Renders the agent's behaviour for a given number of episodes. """
    if nEpisodes == 1:
        print('Rendering 1 episode...')
    else:
        print(f'Rendering {nEpisodes} episodes...')
    
    for _ in range(nEpisodes):
        state, done = env.reset(), False
        while not done:
            action = agent.chooseAction(state)
            state, _, done, _ = env.step(action)
            env.render()
    return


if __name__ == '__main__':
    main()