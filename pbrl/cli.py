#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import warnings

from pbrl import REINFORCEAgent
from pbrl.environment import CatchEnvironment
from pbrl.utils import ParseWrapper, P, DotDict
from pbrl.agents import Network

import torch
import wandb


def main():
    
    args, device = setup()
    
    env = CatchEnvironment()


    agent = REINFORCEAgent(
        # hyperparameters
        alpha = args.alpha,
        beta = args.beta,
        gamma = args.gamma,
        delta = args.delta,

        # experiment-level args
        device = device,
        R = args.render,
        V = args.verbose,
        D = args.debug,
        S = args.save_n
    )

    if args.render: load_agent_and_render(agent)

    else:  agent.train(env, args.nEpisodes)

    wandb.finish()
    
    return

def setup() -> tuple[DotDict, torch.device]:
    
    # parse args
    argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = ParseWrapper(argParser)()

    # create paths
    for path in P.paths:
        path.mkdir(exist_ok=True, parents=True)

    # setup wandb
    wandb.init(
        dir = P.wandb,
        project = f'pbrl-{args.projectID}',
        name = args.runID,
        config = dict(
            alpha = args.alpha,
            beta = args.beta,
            gamma = args.gamma,
            delta = args.delta,
            nEpisodes = args.nEpisodes,
            nRuns = args.nRuns
        )
    )

    # set device
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            warnings.warn('CUDA not found, using CPU')
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    return args, device


def load_agent_and_render(agent):
    # Load the saved network
    rows = 7 
    columns =7 
    speed = 1.0
    max_steps = 250
    max_misses = 10
    observation_type = 'pixel' # 'vector'
    seed = None
    
    # Initialize environment and Q-array
    env = CatchEnvironment(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)
        
    agent.network = Network(98, 3).to(agent.device)

    agent.network.load_state_dict(torch.load('network.pth'))

# Initialize the agent with the loaded network

    for episode in range(10):
        
        # (re)set environment
        state: torch.Tensor = agent.castState(env.reset())
        done = False

        while not done:
            # render environment
            env.render()
            # get action probability distribution
            dist = torch.distributions.Categorical(agent.network(state))
            # sample action
            action: int = dist.sample().item()
            # take action
            next_state, reward, done, _ = env.step(action)
            next_state = agent.castState(next_state)
        
            state = next_state

    # Close the environment
    env.close()


if __name__ == '__main__':
    main()
