#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import warnings

from pbrl import REINFORCEAgent
from pbrl.environment import CatchEnvironment
from pbrl.utils import ParseWrapper, P, DotDict

import torch


def main():
    
    args, device = setup()
    
    env = CatchEnvironment()

    agent = REINFORCEAgent(
        # hyperparameters
        alpha = args.alpha,
        gamma = args.gamma,

        # experiment-level args
        device = device,
        R = args.render,
        V = args.verbose,
        D = args.debug,
    )

    agent.learn(env)
    
    return

def setup() -> tuple[DotDict, torch.device]:
    
    # parse args
    argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = ParseWrapper(argParser)()

    # create paths
    for path in P.paths:
        path.mkdir(exist_ok=True, parents=True)

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


if __name__ == '__main__':
    main()
