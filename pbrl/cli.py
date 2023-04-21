#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from pbrl import REINFORCEAgent
from pbrl import CatchEnvironment
from pbrl.utils import ParseWrapper, P


def main():
    argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = ParseWrapper(argParser)()

    print('arguments passed to cli.py:')
    for k, v in args.items():
        print(k, v)

    print('paths:')
    for path in P.paths:
        print(path)

    V = args.verbose

    env = CatchEnvironment()

    agent = REINFORCEAgent(
        alpha = args.alpha,
    )

    hello_world = agent()

    agent.learn(env)
    
    if V:
        print(f'{hello_world} from {agent.__class__.__name__}')
    else:
        print(hello_world)


if __name__ == '__main__':
    main()
