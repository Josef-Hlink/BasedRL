#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from pbrl import DummyAgent
from pbrl.utils import ParseWrapper

def main():
    argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = ParseWrapper(argParser)()

    print('arguments passed to cli.py:')
    for k, v in args.items():
        print(k, v)

    V = args.verbose

    agent = DummyAgent(
        alpha = args.alpha,
    )
    hello_world = agent()
    if not V:
        print(hello_world)
    if V:
        print(f'{hello_world} from {agent.__class__.__name__}')


if __name__ == '__main__':
    main()
