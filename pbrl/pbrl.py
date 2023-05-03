#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from pbrl.cli import run, render, sweep

def main():
    parser = argparse.ArgumentParser(description='Policy-based Reinforcement Learning', prog='pbrl')
    subparsers = parser.add_subparsers(dest='command')

    run.addSubparser(subparsers)
    render.addSubparser(subparsers)
    sweep.addSubparser(subparsers)

    args = parser.parse_args()

    if args.command == 'run':
        run.main(args)
    elif args.command == 'render':
        render.main(args)
    elif args.command == 'sweep':
        sweep.main(args)
    else:
        raise ValueError('Invalid command')
