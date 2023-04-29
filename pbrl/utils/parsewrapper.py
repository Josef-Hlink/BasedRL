#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

from pbrl.utils import UC, bold, DotDict


class ParseWrapper:
    """ Adds and parses command line arguments for the main script. """
    def __init__(self, parser: ArgumentParser):
        """ Adds arguments to the passed parser object and parses them. """
        
        parser.add_argument('-a', dest='alpha', 
            type=float, default=0.1, help='Learning rate'
        )
        parser.add_argument('-g', dest='gamma',
            type=float, default=0.9, help='Discount factor'
        )
        parser.add_argument('-b', dest='beta',
            type=float, default=0.9, help='Exploration rate')

        parser.add_argument('-G', dest='gpu', action='store_true', help='Try to use GPU')
        parser.add_argument('-V', dest='verbose', action='store_true', help='Verbose output')
        parser.add_argument('-D', dest='debug', action='store_true', help='Print debug statements')
        parser.add_argument('-R', dest='render', action='store_true', help='Render environment')
        parser.add_argument('-S', dest='save', action='store_true', help='Save plots')
        

        self.defaults = ParseWrapper.resolveDefaultNones(vars(parser.parse_args([])))
        self.args = ParseWrapper.resolveDefaultNones(vars(parser.parse_args()))
        self.validate()
        return

    def __call__(self) -> DotDict[str, any]:
        """ Returns the parsed and processed arguments as a standard dictionary. """
        if self.args.verbose:
            print(UC.hd * 80)
            print('Experiment will be ran with the following parameters:')
            for arg, value in self.args.items():
                if self.defaults[arg] != value:
                    print(f'{bold(arg):>28} {UC.vd} {value}')
                else:
                    print(f'{arg:>20} {UC.vd} {value}')
            print(UC.hd * 80)
        return self.args

    @staticmethod
    def resolveDefaultNones(args: dict[str, any]) -> DotDict[str, any]:
        """ Resolves default values for exploration value and run ID. """
        resolvedArgs = args.copy()
        ...
        return DotDict(resolvedArgs)

    def validate(self) -> None:
        """ Checks the validity of all passed values for the experiment. """
        assert 0 <= self.args.alpha <= 1, \
            f'Learning rate {UC.a} must be in [0, 1]'
        return
