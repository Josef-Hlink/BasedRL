#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

from pbrl.utils import UC, bold, DotDict


class ParseWrapper:
    """ Adds and parses command line arguments for the main script. """
    def __init__(self, parser: ArgumentParser):
        """ Adds arguments to the passed parser object and parses them. """
        
        # --- hyperparameters --- #
        parser.add_argument('-a', dest='alpha', 
            type=float, default=0.0001, help=f'Learning rate {bold(UC.a)}'
        )
        parser.add_argument('-b', dest='beta',
            type=float, default=0.1, help=f'Entropy regularization coefficient {bold(UC.b)}'
        )
        parser.add_argument('-g', dest='gamma',
            type=float, default=0.9, help=f'Discount factor {bold(UC.g)}'
        )
        parser.add_argument('-d', dest='delta',
            type=float, default=0.995, help=f'Decay rate {bold(UC.d)} for learning rate {bold(UC.a)}'
        )
        
        # --- experiment-level args --- #
        parser.add_argument('-ne', dest='nEpisodes',
            type=int, default=2500, help='Number of episodes to train for'
        )
        parser.add_argument('-nr', dest='nRuns',
            type=int, default=1, help='Number of runs to average over'
        )

        # --- wandb --- #
        parser.add_argument('--offline', dest='offline',
            default=False, action='store_true', help='Offline mode (no wandb functionality)'
        )
        parser.add_argument('-PID', dest='projectID', default='bl', help='Project ID')
        parser.add_argument('-RID', dest='runID', default=None, help='Run ID')

        # --- misc flags --- #
        parser.add_argument('-G', dest='gpu', action='store_true', help='Try to use GPU')
        parser.add_argument('-V', dest='verbose', action='store_true', help='Verbose output')
        parser.add_argument('-D', dest='debug', action='store_true', help='Print debug statements')
        parser.add_argument('-S', dest='saveNet', action='store_true', help='Save network(s) to disk')
        parser.add_argument('-R', dest='render', action='store_true', help='Render environment')


        # --- parsing --- #
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
        
        # --- hyperparameters --- #
        assert 0 < self.args.alpha <= 1, \
            f'Learning rate {UC.a} must be in (0 .. 1]'
        assert 0 <= self.args.beta <= 1, \
            f'Entropy regularization coefficient {UC.b} must be in [0 .. 1]'
        assert 0 <= self.args.gamma <= 1, \
            f'Discount factor {UC.g} must be in [0 .. 1]'
        
        # --- experiment-level args --- # 
        assert 0 < self.args.nEpisodes <= 1e5, \
            'Number of episodes must be in (0 .. 100,000]'
        assert 0 < self.args.nRuns <= 10, \
            'Number of runs must be in (0 .. 10]'
        
        # --- misc flags --- #
        if self.args.render:
            assert self.args.saveNet == True, \
                'Cannot render without saving network(s) to disk'

        return
