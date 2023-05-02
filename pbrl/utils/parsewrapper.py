#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

from pbrl.utils import UC, DotDict, bold, generateRunID, generateSeed


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
        parser.add_argument('-sd', dest='seed',
            type=int, default=None, help='Seed for random number generators'
        )

        # --- wandb --- #
        parser.add_argument('--offline', dest='offline', action='store_true', help='Offline mode (no wandb functionality)')
        parser.add_argument('-PID', dest='projectID', default='bl', help='Project ID')
        parser.add_argument('-RID', dest='runID', default=None, help='Run ID')

        # --- environment --- #
        parser.add_argument('-et', dest='envType', type=str,
            default='pixel', choices=['pixel', 'vector'], help='Observation type of the environment'
        )
        parser.add_argument('-er', dest='envRows', type=int, default=7, help='Number of rows in the environment')
        parser.add_argument('-ec', dest='envCols', type=int, default=7, help='Number of columns in the environment')
        parser.add_argument('-es', dest='envSpeed', type=float, default=1.0, help='Speed of the ball in the environment')

        # --- misc flags --- #
        parser.add_argument('-G', dest='gpu', action='store_true', help='Try to use GPU')
        parser.add_argument('-V', dest='verbose', action='store_true', help='Verbose output')
        parser.add_argument('-D', dest='debug', action='store_true', help='Print debug statements')
        parser.add_argument('-S', dest='saveModel', action='store_true', help='Save model(s) to disk')

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
        if args['runID'] is None and args['offline']:
            resolvedArgs['runID'] = generateRunID()
        if args['seed'] is None:
            resolvedArgs['seed'] = generateSeed()
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
        assert 100 <= self.args.nEpisodes <= 1e5, \
            'Number of episodes must be in [100 .. 100,000]'
        assert 1 <= self.args.nRuns <= 10, \
            'Number of runs must be in [1 .. 10]'

        # --- environment --- #
        assert 3 <= self.args.envRows <= 100, \
            'Number of rows in the environment must be in [3 .. 100]'
        assert 3 <= self.args.envCols <= 100, \
            'Number of columns in the environment must be in [3 .. 100]'
        assert 0.1 <= self.args.envSpeed <= 10, \
            'Speed of the ball in the environment must be in [0.1 .. 10]'
            

        return
