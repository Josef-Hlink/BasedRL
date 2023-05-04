#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from pbrl.utils import UC, DotDict, bold, generateID, generateSeed


class ParseWrapper:
    """ Adds and parses command line arguments for the pbrl-run script. """
    def __init__(self, parser: argparse.ArgumentParser):
        """ Adds arguments to the passed parser object and parses them. """
        
        parser.description = 'Run an experiment with a PBRL agent.'

        # --- experiment --- #
        parser.add_argument('-PID', dest='projectID',
            type=str, default='glob', help='Project ID'
        )
        parser.add_argument('-RID', dest='runID',
            type=str, default=None, help='Run ID'
        )
        parser.add_argument('-at', dest='agentType',
            type=str, default='RF', choices=['RF', 'AC'], help='Type of agent'
        )
        parser.add_argument('-te', dest='nTrainEps',
            type=int, default=2500, help='Number of episodes to train for'
        )
        parser.add_argument('-ee', dest='nEvalEps',
            type=int, default=10, help='Number of episodes to evaluate on'
        )
        parser.add_argument('-sd', dest='seed',
            type=int, default=None, help='Seed for random number generators'
        )
        
        # --- agent --- #
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
        parser.add_argument('-bs', dest='batchSize',
            type=int, default=8, help='Batch size for training'
        )

        # --- environment --- #
        parser.add_argument('-et', dest='envType', type=str,
            default='pixel', choices=['pixel', 'vector'], help='Observation type of the environment'
        )
        parser.add_argument('-er', dest='envRows', type=int, default=7, help='Number of rows in the environment')
        parser.add_argument('-ec', dest='envCols', type=int, default=7, help='Number of columns in the environment')
        parser.add_argument('-es', dest='envSpeed', type=float, default=1.0, help='Speed of the ball in the environment')

        # --- flags --- #
        parser.add_argument('-G', dest='gpu', action='store_true', help='Try to use GPU')
        parser.add_argument('-Q', dest='quiet', action='store_true', help='Mute all output')
        parser.add_argument('-D', dest='debug', action='store_true', help='Print debug statements')
        parser.add_argument('-W', dest='wandb', action='store_true', help='Use wandb for logging')
        parser.add_argument('-S', dest='saveModel', action='store_true', help='Save model(s) to disk')
        parser.add_argument('-T', dest='trackModel', action='store_true', help='Track model in wandb')

        # --- parsing --- #
        self.defaultConfig = self.createConfig(parser.parse_args([]))
        self.config = self.createConfig(parser.parse_args())
        self.resolveDefaultNones()
        self.validate()
        return

    def __call__(self) -> DotDict[str, any]:
        """ Returns the parsed and processed arguments as a standard dictionary. """
        if not self.config.flags.quiet:
            self.printConfig()
        return self.config

    def createConfig(self, args: argparse.Namespace) -> DotDict:
        """ Creates a nested DotDict config from the passed parsed arguments. """
        config = DotDict(
            exp = DotDict(
                agentType = args.agentType,
                projectID = args.projectID,
                runID = args.runID,
                nTrainEps = args.nTrainEps,
                nEvalEps = args.nEvalEps,
                seed = args.seed,
            ),
            agent = DotDict(
                alpha = args.alpha,
                beta = args.beta,
                gamma = args.gamma,
                delta = args.delta,
                batchSize = args.batchSize,
            ),
            env = DotDict(
                obsType = args.envType,
                nRows = args.envRows,
                nCols = args.envCols,
                speed = args.envSpeed,
            ),
            flags = DotDict(
                gpu = args.gpu,
                quiet = args.quiet,
                debug = args.debug,
                wandb = args.wandb,
                saveModel = args.saveModel,
                trackModel = args.trackModel,
            ),
        )       
        return config

    def printConfig(self) -> None:
        """ Prints the config obtained from parsed args. """
        print(UC.tl + UC.hd * 78 + UC.tr)
        for section, sectionDict in self.config.items():
            line = f'{UC.vd} {section.upper()}:'
            print(line + ' ' * (79 - len(line)) + UC.vd)
            for arg, value in sectionDict.items():
                if self.defaultConfig[section][arg] != value:
                    line = f'{UC.vd}     {bold(arg)}: {value}'
                    nSpaces = 79 - len(line) + len(r'\033[1mx')
                else:
                    line = f'{UC.vd}     {arg}: {value}'
                    nSpaces = 79 - len(line)
                print(line + ' ' * nSpaces + UC.vd)
        print(UC.bl + UC.hd * 78 + UC.br)
        return

    def resolveDefaultNones(self) -> None:
        """ Resolves default values for exploration value and run ID. """
        if self.config.exp.runID is None and not self.config.flags.wandb:
            self.config.exp.runID = generateID()
        if self.config.exp.seed is None:
            self.config.exp.seed = generateSeed()
        return

    def validate(self) -> None:
        """ Checks the validity of all passed values for the experiment. """

        # --- experiment --- #
        assert 100 <= self.config.exp.nTrainEps <= 1e5, \
            'Number of training episodes must be in [100 .. 100,000]'
        assert 10 <= self.config.exp.nEvalEps <= 1e3, \
            'Number of evaluation episodes must be in [10 .. 1,000]'
        assert 0 <= self.config.exp.seed < 1e8, \
            'Seed must be in [0 .. 100,000,000)'
        
        # --- agent --- #
        assert 0 < self.config.agent.alpha <= 1, \
            f'Learning rate {UC.a} must be in (0 .. 1]'
        assert 0 <= self.config.agent.beta <= 1, \
            f'Entropy regularization coefficient {UC.b} must be in [0 .. 1]'
        assert 0 <= self.config.agent.gamma <= 1, \
            f'Discount factor {UC.g} must be in [0 .. 1]'
        assert 0 < self.config.agent.delta <= 1, \
            f'Decay rate {UC.d} for learning rate {UC.a} must be in (0 .. 1]'
        assert 1 <= self.config.agent.batchSize <= 2**7, \
            'Batch size must be in [1 .. 128]'
        assert self.config.agent.batchSize in {2**i for i in range(8)}, \
            'Batch size must be a power of 2'
        
        # --- environment --- #
        assert 3 <= self.config.env.nRows <= 100, \
            'Number of rows in the environment must be in [3 .. 100]'
        assert 3 <= self.config.env.nCols <= 100, \
            'Number of columns in the environment must be in [3 .. 100]'
        assert 0.1 <= self.config.env.speed <= 10, \
            'Speed of the ball in the environment must be in [0.1 .. 10]'
        
        # --- flags --- #
        if self.config.flags.trackModel:
            assert self.config.flags.wandb, \
                'Model tracking requires wandb'
        
        return
