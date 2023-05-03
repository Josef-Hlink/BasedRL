#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
from time import perf_counter
from datetime import datetime

from pbrl.utils import UC, bold, formatRuntime


class ProgressBar:

    def __init__(self, numSteps: int, updateInterval: int = 1, metrics: Optional[list] = None) -> None:
        """ Initializes the progress bar. """
        self.t = numSteps
        self.iterable = range(numSteps)
        self.iterator = iter(self.iterable)
        self.updateInterval = updateInterval
        self.metrics = {m: None for m in metrics} if metrics is not None else None
        
        self.w = 50
        self.i = 0
        
        self.bar = ''
        self.percentage = 0
        self.info = ''

        self.tic = perf_counter()
        print(f'Started training for {numSteps} episodes at {bold(datetime.now().strftime("%H:%M:%S"))}')
        return
    
    ##########
    # PUBLIC #
    ##########
    
    def updateMetrics(self, **kwargs):
        """ Updates the metrics. """
        assert self.metrics is not None, 'No metrics were specified.'
        for k, v in kwargs.items():
            self.metrics[k] = v
        self.info = ' | '.join(f'{k}: {v:.2f}' for k, v in self.metrics.items())
        return
    
    def finish(self):
        """ Finishes the progress bar. """
        print(f'\nFinished training in {bold(formatRuntime(perf_counter() - self.tic))}')
        return

    ###########
    # PRIVATE #
    ###########

    def _updatePercentage(self):
        """ Updates the percentage. """
        self.percentage = int(100 * self.i / self.t)
    
    def _updateBar(self):
        """ Updates the progress bar. """
        n = int(self.w * self.i / self.t)
        self.bar = UC.block * n + UC.empty * (self.w - n) + f' {self.percentage}%'
        if self.info != '':
            self.bar += f' | {self.info}'
        self.bar += ' ' * (80 - len(self.bar))
        return

    def _render(self):
        """ Renders the progress bar. """
        print(f'\r{self.bar}', end='', flush=True)

    ############
    # INTERNAL #
    ############

    def __iter__(self):
        """ Returns the iterator. """
        return self

    def __next__(self):
        """ Returns the next item in the iterable. """
        if self.i % self.updateInterval == 0:
            self._updatePercentage()
            self._updateBar()
            self._render()
        if self.i == self.t:
            self.finish()
        self.i += 1
        return next(self.iterator)
