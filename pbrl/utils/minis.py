#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Even smaller helper methods (split up to prevent circular imports). """

from datetime import datetime


def bold(text: str) -> str:
    """ Returns the passed text in bold. """
    return f'\033[1m{text}\033[0m'

def formatRuntime(seconds: float) -> str:
    """ Returns a formatted string of the passed runtime in (mm:)ss.fff. """
    if seconds < 60:
        return f'{seconds:.3f} sec'
    elif seconds < 3600:
        return datetime.utcfromtimestamp(seconds).strftime('%M:%S.%f')[:-3] + ' min'
    else:
        return datetime.utcfromtimestamp(seconds).strftime('%H:%M:%S.%f')[:-3] + ' hr'

def formattedRuntimeToSeconds(formattedRuntime: str) -> float:
    """ Returns the passed formatted runtime in seconds. """
    if formattedRuntime[-3:] == 'sec':
        return float(formattedRuntime[:-3])
    elif formattedRuntime[-3:] == 'min':
        min_, sec_ = map(int, formattedRuntime[:-8].split(':'))
        return min_ * 60 + sec_ + float(formattedRuntime[-7:-4]) / 1000
    elif formattedRuntime[-2:] == 'hr':
        hr_, min_, sec_ = map(int, formattedRuntime[:-7].split(':'))
        return hr_ * 3600 + min_ * 60 + sec_ + float(formattedRuntime[-6:-3]) / 1000

class DotDict(dict):
    """ `dot.notation` access to dictionary attributes. """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def copy(self):
        return DotDict(super().copy())