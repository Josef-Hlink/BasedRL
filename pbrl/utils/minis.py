#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
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

def generateID() -> str:
    """ Returns a hash of the current time (8 characters long). """
    return hashlib.sha1(str(datetime.now()).encode()).hexdigest()[:8]

def generateSeed() -> int:
    """ Returns a random seed (8 digits long). """
    return int(hashlib.sha1(str(datetime.now()).encode()).hexdigest(), 16) % (10 ** 8)
