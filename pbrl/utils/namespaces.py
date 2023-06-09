#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sys import platform
from pathlib import Path


# fancy terminal output if not on Windows
fancy: bool = platform != 'win32'

class UC:
    """ Namespace with a few UniCode characters for Greek symbols and ASCII art in stdout. """
    e = '\u03b5'  # epsilon
    a = '\u03b1'  # alpha
    b = '\u03b2'  # beta
    g = '\u03b3'  # gamma
    d = '\u03b4'  # delta
    tl = '\u250c'  if fancy else '|'  # ┌
    bl = '\u2514'  if fancy else '|'  # └
    tr = '\u2510'  if fancy else '|'  # ┐
    br = '\u2518'  if fancy else '|'  # ┘
    hd = '\u2500'  if fancy else '-'  # ─
    vd = '\u2502'  if fancy else '|'  # │
    block = '\u25a0'  if fancy else '#'  # ■
    empty = '\u25a1'  if fancy else '='  # □

class P:
    """ Namespace with all paths used in the project. """
    root = Path(__file__).parent.parent.parent
    wandb = root / 'wandb'
    results = root / 'results'
    plots = results / 'plots'
    models = results / 'models'
    sweeps = root / 'sweeps'
    ignored = [results, wandb, plots, models]
