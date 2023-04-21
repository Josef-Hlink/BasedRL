#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sys import platform
from pathlib import Path


mac = platform == 'darwin'

class UC:
    """ Namespace with a few UniCode characters for Greek symbols and ASCII art in stdout. """
    e = '\u03b5'  # epsilon
    g = '\u03b3'  # gamma
    a = '\u03b1'  # alpha
    tl = '\u250c'  if mac else '|'  # ┌
    bl = '\u2514'  if mac else '|'  # └
    tr = '\u2510'  if mac else '|'  # ┐
    br = '\u2518'  if mac else '|'  # ┘
    hd = '\u2500'  if mac else '-'  # ─
    vd = '\u2502'  if mac else '|'  # │
    block = '\u25a0'  if mac else '#'  # ■
    empty = '\u25a1'  if mac else '='  # □

class P:
    """ Namespace with all paths used in the project. """
    root = Path(__file__).parent.parent.parent
    data = root / 'data'
    plots = root / 'plots'
    paths = [root, data, plots]
