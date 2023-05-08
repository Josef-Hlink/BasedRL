#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class DotDict(dict):
    """ `dot.notation` access to dictionary attributes. """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def copy(self):
        return DotDict(super().copy())

    def pop(self, key, default=None):
        if key in self:
            return super().pop(key)
        else:
            return default

    def toDict(self) -> dict:
        """ Converts the DotDict to a regular dict recursively. """
        dct = {}
        def _parse(dct, dotdct):
            for k, v in dotdct.items():
                if isinstance(v, DotDict):
                    dct[k] = {}
                    _parse(dct[k], v)
                else:
                    dct[k] = v
        _parse(dct, self)
        return dct

    def __str__(self):
        """ String representation of the (nested) DotDict. """
        text = 'DOTDICT\n'
        def _walk(dotdct, indent):
            nonlocal text
            for k, v in dotdct.items():
                if isinstance(v, DotDict):
                    text += indent + f'{k}:\n'
                    _walk(v, indent + '    ')
                else:
                    text += indent + f'{k}: {v}\n'
        _walk(self, '')
        return text
