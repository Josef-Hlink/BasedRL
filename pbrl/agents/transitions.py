#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data structures for storing transitions.
"""

from __future__ import annotations
from typing import Iterable, Optional

from torch import Tensor, stack


class Transition:
    """ A transition is a tuple of (state, action, reward, next_state, done). """

    def __init__(self, s: Tensor, a: int, r: float, s_: Tensor, d: bool) -> None:
        """ Initializes a singular transition. """
        self.a = a
        self.s = s
        self.r = r
        self.s_ = s_
        self.d = d
        return

class TransitionBatch:
    """ A batch of transitions. """

    def __init__(self, transitions: Optional[Iterable[Transition]] = None) -> None:
        """
        Initializes a batch of transitions.
        If no transitions are provided, an empty batch is created.
        """
        self.transitions = list(transitions) if transitions is not None else []
        return
    
    def add(self, transition: Transition) -> None:
        """ Adds a transition to the batch. """
        self.transitions.append(transition)
        return

    def unpack(self) -> tuple[Tensor]:
        """
        Unpacks the batch into all their respective tensors.

        Returns:
            - states
            - actions
            - rewards
            - next_states
            - done flags
        """
        return (
            stack([t.s for t in self.transitions]),
            Tensor([t.a for t in self.transitions]).long(),
            Tensor([t.r for t in self.transitions]),
            stack([t.s_ for t in self.transitions]),
            Tensor([t.d for t in self.transitions]),
        )
    
    @property
    def S(self) -> Tensor:
        """ States tensor """
        return stack([t.s for t in self.transitions])
    
    @property
    def A(self) -> Tensor:
        """ Actions tensor """
        return Tensor([t.a for t in self.transitions]).long()
    
    @property
    def R(self) -> Tensor:
        """ Rewards tensor """
        return Tensor([t.r for t in self.transitions])
    
    @property
    def S_(self) -> Tensor:
        """ Next states tensor """
        return stack([t.s_ for t in self.transitions])
    
    @property
    def D(self) -> Tensor:
        """ Done flags tensor """
        return Tensor([t.d for t in self.transitions])

    @property
    def totalReward(self) -> float:
        """ Returns the total reward of the batch. """
        return self.R.sum().item()

    def __len__(self) -> int:
        return len(self.transitions)
    
    def __getitem__(self, index: int | slice) -> Transition | TransitionBatch:
        if isinstance(index, slice):
            return TransitionBatch(self.transitions[index])
        return self.transitions[index]
    
    def __iter__(self) -> Iterable[Transition]:
        return iter(self.transitions)
    
    def __add__(self, other: Transition | TransitionBatch) -> TransitionBatch:
        if isinstance(other, Transition):
            return TransitionBatch(self.transitions + [other])
        return TransitionBatch(self.transitions + other.transitions)
    
    def __delitem__(self, index: int | slice) -> None:
        del self.transitions[index]
