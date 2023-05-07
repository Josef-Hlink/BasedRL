#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from abc import ABC, abstractmethod

from pbrl.environment import CatchEnvironment
from pbrl.agents.transitions import Transition, TransitionBatch
from pbrl.utils import ProgressBar

import numpy as np
import torch


class PBAgent(ABC):
    """ Base class for policy-based agents. """

    @abstractmethod
    def __init__(self,
        alpha: float, beta: float, gamma: float, delta: float, batchSize: int,
        device: torch.device, actor: torch.nn.Module, critic: torch.nn.Module = None,
    ) -> None:
        """ Initializes the agent by setting hyperparameters and creating the model(s).

        ### Args

        Hyperparameters:
            `float` alpha: learning rate
            `float` beta: entropy regularization coefficient
            `float` gamma: discount factor
            `float` delta: learning rate decay rate
            `int` batchSize: number of transitions to use for a single update

        Torch:
            `torch.device` device: device to run on
            `torch.nn.Module` actor: the model to be used for the agent's behavior policy
            `torch.nn.Module` critic: the model to be used for the agent's value function
                (not used by REINFORCEAgent)
        """
        self.device = device
        # set for both REINFORCEAgent and ActorCriticAgent
        self.actor = actor.to(self.device)
        # to be set only by ActorCriticAgent
        self.critic = critic.to(self.device) if critic is not None else None
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.batchSize = batchSize
        return

    #####################
    # PUBLIC FUNCTIONAL #
    #####################

    def train(self,
        env: CatchEnvironment, nEpisodes: int,
        Q: bool = False, D: bool = False, W: bool = False, T: bool = False,
    ) -> None:
        """ Trains the agent on an environment for a given number of episodes.
        
        ### Args
        `CatchEnvironment` env: environment to train on
        `int` nEpisodes: number of episodes to train for
        `bool` Q: toggle information printing
        `bool` D: toggle debug printing
        `bool` W: toggle wandb logging
        `bool` T: toggle wandb model tracking
        """

        # initialize training
        self._initTrain(nEpisodes, Q, W, T)

        for ep in self.iterator:

            # sample episode and get reward
            transitionBatch = self._sampleEpisode(env)
            epR = transitionBatch.totalReward

            # update policy and learning rate
            epPG, epVL = self._learn(transitionBatch)

            # log to console and/or wandb
            self._logEpisode(ep, epR, epPG, epVL)
            
            # check for convergence
            if self.converged:
                break
        
        # log final summary to console
        self._logFinal()

        return
    
    def evaluate(self, env: CatchEnvironment, nEpisodes: int, R: bool = False) -> float:
        """ Evaluates the agent on an environment on a given number of episodes.

        ### Args
        `CatchEnvironment` env: environment to evaluate for
        `int` nEpisodes: number of episodes to evaluate on
        `bool` R: toggle rendering

        ### Returns
        `float` avgReward: average reward over the evaluation episodes
        """
        avgReward = 0
        for _ in range(nEpisodes):
            transitionBatch = self._sampleEpisode(env, R)
            avgReward += transitionBatch.totalReward / nEpisodes
        return avgReward
    
    @property
    def converged(self) -> bool:
        """ Whether the agent has converged or not. """
        # we can't check convergence in first two windows
        # we also don't want to check every single time bc it's expensive
        if len(self._rT) < 2*self._cI+1 or len(self._rT) % self._uI != 0:
            return False
        currentMedR = np.median(self._rT[-self._cI:])
        self._rI = currentMedR - np.median(self._rT[-2*self._cI-1:-self._cI-1])
        return currentMedR > 0 and self._rI < 1

    ###################
    # PUBLIC ABSTRACT #
    ###################

    @abstractmethod
    def saveModels(self, path: Path) -> None:
        """ Saves the model's / models' parameters to the given path. """
        return NotImplementedError('Must be implemented by subclass')
    
    @abstractmethod
    def loadModels(self, path: Path) -> None:
        """ Loads the model's / models' parameters from the given path. """
        return NotImplementedError('Must be implemented by subclass')
    
    ######################
    # PRIVATE FUNCTIONAL #
    ######################
    
    def _sampleEpisode(self, env: CatchEnvironment, R: bool = False) -> TransitionBatch:
        """ Samples an episode from the environment and returns the sampled transitions in a `TransitionBatch`. """
        tB = TransitionBatch()
        state = self._castState(env.reset())
        done = False
        while not done:
            action = self._chooseAction(state)
            next_state, reward, done, _ = env.step(action)
            next_state = self._castState(next_state)
            tB.add(Transition(state, action, reward, next_state, done))
            state = next_state
            if R: env.render()
        return tB
    
    def _chooseAction(self, state: torch.Tensor) -> int:
        """ Chooses an action to take based on the current policy. """
        with torch.no_grad():
            state = self._castState(state)
            action = torch.distributions.Categorical(self.actor(state)).sample().item()
        return action
    
    def _castState(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        """ Cast numpy array or a torch tensor to a torch tensor on the correct device.
        If the dimension of the state is 3, it is flattened.
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        if state.ndim == 3:
            state = state.flatten()
        return state

    ####################
    # PRIVATE ABSTRACT #
    ####################

    @abstractmethod
    def _initTrain(self, nEpisodes: int, Q: bool, W: bool, T: bool) -> None:
        """ Initializes some private variables for training. """

        if T: assert W, "Can't track model(s) without logging to wandb"
        
        self._nE = nEpisodes          # total number of episodes
        self._uI = self._nE // 100    # update interval
        self._tR, self._maR = 0, 0    # total reward and moving average reward
        self._tPG, self._maPG = 0, 0  # total policy gradient and moving average policy gradient
        self._tVL, self._maVL = 0, 0  # total value loss and moving average value loss
        self._Q, self._W = Q, W       # whether to use progress bar and wandb

        self._rT = []                 # reward trace
        self._cI = self._nE // 10     # convergence interval
        self._dR = 0                  # difference in reward between current and previous convergence interval

        return
    
    @abstractmethod
    def _learn(self, tB: TransitionBatch) -> tuple[float, float | None]:
        """ Performs a learning step on the given transition batch.

        One transition batch generally represents one episode.

        ### Args
        `TransitionBatch` tB: variable sized batch of transitions to learn from

        ### Returns
        `float` avgPG: average policy gradient over the batch
        `float | None` avgVL: average value loss over the batch
        """
        raise NotImplementedError('Must be implemented by subclass')
    
    @abstractmethod
    def _updatePolicy(self, S: torch.Tensor, A: torch.Tensor, G: torch.Tensor) -> tuple[float, float | None]:
        """ Updates the agent's policy (and value function if applicable).

        ### Args
        `torch.Tensor` S: state tensor
        `torch.Tensor` A: action tensor
        `torch.Tensor` G: target value tensor

        ### Returns
        `float` totalPG: total policy gradient over the batch
        `float | None` totalVL: total value loss over the batch
        """
        raise NotImplementedError('Must be implemented by subclass')
    
    @abstractmethod
    def _logEpisode(self, i: int, r: float, pg: float, vl: float | None) -> None:
        """ Handles all logging after an episode. """
        self._rT.append(r)
        self._tR += r
        self._tPG += pg
        self._tVL += vl if vl is not None else 0
        self._maR += r / self._uI
        self._maPG += pg / self._uI
        self._maVL += vl / self._uI if vl is not None else 0
        if (i+1) % self._uI == 0:
            data = {'r': self._maR, 'pg': self._maPG, 'vl': self._maVL}
            if vl is None: data.pop('vl')
            if not self._Q: self.iterator.updateMetrics(**data)
            self._maR, self._maPG, self._maVL = 0, 0, 0
        return

    @abstractmethod
    def _logFinal(self) -> None:
        """ Handles all logging after training. """
        if self._Q: return
        if self.converged:
            assert isinstance(self.iterator, ProgressBar)
            self.iterator.finish()
            print(f'converged at episode {len(self._rT)}')
        print(f'avg. reward: {self._tR / self._nE:.3f}')
        print(f'avg. policy gradient: {self._tPG / self._nE:.3f}')
        return
