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

        for episode in self.iterator:

            # sample episode and get reward
            transitionBatch = self._sampleEpisode(env)
            episodeR = transitionBatch.totalReward

            # update policy and learning rate
            episodeL = self._learn(transitionBatch)
            # self.optimizer.step()

            # log to console and wandb
            self._logEpisode(episode, episodeR, episodeL)
            
            # check for convergence
            self._checkConvergence(episodeR)
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
        return self._cC >= 50
    
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
    
    def _logFinal(self) -> None:
        """ Handles all logging after training. """
        if self._Q: return
        if self.converged:
            print('converged!')
            self.iterator.finish()
        print(f'avg. reward: {self._tR / self._nE:.2f}')
        print(f'avg. loss: {self._tL / self._nE:.2f}')
        return
    
    def _castState(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        """ Cast numpy array or a torch tensor to a torch tensor on the correct device.
        If the dimension of the state is 3, it is flattened.
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        if state.ndim == 3:
            state = state.flatten()
        return state

    def _checkConvergence(self, reward: float) -> None:
        """ Updates the convergence counter. """
        if reward >= 30: self._cC += 1
        else: self._cC = 0

    ####################
    # PRIVATE ABSTRACT #
    ####################

    @abstractmethod
    def _initTrain(self, nEpisodes: int, Q: bool, W: bool, T: bool) -> None:
        """ Sets up the necessary variables for the training loop.

        1. Initializes some private variables
        2. Tells wandb to track the models if W and T are true
        3. Initializes the episode iterator
        4. Initializes the learning rate scheduler

        ### Args
        `int` nEpisodes: number of episodes to train for
        `bool` Q: whether to use a progress bar or not
        `bool` W: whether to log anything with wandb
        `bool` T: whether to track the model with wandb
        """

        if T: assert W, "Can't track model(s) without logging to wandb"
        
        self._nE = nEpisodes         # total number of episodes
        self._uI = self._nE // 100   # update interval
        self._tR, self._tL = 0, 0    # total reward and loss
        self._maR, self._maL = 0, 0  # moving averages for reward and loss
        self._Q, self._W = Q, W      # whether to use progress bar and wandb
        self._cC = 0                 # convergence counter
        
        if Q: self.iterator = range(self._nE)
        else: self.iterator = ProgressBar(self._nE, updateInterval=self._uI, metrics=['r', 'l'])

        return
    
    @abstractmethod
    def _learn(self, tB: TransitionBatch) -> float:
        """ Performs a learning step on the given transition batch.

        ### Args
        `TransitionBatch` tB: variable sized batch of transitions to learn from

        ### Returns
        `float` avgLoss: average loss over the batch
        """
        raise NotImplementedError('Must be implemented by subclass')
    
    @abstractmethod
    def _updatePolicy(self, S: torch.Tensor, A: torch.Tensor, G: torch.Tensor) -> float:
        """ Updates the agent's policy (and value function if applicable).

        ### Args
        `torch.Tensor` S: state tensor
        `torch.Tensor` A: action tensor
        `torch.Tensor` G: target value tensor

        ### Returns
        `float` totalLoss: total loss over the batch
        """
        raise NotImplementedError('Must be implemented by subclass')
    
    @abstractmethod
    def _logEpisode(self, i: int, r: float, l: float) -> None:
        """ Handles all logging after an episode. """
        self._tR += r
        self._tL += l
        self._maR += r / self._uI
        self._maL += l / self._uI
        if (i+1) % self._uI == 0:
            if not self._Q: self.iterator.updateMetrics(r=self._maR, l=self._maL)
            self._maR, self._maL = 0, 0
        return
