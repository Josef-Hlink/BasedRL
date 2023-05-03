#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from pbrl.environment import CatchEnvironment
from pbrl.agents.transitions import Transition, TransitionBatch 
from pbrl.utils import ProgressBar

import numpy as np
import torch
import wandb


class REINFORCEAgent:

    def __init__(self,
        model: torch.nn.Module, device: torch.device,
        alpha: float, beta: float, gamma: float, delta: float,
        batchSize: int = 1,
    ) -> None:
        """ Initializes the agent by setting hyperparameters and creating the model and optimizer.

        ### Args

        Torch:
            `torch.nn.Module` model: the model to be used as the agent's function approximator
            `torch.device` device: device to run on

        Hyperparameters:
            `float` alpha: learning rate
            `float` beta: entropy regularization coefficient
            `float` gamma: discount factor
            `float` delta: learning rate decay rate
            `int` batchSize: number of transitions to use for a single update
        """
        
        self.device = device
        self.model = model.to(self.device)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.batchSize = batchSize

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)
        self.convergeCounter: int = 0
        return

    ##########
    # PUBLIC #
    ##########

    def train(self,
        env: CatchEnvironment, nEpisodes: int,
        V: bool = False, D: bool = False, W: bool = False, T: bool = False,
    ) -> None:
        """ Trains the agent on an environment for a given number of episodes.
        
        ### Args
        `CatchEnvironment` env: environment to train on
        `int` nEpisodes: number of episodes to train for
        `bool` V: toggle verbose printing
        `bool` D: toggle debug printing
        `bool` W: toggle wandb logging
        `bool` T: toggle wandb model tracking
        """

        # initialize training
        self._initTrain(nEpisodes, V, W, T)

        for episode in self.iterator:

            # sample episode and get reward
            transitionBatch = self._sampleEpisode(env)
            episodeR = transitionBatch.totalReward
            
            # update policy and learning rate
            episodeL = self._learn(transitionBatch)
            self.scheduler.step()

            # log to console and wandb
            self._logEpisode(episode, episodeR, episodeL)
            
            # check for convergence
            self._checkConvergence(episodeR)
            if self.converged:
                break
        
        # log final summary to console
        self._logFinal()

        return
    
    def saveModel(self, path: Path) -> None:
        """ Saves the model's parameters to the given path. """
        torch.save(self.model.state_dict(), path)
        return
    
    def loadModel(self, path: Path) -> None:
        """ Loads the model's parameters from the given path. """
        self.model.load_state_dict(torch.load(path))
        return
    
    def chooseAction(self, state: np.ndarray) -> int:
        """ Chooses an action to take based on the current policy. """
        state = self._castState(state, flatten=True)
        return torch.distributions.Categorical(self.model(state)).sample().item()

    @property
    def converged(self) -> bool:
        """ Hardcoded tolerance for now. """
        return self.convergeCounter >= 50

    ###########
    # PRIVATE #
    ###########

    def _initTrain(self, nEpisodes: int, V: bool, W: bool, T: bool) -> None:
        """ Sets up the necessary variables for the training loop.

        1. Initializes some private variables
        2. Tells wandb to track the model if W and T are true
        3. Initializes the episode iterator
        4. Initializes the learning rate scheduler

        ### Args
        `int` nEpisodes: number of episodes to train for
        `bool` V: whether to use the progress bar, or just use a `range` object to iterate over episodes
        `bool` W: whether to log anything with wandb
        `bool` T: whether to track the model with wandb
        """

        if T: assert W, "Can't track model without logging to wandb"
        
        self._nE = nEpisodes         # total number of episodes
        self._uI = self._nE // 100   # update interval
        self._tR, self._tL = 0, 0    # total reward and loss
        self._maR, self._maL = 0, 0  # moving averages for reward and loss
        self._V, self._W = V, W      # whether to use progress bar and wandb
        
        if W and T: wandb.watch(self.model, log='all', log_freq=self._uI)
        
        if V: self.iterator = ProgressBar(self._nE, updateInterval=self._uI, metrics=['r', 'l'])
        else: self.iterator = range(self._nE)
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size = self._uI,  # fixed for simplicity (TODO: fix this, haha...)
            gamma = self.delta     # yes, this looks weird but it makes sense everywhere else
        )
        
        return

    def _sampleEpisode(self, env: CatchEnvironment) -> TransitionBatch:
        """ Samples an episode from the environment and returns the sampled transitions in a `TransitionBatch`. """
        tB = TransitionBatch()
        state = self._castState(env.reset(), flatten=True)
        done = False
        while not done:
            action = self.chooseAction(state)
            next_state, reward, done, _ = env.step(action)
            next_state = self._castState(next_state, flatten=True)
            tB.add(Transition(state, action, reward, next_state, done))
            state = next_state
        return tB

    def _learn(self, tB: TransitionBatch) -> float:
        """ Updates the agent's policy.

        ### Args
        `TransitionBatch` tB: variable sized batch of transitions to learn from

        ### Returns
        `float` avgLoss: average loss over the batch
        """

        # unpack transition batch into tensors (and put on device)
        S, A, R = map(lambda x: x.to(self.device), (tB.S, tB.A, tB.R))

        # we can't use gradients here
        with torch.no_grad():
            # initialize target value tensor
            G = torch.zeros(len(tB), dtype=torch.float32).to(self.device)
            # calculate (discounted) target values
            for t in range(len(tB)):
                G[t] = sum([self.gamma**t_ * r for t_, r in enumerate(R[t:])])

        totalLoss = 0

        # loop over transitions with batch size
        for i in range(0, len(tB), self.batchSize):
            # slice mini-batch from the full batch
            slc: slice = slice(i, i+self.batchSize)
            _S, _A, _G = map(lambda x: x[slc], (S, A, G))
            # update policy and add loss to total loss
            totalLoss += self._updatePolicy(_S, _A, _G)

        # return average loss
        return totalLoss / len(tB)

    def _updatePolicy(self, S: torch.Tensor, A: torch.Tensor, G: torch.Tensor) -> float:
        """ Updates the agent's policy network.

        ### Args
        `torch.Tensor` S: state tensor
        `torch.Tensor` A: action tensor
        `torch.Tensor` G: target value tensor

        ### Returns
        `float` totalLoss: total loss over the batch
        """

        # forward pass to get action distribution
        dist = torch.distributions.Categorical(self.model(S))
        # calculate base loss
        loss = -dist.log_prob(A) * G
        # add entropy regularization
        loss -= self.beta * dist.entropy()
        # zero_grad used to prevent earlier gradients from affecting the current gradient
        self.optimizer.zero_grad()
        # calculate gradients
        loss.mean().backward()
        # update parameters
        self.optimizer.step()

        # return total loss over the batch
        return loss.sum().item()

    def _logEpisode(self, i: int, r: float, l: float) -> None:
        """ Handles all logging after an episode. """
        self._tR += r
        self._tL += l
        self._maR += r / self._uI
        self._maL += l / self._uI
        if (i+1) % self._uI == 0:
            if self._V: self.iterator.updateMetrics(r=self._maR, l=self._maL)
            self._maR, self._maL = 0, 0
        if self._W:
            wandb.log(dict(reward=r, lr=self.optimizer.param_groups[0]['lr'], loss=l), step=i)
        return

    def _logFinal(self) -> None:
        """ Handles all logging after training. """
        if not self._V: return
        if self.converged:
            print('converged!')
            self.iterator.finish()
        print(f'avg. reward: {self._tR / self._nE:.2f}')
        print(f'avg. loss: {self._tL / self._nE:.2f}')
        return

    def _castState(self, state: np.ndarray | torch.Tensor, flatten: bool = False) -> torch.Tensor:
        """ Cast numpy array or a torch tensor to a torch tensor on the correct device.
        If `flatten`, the tensor is cast to a 1D tensor.
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        if flatten:
            state = state.flatten()
        return state

    def _checkConvergence(self, reward: float) -> None:
        """ Updates the convergence counter. """
        if reward >= 30:
            self.convergeCounter += 1
        else:
            self.convergeCounter = 0
