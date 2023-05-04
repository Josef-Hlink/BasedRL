#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from pbrl.environment import CatchEnvironment
from pbrl.agents.base import PBAgent
from pbrl.agents.transitions import TransitionBatch 

import numpy as np
import torch
import wandb


class REINFORCEAgent(PBAgent):
    """ An implementation of the REINFORCE algorithm. """

    def __init__(self,
        alpha: float, beta: float, gamma: float, delta: float, batchSize: int,
        device: torch.device, actor: torch.nn.Module, critic: torch.nn.Module = None,
    ) -> None:
        super().__init__(alpha, beta, gamma, delta, batchSize, device, actor, critic)
        return

    ##########
    # PUBLIC #
    ##########

    def train(self,
        env: CatchEnvironment, nEpisodes: int,
        Q: bool = False, D: bool = False, W: bool = False, T: bool = False,
    ) -> None:
        return super().train(env, nEpisodes, Q, D, W, T)
    
    def evaluate(self, env: CatchEnvironment, nEpisodes: int, R: bool = False) -> float:
        return super().evaluate(env, nEpisodes, R)
    
    @property
    def converged(self) -> bool:
        return super().converged

    def saveModels(self, path: Path) -> None:
        """ Saves the actor model's parameters to the given path. """
        torch.save(self.actor.state_dict(), path / 'actor.pth')
        return
    
    def loadModels(self, path: Path) -> None:
        """ Loads the actor model's parameters from the given path. """
        self.actor.load_state_dict(torch.load(path / 'actor.pth'))
        return

    ###########
    # PRIVATE #
    ###########
    
    def _initTrain(self, nEpisodes: int, Q: bool, W: bool, T: bool) -> None:
        
        super()._initTrain(nEpisodes, Q, W, T)
        
        if W and T: wandb.watch(self.actor, log='all', log_freq=self._uI)
        
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.alpha)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size = self._uI,  # fixed for simplicity
            gamma = self.delta     # yes, this looks weird but it makes sense everywhere else
        )
        
        return

    def _sampleEpisode(self, env: CatchEnvironment, R: bool = False) -> TransitionBatch:
        return super()._sampleEpisode(env, R)

    def _chooseAction(self, state: np.ndarray) -> int:
        return super()._chooseAction(state)

    def _learn(self, tB: TransitionBatch) -> float:
        """ Performs a learning step on the given transition batch.

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

        # update learning rate
        self.scheduler.step()

        # return average loss
        return totalLoss / len(tB)

    def _updatePolicy(self, S: torch.Tensor, A: torch.Tensor, G: torch.Tensor) -> float:
        """ Updates the agent's policy.

        ### Args
        `torch.Tensor` S: state tensor
        `torch.Tensor` A: action tensor
        `torch.Tensor` G: target value tensor

        ### Returns
        `float` totalLoss: total loss over the batch
        """

        # forward pass to get action distribution
        dist = torch.distributions.Categorical(self.actor(S))
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
        super()._logEpisode(i, r, l)
        if not self._W: return
        lr = self.optimizer.param_groups[0]['lr']
        data = dict(reward=r, lr=lr, loss=l)
        commit: bool = i % self._uI == 0
        wandb.log(data, step=i, commit=commit)
        return

    def _logFinal(self) -> None:
        """ Handles all logging after training. """
        return super()._logFinal()

    def _castState(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        return super()._castState(state)

    def _checkConvergence(self, reward: float) -> None:
        return super()._checkConvergence(reward)
