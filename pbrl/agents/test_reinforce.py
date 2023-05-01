from pbrl.environment import CatchEnvironment
from pbrl.agents.transitions import Transition, TransitionBatch 

import numpy as np
from time import perf_counter

import torch
import torch.nn as nn
import torch.optim as optim


class Network(nn.Module):
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64) -> None:
    
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

        self.loss = nn.MSELoss()
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


class REINFORCEAgent_test:

    def __init__(self,
        alpha: float,
        gamma: float,
        device: torch.device,
        R: bool = False, V: bool = False, D: bool = False,
        beta_init: float = 0.01, beta_max: float = 0.1, beta_inc: float = 0.01,
    ) -> None:
        
        self.alpha = alpha
        self.gamma = gamma

        self.device = device

        self.network = Network(98, 3).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.alpha)
        
        self.R = R
        self.V = V
        self.D = D

        self.beta = beta_init
        self.beta_max = beta_max
        self.beta_inc = beta_inc

    def __call__(self) -> str:
        return 'hello world'

    def learn(self, env: CatchEnvironment, episodes: int = 1000) -> None:
        """
        Learn for 100 episodes.
        """

        rewards = []
        start = perf_counter()

        for episode in range(episodes):
            
            # (re)set environment
            state: torch.Tensor = self.castState(env.reset())
            done = False

            T = TransitionBatch()

            while not done:

                if self.R: env.render()

                # get action probability distribution
                dist = torch.distributions.Categorical(self.network(state))
                # sample action
                action: int = dist.sample().item()
                # take action
                next_state, reward, done, _ = env.step(action)
                next_state = self.castState(next_state)
                
                T.add(Transition(state, action, reward, next_state, done))

                if self.D:
                    print(f'action dist.: {dist.probs}, action: {action}, reward: {reward}, done: {done}')

                state = next_state

            # update policy
            self.update(T)
            rewards.append(T.totalReward)

            if self.V:
                status = f'episode {episode + 1} / {episodes}: reward: {T.totalReward:.2f}'
                print(f'\r{status}' + ' ' * (79-len(status)), end='', flush=True)
                if episode == episodes - 1: print()
        
        print(f'average reward: {np.mean(rewards)}')
        print(f'time elapsed: {perf_counter() - start}')

        return
        
    def update(self, transitionBatch: TransitionBatch, beta: float = 0.01) -> None:
        """ Update the agent's policy. """

        # unpack transition batch into tensors (and put on device)
        S, A, R, S_, D = map(lambda x: x.to(self.device), transitionBatch.unpack())

        # we can't use gradients here
        with torch.no_grad():
            # initialize target value tensor
            G = torch.zeros(len(S), dtype=torch.float32).to(self.device)
            # calculate (discounted) target values
            for i in range(len(R)):
                G[i] = sum([self.gamma**j * r for j, r in enumerate(R[i:])])

        # loop over transitions
        for s, a, g in zip(S, A, G):
            dist = torch.distributions.Categorical(self.network(s))
            # calculate log probability of action
            log_prob = -dist.log_prob(a)
            # compute the entropy of the action distribution
            probs = dist.probs
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            # get loss
            loss = -(log_prob * g + beta * entropy)
            # backprop time baby
            # zero_grad used to prevent earlier gradients from affecting the current gradient
            self.optimizer.zero_grad() 
            loss.backward()
            # update parameters
            self.optimizer.step()
        
        # gradually increase beta over time
        beta *= 1.01
        
        return


    def castState(self, state: np.ndarray) -> torch.Tensor:
        """ Cast 3D state (np.array) to 1D torch tensor on the correct device. """
        return torch.tensor(state, dtype=torch.float32).flatten().to(self.device)
