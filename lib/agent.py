import copy
from pathlib import Path
import random

from gym.spaces import Discrete
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def polyak_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


class DQN(object):
    def __init__(
        self,
        value_network,
        action_space,
        eps,
        eps_decay,
        batch_size,
        learning_rate,
        discount_factor,
        polyak_tau=1.,
        double=False,
        cuda=False
    ):
        """Abstraction for DeepQNetwork agent.
        
        Arguments:
            value_network (nn.Module): NN for value-function approximation.
            action_space (gym.Space): Action space of the task.
            eps (float): Initial value for the probability of taking a
                random action.
            eps_decay (float): Is subtracted from epsilon on each iteration.
            batch_size (int): Size of batch that is sampled from ReplayBuffer.
            learning_rate (float): Adam initial learning rate.
            discount_factor (float): Gamma from MDPs.
        Parameters:
            polyak_tau (float): softness of target network updates.
                1.0 - hard copy of weights. 0.0 - no copying at all.
            cuda (bool): whether to use Cuda.

        """
        self.value_network = value_network
        self.target_network = copy.deepcopy(value_network)
        self.eps = eps
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.tau = polyak_tau
        self.double = double
        self.cuda = cuda

        assert isinstance(action_space, Discrete), \
            "Action space has to be discrete"

        self.action_space = action_space

        if cuda:
            self.value_network.cuda()
            self.target_network.cuda()

        ##########################################################################
        ########                        TASK 2                            ########
        ##########################################################################
        # Define a loss (Huber loss is preferred) and Adam optimizer:            #
        # I decided to use F.smooth_l1_loss, which is Huber loss

        self.optimizer = torch.optim.Adam(self.value_network.parameters(), lr=learning_rate)
        ##########################################################################
        ########                        TASK 2                            ########
        ##########################################################################

    def update_target(self):
        polyak_update(
            self.target_network,
            self.value_network,
            self.tau
        )

    def update_value(self, replay_buffer):
        batch = replay_buffer.sample(self.batch_size)
        v_s0, v_a, v_s1, v_r, v_d = zip(*batch)

        if self.cuda:
            FloatTensor = torch.cuda.FloatTensor
            LongTensor = torch.cuda.LongTensor
        else:
            FloatTensor = torch.FloatTensor
            LongTensor = torch.LongTensor

        v_s0 = FloatTensor(v_s0)
        v_a = LongTensor(v_a)
        v_s1 = FloatTensor(v_s1)
        v_r = FloatTensor(v_r)
        v_d = FloatTensor(v_d)

        ##########################################################################
        ########                        TASK 2                            ########
        ##########################################################################
        #   Here, you should implement the estimation of y_hat - predicted Q     #
        # value and y - target, i.e. the right-hand side of Bellman equation     #

        #import ipdb; ipdb.set_trace()

        y_hat = self.value_network(v_s0).gather(1, v_a.unsqueeze(1))
        y = self.target_network(v_s1).detach().max(1)[0] * self.discount_factor * (1-v_d) + v_r

        if not self.double:
            pass
        else:
            ##########################################################################
            ########                        TASK 4                            ########
            ##########################################################################
            # Double DQN estimation                                                  #
            pass
            ##########################################################################
            ########                        TASK 4                            ########
            ##########################################################################

        #y = None
        #y_hat = None
        ##########################################################################
        ########                        TASK 2                            ########
        ##########################################################################

        loss = F.smooth_l1_loss(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return (
            loss.detach().cpu().item(),
            y.detach().mean().cpu().item(),
            y_hat.detach().mean().cpu().item()
        )

    def pick_action(self, s, force_greedy=False):
        if self.cuda:
            s = s.cuda()


        randompolicy = bool(np.random.binomial(n=1, p=self.eps)) and not force_greedy

        if randompolicy:
            action =  self.action_space.sample()
        else:
            action = self.value_network(s).argmax(1).item()


        return action

    def update_eps(self):
        self.eps = max(self.eps - self.eps_decay, 1e-2)

    def save_to(self, path, prefix=None):
        if prefix is None:
            path_q = Path(path) / "best.pth"
        elif isinstance(prefix, str):
            path_q = Path(path) / ("%s.pth" % prefix)
        else:
            raise NotImplementedError

        torch.save(self.value_network.state_dict(), str(path_q))
