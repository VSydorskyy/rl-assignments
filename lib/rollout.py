import random
import numpy as np


##########################################################################
########                        TASK 0                            ########
##########################################################################
# Implement ReplayBuffer class. See docstrings for details               #

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, obs_prev, act, obs_cur, rew, done):
        self.memory.append((obs_prev, act, obs_cur, rew, done))
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

##########################################################################
########                        TASK 0                            ########
##########################################################################
