import random
from collections import namedtuple


Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class ReplayBufferM(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.total_count = 0

    def push(self, experience):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        self.total_count += 1

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
