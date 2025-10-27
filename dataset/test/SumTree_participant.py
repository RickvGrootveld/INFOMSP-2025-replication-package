from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random
from six.moves import xrange
import numpy as np
from collections import namedtuple

from tensorforce import util, TensorForceError
from tensorforce.core.memories import Memory

_SumRow = namedtuple('SumRow', ['item', 'priority'])

class SumTree(object):
    """
    Sum tree data structure where data is stored in leaves and each node on the
    tree contains a sum of the children.

    Items and priorities are stored in leaf nodes, while internal nodes store
    the sum of priorities from all its descendants. Internally a single list
    stores the internal nodes followed by leaf nodes.

    See:
    - [Binary heap trees](https://en.wikipedia.org/wiki/Binary_heap)
    - [Section B.2.1 in the prioritized replay paper](https://arxiv.org/pdf/1511.05952.pdf)
    - [The CNTK implementation](https://github.com/Microsoft/CNTK/blob/258fbec7600fe525b50c3e12d4df0c971a42b96a/bindings/python/cntk/contrib/deeprl/agent/shared/replay_memory.py)

    Usage:
        tree = SumTree(100)
        tree.push('item1', priority=0.5)
        tree.push('item2', priority=0.6)
        item, priority = tree[0]
        batch = tree.sample_minibatch(2)
    """

    def __init__(self, capacity):
        self._capacity = capacity

        # Initializes all internal nodes to have value 0.
        self._memory = [0] * (capacity - 1)
        self._position = 0
        self._actual_capacity = 2 * self._capacity - 1

    def put(self, item, priority=None):
        """
        Stores a transition in replay memory.

        If the memory is full, the oldest entry is replaced.
        """
        if not self._isfull():
            self._memory.append(None)
        position = self._next_position_then_increment()
        old_priority = 0 if self._memory[position] is None \
            else (self._memory[position].priority or 0)
        row = _SumRow(item, priority)
        self._memory[position] = row
        self._update_internal_nodes(
            position, (row.priority or 0) - old_priority)

    def move(self, external_index, new_priority):
        """
        Change the priority of a leaf node
        """
        index = external_index + (self._capacity - 1)
        return self._move(index, new_priority)

    def _move(self, index, new_priority):
        """
        Change the priority of a leaf node.
        """
        item, old_priority = self._memory[index]
        old_priority = old_priority or 0
        self._memory[index] = _SumRow(item, new_priority)
        self._update_internal_nodes(index, new_priority - old_priority)

    def _update_internal_nodes(self, index, delta):
        """
        Update internal priority sums when leaf priority has been changed.
        Args:
            index: leaf node index
            delta: change in priority
        """
        # Move up tree, increasing position, updating sum
        while index > 0:
            index = (index - 1) // 2
            self._memory[index] += delta

    def _isfull(self):
        return len(self) == self._capacity

    def _next_position_then_increment(self):
        """
        Similar to position++.
        """
        start = self._capacity - 1
        position = start + self._position
        self._position = (self._position + 1) % self._capacity
        return position

    def _sample_with_priority(self, p):
        """
        Sample random element with priority greater than p.
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            if left >= len(self._memory):
                # parent points to a leaf node already.
                return parent

            left_p = self._memory[left] if left < self._capacity - 1 \
                else (self._memory[left].priority or 0)
            if p <= left_p:
                parent = left
            else:
                if left + 1 >= len(self._memory):
                    raise RuntimeError('Right child is expected to exist.')
                p -= left_p
                parent = left + 1

    def sample_minibatch(self, batch_size):
        """
        Sample minibatch of size batch_size.
        """
        pool_size = len(self)
        if pool_size == 0:
            return []

        delta_p = self._memory[0] / batch_size
        chosen_idx = []
        # if all priorities sum to ~0  choose randomly otherwise random sample
        if abs(self._memory[0]) < util.epsilon:
            chosen_idx = np.random.randint(self._capacity - 1, self._capacity - 1 + len(self), size=batch_size).tolist()
        else:
            for i in xrange(batch_size):
                lower = max(i * delta_p, 0)
                upper = min((i + 1) * delta_p, self._memory[0])
                p = random.uniform(lower, upper)
                chosen_idx.append(self._sample_with_priority(p))
        return [(i, self._memory[i]) for i in chosen_idx]

    def __len__(self):
        """
        Return the current number of transitions.
        """
        return len(self._memory) - (self._capacity - 1)

    def __getitem__(self, index):
        return self._memory[self._capacity - 1:][index]

    def __getslice__(self, start, end):
        self.memory[self._capacity - 1:][start:end]
