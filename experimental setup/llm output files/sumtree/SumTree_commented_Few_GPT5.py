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
    """SumTree data structure for prioritized sampling.

    This class implements a binary sum tree commonly used for prioritized
    experience replay. Leaf nodes store (item, priority) tuples, while
    internal nodes store the sum of priorities of their children. This
    enables O(log N) updates and sampling by priority mass.
    """
    
    def __init__(self, capacity):
        """Initialize an empty SumTree.

        Arguments:
            capacity (int): Maximum number of leaf items the tree can hold.
                            Internally, the tree uses an array of size
                            (2 * capacity - 1). The first (capacity - 1)
                            entries are internal nodes holding priority sums,
                            and the remaining entries are leaves.
        """
        self._capacity = capacity

        # Internal nodes array initialized to zeros; leaves will be appended.
        self._memory = [0] * (capacity - 1)
        # Next insertion position among leaves (circular buffer behavior).
        self._position = 0
        # Total array size when full: internal nodes + leaves.
        self._actual_capacity = 2 * self._capacity - 1

    def put(self, item, priority=None):
        """Insert or overwrite a leaf with given item and priority.

        If the tree is not yet full, a new leaf slot is appended.
        Otherwise, insertion overwrites the next position in a circular manner.

        Arguments:
            item: The payload to store at the leaf (e.g., an experience).
            priority (float or None): Priority mass for sampling; if None,
                                      treated as zero for sum updates.
        """
        if not self._isfull():
            # Expand leaves region until full capacity is reached.
            self._memory.append(None)
        position = self._next_position_then_increment()
        # Determine old priority (for delta update); treat missing as zero.
        old_priority = 0 if self._memory[position] is None \
            else (self._memory[position].priority or 0)
        row = _SumRow(item, priority)
        self._memory[position] = row
        # Propagate the priority delta up the internal nodes.
        self._update_internal_nodes(
            position, (row.priority or 0) - old_priority)

    def move(self, external_index, new_priority):
        """Update the priority of an existing leaf by external index.

        Arguments:
            external_index (int): Index within the leaves (0-based).
            new_priority (float): New priority value to set.

        Returns:
            None
        """
        index = external_index + (self._capacity - 1)
        return self._move(index, new_priority)

    def _move(self, index, new_priority):
        """Internal helper to update a leaf priority by absolute array index.

        Arguments:
            index (int): Absolute index into the underlying array.
            new_priority (float): New priority value to set.
        """
        item, old_priority = self._memory[index]
        old_priority = old_priority or 0
        self._memory[index] = _SumRow(item, new_priority)
        # Propagate priority difference to the root.
        self._update_internal_nodes(index, new_priority - old_priority)

    def _update_internal_nodes(self, index, delta):
        """Propagate a priority delta from a leaf up to the root.

        Arguments:
            index (int): Leaf index where the change occurred.
            delta (float): Difference to add to each ancestor internal node.
        """
        # Walk up parent chain and add delta to each internal node.
        while index > 0:
            index = (index - 1) // 2
            self._memory[index] += delta

    def _isfull(self):
        """Check whether the leaves region has reached capacity.

        Returns:
            bool: True if number of stored leaves equals capacity.
        """
        return len(self) == self._capacity

    def _next_position_then_increment(self):
        """Return current leaf write position and advance circular index.

        Returns:
            int: Absolute array index for the next write into leaves.
        """
        start = self._capacity - 1
        position = start + self._position
        # Circularly advance the write pointer.
        self._position = (self._position + 1) % self._capacity
        return position

    def _sample_with_priority(self, p):
        """Traverse the tree to find the leaf where cumulative mass reaches p.

        This performs a binary search on the implicit tree:
        starting at the root, walk left/right depending on cumulative mass.

        Arguments:
            p (float): Target cumulative priority mass to sample.

        Returns:
            int: Absolute index of the selected leaf in the underlying array.
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            # If there is no left child, parent is a leaf.
            if left >= len(self._memory):
                return parent

            # Get left child mass: internal nodes store sums; leaves store rows.
            left_p = self._memory[left] if left < self._capacity - 1 \
                else (self._memory[left].priority or 0)
            if p <= left_p:
                parent = left
            else:
                # Ensure right child exists in a proper binary tree.
                if left + 1 >= len(self._memory):
                    raise RuntimeError('Right child is expected to exist.')
                p -= left_p
                parent = left + 1

    def sample_minibatch(self, batch_size):
        """Sample a minibatch of leaves approximately stratified by priority.

        The total priority mass is divided into equal segments and one sample
        is drawn uniformly from each segment, yielding reduced variance.

        Arguments:
            batch_size (int): Number of samples to draw.

        Returns:
            list[tuple[int, _SumRow]]: Pairs of (absolute index, (item, priority)).
                                       If the tree is empty, returns an empty list.
        """
        pool_size = len(self)
        if pool_size == 0:
            return []

        # Total mass is at root (index 0). Divide into equal segments.
        delta_p = self._memory[0] / batch_size
        chosen_idx = []
        # If total mass is effectively zero, sample uniformly among current leaves.
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
        """Number of leaves currently stored.

        Returns:
            int: Count of stored items (leaves), up to capacity.
        """
        return len(self._memory) - (self._capacity - 1)

    def __getitem__(self, index):
        """Access leaves by relative index (0-based among leaves).

        Arguments:
            index (int or slice): Relative index within leaves.

        Returns:
            _SumRow or list[_SumRow]: Requested leaf row(s).
        """
        return self._memory[self._capacity - 1:][index]

    def __getslice__(self, start, end):
        """Support old-style slicing for leaves [start:end].

        Arguments:
            start (int): Start index within leaves.
            end (int): End index within leaves (exclusive).

        Returns:
            list[_SumRow]: Slice of leaf rows.
        """
        self.memory[self._capacity - 1:][start:end]
