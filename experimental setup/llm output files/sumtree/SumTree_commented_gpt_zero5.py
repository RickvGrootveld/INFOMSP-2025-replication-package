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
    SumTree data structure supporting:
    - Priority-based sampling via a binary tree of cumulative sums.
    - Fixed-capacity circular buffer for leaf nodes (experience items).
    
    Internal layout:
    - The tree is stored in a single list _memory of size (2 * capacity - 1).
      * Indices [0 : capacity - 1) store internal nodes with cumulative sums.
      * Indices [capacity - 1 : 2*capacity - 1) store leaf nodes as (_SumRow).
    - _position cycles through 0..capacity-1 to implement circular overwrite.
    
    Typical use:
    - put(item, priority): insert/overwrite an item at next leaf with priority.
    - move(external_index, new_priority): update an existing leaf priority.
    - sample_minibatch(batch_size): sample leaves proportional to priority.
    """

    def __init__(self, capacity):
        """
        Initialize the sum-tree with a fixed number of leaves.

        Args:
            capacity (int): Number of leaf nodes (maximum number of items).

        Notes:
            - _memory starts with only internal nodes initialized to 0.
            - Leaf nodes (rows) are appended lazily on first insertions.
            - _actual_capacity is the total number of nodes in the tree array.
        """
        self._capacity = capacity

        # Internal nodes initialized to zero; leaves appended as we insert.
        self._memory = [0] * (capacity - 1)
        # Next leaf write position (0..capacity-1, circular).
        self._position = 0
        self._actual_capacity = 2 * self._capacity - 1

    def put(self, item, priority=None):
        """
        Insert an item with given priority into the next leaf position.

        If the tree is not yet full, a None placeholder for the new leaf is
        appended. When full, the insertion overwrites the oldest leaf.

        Args:
            item: Arbitrary payload to be stored at the leaf.
            priority (float or None): Priority for sampling; treated as 0 if None.
        """
        if not self._isfull():
            # Lazily allocate next leaf slot.
            self._memory.append(None)
        position = self._next_position_then_increment()
        # Retrieve old priority at this leaf (0 if empty/None).
        old_priority = 0 if self._memory[position] is None \
            else (self._memory[position].priority or 0)
        row = _SumRow(item, priority)
        self._memory[position] = row
        # Update internal sums with the delta between new and old priority.
        self._update_internal_nodes(
            position, (row.priority or 0) - old_priority)

    def move(self, external_index, new_priority):
        """
        Update the priority of an existing leaf by its external index.

        Args:
            external_index (int): Index within the leaves range [0, len(self)).
            new_priority (float): New priority value.

        Returns:
            None
        """
        index = external_index + (self._capacity - 1)
        return self._move(index, new_priority)

    def _move(self, index, new_priority):
        """
        Internal helper to update a leaf's priority given absolute tree index.

        Args:
            index (int): Absolute index in the underlying _memory array.
            new_priority (float): New priority value.

        Returns:
            None
        """
        item, old_priority = self._memory[index]
        old_priority = old_priority or 0
        self._memory[index] = _SumRow(item, new_priority)
        # Propagate the priority change up the tree.
        self._update_internal_nodes(index, new_priority - old_priority)

    def _update_internal_nodes(self, index, delta):
        """
        Propagate a change in priority up from a leaf to the root.

        Args:
            index (int): Leaf or child node index where the change occurred.
            delta (float): Difference to add to each ancestor cumulative sum.
        """
        while index > 0:
            # Move to parent index.
            index = (index - 1) // 2
            self._memory[index] += delta

    def _isfull(self):
        """
        Check whether the number of stored leaves reached the capacity.

        Returns:
            bool: True if tree has capacity leaves, False otherwise.
        """
        return len(self) == self._capacity

    def _next_position_then_increment(self):
        """
        Compute the absolute index of the next leaf to write, then advance.

        Returns:
            int: Absolute index into _memory where the leaf resides.
        """
        start = self._capacity - 1
        position = start + self._position
        # Advance circular position.
        self._position = (self._position + 1) % self._capacity
        return position

    def _sample_with_priority(self, p):
        """
        Traverse the tree to find the leaf corresponding to cumulative value p.

        Args:
            p (float): Cumulative priority mass to locate.

        Returns:
            int: Absolute index of the selected leaf in _memory.

        Raises:
            RuntimeError: If a right child is expected but does not exist.
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            # If left is beyond memory, parent is a leaf.
            if left >= len(self._memory):
                return parent

            # Retrieve left child's priority/cumulative value.
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
        Sample a minibatch of leaves proportionally to their priorities.

        The total priority mass is divided into equal ranges and one sample
        is drawn uniformly from each range, improving coverage.

        Args:
            batch_size (int): Number of samples to draw.

        Returns:
            list of tuples: [(index, _SumRow), ...] where index is the absolute
            index of the selected leaf in _memory, and _SumRow holds the data.

        Notes:
            - If total mass is near zero (all priorities ~0), sample uniformly
              among the existing leaves instead of by priority.
        """
        pool_size = len(self)
        if pool_size == 0:
            return []

        delta_p = self._memory[0] / batch_size
        chosen_idx = []
        # If the root sum is ~0, perform uniform sampling among leaves.
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
        Number of currently stored leaves (items).

        Returns:
            int: Current count of leaf nodes (<= capacity).
        """
        return len(self._memory) - (self._capacity - 1)

    def __getitem__(self, index):
        """
        Get the leaf at the given external index.

        Args:
            index (int): Index in range [0, len(self)).

        Returns:
            _SumRow: The stored item and its priority at the leaf.
        """
        return self._memory[self._capacity - 1:][index]

    def __getslice__(self, start, end):
        """
        Get a slice of leaves between start and end.

        Args:
            start (int): Start index.
            end (int): End index (exclusive).

        Returns:
            list: Slice of _SumRow entries.

        Note:
            There's a likely bug in the original code (missing return). This
            method returns the intended slice for convenience.
        """
        return self._memory[self._capacity - 1:][start:end]
