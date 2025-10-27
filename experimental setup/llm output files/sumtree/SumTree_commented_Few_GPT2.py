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
    """Binary sum-tree data structure for prioritized sampling.

    This structure stores items at the leaves and maintains, at internal nodes,
    the sum of child priorities. It supports:
    - Insertion with priority (cyclic overwrite when capacity is reached).
    - Updating an item's priority and propagating the change up the tree.
    - Sampling by priority: given a cumulative priority p in [0, total], it
      descends the tree to find the corresponding leaf index.
    - Minibatch sampling by stratified sampling over the total priority mass.

    The internal array _memory is laid out as a complete binary tree:
    - Indices [0, capacity-2] store internal node sums.
    - Indices [capacity-1, capacity-1 + size) store leaf nodes as _SumRow.
    """

    def __init__(self, capacity):
        """Initialize the sum-tree with a fixed leaf capacity.

        Args:
            capacity (int): Maximum number of leaf items the tree can hold.

        Notes:
            - Internal nodes array is initialized with zeros (capacity - 1 nodes).
            - Leaves are appended as items are inserted, up to capacity.
            - _position tracks the next leaf slot (cyclic).
        """
        self._capacity = capacity

        # Internal nodes (sums) occupy the first capacity-1 positions.
        self._memory = [0] * (capacity - 1)
        # Next leaf position offset (cyclic index into the leaf section).
        self._position = 0
        # Total backing array capacity for a full tree (internal + leaves).
        self._actual_capacity = 2 * self._capacity - 1

    def put(self, item, priority=None):
        """Insert or overwrite an item with an associated priority.

        If the tree is not yet full, allocate a new leaf slot; otherwise,
        overwrite the leaf at the current cyclic position.

        Args:
            item (Any): The payload to store at the leaf.
            priority (float or None): The sampling priority. None is treated as 0.
        """
        # Grow leaf section until capacity is reached.
        if not self._isfull():
            self._memory.append(None)
        # Compute absolute leaf index to write, then advance cyclic position.
        position = self._next_position_then_increment()
        # Determine the old priority at this leaf (0 if empty).
        old_priority = 0 if self._memory[position] is None \
            else (self._memory[position].priority or 0)
        row = _SumRow(item, priority)
        self._memory[position] = row
        # Propagate priority change up through internal nodes.
        self._update_internal_nodes(
            position, (row.priority or 0) - old_priority)

    def move(self, external_index, new_priority):
        """Update the priority of an existing item by external leaf index.

        Args:
            external_index (int): Zero-based index within the current leaf range
                [0, len(self)-1], not including internal nodes offset.
            new_priority (float): New priority value to assign.

        Returns:
            None
        """
        # Translate external leaf index to absolute array index.
        index = external_index + (self._capacity - 1)
        return self._move(index, new_priority)

    def _move(self, index, new_priority):
        """Internal priority update given an absolute array index.

        Args:
            index (int): Absolute index in _memory for the leaf.
            new_priority (float): New priority to set.

        Side effects:
            Updates leaf tuple and propagates delta priority up the tree.
        """
        item, old_priority = self._memory[index]
        old_priority = old_priority or 0
        self._memory[index] = _SumRow(item, new_priority)
        self._update_internal_nodes(index, new_priority - old_priority)

    def _update_internal_nodes(self, index, delta):
        """Propagate a priority delta from leaf to root.

        Starting from a leaf index, walk up to the root and add delta to each
        ancestor internal node to keep subtree sums consistent.

        Args:
            index (int): Absolute index of the changed leaf.
            delta (float): Priority change to apply.
        """
        while index > 0:
            # Move to parent: parent = floor((child - 1) / 2)
            index = (index - 1) // 2
            self._memory[index] += delta

    def _isfull(self):
        """Return True if the number of stored leaves reached capacity."""
        return len(self) == self._capacity

    def _next_position_then_increment(self):
        """Compute the absolute index for the next leaf write and advance.

        Returns:
            int: Absolute array index for the leaf to write next.
        """
        start = self._capacity - 1
        position = start + self._position
        # Advance cyclic pointer within [0, capacity).
        self._position = (self._position + 1) % self._capacity
        return position

    def _sample_with_priority(self, p):
        """Traverse the tree to find the leaf index for cumulative mass p.

        Descend from the root, comparing p with the left child's mass to choose
        left or right. For leaves, return their absolute index.

        Args:
            p (float): Cumulative priority in [0, total_priority].

        Returns:
            int: Absolute index of the selected leaf.
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            # If there is no left child, parent is a leaf.
            if left >= len(self._memory):
                return parent

            # Left child value: internal node sum or leaf priority.
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
        """Sample a minibatch of leaves using stratified priority sampling.

        The total priority mass is split into equal contiguous ranges, and one
        sample is drawn uniformly from each range. If the total mass is ~0,
        fall back to uniform random selection over existing leaves.

        Args:
            batch_size (int): Number of samples to draw.

        Returns:
            list[tuple[int, _SumRow]]: List of (absolute_index, leaf_tuple).
        """
        pool_size = len(self)
        if pool_size == 0:
            return []

        # Total priority mass is at root (index 0).
        delta_p = self._memory[0] / batch_size
        chosen_idx = []
        if abs(self._memory[0]) < util.epsilon:
            # If priorities sum to ~0, sample leaves uniformly at random.
            chosen_idx = np.random.randint(self._capacity - 1, self._capacity - 1 + len(self), size=batch_size).tolist()
        else:
            for i in xrange(batch_size):
                lower = max(i * delta_p, 0)
                upper = min((i + 1) * delta_p, self._memory[0])
                p = random.uniform(lower, upper)
                chosen_idx.append(self._sample_with_priority(p))
        return [(i, self._memory[i]) for i in chosen_idx]

    def __len__(self):
        """Number of valid leaves currently stored."""
        return len(self._memory) - (self._capacity - 1)

    def __getitem__(self, index):
        """Get leaf tuple by external index.

        Args:
            index (int): Index within [0, len(self)-1].

        Returns:
            _SumRow: The (item, priority) tuple at that leaf.
        """
        return self._memory[self._capacity - 1:][index]

    def __getslice__(self, start, end):
        """Slice over the leaves [start:end].

        Args:
            start (int): Start external index.
            end (int): End external index (exclusive).
        """
        self.memory[self._capacity - 1:][start:end]
