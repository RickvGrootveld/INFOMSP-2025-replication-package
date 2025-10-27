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
    SumTree stores items with associated priorities in a binary tree structure that
    supports efficient weighted random sampling and priority updates.

    Tree layout (array-based):
    - Internal nodes [0 .. capacity-2] store the sum of priorities of their children.
    - Leaf nodes [capacity-1 .. capacity-1+size-1] store (_SumRow(item, priority)).
    - The root (index 0) stores the total sum of priorities.
    - New items are inserted in a cyclic fashion over the leaf range.

    Typical use: prioritized experience replay, where items are sampled with
    probability proportional to their priority.
    """
    
    def __init__(self, capacity):
        """
        Initialize the sum tree.

        Args:
            capacity (int): Maximum number of items (leaf nodes) that can be stored.
        """
        self._capacity = capacity

        # Internal nodes initially (capacity - 1) zeros for the segment tree sums.
        self._memory = [0] * (capacity - 1)
        # Next write position (relative to the leaf section).
        self._position = 0
        # Total size of array when full: internal nodes + leaves.
        self._actual_capacity = 2 * self._capacity - 1

    def put(self, item, priority=None):
        """
        Insert or overwrite an item at the current write position with given priority.

        If the structure is not yet full, append a new leaf; otherwise, overwrite
        the leaf at the current cyclic position. Internal sums are updated accordingly.

        Args:
            item (Any): The value to store.
            priority (float or None): Priority associated with the item (defaults to 0 if None).
        """
        if not self._isfull():
            # Until full, grow the leaf section by appending actual leaf rows.
            self._memory.append(None)
        position = self._next_position_then_increment()
        # Old priority for delta update (0 if leaf is new or had None).
        old_priority = 0 if self._memory[position] is None \
            else (self._memory[position].priority or 0)
        row = _SumRow(item, priority)
        self._memory[position] = row
        # Update internal node sums with the change in priority.
        self._update_internal_nodes(
            position, (row.priority or 0) - old_priority)

    def move(self, external_index, new_priority):
        """
        Update the priority of an existing item by external (leaf-relative) index.

        Args:
            external_index (int): Index relative to the leaf section (0-based).
            new_priority (float): New priority value.

        Returns:
            None
        """
        index = external_index + (self._capacity - 1)
        return self._move(index, new_priority)

    def _move(self, index, new_priority):
        """
        Internal method to update priority at an absolute array index.

        Args:
            index (int): Absolute index in the underlying array (leaf node index).
            new_priority (float): New priority.

        Returns:
            None
        """
        item, old_priority = self._memory[index]
        old_priority = old_priority or 0
        self._memory[index] = _SumRow(item, new_priority)
        # Propagate priority difference to internal nodes.
        self._update_internal_nodes(index, new_priority - old_priority)

    def _update_internal_nodes(self, index, delta):
        """
        Propagate a priority change from a leaf to the root by updating internal sums.

        Args:
            index (int): Starting index (leaf) where delta applies.
            delta (float): Change in priority to be added to ancestors.

        Returns:
            None
        """
        # Walk up the tree, updating each parent.
        while index > 0:
            index = (index - 1) // 2
            self._memory[index] += delta

    def _isfull(self):
        """
        Check if the tree has reached its capacity (number of stored items).

        Returns:
            bool: True if the number of items equals capacity, else False.
        """
        return len(self) == self._capacity

    def _next_position_then_increment(self):
        """
        Compute the absolute array position for the next write, then advance the pointer.

        Returns:
            int: Absolute index of the next leaf slot to write into.
        """
        start = self._capacity - 1  # First leaf index
        position = start + self._position
        # Cyclic increment within the leaf range.
        self._position = (self._position + 1) % self._capacity
        return position

    def _sample_with_priority(self, p):
        """
        Traverse the tree to find the leaf corresponding to cumulative mass p.

        Args:
            p (float): Target cumulative priority in [0, total_priority].

        Returns:
            int: Absolute index of the selected leaf node.
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            # If no children, we've reached a leaf (in the internal segment area).
            if left >= len(self._memory):
                return parent

            # Get left child priority or subtree sum.
            left_p = self._memory[left] if left < self._capacity - 1 \
                else (self._memory[left].priority or 0)
            if p <= left_p:
                parent = left
            else:
                # Move to right child after subtracting left mass.
                if left + 1 >= len(self._memory):
                    raise RuntimeError('Right child is expected to exist.')
                p -= left_p
                parent = left + 1

    def sample_minibatch(self, batch_size):
        """
        Sample a minibatch of leaves proportional to their priority.

        The total priority range [0, total] is split into equal segments, and one
        sample is drawn uniformly from each segment. If the total priority is ~0,
        resort to uniform sampling over existing leaves.

        Args:
            batch_size (int): Number of samples to draw.

        Returns:
            list[tuple[int, _SumRow]]: List of (absolute_index, _SumRow(item, priority)).
        """
        pool_size = len(self)
        if pool_size == 0:
            return []

        # Root holds total priority mass for the tree.
        delta_p = self._memory[0] / batch_size
        chosen_idx = []
        if abs(self._memory[0]) < util.epsilon:
            # If total priority is near zero, sample uniformly among existing leaves.
            chosen_idx = np.random.randint(self._capacity - 1, self._capacity - 1 + len(self), size=batch_size).tolist()
        else:
            for i in xrange(batch_size):
                # Define segment [lower, upper] for stratified sampling.
                lower = max(i * delta_p, 0)
                upper = min((i + 1) * delta_p, self._memory[0])
                p = random.uniform(lower, upper)
                chosen_idx.append(self._sample_with_priority(p))
        return [(i, self._memory[i]) for i in chosen_idx]

    def __len__(self):
        """
        Number of items currently stored (number of valid leaves).

        Returns:
            int: Current size (<= capacity).
        """
        return len(self._memory) - (self._capacity - 1)

    def __getitem__(self, index):
        """
        Get the leaf at a given external index (0-based over stored items).

        Args:
            index (int): External index into the leaf section.

        Returns:
            _SumRow: The stored (item, priority) at the requested index.
        """
        return self._memory[self._capacity - 1:][index]

    def __getslice__(self, start, end):
        """
        Slice over the leaf section [start:end].

        Args:
            start (int): Start index (inclusive).
            end (int): End index (exclusive).

        Returns:
            list[_SumRow]: Sliced list of stored rows.

        Note:
            As written, this method appears to be a bug (uses self.memory). It should
            likely return self._memory[self._capacity - 1:][start:end].
        """
        self.memory[self._capacity - 1:][start:end]
