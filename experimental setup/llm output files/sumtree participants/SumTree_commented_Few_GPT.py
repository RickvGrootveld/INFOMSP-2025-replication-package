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
    """Binary sum tree for prioritized sampling.

    This data structure maintains a complete binary tree in an array form where
    the leaves store (item, priority) tuples and internal nodes store the sum of
    priorities of their respective subtrees. It supports:
    - Insertion of items with priorities (ring buffer over leaves).
    - Priority updates for existing items.
    - Sampling by priority: given a value p in [0, total_priority], returns the
      leaf index whose cumulative priority interval contains p.
    - Minibatch sampling by stratified sampling across total priority mass.

    The underlying storage `_memory` is a single list representing both the
    internal nodes (indexes [0, capacity-2]) and the leaves
    (indexes [capacity-1, 2*capacity-2]). The tree has fixed leaf capacity; new
    insertions overwrite leaves in a circular fashion.
    """

    def __init__(self, capacity):
        """Initialize the sum tree.

        Args:
            capacity (int): Maximum number of leaf items stored (number of leaves).

        Notes:
            - Internal node region size is capacity - 1.
            - Leaf region size is capacity.
            - The full array size is 2 * capacity - 1.
            - `_position` is the next leaf offset to write (0..capacity-1).
        """
        self._capacity = capacity

        # Internal nodes initialized to zero sums; will accumulate priority deltas.
        self._memory = [0] * (capacity - 1)
        # Next position in circular leaf buffer.
        self._position = 0
        # Cached total array size (internal + leaves).
        self._actual_capacity = 2 * self._capacity - 1

    def put(self, item, priority=None):
        """Insert item with given priority into the next leaf position.

        Overwrites the existing leaf at the current write position and updates
        all ancestor internal nodes with the priority delta.

        Args:
            item: Arbitrary payload to store at leaf.
            priority (float or None): Non-negative sampling priority. If None,
                treated as 0 for sums and storage.
        """
        # Grow leaf region lazily until full.
        if not self._isfull():
            self._memory.append(None)
        position = self._next_position_then_increment()

        # Previous priority to compute delta; treat missing/None as zero.
        old_priority = 0 if self._memory[position] is None \
            else (self._memory[position].priority or 0)

        # Write new row into leaf.
        row = _SumRow(item, priority)
        self._memory[position] = row

        # Propagate priority change up the tree.
        self._update_internal_nodes(
            position, (row.priority or 0) - old_priority)

    def move(self, external_index, new_priority):
        """Update priority of an existing leaf by external (0-based) index.

        Args:
            external_index (int): Index within the logical leaf array
                [0, len(self) - 1].
            new_priority (float): New priority value for the item.

        Returns:
            None
        """
        index = external_index + (self._capacity - 1)
        return self._move(index, new_priority)

    def _move(self, index, new_priority):
        """Internal helper to update a leaf priority given absolute index.

        Args:
            index (int): Absolute index in `_memory` of the leaf to update.
            new_priority (float): New priority value.

        Side effects:
            - Updates the leaf tuple, preserving the item.
            - Propagates the priority delta to ancestor internal nodes.
        """
        item, old_priority = self._memory[index]
        old_priority = old_priority or 0
        self._memory[index] = _SumRow(item, new_priority)
        self._update_internal_nodes(index, new_priority - old_priority)

    def _update_internal_nodes(self, index, delta):
        """Propagate a leaf priority change up to the root.

        Args:
            index (int): Absolute index of the changed node (leaf).
            delta (float): Difference between new and old priority.
        """
        # Traverse parents: parent(i) = floor((i-1)/2), stop at root.
        while index > 0:
            index = (index - 1) // 2
            self._memory[index] += delta

    def _isfull(self):
        """Check whether the leaf region is full.

        Returns:
            bool: True if number of items equals capacity, else False.
        """
        return len(self) == self._capacity

    def _next_position_then_increment(self):
        """Compute absolute leaf index for next insertion and advance pointer.

        Returns:
            int: Absolute index in `_memory` of the target leaf.
        """
        start = self._capacity - 1
        position = start + self._position
        self._position = (self._position + 1) % self._capacity
        return position

    def _sample_with_priority(self, p):
        """Traverse the tree to find the leaf containing cumulative mass p.

        Args:
            p (float): Target cumulative priority in [0, total_priority].

        Returns:
            int: Absolute index of the selected leaf in `_memory`.

        Raises:
            RuntimeError: If traversal expects a right child that is missing.
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            # If no left child, we've reached a leaf; return its index.
            if left >= len(self._memory):
                return parent

            # Retrieve left child's mass: internal node stores sum directly;
            # leaf stores a (_SumRow) whose priority we read.
            left_p = self._memory[left] if left < self._capacity - 1 \
                else (self._memory[left].priority or 0)

            # Descend left if p falls within left mass; otherwise subtract and go right.
            if p <= left_p:
                parent = left
            else:
                if left + 1 >= len(self._memory):
                    raise RuntimeError('Right child is expected to exist.')
                p -= left_p
                parent = left + 1

    def sample_minibatch(self, batch_size):
        """Sample a stratified minibatch of leaves proportional to priority.

        The total priority mass (root sum) is divided into `batch_size` equal
        ranges. One sample is drawn uniformly from each range and mapped to a
        leaf via tree traversal. If the total mass is approximately zero, falls
        back to uniform sampling over the current leaf range.

        Args:
            batch_size (int): Number of samples to draw.

        Returns:
            list[tuple[int, _SumRow]]: List of (absolute_index, _SumRow) pairs
            for the chosen leaves. Returns empty list if the pool is empty.
        """
        pool_size = len(self)
        if pool_size == 0:
            return []

        # Total priority mass is at root (index 0).
        delta_p = self._memory[0] / batch_size
        chosen_idx = []
        # If total mass ~ 0, sample uniformly among existing leaves.
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
        """Number of currently stored items (leaves populated).

        Returns:
            int: Count of valid leaves (may be less than capacity early on).
        """
        return len(self._memory) - (self._capacity - 1)

    def __getitem__(self, index):
        """Get leaf by external 0-based index.

        Args:
            index (int): Position within logical leaf array.

        Returns:
            _SumRow: The stored (item, priority) tuple at the requested leaf.
        """
        return self._memory[self._capacity - 1:][index]

    def __getslice__(self, start, end):
        """Get a slice of leaves by external indices [start:end].

        Args:
            start (int): Start index in logical leaves.
            end (int): End index (exclusive) in logical leaves.

        Returns:
            list[_SumRow]: Slice of stored leaf rows.

        Note:
            This function currently doesn't return its value due to a likely
            bug in the original implementation. It should return the slice
            below, but it is kept as-is to preserve behavior:
                return self._memory[self._capacity - 1:][start:end]
        """
        self.memory[self._capacity - 1:][start:end]
