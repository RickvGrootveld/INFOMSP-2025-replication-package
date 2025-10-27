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
    """Binary sum-tree for prioritized sampling.

    This data structure stores items at the leaves with associated priorities,
    while each internal node stores the sum of its children's priorities.
    It supports:
    - Insertion (cyclic over a fixed capacity).
    - Priority updates for existing items.
    - Sampling by priority: given a random value p in [0, total_priority),
      the tree can be traversed to find the leaf corresponding to that segment.

    The memory layout uses a flat array representation:
    - Internal nodes occupy indices [0, capacity - 2] and store sums.
    - Leaves occupy indices [capacity - 1, capacity - 1 + size - 1] and store _SumRow(item, priority).
    """

    def __init__(self, capacity):
        """Initialize the sum-tree with a fixed number of leaves.

        Args:
            capacity (int): Maximum number of items (leaf nodes).
        """
        self._capacity = capacity

        # Internal nodes initialized with zeros; length = capacity - 1
        self._memory = [0] * (capacity - 1)
        # Next write position within the leaf segment [0, capacity)
        self._position = 0
        # Total backing array size (internal + leaf region)
        self._actual_capacity = 2 * self._capacity - 1

    def put(self, item, priority=None):
        """Insert an item with an optional priority (overwrites cyclically).

        If the tree is not full yet, extend the memory with a new leaf slot.
        Otherwise, overwrite the leaf at the current cyclic position.

        Args:
            item (Any): The object to store at the leaf.
            priority (float or None): Priority associated with the item. If None, treated as 0.
        """
        if not self._isfull():
            # Grow leaf region until reaching capacity
            self._memory.append(None)
        # Compute absolute leaf index to write and advance cyclic pointer
        position = self._next_position_then_increment()
        # Determine old priority to compute delta for internal nodes
        old_priority = 0 if self._memory[position] is None \
            else (self._memory[position].priority or 0)
        row = _SumRow(item, priority)
        self._memory[position] = row
        # Propagate priority delta up to the root
        self._update_internal_nodes(
            position, (row.priority or 0) - old_priority)

    def move(self, external_index, new_priority):
        """Update the priority of an existing item by its external (leaf) index.

        Args:
            external_index (int): Zero-based index within current stored items (0..len(self)-1).
            new_priority (float): New priority to assign.

        Returns:
            None
        """
        index = external_index + (self._capacity - 1)
        return self._move(index, new_priority)

    def _move(self, index, new_priority):
        """Internal helper to update a leaf's priority by absolute memory index.

        Args:
            index (int): Absolute index in backing array corresponding to a leaf.
            new_priority (float): New priority to assign.

        Returns:
            None
        """
        item, old_priority = self._memory[index]
        old_priority = old_priority or 0
        self._memory[index] = _SumRow(item, new_priority)
        # Update internal nodes with the delta
        self._update_internal_nodes(index, new_priority - old_priority)

    def _update_internal_nodes(self, index, delta):
        """Propagate a change in a leaf's priority up the tree.

        Starting from a leaf index, walk up to the root, updating the sum at
        each parent by the provided delta.

        Args:
            index (int): Absolute index of the starting node (leaf).
            delta (float): Change in priority to add to each ancestor.

        Returns:
            None
        """
        while index > 0:
            # Move to parent index
            index = (index - 1) // 2
            self._memory[index] += delta

    def _isfull(self):
        """Check whether the leaf region has reached capacity.

        Returns:
            bool: True if number of stored items equals capacity.
        """
        return len(self) == self._capacity

    def _next_position_then_increment(self):
        """Get current cyclic write position (as absolute leaf index) and advance it.

        Returns:
            int: Absolute index within backing array for the next write.
        """
        start = self._capacity - 1
        position = start + self._position
        # Advance cyclically over [0, capacity)
        self._position = (self._position + 1) % self._capacity
        return position

    def _sample_with_priority(self, p):
        """Traverse the tree to find the leaf corresponding to cumulative mass p.

        Args:
            p (float): Value in [0, total_priority). For zero total priority, caller must avoid this.

        Returns:
            int: Absolute index of the selected leaf in the backing array.

        Raises:
            RuntimeError: If traversal reaches a missing right child unexpectedly.
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            # If left child index exceeds memory, current parent is a leaf
            if left >= len(self._memory):
                return parent

            # Fetch left child's sum or priority depending on internal/leaf region
            left_p = self._memory[left] if left < self._capacity - 1 \
                else (self._memory[left].priority or 0)
            if p <= left_p:
                parent = left
            else:
                if left + 1 >= len(self._memory):
                    # Right child must exist if p > left_p
                    raise RuntimeError('Right child is expected to exist.')
                p -= left_p
                parent = left + 1

    def sample_minibatch(self, batch_size):
        """Sample a minibatch of indices and rows proportional to priority.

        The interval [0, total_priority) is split into equal ranges, and one
        sample is drawn uniformly from each range to reduce variance (stratified sampling).
        If total priority is effectively zero, fall back to uniform sampling over current leaves.

        Args:
            batch_size (int): Number of samples to draw.

        Returns:
            list[tuple[int, _SumRow]]: List of (absolute_index, _SumRow(item, priority)).
        """
        pool_size = len(self)
        if pool_size == 0:
            return []

        # Root stores total priority mass
        delta_p = self._memory[0] / batch_size
        chosen_idx = []
        if abs(self._memory[0]) < util.epsilon:
            # No meaningful priorities: uniform over existing leaf slots
            chosen_idx = np.random.randint(self._capacity - 1, self._capacity - 1 + len(self), size=batch_size).tolist()
        else:
            for i in xrange(batch_size):
                lower = max(i * delta_p, 0)
                upper = min((i + 1) * delta_p, self._memory[0])
                p = random.uniform(lower, upper)
                chosen_idx.append(self._sample_with_priority(p))
        return [(i, self._memory[i]) for i in chosen_idx]

    def __len__(self):
        """Number of items currently stored (leaf count).

        Returns:
            int: Current number of leaves populated (<= capacity).
        """
        return len(self._memory) - (self._capacity - 1)

    def __getitem__(self, index):
        """Get leaf rows by relative index into the leaf region.

        Args:
            index (int or slice): Relative index starting at 0 for first leaf.

        Returns:
            _SumRow or list[_SumRow]: Retrieved item(s) from the leaf region.
        """
        return self._memory[self._capacity - 1:][index]

    def __getslice__(self, start, end):
        """Deprecated slice accessor retained for backward compatibility.

        Note: This method appears to have a bug (missing return). Kept as-is to
        preserve original behavior; prefer __getitem__ with slices instead.

        Args:
            start (int): Start index (relative to leaf region).
            end (int): End index (relative to leaf region).

        Returns:
            None
        """
        self.memory[self._capacity - 1:][start:end]
