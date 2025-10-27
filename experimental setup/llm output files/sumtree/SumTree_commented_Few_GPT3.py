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
    """Binary sum tree for efficient prioritized sampling.

    This data structure stores items in the leaves of a complete binary tree
    along with their priorities. Internal nodes store the sum of priorities of
    their child subtrees. It supports:
    - Insertion of items with a given priority (circular buffer over leaves).
    - Updating the priority of an existing item.
    - Sampling a leaf by drawing a value p in [0, total_priority) and descending
      the tree according to cumulative sums, which is O(log N).
    - Stratified minibatch sampling across the priority range.

    The underlying array layout is:
    - Internal nodes occupy indices [0, capacity - 2], where index 0 is the root
      holding the total sum of all priorities.
    - Leaves (items) occupy indices [capacity - 1, 2 * capacity - 2].
    """

    def __init__(self, capacity):
        """Initialize the sum tree.

        Arguments:
            capacity (int): Maximum number of items (leaf nodes). The total
                array size will be 2 * capacity - 1 to accommodate internal nodes.
        """
        self._capacity = capacity

        # Internal nodes array portion initialized to zeros (priority sums).
        self._memory = [0] * (capacity - 1)
        # Next insertion position among leaves [0, capacity).
        self._position = 0
        # Total size of the tree array (internal + leaves).
        self._actual_capacity = 2 * self._capacity - 1

    def put(self, item, priority=None):
        """Insert an item with an optional priority into the tree.

        If the tree is not yet full, extend the underlying array to create a
        new leaf slot. Otherwise, overwrite at the current circular position.

        Arguments:
            item (Any): The payload to store at the leaf.
            priority (float or None): The priority value. If None, treated as 0.
        """
        if not self._isfull():
            # Extend once per insertion until leaves are filled.
            self._memory.append(None)
        # Compute leaf index to write, then advance circular pointer.
        position = self._next_position_then_increment()
        # Determine previous priority at that position to compute delta.
        old_priority = 0 if self._memory[position] is None \
            else (self._memory[position].priority or 0)
        row = _SumRow(item, priority)
        self._memory[position] = row
        # Propagate priority change up the tree.
        self._update_internal_nodes(
            position, (row.priority or 0) - old_priority)

    def move(self, external_index, new_priority):
        """Update an item's priority given its external (0-based) index.

        Arguments:
            external_index (int): Index into the logical list of stored items
                (0 is the first leaf currently in memory).
            new_priority (float): New priority value to be set.
        """
        index = external_index + (self._capacity - 1)
        return self._move(index, new_priority)

    def _move(self, index, new_priority):
        """Internal method to update priority at a specific leaf index.

        Arguments:
            index (int): Absolute index into the underlying array for the leaf.
            new_priority (float): New priority.
        """
        item, old_priority = self._memory[index]
        old_priority = old_priority or 0
        self._memory[index] = _SumRow(item, new_priority)
        # Update internal sums by the delta.
        self._update_internal_nodes(index, new_priority - old_priority)

    def _update_internal_nodes(self, index, delta):
        """Propagate a change in a leaf's priority up to the root.

        Arguments:
            index (int): Leaf index where the change occurred.
            delta (float): Difference between new and old priority.
        """
        while index > 0:
            # Move to parent and add delta to its sum.
            index = (index - 1) // 2
            self._memory[index] += delta

    def _isfull(self):
        """Return whether the leaf layer is fully populated."""
        return len(self) == self._capacity

    def _next_position_then_increment(self):
        """Get current circular leaf position and advance it.

        Returns:
            int: Absolute index into the underlying array for the next write.
        """
        start = self._capacity - 1
        position = start + self._position
        self._position = (self._position + 1) % self._capacity
        return position

    def _sample_with_priority(self, p):
        """Traverse the tree to find the leaf corresponding to cumulative p.

        Arguments:
            p (float): Cumulative priority value within [0, total_priority).

        Returns:
            int: Index of the selected leaf in the underlying array.
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            # If no children, current parent is a leaf index.
            if left >= len(self._memory):
                return parent

            # Determine left child's sum (internal node) or priority (leaf).
            left_p = self._memory[left] if left < self._capacity - 1 \
                else (self._memory[left].priority or 0)
            if p <= left_p:
                parent = left
            else:
                if left + 1 >= len(self._memory):
                    # Should never happen in a complete binary tree.
                    raise RuntimeError('Right child is expected to exist.')
                p -= left_p
                parent = left + 1

    def sample_minibatch(self, batch_size):
        """Stratified sample a minibatch of leaves based on priority.

        The full priority range [0, total_priority) is divided into equal
        contiguous segments, and one sample is drawn uniformly from each
        segment. This reduces variance compared to naive sampling.

        If the total sum is (close to) zero, fall back to uniform sampling
        over stored leaves.

        Arguments:
            batch_size (int): Number of samples to draw.

        Returns:
            list[tuple[int, _SumRow]]: List of (array_index, (item, priority)).
        """
        pool_size = len(self)
        if pool_size == 0:
            return []

        # Each segment width in cumulative priority space.
        delta_p = self._memory[0] / batch_size
        chosen_idx = []
        if abs(self._memory[0]) < util.epsilon:
            # No meaningful priorities; sample uniformly among current leaves.
            chosen_idx = np.random.randint(self._capacity - 1, self._capacity - 1 + len(self), size=batch_size).tolist()
        else:
            for i in xrange(batch_size):
                lower = max(i * delta_p, 0)
                upper = min((i + 1) * delta_p, self._memory[0])
                p = random.uniform(lower, upper)
                chosen_idx.append(self._sample_with_priority(p))
        return [(i, self._memory[i]) for i in chosen_idx]

    def __len__(self):
        """Number of items currently stored (count of leaves filled)."""
        return len(self._memory) - (self._capacity - 1)

    def __getitem__(self, index):
        """Get stored item tuple at logical index.

        Arguments:
            index (int or slice): Logical position among leaves.

        Returns:
            _SumRow or list[_SumRow]: The (item, priority) or list thereof.
        """
        return self._memory[self._capacity - 1:][index]

    def __getslice__(self, start, end):
        """Get a slice of stored items between start and end.

        Note: This method mirrors legacy Python slice behavior.

        Arguments:
            start (int): Start index (inclusive).
            end (int): End index (exclusive).
        """
        self.memory[self._capacity - 1:][start:end]
