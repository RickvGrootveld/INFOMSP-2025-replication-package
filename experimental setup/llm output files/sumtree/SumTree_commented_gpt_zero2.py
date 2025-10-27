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
    SumTree data structure for prioritized sampling.

    This binary tree stores leaf nodes as (item, priority) pairs and internal
    nodes as the sum of priorities of their children. It supports:
    - Insertion with priority (circular buffer behavior).
    - Updating an item's priority.
    - Sampling leaves proportional to priority via prefix-sum lookup.

    The internal array layout is:
    - Indices [0 .. capacity-2]: internal nodes (priority sums).
    - Indices [capacity-1 .. 2*capacity-2]: leaf nodes (items with priorities).

    The tree grows from an initial list of zeros for internal nodes and appends
    leaves as items are inserted up to the configured capacity. After reaching
    capacity, insertions overwrite leaves in a round-robin manner.
    """
    
    def __init__(self, capacity):
        """
        Initialize the SumTree.

        Args:
            capacity (int): Maximum number of items (leaf nodes) the tree can hold.
        """
        self._capacity = capacity

        # Internal nodes initialized to 0; length is capacity - 1 (full binary tree property)
        self._memory = [0] * (capacity - 1)
        # Next circular position among leaves to write to
        self._position = 0
        # Total nodes in a full tree = 2 * capacity - 1
        self._actual_capacity = 2 * self._capacity - 1

    def put(self, item, priority=None):
        """
        Insert an item with a given priority into the tree.

        If the tree has not reached capacity, append a new leaf. Once at capacity,
        overwrite the next leaf in circular order.

        Args:
            item: Arbitrary payload to store at the leaf.
            priority (float or None): Priority value; if None or falsy, treated as 0.
        """
        if not self._isfull():
            # Append a placeholder for a new leaf node until capacity is reached
            self._memory.append(None)
        position = self._next_position_then_increment()
        # Determine previous priority to compute delta for internal nodes
        old_priority = 0 if self._memory[position] is None \
            else (self._memory[position].priority or 0)
        row = _SumRow(item, priority)
        self._memory[position] = row
        # Propagate the change in priority up the tree
        self._update_internal_nodes(
            position, (row.priority or 0) - old_priority)

    def move(self, external_index, new_priority):
        """
        Update the priority of a leaf, given its external (leaf-level) index.

        Args:
            external_index (int): Index into the leaf array [0 .. size-1].
            new_priority (float): New priority to assign.

        Returns:
            None
        """
        index = external_index + (self._capacity - 1)
        return self._move(index, new_priority)

    def _move(self, index, new_priority):
        """
        Internal helper to update a leaf's priority by absolute array index.

        Args:
            index (int): Absolute index into the backing array for a leaf.
            new_priority (float): New priority to assign.

        Returns:
            None
        """
        item, old_priority = self._memory[index]
        old_priority = old_priority or 0
        self._memory[index] = _SumRow(item, new_priority)
        # Propagate the priority delta up the tree
        self._update_internal_nodes(index, new_priority - old_priority)

    def _update_internal_nodes(self, index, delta):
        """
        Propagate a priority change from a leaf up to the root.

        Args:
            index (int): Starting index (leaf) from which to propagate.
            delta (float): Change in priority to add to each ancestor.

        Returns:
            None
        """
        while index > 0:
            # Move to parent
            index = (index - 1) // 2
            self._memory[index] += delta

    def _isfull(self):
        """
        Check whether the leaf capacity has been reached.

        Returns:
            bool: True if the number of stored leaves equals capacity.
        """
        return len(self) == self._capacity

    def _next_position_then_increment(self):
        """
        Compute the absolute array position for the next leaf insertion,
        then advance the circular write pointer.

        Returns:
            int: Absolute index in the backing array for the next leaf.
        """
        start = self._capacity - 1  # First leaf index
        position = start + self._position
        # Circular increment across the capacity range
        self._position = (self._position + 1) % self._capacity
        return position

    def _sample_with_priority(self, p):
        """
        Perform a prefix-sum lookup to find the leaf corresponding to cumulative mass p.

        Traverses from the root, descending left or right depending on p relative
        to the left child's sum, until a leaf index is reached.

        Args:
            p (float): Target cumulative priority within [0, total_priority].

        Returns:
            int: Absolute index of the selected leaf in the backing array.

        Raises:
            RuntimeError: If expected right child is missing (consistency error).
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            # If left child is out of range, parent is a leaf
            if left >= len(self._memory):
                return parent

            # Fetch left child's sum; if at leaf level, use the stored leaf priority
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
        Sample a minibatch of leaves proportional to their priority.

        The total priority mass is split into contiguous segments and one sample
        is drawn uniformly from each segment (stratified sampling). If the total
        mass is effectively zero, sampling falls back to uniform over the currently
        filled leaf range.

        Args:
            batch_size (int): Number of samples to draw.

        Returns:
            list of tuples: Each entry is (absolute_index, _SumRow(item, priority)).
        """
        pool_size = len(self)
        if pool_size == 0:
            return []

        delta_p = self._memory[0] / batch_size  # Total mass per stratum
        chosen_idx = []
        # If total priority is near zero, sample uniformly among existing leaves
        if abs(self._memory[0]) < util.epsilon:
            chosen_idx = np.random.randint(self._capacity - 1, self._capacity - 1 + len(self), size=batch_size).tolist()
        else:
            for i in xrange(batch_size):
                # Define stratum bounds and sample a target mass
                lower = max(i * delta_p, 0)
                upper = min((i + 1) * delta_p, self._memory[0])
                p = random.uniform(lower, upper)
                chosen_idx.append(self._sample_with_priority(p))
        return [(i, self._memory[i]) for i in chosen_idx]

    def __len__(self):
        """
        Number of stored leaves (items) currently in the tree.

        Returns:
            int: Count of items inserted (up to capacity).
        """
        return len(self._memory) - (self._capacity - 1)

    def __getitem__(self, index):
        """
        Get a leaf by relative index among stored items.

        Args:
            index (int): Index in [0 .. len(self)-1].

        Returns:
            _SumRow: The (item, priority) tuple at the requested position.
        """
        return self._memory[self._capacity - 1:][index]

    def __getslice__(self, start, end):
        """
        Get a slice of leaves by relative indices.

        Note: This method returns a slice but does not assign it to a member.
        It appears unused or may be a bug; kept for API parity.

        Args:
            start (int): Start index (inclusive).
            end (int): End index (exclusive).

        Returns:
            list: Slice of _SumRow entries from leaves, if used.
        """
        self.memory[self._capacity - 1:][start:end]
