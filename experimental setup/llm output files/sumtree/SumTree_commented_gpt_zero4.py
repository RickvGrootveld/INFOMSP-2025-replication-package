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
    SumTree implements a binary tree data structure that stores priorities in internal
    nodes and (item, priority) pairs in leaf nodes. It supports:
    - Insertion of items with associated priorities (with ring-buffer overwrite behavior).
    - Updating priorities of existing items.
    - Efficient sampling of leaf indices proportional to their priority mass (stratified).
    
    The underlying storage layout:
    - A complete binary tree stored in a single list `self._memory`.
    - Internal nodes (length = capacity - 1) store sums of child priorities.
    - Leaf nodes (length = capacity) store _SumRow(item, priority).
    - The total length once full is 2 * capacity - 1.
    
    This data structure is commonly used for prioritized experience replay.
    """
    
    def __init__(self, capacity):
        """
        Initialize a SumTree with a fixed capacity (number of leaves).

        Args:
            capacity (int): Maximum number of leaf items the tree can hold.
                            Must be a positive integer.
        """
        self._capacity = capacity

        # Initialize internal nodes (sum tree) with zeros; leaves will be appended later.
        self._memory = [0] * (capacity - 1)
        # Next insertion position among leaves (ring buffer index).
        self._position = 0
        # Precomputed actual total storage size (internal + leaves) once full.
        self._actual_capacity = 2 * self._capacity - 1

    def put(self, item, priority=None):
        """
        Insert an item with an optional priority. If the tree is not full yet,
        append a new leaf; otherwise, overwrite the leaf at the current ring-buffer position.

        Args:
            item (Any): The payload to store at a leaf.
            priority (float or None): Priority mass associated with the item. If None,
                                      treated as 0 for summation/sampling.
        """
        # If not full, extend storage with a new leaf slot.
        if not self._isfull():
            self._memory.append(None)
        # Compute the leaf index to write to and advance the ring-buffer pointer.
        position = self._next_position_then_increment()
        # Determine the prior priority at this position (for delta update of sums).
        old_priority = 0 if self._memory[position] is None \
            else (self._memory[position].priority or 0)
        # Store the (item, priority) at the leaf.
        row = _SumRow(item, priority)
        self._memory[position] = row
        # Propagate the priority delta up to internal sum nodes.
        self._update_internal_nodes(
            position, (row.priority or 0) - old_priority)

    def move(self, external_index, new_priority):
        """
        Update the priority of an existing item by its external (leaf) index.

        Args:
            external_index (int): Zero-based index into the list of leaves (0..len(self)-1).
            new_priority (float): The new priority value to assign.

        Returns:
            None
        """
        # Translate external leaf index to the absolute array index.
        index = external_index + (self._capacity - 1)
        return self._move(index, new_priority)

    def _move(self, index, new_priority):
        """
        Internal helper to update a leaf's priority and propagate changes upward.

        Args:
            index (int): Absolute index in self._memory corresponding to a leaf.
            new_priority (float): The new priority value to assign.

        Returns:
            None
        """
        item, old_priority = self._memory[index]
        old_priority = old_priority or 0
        # Update leaf with new priority.
        self._memory[index] = _SumRow(item, new_priority)
        # Update internal nodes by the delta.
        self._update_internal_nodes(index, new_priority - old_priority)

    def _update_internal_nodes(self, index, delta):
        """
        Propagate a priority delta from a leaf up to the root, updating internal sums.

        Args:
            index (int): Absolute index of the starting node (usually a leaf).
            delta (float): Change to apply to all ancestor sums.

        Returns:
            None
        """
        # Climb towards the root, updating parent sums along the path.
        while index > 0:
            index = (index - 1) // 2
            self._memory[index] += delta

    def _isfull(self):
        """
        Check whether the leaf-level capacity has been reached.

        Returns:
            bool: True if the number of stored leaves equals capacity, else False.
        """
        return len(self) == self._capacity

    def _next_position_then_increment(self):
        """
        Compute the absolute index for the next leaf insertion and advance the pointer.

        Returns:
            int: Absolute index in self._memory where the next item should be stored.
        """
        start = self._capacity - 1  # Starting index of leaves in the array layout.
        position = start + self._position
        # Advance ring buffer position modulo capacity.
        self._position = (self._position + 1) % self._capacity
        return position

    def _sample_with_priority(self, p):
        """
        Traverse the sum tree to find the leaf index corresponding to cumulative mass p.

        Args:
            p (float): A value in [0, total_priority] used to locate a leaf proportionally.

        Returns:
            int: Absolute index of the located leaf in self._memory.
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            # If left child is beyond array, we've reached a leaf (since internal nodes precede leaves).
            if left >= len(self._memory):
                return parent

            # left_p: priority mass of the left subtree.
            # For internal nodes, it's stored directly.
            # For leaf nodes, use the leaf's priority (or 0 if None).
            left_p = self._memory[left] if left < self._capacity - 1 \
                else (self._memory[left].priority or 0)
            if p <= left_p:
                parent = left
            else:
                # Ensure right child exists.
                if left + 1 >= len(self._memory):
                    raise RuntimeError('Right child is expected to exist.')
                p -= left_p
                parent = left + 1

    def sample_minibatch(self, batch_size):
        """
        Sample a minibatch of leaf indices and rows proportionally to their priorities,
        using stratified sampling across the total priority mass. If the total mass is
        (near) zero, sample uniformly among existing leaves.

        Args:
            batch_size (int): Number of samples to draw.

        Returns:
            list[tuple[int, _SumRow]]: A list of (absolute_index, _SumRow(item, priority)).
        """
        pool_size = len(self)
        if pool_size == 0:
            return []

        # Total priority mass is stored at the root (index 0). Each stratum has width delta_p.
        delta_p = self._memory[0] / batch_size
        chosen_idx = []
        # If total mass is effectively zero, fallback to uniform sampling over existing leaves.
        if abs(self._memory[0]) < util.epsilon:
            chosen_idx = np.random.randint(self._capacity - 1, self._capacity - 1 + len(self), size=batch_size).tolist()
        else:
            # Stratified sampling: draw one sample uniformly from each consecutive mass interval.
            for i in xrange(batch_size):
                lower = max(i * delta_p, 0)
                upper = min((i + 1) * delta_p, self._memory[0])
                p = random.uniform(lower, upper)
                chosen_idx.append(self._sample_with_priority(p))
        return [(i, self._memory[i]) for i in chosen_idx]

    def __len__(self):
        """
        Number of currently stored leaves.

        Returns:
            int: Count of leaves inserted (up to capacity).
        """
        return len(self._memory) - (self._capacity - 1)

    def __getitem__(self, index):
        """
        Index into the leaves as a contiguous list.

        Args:
            index (int): Zero-based index among current leaves.

        Returns:
            _SumRow: The (item, priority) tuple at the given leaf index.
        """
        return self._memory[self._capacity - 1:][index]

    def __getslice__(self, start, end):
        """
        Slice into the leaves as a contiguous list.

        Args:
            start (int): Start index (inclusive).
            end (int): End index (exclusive).

        Returns:
            list[_SumRow]: Slice of (item, priority) tuples among the leaves.
        """
        self.memory[self._capacity - 1:][start:end]
