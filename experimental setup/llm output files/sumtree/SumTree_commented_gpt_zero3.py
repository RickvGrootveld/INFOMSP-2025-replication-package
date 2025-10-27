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

    This implementation represents a complete binary tree in a flat array where:
    - Internal nodes (indices [0, capacity-2]) store the sum of priorities of their children.
    - Leaf nodes (indices [capacity-1, capacity-1 + size)) store (_SumRow) tuples of (item, priority).
    - Sampling by priority is done by traversing the tree top-down with a random value in [0, total_priority].

    Key properties:
    - Fixed capacity for leaf nodes; older entries are overwritten in a circular manner.
    - Supports updating priorities and drawing stratified samples for a minibatch.
    """

    def __init__(self, capacity):
        """
        Initialize the SumTree with a fixed capacity.

        Args:
            capacity (int): The maximum number of items (leaf nodes) the tree can hold.

        Internal layout:
            - _memory holds both internal nodes and leaf nodes in a single array.
              Initially, internal nodes are zeros and no leaves are present.
            - _position points to the next leaf slot to overwrite (circular buffer).
            - _actual_capacity is the full tree array size when at max capacity.
        """
        self._capacity = capacity

        # Internal nodes initialized to zero. Leaves will be appended as items are added.
        self._memory = [0] * (capacity - 1)
        # Next leaf insertion position among the capacity leaf slots.
        self._position = 0
        # Total array length when full: internal nodes + leaf nodes.
        self._actual_capacity = 2 * self._capacity - 1

    def put(self, item, priority=None):
        """
        Insert an item with an optional priority into the tree.

        - If the tree isn't full, append a new leaf slot.
        - Otherwise, overwrite the next leaf in a circular manner.
        - Update internal node sums to reflect the change in leaf priority.

        Args:
            item (any): The payload to store at the leaf.
            priority (float or None): The priority value; treated as 0 if None.
        """
        if not self._isfull():
            # Lazily append a new leaf node until we reach full capacity.
            self._memory.append(None)
        position = self._next_position_then_increment()
        # Get old priority at this leaf to compute the delta for internal updates.
        old_priority = 0 if self._memory[position] is None \
            else (self._memory[position].priority or 0)
        row = _SumRow(item, priority)
        self._memory[position] = row
        # Propagate priority change up the tree.
        self._update_internal_nodes(
            position, (row.priority or 0) - old_priority)

    def move(self, external_index, new_priority):
        """
        Update the priority of an existing item by external (leaf) index.

        Args:
            external_index (int): Index into the logical leaf array [0, len(self)).
            new_priority (float): The new priority to set.

        Returns:
            None
        """
        # Translate external leaf index to internal array index.
        index = external_index + (self._capacity - 1)
        return self._move(index, new_priority)

    def _move(self, index, new_priority):
        """
        Internal helper to update priority at a given absolute index.

        Args:
            index (int): Absolute index into _memory for the leaf node.
            new_priority (float): The new priority to set.

        Side effects:
            Updates the leaf value and adjusts internal sums by the delta.
        """
        item, old_priority = self._memory[index]
        old_priority = old_priority or 0
        self._memory[index] = _SumRow(item, new_priority)
        # Update internal nodes with the delta in priority.
        self._update_internal_nodes(index, new_priority - old_priority)

    def _update_internal_nodes(self, index, delta):
        """
        Propagate a priority change (delta) from a leaf up to the root.

        Args:
            index (int): The starting index (leaf) from which to update upward.
            delta (float): The change in priority to add to parent sums.
        """
        while index > 0:
            # Move to parent index.
            index = (index - 1) // 2
            # Internal nodes store sums; add delta.
            self._memory[index] += delta

    def _isfull(self):
        """
        Check whether the tree currently stores capacity number of leaves.

        Returns:
            bool: True if number of leaves equals capacity, else False.
        """
        return len(self) == self._capacity

    def _next_position_then_increment(self):
        """
        Get the absolute index for the next leaf position and advance the pointer.

        Returns:
            int: Absolute index into _memory where the next leaf should be written.
        """
        start = self._capacity - 1  # Start of leaf region.
        position = start + self._position
        # Advance circular position among leaf slots.
        self._position = (self._position + 1) % self._capacity
        return position

    def _sample_with_priority(self, p):
        """
        Traverse the tree to find the leaf corresponding to cumulative priority p.

        Args:
            p (float): Target cumulative priority in [0, total_priority].

        Returns:
            int: Absolute index in _memory of the selected leaf node.

        Raises:
            RuntimeError: If the tree shape invariant is broken (missing right child).
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            # If left child index points into leaf region or beyond memory, we are at a leaf.
            if left >= len(self._memory):
                return parent

            # Get left child's sum or leaf priority.
            left_p = self._memory[left] if left < self._capacity - 1 \
                else (self._memory[left].priority or 0)
            if p <= left_p:
                # Go left if cumulative target is within left subtree sum.
                parent = left
            else:
                if left + 1 >= len(self._memory):
                    # Right child must exist for a valid binary tree traversal.
                    raise RuntimeError('Right child is expected to exist.')
                # Otherwise, subtract left sum and go right.
                p -= left_p
                parent = left + 1

    def sample_minibatch(self, batch_size):
        """
        Sample a minibatch of leaves using stratified priority sampling.

        Behavior:
            - If the total priority is approximately zero (all priorities zero/None),
              sample uniformly among existing leaves.
            - Otherwise, divide total priority into equal segments and draw one sample
              uniformly from each segment to reduce variance.

        Args:
            batch_size (int): Number of samples to draw.

        Returns:
            list of tuples: Each element is (absolute_index, _SumRow(item, priority)).
        """
        pool_size = len(self)
        if pool_size == 0:
            return []

        # Total priority is stored at the root (index 0). Segment width:
        delta_p = self._memory[0] / batch_size
        chosen_idx = []
        if abs(self._memory[0]) < util.epsilon:
            # Degenerate case: no priority mass; sample uniformly among current leaves.
            chosen_idx = np.random.randint(self._capacity - 1, self._capacity - 1 + len(self), size=batch_size).tolist()
        else:
            # Stratified sampling across [0, total_priority] segments.
            for i in xrange(batch_size):
                lower = max(i * delta_p, 0)
                upper = min((i + 1) * delta_p, self._memory[0])
                p = random.uniform(lower, upper)
                chosen_idx.append(self._sample_with_priority(p))
        return [(i, self._memory[i]) for i in chosen_idx]

    def __len__(self):
        """
        Number of current leaves stored in the tree.

        Returns:
            int: Count of leaf nodes currently present.
        """
        return len(self._memory) - (self._capacity - 1)

    def __getitem__(self, index):
        """
        Get a leaf by logical index.

        Args:
            index (int): Logical index into the leaf array [0, len(self)).

        Returns:
            _SumRow: The (item, priority) pair at the requested leaf.
        """
        return self._memory[self._capacity - 1:][index]

    def __getslice__(self, start, end):
        """
        Slice access to leaves by logical indices [start:end].

        Args:
            start (int): Start index (inclusive).
            end (int): End index (exclusive).

        Note:
            This method appears to have a bug (missing return). Kept as-is to
            preserve original behavior; callers should not rely on this method.
        """
        self.memory[self._capacity - 1:][start:end]
