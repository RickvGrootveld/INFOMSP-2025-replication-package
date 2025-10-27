from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random
from six.moves import xrange
import numpy as np
from collections import namedtuple

from tensorforce import util, TensorForceError
from tensorforce.core.memories import Memory

# Named tuple representing a row in the sum tree with an item and its priority
_SumRow = namedtuple('SumRow', ['item', 'priority'])


class SumTree(object):
    """A sum tree data structure for efficient prioritized sampling.
    
    The sum tree is a binary tree where leaf nodes store items with priorities,
    and internal nodes store the sum of their children's priorities. This enables
    O(log n) sampling proportional to priority and O(log n) priority updates.
    
    The tree is stored in a flat array where:
    - Indices [0, capacity-1) are internal nodes
    - Indices [capacity-1, 2*capacity-1) are leaf nodes containing actual items
    """
    
    def __init__(self, capacity):
        """Initialize a sum tree with the given capacity.
        
        Args:
            capacity: Maximum number of items the tree can hold.
        """
        self._capacity = capacity

        # Initialize internal nodes (sum nodes) with zeros
        self._memory = [0] * (capacity - 1)
        # Current position for circular buffer insertion
        self._position = 0
        # Total size of the tree (internal nodes + leaf nodes)
        self._actual_capacity = 2 * self._capacity - 1

    def put(self, item, priority=None):
        """Insert or update an item in the tree with the given priority.
        
        Items are inserted in a circular buffer fashion at leaf nodes.
        When the tree is full, the oldest item is overwritten.
        
        Args:
            item: The item to store in the tree.
            priority: The priority value for this item. If None, treated as 0.
        """
        # Expand memory if not yet at full capacity
        if not self._isfull():
            self._memory.append(None)
        
        # Get the next leaf position and increment the circular buffer pointer
        position = self._next_position_then_increment()
        
        # Get the old priority (0 if position was empty)
        old_priority = 0 if self._memory[position] is None \
            else (self._memory[position].priority or 0)
        
        # Create and store the new row
        row = _SumRow(item, priority)
        self._memory[position] = row
        
        # Update all ancestor nodes with the priority delta
        self._update_internal_nodes(
            position, (row.priority or 0) - old_priority)

    def move(self, external_index, new_priority):
        """Update the priority of an item at the given external index.
        
        Args:
            external_index: The index relative to the leaf nodes (0-based).
            new_priority: The new priority value for the item.
        """
        # Convert external index to internal tree index
        index = external_index + (self._capacity - 1)
        return self._move(index, new_priority)

    def _move(self, index, new_priority):
        """Update the priority of an item at the given internal tree index.
        
        Args:
            index: The absolute index in the tree array.
            new_priority: The new priority value for the item.
        """
        item, old_priority = self._memory[index]
        old_priority = old_priority or 0
        self._memory[index] = _SumRow(item, new_priority)
        # Propagate the priority change up the tree
        self._update_internal_nodes(index, new_priority - old_priority)

    def _update_internal_nodes(self, index, delta):
        """Propagate a priority change up the tree to all ancestor nodes.
        
        Args:
            index: The starting index (typically a leaf node).
            delta: The change in priority to propagate.
        """
        while index > 0:
            # Move to parent node
            index = (index - 1) // 2
            # Add the delta to the parent's sum
            self._memory[index] += delta

    def _isfull(self):
        """Check if the tree has reached its capacity.
        
        Returns:
            True if the tree is full, False otherwise.
        """
        return len(self) == self._capacity

    def _next_position_then_increment(self):
        """Get the next leaf position for insertion and increment the pointer.
        
        Returns:
            The absolute index in the tree array for the next insertion.
        """
        start = self._capacity - 1
        position = start + self._position
        # Circular buffer: wrap around when reaching capacity
        self._position = (self._position + 1) % self._capacity
        return position

    def _sample_with_priority(self, p):
        """Sample a leaf node index based on a priority value.
        
        Traverses the tree from root to leaf, choosing left or right child
        based on the cumulative priority sums.
        
        Args:
            p: A priority value in the range [0, total_priority].
            
        Returns:
            The index of the sampled leaf node.
            
        Raises:
            RuntimeError: If tree structure is inconsistent.
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            # If left child doesn't exist, we've reached a leaf
            if left >= len(self._memory):
                return parent

            # Get priority of left child (sum if internal node, priority if leaf)
            left_p = self._memory[left] if left < self._capacity - 1 \
                else (self._memory[left].priority or 0)
            
            if p <= left_p:
                # Sample from left subtree
                parent = left
            else:
                # Sample from right subtree
                if left + 1 >= len(self._memory):
                    raise RuntimeError('Right child is expected to exist.')
                p -= left_p
                parent = left + 1

    def sample_minibatch(self, batch_size):
        """Sample a minibatch of items proportional to their priorities.
        
        Divides the total priority range into batch_size segments and samples
        one item from each segment to ensure diversity.
        
        Args:
            batch_size: Number of items to sample.
            
        Returns:
            A list of tuples (index, SumRow) for the sampled items.
        """
        pool_size = len(self)
        if pool_size == 0:
            return []

        # Divide total priority into batch_size segments
        delta_p = self._memory[0] / batch_size
        chosen_idx = []
        
        # If total priority is near zero, sample uniformly
        if abs(self._memory[0]) < util.epsilon:
            chosen_idx = np.random.randint(
                self._capacity - 1, 
                self._capacity - 1 + len(self), 
                size=batch_size
            ).tolist()
        else:
            # Sample one item from each priority segment
            for i in xrange(batch_size):
                lower = max(i * delta_p, 0)
                upper = min((i + 1) * delta_p, self._memory[0])
                p = random.uniform(lower, upper)
                chosen_idx.append(self._sample_with_priority(p))
        
        return [(i, self._memory[i]) for i in chosen_idx]

    def __len__(self):
        """Return the number of items currently stored in the tree.
        
        Returns:
            The number of leaf nodes containing items.
        """
        return len(self._memory) - (self._capacity - 1)

    def __getitem__(self, index):
        """Get an item by its external index.
        
        Args:
            index: The index relative to the leaf nodes (0-based).
            
        Returns:
            The SumRow at the given index.
        """
        return self._memory[self._capacity - 1:][index]

    def __getslice__(self, start, end):
        """Get a slice of items from the tree.
        
        Args:
            start: Starting index (inclusive).
            end: Ending index (exclusive).
            
        Returns:
            A slice of SumRow objects.
        """
        self.memory[self._capacity - 1:][start:end]
