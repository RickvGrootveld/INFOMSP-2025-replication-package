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
    """Binary heap implementation of a sum tree for prioritized experience replay.
    
    A sum tree is a binary tree data structure where the value of each parent node
    is the sum of its children. Leaf nodes store actual data items with associated
    priorities, while internal nodes store cumulative sums. This structure enables
    efficient O(log n) sampling proportional to priorities.
    """
    
    def __init__(self, capacity):
        """Initializes a new sum tree with the specified capacity.
        
        Arguments:
            capacity:
                Maximum number of items the tree can hold.
        """
        self._capacity = capacity

        # Initialize internal nodes (capacity - 1 of them) with zeros
        self._memory = [0] * (capacity - 1)
        # Current position for circular buffer insertion
        self._position = 0
        # Total capacity including internal nodes and leaf nodes
        self._actual_capacity = 2 * self._capacity - 1

    def put(self, item, priority=None):
        """Inserts an item into the tree with the given priority.
        
        If the tree is not full, appends a new leaf node. Otherwise, overwrites
        the oldest item in circular buffer fashion. Updates all parent nodes
        to reflect the new priority sum.
        
        Arguments:
            item:
                The data item to store.
            priority:
                Priority value for sampling. Higher priorities are sampled more often.
        """
        if not self._isfull():
            self._memory.append(None)
        position = self._next_position_then_increment()
        old_priority = 0 if self._memory[position] is None \
            else (self._memory[position].priority or 0)
        row = _SumRow(item, priority)
        self._memory[position] = row
        # Update parent nodes with the priority delta
        self._update_internal_nodes(
            position, (row.priority or 0) - old_priority)

    def move(self, external_index, new_priority):
        """Updates the priority of an item at the given external index.
        
        External index refers to the position in the leaf nodes only,
        not including internal nodes.
        
        Arguments:
            external_index:
                Index of the item in the leaf node array.
            new_priority:
                New priority value to assign.
        """
        # Convert external index to internal memory index
        index = external_index + (self._capacity - 1)
        return self._move(index, new_priority)

    def _move(self, index, new_priority):
        """Internal method to update priority at a specific memory index.
        
        Arguments:
            index:
                Absolute index in the internal memory array.
            new_priority:
                New priority value to assign.
        """
        item, old_priority = self._memory[index]
        old_priority = old_priority or 0
        self._memory[index] = _SumRow(item, new_priority)
        # Propagate the priority change up the tree
        self._update_internal_nodes(index, new_priority - old_priority)

    def _update_internal_nodes(self, index, delta):
        """Propagates a priority change up the tree to all parent nodes.
        
        Starting from the given index, traverses up to the root, adding
        the delta to each parent node's sum.
        
        Arguments:
            index:
                Starting index (typically a leaf node).
            delta:
                Change in priority to propagate.
        """
        while index > 0:
            # Move to parent node
            index = (index - 1) // 2
            self._memory[index] += delta

    def _isfull(self):
        """Checks whether the tree has reached its capacity.
        
        Returns:
            True if the tree is full, False otherwise.
        """
        return len(self) == self._capacity

    def _next_position_then_increment(self):
        """Gets the next insertion position and increments the position counter.
        
        Implements circular buffer behavior by wrapping around when capacity is reached.
        
        Returns:
            The absolute memory index for the next insertion.
        """
        start = self._capacity - 1
        position = start + self._position
        self._position = (self._position + 1) % self._capacity
        return position

    def _sample_with_priority(self, p):
        """Samples a leaf node index based on a priority value.
        
        Traverses the tree from root to leaf, choosing left or right child
        based on cumulative priority sums. This enables O(log n) sampling
        proportional to priorities.
        
        Arguments:
            p:
                Priority value to search for (between 0 and total priority sum).
                
        Returns:
            Index of the sampled leaf node.
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            # Reached a leaf node
            if left >= len(self._memory):
                return parent

            # Get priority of left child (internal node or leaf)
            left_p = self._memory[left] if left < self._capacity - 1 \
                else (self._memory[left].priority or 0)
            if p <= left_p:
                parent = left
            else:
                if left + 1 >= len(self._memory):
                    raise RuntimeError('Right child is expected to exist.')
                # Subtract left priority and move to right child
                p -= left_p
                parent = left + 1

    def sample_minibatch(self, batch_size):
        """Samples a minibatch of items proportional to their priorities.
        
        Divides the total priority range into equal segments and samples
        one item from each segment. This ensures diverse sampling across
        the priority distribution.
        
        Arguments:
            batch_size:
                Number of items to sample.
                
        Returns:
            List of tuples (index, SumRow) for sampled items.
        """
        pool_size = len(self)
        if pool_size == 0:
            return []

        # Divide total priority into equal segments
        delta_p = self._memory[0] / batch_size
        chosen_idx = []
        # Handle edge case where all priorities are zero
        if abs(self._memory[0]) < util.epsilon:
            chosen_idx = np.random.randint(self._capacity - 1, self._capacity - 1 + len(self), size=batch_size).tolist()
        else:
            # Sample one item from each priority segment
            for i in xrange(batch_size):
                lower = max(i * delta_p, 0)
                upper = min((i + 1) * delta_p, self._memory[0])
                p = random.uniform(lower, upper)
                chosen_idx.append(self._sample_with_priority(p))
        return [(i, self._memory[i]) for i in chosen_idx]

    def __len__(self):
        """Returns the number of items currently stored in the tree.
        
        Returns:
            Number of leaf nodes (excluding internal nodes).
        """
        return len(self._memory) - (self._capacity - 1)

    def __getitem__(self, index):
        """Gets an item by its external index (leaf node position).
        
        Arguments:
            index:
                External index in the leaf node array.
                
        Returns:
            The SumRow at the specified index.
        """
        return self._memory[self._capacity - 1:][index]

    def __getslice__(self, start, end):
        """Gets a slice of items from the leaf nodes.
        
        Arguments:
            start:
                Starting index in the leaf node array.
            end:
                Ending index in the leaf node array.
                
        Returns:
            Slice of SumRow objects.
        """
        self.memory[self._capacity - 1:][start:end]
