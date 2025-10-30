from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorforce.environments import Environment
import numpy as np

ACTION_NAMES = ["left", "up", "right", "down"]
ACTION_LEFT = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3


class Game2048(Environment):
    """Tensorforce-compatible environment for the 2048 game.

    The environment represents a 4x4 2048 board using a NumPy array of exponents,
    where a cell value n corresponds to a tile with value 2**n (0 denotes empty).
    It exposes the standard Environment API: reset, execute, states, actions.
    """

    def __str__(self):
        """Return a string representation of the board by printing it to stdout.

        Note:
            This method prints the state and does not explicitly return a string.
            It is primarily useful for quick debugging/inspection.
        """
        self.print_state()

    def reset(self):
        """Reset the environment to an initial state and return the observation.

        Returns:
            np.ndarray: The initial 4x4 board state after placing two random tiles.
        """
        self.__init__()
        return self._state

    def execute(self, action):
        """Apply an action to the environment.

        The action is one of: 0=left, 1=up, 2=right, 3=down. If the game is over
        or the action has no effect (i.e., is not available), the state is
        returned unchanged with zero reward.

        Args:
            action (int): The action index in [0, 3].

        Returns:
            tuple:
                - state (np.ndarray): Current board state after the step.
                - terminal (bool): Whether the game is over.
                - reward (float): Reward obtained from the action (sum of merged tile values).
        """
        reward = 0
        terminal = self.game_over()
        if terminal:
            return self._state, terminal, reward

        action_available = self.is_action_available(action)
        if not action_available:
            # Invalid/no-op action yields zero reward and no state change
            return self._state, terminal, reward

        reward = self.do_action(action)

        return self._state, terminal, reward

    @property
    def largest_tile(self):
        """Return the largest tile value currently on the board."""
        return 2**np.amax(self._state)

    @property
    def states(self):
        """Tensorforce 'states' specification.

        Returns:
            dict: Shape and type of the observation tensor.
        """
        return dict(shape=self._state.shape, type='float')

    @property
    def actions(self):
        """Tensorforce 'actions' specification.

        Returns:
            dict: Number of discrete actions and their type.
        """
        return dict(num_actions=len(ACTION_NAMES), type='int')

    def __init__(self, state=None, initial_score=0):
        """Initialize the 2048 environment.

        Args:
            state (np.ndarray, optional): Existing board state (4x4 of exponents).
                If None, a fresh board is created with two random tiles.
            initial_score (int, optional): Starting score, default 0.
        """
        self._score = initial_score

        if state is None:
            # Use exponent representation: 0=empty, 1=2, 2=4, ...
            self._state = np.zeros((4, 4), dtype=np.int)
            self.add_random_tile()
            self.add_random_tile()
        else:
            self._state = state

    def copy(self):
        """Create a deep copy of the environment with identical state and score.

        Returns:
            Game2048: A new environment instance.
        """
        return Game2048(np.copy(self._state), self._score)

    def game_over(self):
        """Check whether no valid moves remain.

        Returns:
            bool: True if no actions are available, else False.
        """
        for action in range(4):
            if self.is_action_available(action):
                return False
        return True

    def available_actions(self):
        """List all currently available actions.

        Returns:
            list[int]: Action indices that would change the state.
        """
        return [action for action in range(4) if self.is_action_available(action)]

    def is_action_available(self, action):
        """Determine if an action would change the board state.

        The board is rotated so that checking left-move availability is sufficient.

        Args:
            action (int): Action index in [0, 3].

        Returns:
            bool: True if the action would move or merge at least one tile.
        """
        temp_state = np.rot90(self._state, action)
        return self._is_action_available_left(temp_state)

    def _is_action_available_left(self, state):
        """Check if a 'left' move is available on the given rotated state.

        A move is available if:
          - there exists a non-empty tile with at least one empty cell to its left, or
          - there exists at least one adjacent equal pair horizontally.

        Args:
            state (np.ndarray): Board state oriented such that action 'left' is tested.

        Returns:
            bool: True if a left move can be performed.
        """
        for row in range(4):
            has_empty = False  # Tracks if any empty cell has been encountered in this row while scanning left-to-right
            for col in range(4):
                has_empty |= state[row, col] == 0
                # If we see a non-empty after an empty, a slide is possible
                if state[row, col] != 0 and has_empty:
                    return True
                # If two adjacent tiles are equal, a merge is possible
                if (state[row, col] != 0 and col > 0 and
                        state[row, col] == state[row, col - 1]):
                    return True

        return False

    def do_action(self, action):
        """Execute an action and update the board, score, and add a random tile.

        Args:
            action (int): Action index in [0, 3].

        Returns:
            int: Reward accumulated from merges during this action.
        """
        # Rotate so we can always process as a 'left' move
        temp_state = np.rot90(self._state, action)
        reward = self._do_action_left(temp_state)
        # Rotate back to the original orientation
        self._state = np.rot90(temp_state, -action)
        self._score += reward

        # Add a new random tile after a successful move
        self.add_random_tile()

        return reward

    def _do_action_left(self, state):
        """Apply a 'left' move on the given state in-place and compute reward.

        The algorithm compacts tiles to the left and merges equal adjacent tiles
        at most once per tile per move. Tiles are stored as exponents of 2.

        Args:
            state (np.ndarray): Board state (rotated orientation for left move).

        Returns:
            int: Sum of merged tile values produced during the move.
        """
        reward = 0

        for row in range(4):
            merge_candidate = -1  # Index of the next position to move/merge into
            merged = np.zeros((4,), dtype=np.bool)  # Track if a position already merged this move

            for col in range(4):
                if state[row, col] == 0:
                    continue

                # Try to merge into the latest candidate if same value and not already merged
                if (merge_candidate != -1 and
                        not merged[merge_candidate] and
                        state[row, merge_candidate] == state[row, col]):
                    # Merge: clear source, increment exponent, mark as merged
                    state[row, col] = 0
                    merged[merge_candidate] = True
                    state[row, merge_candidate] += 1
                    reward += 2 ** state[row, merge_candidate]

                else:
                    # Move tile left into next free position (which may be its current position)
                    merge_candidate += 1
                    if col != merge_candidate:
                        state[row, merge_candidate] = state[row, col]
                        state[row, col] = 0

        return reward

    def add_random_tile(self):
        """Add a new random tile (2 with 90% or 4 with 10%) to a random empty cell.

        The board stores exponents, so this adds value 1 (2) with p=0.9 or 2 (4) with p=0.1.
        """
        x_pos, y_pos = np.where(self._state == 0)
        assert len(x_pos) != 0
        empty_index = np.random.choice(len(x_pos))
        value = np.random.choice([1, 2], p=[0.9, 0.1])

        self._state[x_pos[empty_index], y_pos[empty_index]] = value

    def print_state(self):
        """Pretty-print the current board to stdout."""
        def tile_string(value):
            if value > 0:
                return '% 5d' % (2 ** value,)
            return "     "

        separator_line = '-' * 25
        print(separator_line)
        for row in range(4):
            print("|" + "|".join([tile_string(v) for v in self._state[row, :]]) + "|")
            print(separator_line)

    def state(self):
        """Return the internal board state array.

        Returns:
            np.ndarray: Current 4x4 grid of exponents.
        """
        return self._state

    def score(self):
        """Return the current accumulated score.

        Returns:
            int: The score, equal to the sum of values gained from merges.
        """
        return self._score
