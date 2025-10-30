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
    """
    Tensorforce-compatible environment implementing the 2048 game mechanics.

    The board is a 4x4 numpy array where each cell stores the exponent of 2 for the tile value.
    For example, a stored value of 1 represents the tile '2', 2 represents '4', etc.
    Empty cells are stored as 0. Actions are 0: left, 1: up, 2: right, 3: down.
    Rewards are the sum of the values of merged tiles during a single move.
    """

    def __str__(self):
        """
        Return a string representation by printing the current board state to stdout.

        Note: This method prints the state and implicitly returns None, which is slightly
        unconventional for __str__, but kept to preserve original behavior.
        """
        self.print_state()

    def reset(self):
        """
        Reset the game to a new initial state.

        Returns:
            numpy.ndarray: The new initial board state.
        """
        self.__init__()
        return self._state

    def execute(self, action):
        """
        Execute a single action in the environment.

        Args:
            action (int): One of {0: left, 1: up, 2: right, 3: down}.

        Returns:
            tuple: (state, terminal, reward) where
                - state (numpy.ndarray): current board state after the action,
                - terminal (bool): True if no further actions are possible (game over),
                - reward (float): reward obtained from the action (sum of merged tile values).

        Behavior:
            - If the game is already over, returns immediately with zero reward.
            - If the action makes no change (not available), returns state with zero reward.
            - Otherwise, performs the action, spawns a random tile, and returns resulting values.
        """
        reward = 0
        terminal = self.game_over()
        if terminal:
            return self._state, terminal, reward
        action_available = self.is_action_available(action)
        if not action_available:
            return self._state, terminal, reward

        reward = self.do_action(action)

        return self._state, terminal, reward

    @property
    def largest_tile(self):
        """
        Get the largest tile value on the board.

        Returns:
            int: The largest tile value (as an integer power of 2).
        """
        return 2**np.amax(self._state)

    @property
    def states(self):
        """
        Tensorforce states specification.

        Returns:
            dict: Specification dict containing shape and type.
        """
        return dict(shape=self._state.shape, type='float')

    @property
    def actions(self):
        """
        Tensorforce actions specification.

        Returns:
            dict: Specification dict containing number of discrete actions and type.
        """
        return dict(num_actions=len(ACTION_NAMES), type='int')

    def __init__(self, state=None, initial_score=0):
        """
        Initialize a new 2048 game instance.

        Args:
            state (numpy.ndarray or None): Optional pre-defined board state to start with.
                If None, a fresh 4x4 board with two random tiles is created.
            initial_score (int): Optional initial score, used for cloning/copying states.
        """
        self._score = initial_score

        if state is None:
            # Create empty board and add two starting tiles (2 or 4, with 90% chance for 2).
            self._state = np.zeros((4, 4), dtype=np.int)
            self.add_random_tile()
            self.add_random_tile()
        else:
            # Use provided state (assumed valid).
            self._state = state

    def copy(self):
        """
        Create a deep copy of the current environment state.

        Returns:
            Game2048: New instance with copied board and score.
        """
        return Game2048(np.copy(self._state), self._score)

    def game_over(self):
        """
        Check if no moves are available.

        Returns:
            bool: True if no action can change the board (terminal state), False otherwise.
        """
        for action in range(4):
            if self.is_action_available(action):
                return False
        return True

    def available_actions(self):
        """
        List all actions that would change the board state.

        Returns:
            list[int]: Actions in {0, 1, 2, 3} that are currently available.
        """
        return [action for action in range(4) if self.is_action_available(action)]

    def is_action_available(self, action):
        """
        Determine if a given action would change the board.

        Args:
            action (int): One of {0: left, 1: up, 2: right, 3: down}.

        Returns:
            bool: True if the action results in any tile movement or merge, False otherwise.
        """
        # Rotate state so that checking left-move logic applies for the given direction.
        temp_state = np.rot90(self._state, action)
        return self._is_action_available_left(temp_state)

    def _is_action_available_left(self, state):
        """
        Internal helper to check availability of a 'left' move on a rotated state.

        Args:
            state (numpy.ndarray): Board state oriented so that intended move is 'left'.

        Returns:
            bool: True if at least one tile can move or merge to the left, False otherwise.
        """
        for row in range(4):
            has_empty = False
            for col in range(4):
                # Track if any empty has been seen to the left of current cell
                has_empty |= state[row, col] == 0
                # If a non-empty tile has an empty to the left, it can move
                if state[row, col] != 0 and has_empty:
                    return True
                # If adjacent equal tiles exist, a merge is possible
                if (state[row, col] != 0 and col > 0 and
                        state[row, col] == state[row, col - 1]):
                    return True

        return False

    def do_action(self, action):
        """
        Apply an action to the environment, update state, score, and spawn a new tile.

        Args:
            action (int): One of {0: left, 1: up, 2: right, 3: down}.

        Returns:
            int: Reward obtained from the move (sum of merged tile values).
        """
        # Rotate board so that the move can be processed as a 'left' move.
        temp_state = np.rot90(self._state, action)
        reward = self._do_action_left(temp_state)
        # Rotate back to original orientation.
        self._state = np.rot90(temp_state, -action)
        self._score += reward

        # After a valid move, add a new random tile at an empty position.
        self.add_random_tile()

        return reward

    def _do_action_left(self, state):
        """
        Internal helper to perform a 'left' move on the given state.

        This function compacts tiles to the left, merging identical adjacent tiles once per move.

        Args:
            state (numpy.ndarray): Board state oriented such that the move is to the left.

        Returns:
            int: Reward earned by merging tiles during this move.
        """
        reward = 0

        for row in range(4):
            merge_candidate = -1
            # Track whether a tile at a given position has already been merged.
            merged = np.zeros((4,), dtype=np.bool)

            for col in range(4):
                if state[row, col] == 0:
                    continue

                # Merge with the current merge candidate if possible and not yet merged.
                if (merge_candidate != -1 and
                        not merged[merge_candidate] and
                        state[row, merge_candidate] == state[row, col]):
                    # Perform the merge: clear source, mark merged, increment exponent.
                    state[row, col] = 0
                    merged[merge_candidate] = True
                    state[row, merge_candidate] += 1
                    # Reward is the value of the resulting tile (2^exponent).
                    reward += 2 ** state[row, merge_candidate]

                else:
                    # Slide tile left to the next available spot.
                    merge_candidate += 1
                    if col != merge_candidate:
                        state[row, merge_candidate] = state[row, col]
                        state[row, col] = 0

        return reward

    def add_random_tile(self):
        """
        Spawn a new random tile (2 with 90% chance, 4 with 10% chance) in an empty cell.

        Raises:
            AssertionError: If there are no empty cells available.
        """
        x_pos, y_pos = np.where(self._state == 0)
        assert len(x_pos) != 0
        empty_index = np.random.choice(len(x_pos))
        value = np.random.choice([1, 2], p=[0.9, 0.1])

        self._state[x_pos[empty_index], y_pos[empty_index]] = value

    def print_state(self):
        """
        Pretty-print the current board to stdout, showing tile values and separators.
        """

        def tile_string(value):
            # Convert internal exponent representation to human-readable tile value.
            if value > 0:
                return '% 5d' % (2 ** value,)
            return "     "

        separator_line = '-' * 25
        print(separator_line)
        for row in range(4):
            print("|" + "|".join([tile_string(v) for v in self._state[row, :]]) + "|")
            print(separator_line)

    def state(self):
        """
        Get the current board state array.

        Returns:
            numpy.ndarray: The 4x4 board with exponents of 2, 0 meaning empty.
        """
        return self._state

    def score(self):
        """
        Get the current accumulated score.

        Returns:
            int: The total score accumulated from merges.
        """
        return self._score
