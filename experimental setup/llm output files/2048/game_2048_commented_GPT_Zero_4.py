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
    Tensorforce-compatible 2048 environment.

    The board is a 4x4 numpy array storing exponents of 2 (i.e., value k represents tile 2**k).
    The environment supports four actions: left, up, right, down. After each valid move, a
    new tile (2 with prob 0.9, 4 with prob 0.1) is added at a random empty position.

    State:
      - self._state: np.ndarray shape (4,4) with integers (exponents), 0 represents empty cell.
      - self._score: cumulative score; increases by the merged tile value (e.g., merging two 4s yields 8).

    API methods align with Tensorforce's Environment:
      - reset() -> initial state
      - execute(action) -> (state, terminal, reward)
      - states, actions properties for specs
    """

    def __str__(self):
        """
        Print the current board to stdout and return None.

        Note: __str__ conventionally returns a string; here it prints for convenience.
        """
        self.print_state()

    def reset(self):
        """
        Reset the environment to a new game and return the initial state.

        Returns:
            np.ndarray: The 4x4 board state after placing two random starting tiles.
        """
        self.__init__()
        return self._state

    def execute(self, action):
        """
        Apply an action and return the resulting transition tuple.

        Args:
            action (int): One of {0:left, 1:up, 2:right, 3:down}.

        Returns:
            tuple: (state, terminal, reward)
                - state (np.ndarray): New board state after the action (or unchanged if invalid/terminal).
                - terminal (bool): True if no further actions are available.
                - reward (float): Sum of merged tile values produced by this move; 0 for invalid move or terminal.
        """
        reward = 0
        terminal = self.game_over()
        if terminal:
            # No moves possible; return terminal state without change
            return self._state, terminal, reward
        action_available = self.is_action_available(action)
        if not action_available:
            # Invalid move (no tiles would move/merge); return zero-reward, non-terminal transition
            return self._state, terminal, reward

        reward = self.do_action(action)

        return self._state, terminal, reward

    @property
    def largest_tile(self):
        """
        Get the largest tile value currently on the board.

        Returns:
            int: Maximum tile value (as power of two).
        """
        return 2**np.amax(self._state)

    @property
    def states(self):
        """
        Tensorforce states specification.

        Returns:
            dict: Dictionary with shape and type of the state tensor.
        """
        return dict(shape=self._state.shape, type='float')

    @property
    def actions(self):
        """
        Tensorforce actions specification.

        Returns:
            dict: Dictionary with number of discrete actions and type.
        """
        return dict(num_actions=len(ACTION_NAMES), type='int')

    def __init__(self, state=None, initial_score=0):
        """
        Initialize a new 2048 game state.

        Args:
            state (np.ndarray or None): Optional existing 4x4 board (exponents). If None, a new game is created.
            initial_score (int): Starting score, useful for cloning/copying environments.
        """
        self._score = initial_score

        if state is None:
            # Start with an empty board and add two random tiles
            self._state = np.zeros((4, 4), dtype=np.int)
            self.add_random_tile()
            self.add_random_tile()
        else:
            # Use provided state as is
            self._state = state

    def copy(self):
        """
        Create a deep copy of the environment.

        Returns:
            Game2048: New instance with copied state and score.
        """
        return Game2048(np.copy(self._state), self._score)

    def game_over(self):
        """
        Check whether no valid actions remain.

        Returns:
            bool: True if the game has no available moves, False otherwise.
        """
        for action in range(4):
            if self.is_action_available(action):
                return False
        return True

    def available_actions(self):
        """
        List all currently available (valid) actions.

        Returns:
            list[int]: List of action indices that would change the state.
        """
        return [action for action in range(4) if self.is_action_available(action)]

    def is_action_available(self, action):
        """
        Determine whether a given action would change the state (i.e., is valid).

        This rotates the board so that checking "left" suffices for all directions.

        Args:
            action (int): One of {0:left, 1:up, 2:right, 3:down}.

        Returns:
            bool: True if tiles would move or merge under this action.
        """
        # Rotate so that "action" direction becomes "left" and reuse left-check logic
        temp_state = np.rot90(self._state, action)
        return self._is_action_available_left(temp_state)

    def _is_action_available_left(self, state):
        """
        Internal helper: check if a left action is available on the provided state.

        A left move is available if:
          - There exists at least one empty cell to the left of a non-empty tile in a row (tile can slide), or
          - There exist adjacent equal non-zero tiles in a row (tiles can merge).

        Args:
            state (np.ndarray): 4x4 board (exponents) to evaluate.

        Returns:
            bool: True if a left move would change the state.
        """
        for row in range(4):
            has_empty = False
            for col in range(4):
                # Track if we've seen an empty cell earlier in the row
                has_empty |= state[row, col] == 0
                # Non-empty tile can slide left if empty encountered before
                if state[row, col] != 0 and has_empty:
                    return True
                # Adjacent equal tiles can merge
                if (state[row, col] != 0 and col > 0 and
                        state[row, col] == state[row, col - 1]):
                    return True

        return False

    def do_action(self, action):
        """
        Apply an action to the environment, add a random tile, and update score.

        Args:
            action (int): One of {0:left, 1:up, 2:right, 3:down}.

        Returns:
            int: Reward gained from merges performed during this move.
        """
        # Rotate to align action direction with left
        temp_state = np.rot90(self._state, action)
        reward = self._do_action_left(temp_state)
        # Rotate back to original orientation
        self._state = np.rot90(temp_state, -action)
        self._score += reward

        # Add a new random tile to an empty position
        self.add_random_tile()

        return reward

    def _do_action_left(self, state):
        """
        Internal helper: perform a left move on the given state and return reward.

        The algorithm compacts tiles to the left, merging equal adjacent tiles at most once
        per position per move, and sums the values of newly created tiles as reward.

        Args:
            state (np.ndarray): 4x4 board to mutate in-place (left move applied).

        Returns:
            int: Total reward from merges (sum of values of created tiles).
        """
        reward = 0

        for row in range(4):
            merge_candidate = -1  # Index where the next tile should attempt to settle/merge
            merged = np.zeros((4,), dtype=np.bool)  # Track if a position has already merged this move

            for col in range(4):
                if state[row, col] == 0:
                    continue

                # If current tile matches the tile at merge_candidate and that position hasn't merged yet, merge them
                if (merge_candidate != -1 and
                        not merged[merge_candidate] and
                        state[row, merge_candidate] == state[row, col]):
                    state[row, col] = 0
                    merged[merge_candidate] = True
                    state[row, merge_candidate] += 1  # Increase exponent by 1 (e.g., 2**k + 2**k -> 2**(k+1))
                    reward += 2 ** state[row, merge_candidate]

                else:
                    # Move/compact current tile to the next available position to the left
                    merge_candidate += 1
                    if col != merge_candidate:
                        state[row, merge_candidate] = state[row, col]
                        state[row, col] = 0

        return reward

    def add_random_tile(self):
        """
        Add a new random tile to an empty cell.

        A tile with exponent 1 (value 2) is added with probability 0.9,
        and exponent 2 (value 4) with probability 0.1.

        Raises:
            AssertionError: If there are no empty cells (should not be called in that case).
        """
        x_pos, y_pos = np.where(self._state == 0)
        assert len(x_pos) != 0
        empty_index = np.random.choice(len(x_pos))
        value = np.random.choice([1, 2], p=[0.9, 0.1])

        self._state[x_pos[empty_index], y_pos[empty_index]] = value

    def print_state(self):
        """
        Pretty-print the current board to stdout, showing tile values as powers of two.
        """

        def tile_string(value):
            # Render empty cells as blanks, non-empty as right-aligned integers
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
            np.ndarray: The 4x4 board (exponents).
        """
        return self._state

    def score(self):
        """
        Get the current cumulative score.

        Returns:
            int: Score accumulated from merges so far.
        """
        return self._score
