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
    """Tensorforce-compatible 2048 environment.

    The environment stores the board as a 4x4 numpy array of exponents, where a
    tile with value 2^k is represented by integer k, and 0 denotes an empty cell.
    Actions are integers in [0, 3] corresponding to left, up, right, down.
    Rewards are the sums of merged tile values produced by a move.
    """

    def __str__(self):
        """Pretty-print the current board to stdout and return None.

        Note: This method prints the state and does not return a string,
        which is atypical for __str__. It is used for quick debugging.
        """
        self.print_state()

    def reset(self):
        """Reset the environment to a fresh game state.

        Returns:
            np.ndarray: The new 4x4 board state (exponent representation).
        """
        self.__init__()
        return self._state

    def execute(self, action):
        """Apply an action and advance the environment by one step.

        Args:
            action (int): One of {0:left, 1:up, 2:right, 3:down}.

        Returns:
            tuple:
                - state (np.ndarray): Current board after the step.
                - terminal (bool): True if no further moves are available.
                - reward (float): Reward obtained by this action (sum of merged tile values).
        """
        reward = 0
        terminal = self.game_over()
        if terminal:
            return self._state, terminal, reward
        action_available = self.is_action_available(action)
        if not action_available:
            # Invalid or no-op action yields no change and no reward.
            return self._state, terminal, reward

        reward = self.do_action(action)

        return self._state, terminal, reward

    @property
    def largest_tile(self):
        """Return the numeric value of the largest tile on the board."""
        return 2**np.amax(self._state)

    @property
    def states(self):
        """Tensorforce states spec: 4x4 float array."""
        return dict(shape=self._state.shape, type='float')

    @property
    def actions(self):
        """Tensorforce actions spec: discrete with 4 possible moves."""
        return dict(num_actions=len(ACTION_NAMES), type='int')

    def __init__(self, state=None, initial_score=0):
        """Create a new 2048 environment.

        Args:
            state (np.ndarray or None): Optional existing board to start from.
                If None, initializes an empty board and adds two random tiles.
            initial_score (int): Initial score to start with.
        """
        self._score = initial_score

        if state is None:
            # Board stores exponents; zeros mean empty.
            self._state = np.zeros((4, 4), dtype=np.int)
            self.add_random_tile()
            self.add_random_tile()
        else:
            self._state = state

    def copy(self):
        """Deep-copy the environment.

        Returns:
            Game2048: A new environment instance with copied state and score.
        """
        return Game2048(np.copy(self._state), self._score)

    def game_over(self):
        """Check if no moves are available.

        Returns:
            bool: True if the game is over (no valid moves), else False.
        """
        for action in range(4):
            if self.is_action_available(action):
                return False
        return True

    def available_actions(self):
        """List all actions that would change the board.

        Returns:
            list[int]: Actions in {0,1,2,3} that are currently available.
        """
        return [action for action in range(4) if self.is_action_available(action)]

    def is_action_available(self, action):
        """Determine if a given action will change the board.

        Uses rotation so that availability is checked as if moving left.

        Args:
            action (int): One of {0:left, 1:up, 2:right, 3:down}.

        Returns:
            bool: True if the action would cause any movement/merge.
        """
        # Rotate the board so that the desired action becomes "left".
        temp_state = np.rot90(self._state, action)
        return self._is_action_available_left(temp_state)

    def _is_action_available_left(self, state):
        """Check if a left move is possible for the given state.

        Args:
            state (np.ndarray): 4x4 board (exponent representation).

        Returns:
            bool: True if some tile can move left or merge with its neighbor.
        """
        for row in range(4):
            has_empty = False
            for col in range(4):
                # Track if we've seen an empty cell; any nonzero after an empty can slide.
                has_empty |= state[row, col] == 0
                if state[row, col] != 0 and has_empty:
                    return True
                # Adjacent equal tiles can merge.
                if (state[row, col] != 0 and col > 0 and
                        state[row, col] == state[row, col - 1]):
                    return True

        return False

    def do_action(self, action):
        """Execute an action, update state and score, and spawn a random tile.

        Args:
            action (int): One of {0:left, 1:up, 2:right, 3:down}.

        Returns:
            int: Reward obtained from merges during this action.
        """
        # Rotate so action is a left move.
        temp_state = np.rot90(self._state, action)
        reward = self._do_action_left(temp_state)
        # Rotate back to original orientation.
        self._state = np.rot90(temp_state, -action)
        self._score += reward

        # Add a new random tile (2 with p=0.9 or 4 with p=0.1 in standard 2048).
        self.add_random_tile()

        return reward

    def _do_action_left(self, state):
        """Slide and merge tiles to the left, mutating the given state in-place.

        Merge rules:
        - Each row slides all non-zero tiles as far left as possible.
        - Adjacent equal tiles merge once per move (left to right), creating a tile
          with exponent +1, and reward equals the resulting tile value (2^new_exp).

        Args:
            state (np.ndarray): 4x4 board rotated so that the action is 'left'.

        Returns:
            int: Sum of values produced by merges for this move.
        """
        reward = 0

        for row in range(4):
            merge_candidate = -1  # Index where next non-zero will try to occupy/merge.
            merged = np.zeros((4,), dtype=np.bool)  # Track if a position already merged.

            for col in range(4):
                if state[row, col] == 0:
                    continue

                # Try to merge with the latest candidate if same exponent and not already merged.
                if (merge_candidate != -1 and
                        not merged[merge_candidate] and
                        state[row, merge_candidate] == state[row, col]):
                    # Merge: increment exponent, clear current, mark merged.
                    state[row, col] = 0
                    merged[merge_candidate] = True
                    state[row, merge_candidate] += 1
                    reward += 2 ** state[row, merge_candidate]

                else:
                    # Move tile left into next candidate slot (compacting step).
                    merge_candidate += 1
                    if col != merge_candidate:
                        state[row, merge_candidate] = state[row, col]
                        state[row, col] = 0

        return reward

    def add_random_tile(self):
        """Add a random tile (2 with 90% or 4 with 10%) to an empty cell."""
        x_pos, y_pos = np.where(self._state == 0)
        assert len(x_pos) != 0
        empty_index = np.random.choice(len(x_pos))
        value = np.random.choice([1, 2], p=[0.9, 0.1])  # exponents: 1->2, 2->4

        self._state[x_pos[empty_index], y_pos[empty_index]] = value

    def print_state(self):
        """Print the board to stdout as a human-readable 2048 grid."""

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
        """Return the current board state array (exponent representation)."""
        return self._state

    def score(self):
        """Return the current accumulated score."""
        return self._score
