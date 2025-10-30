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

    The board is represented as a 4x4 NumPy array of integers where each cell
    stores log2(tile_value). For example, an empty cell is 0, tile 2 is 1, tile 4 is 2, etc.
    Actions are encoded as integers [0..3] corresponding to left, up, right, down.
    Rewards are the sum of newly created tile values during a move.
    """

    def __str__(self):
        """Return string representation by printing the current board to stdout.

        Note: This method prints and returns None, which is unusual for __str__.
        """
        self.print_state()

    def reset(self):
        """Reset the environment to a new game state.

        Returns:
            numpy.ndarray: The new initial state after reset.
        """
        self.__init__()
        return self._state

    def execute(self, action):
        """Apply an action to the environment.

        Args:
            action (int): One of {0: left, 1: up, 2: right, 3: down}.

        Returns:
            tuple:
                - state (numpy.ndarray): The resulting state.
                - terminal (bool): Whether the game is over.
                - reward (float): Reward obtained by this action.

        Notes:
            - If the game is already terminal or the action has no effect,
              the state is returned unchanged with zero reward.
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
        """Return the largest tile value currently on the board."""
        return 2**np.amax(self._state)

    @property
    def states(self):
        """Environment state specification used by Tensorforce."""
        return dict(shape=self._state.shape, type='float')

    @property
    def actions(self):
        """Environment action specification used by Tensorforce."""
        return dict(num_actions=len(ACTION_NAMES), type='int')

    def __init__(self, state=None, initial_score=0):
        """Initialize a new 2048 game.

        Args:
            state (numpy.ndarray, optional): Existing board to start from.
                If None, a fresh 4x4 board is created with two random tiles.
            initial_score (int): Starting score.
        """
        self._score = initial_score

        if state is None:
            # Board stores exponents: 0 for empty, 1 for 2, 2 for 4, etc.
            self._state = np.zeros((4, 4), dtype=np.int)
            self.add_random_tile()
            self.add_random_tile()
        else:
            self._state = state

    def copy(self):
        """Create a deep copy of the current game.

        Returns:
            Game2048: Copied game instance with duplicated state and score.
        """
        return Game2048(np.copy(self._state), self._score)

    def game_over(self):
        """Check whether no actions are available (terminal state)."""
        for action in range(4):
            if self.is_action_available(action):
                return False
        return True

    def available_actions(self):
        """List all currently available actions.

        Returns:
            list[int]: Actions that would change the board.
        """
        return [action for action in range(4) if self.is_action_available(action)]

    def is_action_available(self, action):
        """Determine if applying the given action changes the board.

        Args:
            action (int): One of {0: left, 1: up, 2: right, 3: down}.

        Returns:
            bool: True if a move in this direction is possible.
        """
        # Rotate so that checking "left" on the rotated board corresponds to the desired action.
        temp_state = np.rot90(self._state, action)
        return self._is_action_available_left(temp_state)

    def _is_action_available_left(self, state):
        """Check if there is any valid left move on a given board orientation.

        A left move is available if there exists:
        - an empty space before a non-empty tile (allowing a slide), or
        - two adjacent tiles with the same value (allowing a merge).

        Args:
            state (numpy.ndarray): Board oriented such that the attempted move is left.

        Returns:
            bool: True if a left move is possible.
        """
        for row in range(4):
            has_empty = False
            for col in range(4):
                has_empty |= state[row, col] == 0
                # Can slide left into an empty seen earlier in the row
                if state[row, col] != 0 and has_empty:
                    return True
                # Can merge with immediate left neighbor
                if (state[row, col] != 0 and col > 0 and
                        state[row, col] == state[row, col - 1]):
                    return True

        return False

    def do_action(self, action):
        """Apply an action to the board, update score, and spawn a new tile.

        Args:
            action (int): One of {0: left, 1: up, 2: right, 3: down}.

        Returns:
            int: Reward obtained (sum of newly created tile values).
        """
        # Rotate to align action with "left" processing
        temp_state = np.rot90(self._state, action)
        reward = self._do_action_left(temp_state)
        # Rotate back to original orientation
        self._state = np.rot90(temp_state, -action)
        self._score += reward

        # Spawn a random tile (2 with 0.9 prob, 4 with 0.1 prob)
        self.add_random_tile()

        return reward

    def _do_action_left(self, state):
        """Execute a left move on the given board orientation.

        Slides tiles to the left, merging equal adjacent tiles once per move per cell.
        Accumulates reward equal to the numeric value of created tiles.

        Args:
            state (numpy.ndarray): Board oriented such that the move is left.

        Returns:
            int: Total reward from merges during this move.
        """
        reward = 0

        for row in range(4):
            merge_candidate = -1  # index of the next position to try to fill/merge
            merged = np.zeros((4,), dtype=np.bool)  # track merges to prevent double-merge

            for col in range(4):
                if state[row, col] == 0:
                    continue

                # Try to merge into the last placed tile if it matches and hasn't merged yet
                if (merge_candidate != -1 and
                        not merged[merge_candidate] and
                        state[row, merge_candidate] == state[row, col]):
                    state[row, col] = 0
                    merged[merge_candidate] = True
                    state[row, merge_candidate] += 1  # increase exponent
                    reward += 2 ** state[row, merge_candidate]  # reward is actual tile value
                else:
                    # Move current tile to next free slot (or keep if already aligned)
                    merge_candidate += 1
                    if col != merge_candidate:
                        state[row, merge_candidate] = state[row, col]
                        state[row, col] = 0

        return reward

    def add_random_tile(self):
        """Add a random tile to an empty position.

        Chooses a random empty cell and sets it to:
        - 1 (tile value 2) with probability 0.9
        - 2 (tile value 4) with probability 0.1
        """
        x_pos, y_pos = np.where(self._state == 0)
        assert len(x_pos) != 0
        empty_index = np.random.choice(len(x_pos))
        value = np.random.choice([1, 2], p=[0.9, 0.1])

        self._state[x_pos[empty_index], y_pos[empty_index]] = value

    def print_state(self):
        """Pretty-print the board to stdout using actual tile values."""
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
        """Get the current state array (view, not copied)."""
        return self._state

    def score(self):
        """Get the current cumulative score."""
        return self._score
