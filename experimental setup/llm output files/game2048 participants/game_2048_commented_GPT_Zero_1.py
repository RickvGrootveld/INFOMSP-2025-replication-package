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
    Tensorforce-compatible 2048 game environment.

    The board is a 4x4 grid where each cell stores the exponent of 2
    (e.g., a value of 1 represents tile 2, 2 represents tile 4, ...).
    The environment supports rotations to generalize movement handling,
    provides available actions, executes moves, computes rewards, and
    tracks score and terminal conditions.
    """

    def __str__(self):
        """
        Return a human-readable representation by printing the current board state.
        """
        self.print_state()

    def reset(self):
        """
        Reset the environment to a new initial state.

        Returns:
            np.ndarray: The initial 4x4 state after spawning two random tiles.
        """
        self.__init__()
        return self._state

    def execute(self, action):
        """
        Execute a single environment step given an action.

        Args:
            action (int): One of {0:left, 1:up, 2:right, 3:down}.

        Returns:
            tuple:
                - state (np.ndarray): Updated 4x4 board.
                - terminal (bool): Whether the game is over.
                - reward (float): Reward obtained from the action (sum of merged tile values).
        """
        reward = 0
        terminal = self.game_over()
        if terminal:
            # No-op if already terminal.
            return self._state, terminal, reward
        action_available = self.is_action_available(action)
        if not action_available:
            # Invalid move yields zero reward and no state change.
            return self._state, terminal, reward

        reward = self.do_action(action)

        return self._state, terminal, reward

    @property
    def largest_tile(self):
        """
        Return the largest tile value on the board as power-of-two value.

        Returns:
            int: Largest tile value (e.g., 2048).
        """
        return 2**np.amax(self._state)

    @property
    def states(self):
        """
        Tensorforce states spec.

        Returns:
            dict: State specification with shape and dtype information.
        """
        return dict(shape=self._state.shape, type='float')

    @property
    def actions(self):
        """
        Tensorforce actions spec.

        Returns:
            dict: Number of discrete actions and type.
        """
        return dict(num_actions=len(ACTION_NAMES), type='int')

    def __init__(self, state=None, initial_score=0):
        """
        Initialize the 2048 environment.

        Args:
            state (np.ndarray or None): Optional existing 4x4 state to load. If None,
                a fresh board is created with two random tiles.
            initial_score (int): Starting score.
        """
        self._score = initial_score

        if state is None:
            # Use exponent representation for tiles. 0 means empty.
            self._state = np.zeros((4, 4), dtype=np.int)
            self.add_random_tile()
            self.add_random_tile()
        else:
            self._state = state

    def copy(self):
        """
        Create a deep copy of the environment (state and score).

        Returns:
            Game2048: Copied environment instance.
        """
        return Game2048(np.copy(self._state), self._score)

    def game_over(self):
        """
        Check if there are no legal actions left.

        Returns:
            bool: True if no moves are available; False otherwise.
        """
        for action in range(4):
            if self.is_action_available(action):
                return False
        return True

    def available_actions(self):
        """
        List all currently available actions.

        Returns:
            list[int]: Actions in {0,1,2,3} that would change the state.
        """
        return [action for action in range(4) if self.is_action_available(action)]

    def is_action_available(self, action):
        """
        Determine if taking the given action would change the state.

        This is performed by rotating the board so that the action corresponds
        to a left move, and then checking for availability of a left move.

        Args:
            action (int): One of {0:left, 1:up, 2:right, 3:down}.

        Returns:
            bool: True if the move would result in any shift/merge, False otherwise.
        """
        # Rotate so that 'action' aligns with a left move.
        temp_state = np.rot90(self._state, action)
        return self._is_action_available_left(temp_state)

    def _is_action_available_left(self, state):
        """
        Check if a left move is possible on the given state.

        Conditions for availability:
          - There exists an empty space to the left of a tile, or
          - Adjacent tiles of equal value can be merged.

        Args:
            state (np.ndarray): 4x4 board in exponent representation.

        Returns:
            bool: True if a left move is possible.
        """
        for row in range(4):
            has_empty = False
            for col in range(4):
                has_empty |= state[row, col] == 0
                # If we see a non-empty tile after an empty slot, it can shift left.
                if state[row, col] != 0 and has_empty:
                    return True
                # If two adjacent tiles are equal, they can merge.
                if (state[row, col] != 0 and col > 0 and
                        state[row, col] == state[row, col - 1]):
                    return True

        return False

    def do_action(self, action):
        """
        Apply the given action, update state and score, and spawn a new random tile.

        Args:
            action (int): One of {0:left, 1:up, 2:right, 3:down}.

        Returns:
            int: Reward obtained from this move (sum of merged tile values).
        """
        # Rotate so that we can process as a left move.
        temp_state = np.rot90(self._state, action)
        reward = self._do_action_left(temp_state)
        # Rotate back to original orientation.
        self._state = np.rot90(temp_state, -action)
        self._score += reward

        # Add a new random tile after a successful move.
        self.add_random_tile()

        return reward

    def _do_action_left(self, state):
        """
        Perform a left move on the given state in-place.

        Implements shifting tiles left and merging equal adjacent tiles once per move.
        Rewards are computed as the sum of the values of newly formed tiles
        (using actual tile value, i.e., 2**exponent).

        Args:
            state (np.ndarray): 4x4 board in exponent representation; modified in-place.

        Returns:
            int: Total reward generated by merges during this move.
        """
        reward = 0

        for row in range(4):
            merge_candidate = -1
            merged = np.zeros((4,), dtype=np.bool)

            for col in range(4):
                if state[row, col] == 0:
                    # Skip empty cells.
                    continue

                # If the current cell can merge with the last candidate and hasn't merged yet.
                if (merge_candidate != -1 and
                        not merged[merge_candidate] and
                        state[row, merge_candidate] == state[row, col]):
                    # Merge: increase exponent, zero out current cell, mark merged.
                    state[row, col] = 0
                    merged[merge_candidate] = True
                    state[row, merge_candidate] += 1
                    reward += 2 ** state[row, merge_candidate]

                else:
                    # Shift: move current tile to the next merge_candidate position.
                    merge_candidate += 1
                    if col != merge_candidate:
                        state[row, merge_candidate] = state[row, col]
                        state[row, col] = 0

        return reward

    def add_random_tile(self):
        """
        Spawn a new random tile (2 with 90% probability, 4 with 10%) in an empty cell.
        Tiles are represented as exponents: 1 -> 2, 2 -> 4.
        """
        x_pos, y_pos = np.where(self._state == 0)
        assert len(x_pos) != 0
        empty_index = np.random.choice(len(x_pos))
        value = np.random.choice([1, 2], p=[0.9, 0.1])

        self._state[x_pos[empty_index], y_pos[empty_index]] = value

    def print_state(self):
        """
        Pretty-print the current board to stdout using base-2 tile values.
        """
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
        """
        Get the current state array.

        Returns:
            np.ndarray: 4x4 grid in exponent representation.
        """
        return self._state

    def score(self):
        """
        Get the current game score.

        Returns:
            int: Accumulated score.
        """
        return self._score
