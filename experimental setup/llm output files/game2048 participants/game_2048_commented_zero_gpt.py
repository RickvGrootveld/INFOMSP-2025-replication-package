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

    The environment represents the 2048 game using a 4x4 numpy array of exponents,
    where a cell value n corresponds to a tile with face value 2**n. Zeros represent
    empty tiles. Actions are encoded as integers: 0=left, 1=up, 2=right, 3=down.

    The environment provides:
    - reset(): reinitializes the board with two random tiles.
    - execute(action): applies an action if legal, spawns a new tile, and returns (state, terminal, reward).
    - properties states/actions for Tensorforce API compatibility.
    - utility methods for copying state, checking game over, available actions, etc.
    """

    def __str__(self):
        """String representation hook.

        Prints the current board state to stdout using print_state().
        Returns None because the intention is side-effect printing rather than producing a string.
        """
        self.print_state()

    def reset(self):
        """Reset the environment to an initial state.

        Re-initializes the board and score and spawns two random tiles.

        Returns:
            np.ndarray: The current 4x4 board state (exponents).
        """
        self.__init__()
        return self._state

    def execute(self, action):
        """Apply an action to the environment step.

        If the game is over or the action is not available (i.e., would not change the board),
        the state is returned unchanged with zero reward. Otherwise, applies the action, merges
        tiles, updates score, spawns a random tile, and returns the new state and reward.

        Args:
            action (int): One of {0: left, 1: up, 2: right, 3: down}.

        Returns:
            tuple:
                - state (np.ndarray): The 4x4 board after the step.
                - terminal (bool): True if no further actions are available.
                - reward (int): Sum of merged tile values produced this step.
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
        """Return the largest face value tile currently on the board.

        Returns:
            int: Largest tile value as 2**(max exponent), or 1 if board is empty.
        """
        return 2**np.amax(self._state)

    @property
    def states(self):
        """Tensorforce API: describe the state space.

        Returns:
            dict: Contains the shape and dtype type descriptor for the state.
        """
        return dict(shape=self._state.shape, type='float')

    @property
    def actions(self):
        """Tensorforce API: describe the action space.

        Returns:
            dict: Contains the number of discrete actions and type descriptor.
        """
        return dict(num_actions=len(ACTION_NAMES), type='int')

    def __init__(self, state=None, initial_score=0):
        """Initialize a new 2048 environment.

        If no state is provided, creates an empty 4x4 board, then spawns two random tiles
        with values 2 (90%) or 4 (10%). The internal board uses exponents, i.e., stores 1 for tile 2 and 2 for tile 4.

        Args:
            state (np.ndarray or None): Optional 4x4 board of exponents to start from.
            initial_score (int): Starting score value.
        """
        self._score = initial_score

        if state is None:
            # np.int is used here to mirror original code; in newer NumPy versions, prefer int or np.int64
            self._state = np.zeros((4, 4), dtype=np.int)
            self.add_random_tile()
            self.add_random_tile()
        else:
            self._state = state

    def copy(self):
        """Create a deep copy of the current environment.

        Returns:
            Game2048: A new environment instance with copied state and score.
        """
        return Game2048(np.copy(self._state), self._score)

    def game_over(self):
        """Check whether no actions are available.

        Returns:
            bool: True if no valid action can change the board; otherwise False.
        """
        for action in range(4):
            if self.is_action_available(action):
                return False
        return True

    def available_actions(self):
        """List all currently available actions.

        Returns:
            list[int]: Subset of {0,1,2,3} that would change the board.
        """
        return [action for action in range(4) if self.is_action_available(action)]

    def is_action_available(self, action):
        """Determine whether the given action would change the board.

        Strategy: Rotate the board so that the action aligns with a left move,
        then test using _is_action_available_left.

        Args:
            action (int): One of {0: left, 1: up, 2: right, 3: down}.

        Returns:
            bool: True if the move would result in any shift or merge; False otherwise.
        """
        temp_state = np.rot90(self._state, action)
        return self._is_action_available_left(temp_state)

    def _is_action_available_left(self, state):
        """Internal helper to detect availability of a left move on a given board.

        A left move is available if:
        - Any non-zero tile has an empty space to its left (i.e., compaction would occur), or
        - Any two adjacent tiles with the same value can be merged.

        Args:
            state (np.ndarray): 4x4 board oriented so that the move is to the left.

        Returns:
            bool: True if a left move would change the board; False otherwise.
        """
        for row in range(4):
            has_empty = False
            for col in range(4):
                has_empty |= state[row, col] == 0
                # If we encounter a non-zero after an empty in the same row, shifting is possible
                if state[row, col] != 0 and has_empty:
                    return True
                # If two adjacent tiles are equal, a merge is possible
                if (state[row, col] != 0 and col > 0 and
                        state[row, col] == state[row, col - 1]):
                    return True

        return False

    def do_action(self, action):
        """Execute an action, update state and score, and spawn a random tile.

        Rotates the board to align the action with a left move, performs the move/merges,
        rotates back, updates the cumulative score, and adds a new random tile.

        Args:
            action (int): One of {0: left, 1: up, 2: right, 3: down}.

        Returns:
            int: Reward for this move (sum of merged face values).
        """
        temp_state = np.rot90(self._state, action)
        reward = self._do_action_left(temp_state)
        self._state = np.rot90(temp_state, -action)
        self._score += reward

        self.add_random_tile()

        return reward

    def _do_action_left(self, state):
        """Perform the left move on the provided state (in-place).

        This compacts tiles to the left, merging equal adjacent tiles once per move per tile.
        The board uses exponents; when two equal exponents merge, the destination exponent
        increments by one, and the reward increases by the face value 2**new_exponent.

        Args:
            state (np.ndarray): 4x4 board oriented for a left move.

        Returns:
            int: Reward accumulated from merges during this move.
        """
        reward = 0

        for row in range(4):
            merge_candidate = -1  # index of the next write/merge position in the row
            merged = np.zeros((4,), dtype=np.bool)  # track whether a position has already merged

            for col in range(4):
                if state[row, col] == 0:
                    continue

                # If current tile can merge with the last placed tile and that tile hasn't merged yet
                if (merge_candidate != -1 and
                        not merged[merge_candidate] and
                        state[row, merge_candidate] == state[row, col]):
                    # Merge into the candidate position
                    state[row, col] = 0
                    merged[merge_candidate] = True
                    state[row, merge_candidate] += 1  # exponent increments by 1
                    reward += 2 ** state[row, merge_candidate]

                else:
                    # Move/compact the tile to the next available position
                    merge_candidate += 1
                    if col != merge_candidate:
                        state[row, merge_candidate] = state[row, col]
                        state[row, col] = 0

        return reward

    def add_random_tile(self):
        """Spawn a new random tile (2 with 90% probability, 4 with 10%) at an empty position.

        Internally, stores 1 for '2' and 2 for '4'.
        """
        x_pos, y_pos = np.where(self._state == 0)
        assert len(x_pos) != 0
        empty_index = np.random.choice(len(x_pos))
        value = np.random.choice([1, 2], p=[0.9, 0.1])

        self._state[x_pos[empty_index], y_pos[empty_index]] = value

    def print_state(self):
        """Print the current board state in a human-readable grid.

        Displays the face values (powers of two) or blanks for empty cells,
        surrounded by separators for clarity.
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
        """Get the current raw board state (exponents).

        Returns:
            np.ndarray: 4x4 array of tile exponents (0 for empty).
        """
        return self._state

    def score(self):
        """Get the cumulative score for the current game.

        Returns:
            int: Total score accumulated from merges.
        """
        return self._score
