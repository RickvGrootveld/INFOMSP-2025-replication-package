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

    The board is represented as a 4x4 numpy array of integers where each entry
    stores the exponent of 2 for the tile at that position, i.e., a stored
    value of n corresponds to a tile value of 2**n. Empty cells are stored as 0.

    Actions:
    - 0: left
    - 1: up
    - 2: right
    - 3: down

    Rewards are gained when tiles merge (equal to the value of the merged tile).
    """

    def __str__(self):
        """
        String representation helper: prints the current board state.

        Note: Returns None implicitly since it only prints; primarily useful
        for debugging via print(Game2048()).
        """
        self.print_state()

    def reset(self):
        """
        Reset the environment to the initial state with two random tiles placed.

        Returns:
            numpy.ndarray: The initial board state after reset.
        """
        self.__init__()
        return self._state

    def execute(self, action):
        """
        Execute one environment step given an action.

        Args:
            action (int): One of {0: left, 1: up, 2: right, 3: down}.

        Returns:
            tuple:
                - state (numpy.ndarray): The current board state after action.
                - terminal (bool): True if no more moves are available.
                - reward (int): Reward obtained from the action (sum of merged tiles).
        """
        reward = 0
        terminal = self.game_over()
        if terminal:
            # If the game is already over, no action is applied and no reward is given.
            return self._state, terminal, reward
        action_available = self.is_action_available(action)
        if not action_available:
            # Invalid action (no tiles would move/merge); treat as no-op with zero reward.
            return self._state, terminal, reward

        reward = self.do_action(action)

        return self._state, terminal, reward

    @property
    def largest_tile(self):
        """
        Get the largest tile value currently on the board.

        Returns:
            int: The largest tile (actual value, e.g., 2048), or 1 if board is empty.
        """
        return 2**np.amax(self._state)

    @property
    def states(self):
        """
        Specification of the state space for Tensorforce.

        Returns:
            dict: Dictionary describing the shape and type of the state.
        """
        return dict(shape=self._state.shape, type='float')

    @property
    def actions(self):
        """
        Specification of the action space for Tensorforce.

        Returns:
            dict: Dictionary describing the number of discrete actions.
        """
        return dict(num_actions=len(ACTION_NAMES), type='int')

    def __init__(self, state=None, initial_score=0):
        """
        Initialize the 2048 environment.

        Args:
            state (numpy.ndarray, optional): Custom 4x4 board state of exponents.
                If None, a fresh board is created with two random tiles.
            initial_score (int, optional): Starting score. Defaults to 0.
        """
        self._score = initial_score

        if state is None:
            # Use deprecated np.int to match original code behavior; could be np.int64.
            self._state = np.zeros((4, 4), dtype=np.int)
            # Start with two random tiles on an empty board.
            self.add_random_tile()
            self.add_random_tile()
        else:
            self._state = state

    def copy(self):
        """
        Create a deep copy of the environment.

        Returns:
            Game2048: A new Game2048 instance with copied state and score.
        """
        return Game2048(np.copy(self._state), self._score)

    def game_over(self):
        """
        Determine if the game has ended (no moves available).

        Returns:
            bool: True if no action can change the board; False otherwise.
        """
        for action in range(4):
            if self.is_action_available(action):
                return False
        return True

    def available_actions(self):
        """
        List all currently available actions that produce a state change.

        Returns:
            list[int]: Actions in {0,1,2,3} that would move/merge tiles.
        """
        return [action for action in range(4) if self.is_action_available(action)]

    def is_action_available(self, action):
        """
        Check if applying the specified action would change the state.

        Internally, rotates the board so that "action" aligns with "left",
        then checks movement/merge availability on that aligned board.

        Args:
            action (int): One of {0: left, 1: up, 2: right, 3: down}.

        Returns:
            bool: True if any tile would move or merge; False otherwise.
        """
        # Rotate so that checking "left" logic applies for the chosen action.
        temp_state = np.rot90(self._state, action)
        return self._is_action_available_left(temp_state)

    def _is_action_available_left(self, state):
        """
        Check if a left move is available on the given state.

        A left move is available if:
        - There exists a non-zero tile with an empty space to its left (after compaction),
          meaning at least one tile would shift; or
        - There exists at least one pair of adjacent equal tiles that can merge.

        Args:
            state (numpy.ndarray): 4x4 board (exponents), not modified.

        Returns:
            bool: True if a left action would change the board; False otherwise.
        """
        for row in range(4):
            has_empty = False
            for col in range(4):
                # Track whether we've encountered an empty cell before current position.
                has_empty |= state[row, col] == 0
                # If current is non-zero and there was an empty before, it can move left.
                if state[row, col] != 0 and has_empty:
                    return True
                # If current equals immediate left neighbor, a merge is possible.
                if (state[row, col] != 0 and col > 0 and
                        state[row, col] == state[row, col - 1]):
                    return True

        return False

    def do_action(self, action):
        """
        Apply an action to the environment, update state and score, and spawn a new tile.

        The board is rotated to align the action with a left move, processed,
        then rotated back. A new random tile (2 with 90% probability or 4 with 10%)
        is added after a successful move.

        Args:
            action (int): One of {0: left, 1: up, 2: right, 3: down}.

        Returns:
            int: Reward obtained from all merges during this move.
        """
        # Rotate to align selected action with "left" processing.
        temp_state = np.rot90(self._state, action)
        reward = self._do_action_left(temp_state)
        # Rotate back to original orientation.
        self._state = np.rot90(temp_state, -action)
        self._score += reward

        # Add a new random tile after every valid move.
        self.add_random_tile()

        return reward

    def _do_action_left(self, state):
        """
        Execute a left move on the given state in-place and compute reward.

        The algorithm compacts tiles to the left, merging equal adjacent tiles
        at most once per tile per move, and accumulates the merge rewards.

        Args:
            state (numpy.ndarray): 4x4 board (exponents); modified in-place.

        Returns:
            int: Total reward (sum of values of created tiles, i.e., 2**new_exp).
        """
        reward = 0

        for row in range(4):
            merge_candidate = -1  # Position where the next non-zero tile may land/merge.
            merged = np.zeros((4,), dtype=np.bool)  # Track if a position already merged.

            for col in range(4):
                if state[row, col] == 0:
                    # Skip empty cells.
                    continue

                # If the last placed tile equals current and hasn't merged yet, merge them.
                if (merge_candidate != -1 and
                        not merged[merge_candidate] and
                        state[row, merge_candidate] == state[row, col]):
                    # Consume current tile by merging into merge_candidate.
                    state[row, col] = 0
                    merged[merge_candidate] = True
                    state[row, merge_candidate] += 1  # Increase exponent: 2^n + 2^n -> 2^(n+1)
                    reward += 2 ** state[row, merge_candidate]

                else:
                    # Move tile to the next available position to the left.
                    merge_candidate += 1
                    if col != merge_candidate:
                        state[row, merge_candidate] = state[row, col]
                        state[row, col] = 0

        return reward

    def add_random_tile(self):
        """
        Add a new random tile to an empty cell.

        The tile is 2 (exponent 1) with probability 0.9, or 4 (exponent 2) with probability 0.1.
        """
        x_pos, y_pos = np.where(self._state == 0)
        assert len(x_pos) != 0
        empty_index = np.random.choice(len(x_pos))
        value = np.random.choice([1, 2], p=[0.9, 0.1])

        self._state[x_pos[empty_index], y_pos[empty_index]] = value

    def print_state(self):
        """
        Pretty-print the current board to stdout.
        """

        def tile_string(value):
            # Render 0 as empty, otherwise show 2**value right-aligned in 5 spaces.
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
        Get the current internal state array.

        Returns:
            numpy.ndarray: 4x4 board of exponents.
        """
        return self._state

    def score(self):
        """
        Get the current cumulative score.

        Returns:
            int: Accumulated reward from all moves so far.
        """
        return self._score
