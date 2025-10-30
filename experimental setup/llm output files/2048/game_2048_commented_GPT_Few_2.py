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

    The board is a 4x4 numpy array of integer exponents. A tile with value v
    represents the number 2**v in the classic 2048 game. Empty tiles are 0.
    Actions are encoded as:
      0: left, 1: up, 2: right, 3: down

    Rewards returned by execute/do_action are the sum of merged tile values
    produced by the move (in standard 2048 scoring).
    """

    def __str__(self):
        """Return a string representation by printing the current state.

        Note:
            This method prints the board to stdout and returns None, which is
            unusual for __str__. It preserves the original behavior.
        """
        self.print_state()

    def reset(self):
        """Reset the environment to an initial state.

        Two random tiles (2 with 90% chance, 4 with 10% chance) are placed.

        Returns:
            np.ndarray: The initial board state (4x4 array of exponents).
        """
        self.__init__()
        return self._state

    def execute(self, action):
        """Apply an action to the environment.

        If the game is over or the action has no effect (illegal/no-op), the
        environment returns zero reward and unchanged state.

        Args:
            action (int): One of {0: left, 1: up, 2: right, 3: down}.

        Returns:
            tuple:
                - state (np.ndarray): New board state after the action.
                - terminal (bool): Whether the game is over (no legal moves).
                - reward (float): Score gained by the action (sum of merged tiles).
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
        """Get the numeric value of the largest tile currently on the board.

        Returns:
            int: Largest tile value as a power of two.
        """
        return 2**np.amax(self._state)

    @property
    def states(self):
        """Tensorforce states specification for this environment.

        Returns:
            dict: Specification with shape and dtype information.
        """
        return dict(shape=self._state.shape, type='float')

    @property
    def actions(self):
        """Tensorforce actions specification for this environment.

        Returns:
            dict: Specification with number of discrete actions.
        """
        return dict(num_actions=len(ACTION_NAMES), type='int')

    def __init__(self, state=None, initial_score=0):
        """Initialize a new 2048 game instance.

        Args:
            state (np.ndarray, optional): Existing board (4x4) of exponents.
                If None, a fresh game is started with two random tiles placed.
            initial_score (int): Starting score, used when creating a copy.
        """
        self._score = initial_score

        if state is None:
            # Initialize empty board and place two random tiles
            self._state = np.zeros((4, 4), dtype=np.int)
            self.add_random_tile()
            self.add_random_tile()
        else:
            self._state = state

    def copy(self):
        """Create a deep copy of the game state and score.

        Returns:
            Game2048: A new instance with copied state and score.
        """
        return Game2048(np.copy(self._state), self._score)

    def game_over(self):
        """Check if there are no legal moves left.

        Returns:
            bool: True if no moves are available, False otherwise.
        """
        for action in range(4):
            if self.is_action_available(action):
                return False
        return True

    def available_actions(self):
        """List all currently available actions (non-no-op moves).

        Returns:
            list[int]: Actions in {0,1,2,3} that change the board.
        """
        return [action for action in range(4) if self.is_action_available(action)]

    def is_action_available(self, action):
        """Check whether an action would change the board.

        Rotates the board so the action maps to a left move, then checks if a
        left move would be effective.

        Args:
            action (int): One of {0: left, 1: up, 2: right, 3: down}.

        Returns:
            bool: True if the move changes the board; False otherwise.
        """
        # Rotate the board so that the desired action corresponds to "left"
        temp_state = np.rot90(self._state, action)
        return self._is_action_available_left(temp_state)

    def _is_action_available_left(self, state):
        """Check if a left move is available on the given rotated board.

        A left move is available if there exists:
          - a non-empty tile with an empty cell to its left (i.e., can slide), or
          - two adjacent equal non-empty tiles (i.e., can merge).

        Args:
            state (np.ndarray): Board rotated such that the intended direction is left.

        Returns:
            bool: True if a left move would change the board.
        """
        for row in range(4):
            has_empty = False  # whether we've seen an empty tile to the left
            for col in range(4):
                has_empty |= state[row, col] == 0
                # If current cell is non-empty and there's empty space to the left -> slide possible
                if state[row, col] != 0 and has_empty:
                    return True
                # If current and previous cells are equal and non-empty -> merge possible
                if (state[row, col] != 0 and col > 0 and
                        state[row, col] == state[row, col - 1]):
                    return True

        return False

    def do_action(self, action):
        """Apply an action to the board and spawn a new random tile.

        The board is rotated so that the action maps to a left move, the left
        move is applied, then rotated back. If the move is valid, a new random
        tile (2 or 4) is added.

        Args:
            action (int): One of {0: left, 1: up, 2: right, 3: down}.

        Returns:
            int: Reward (sum of merged tile values) gained by this move.
        """
        # Rotate, apply left, then rotate back to original orientation
        temp_state = np.rot90(self._state, action)
        reward = self._do_action_left(temp_state)
        self._state = np.rot90(temp_state, -action)
        self._score += reward

        # Add a new tile after a successful move
        self.add_random_tile()

        return reward

    def _do_action_left(self, state):
        """Execute a left move in-place on the provided board.

        Implements the slide-and-merge logic used in 2048:
        - Tiles slide left to fill empty spaces.
        - Adjacent equal tiles merge once per move per cell (from left to right).
        - Reward accumulates as the sum of merged tile values (2**exponent).

        Args:
            state (np.ndarray): Board rotated such that the intended direction is left.

        Returns:
            int: Reward earned by merges during this left move.
        """
        reward = 0

        for row in range(4):
            merge_candidate = -1  # index of the last compacted/placed column
            merged = np.zeros((4,), dtype=np.bool)  # track merges to prevent double-merge

            for col in range(4):
                if state[row, col] == 0:
                    continue

                # Attempt to merge with the last placed tile if equal and not merged yet
                if (merge_candidate != -1 and
                        not merged[merge_candidate] and
                        state[row, merge_candidate] == state[row, col]):
                    # Merge into the candidate position
                    state[row, col] = 0
                    merged[merge_candidate] = True
                    state[row, merge_candidate] += 1  # exponent increases by 1
                    reward += 2 ** state[row, merge_candidate]

                else:
                    # Move/compact current tile to the next position to the left
                    merge_candidate += 1
                    if col != merge_candidate:
                        state[row, merge_candidate] = state[row, col]
                        state[row, col] = 0

        return reward

    def add_random_tile(self):
        """Add a random tile (2 with 90% chance, 4 with 10% chance) to an empty cell.

        The internal board stores exponents, so 2 -> 1 and 4 -> 2.
        Raises:
            AssertionError: If there are no empty cells available.
        """
        x_pos, y_pos = np.where(self._state == 0)
        assert len(x_pos) != 0
        empty_index = np.random.choice(len(x_pos))
        value = np.random.choice([1, 2], p=[0.9, 0.1])

        self._state[x_pos[empty_index], y_pos[empty_index]] = value

    def print_state(self):
        """Pretty-print the current board to stdout.

        Displays actual tile values (powers of two), with blanks for empties.
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
        """Get the current board.

        Returns:
            np.ndarray: The current 4x4 board of exponents.
        """
        return self._state

    def score(self):
        """Get the current score.

        Returns:
            int: Accumulated reward from all moves so far.
        """
        return self._score
