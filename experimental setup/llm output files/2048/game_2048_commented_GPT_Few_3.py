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
    """Tensorforce-compatible environment implementing the 2048 game logic.

    The board is represented as a 4x4 numpy array of integers storing exponents
    of powers of two (i.e., a cell with value n corresponds to tile 2**n, and 0
    represents an empty cell). The environment exposes standard methods required
    by Tensorforce: reset(), execute(), states, and actions, as well as helpers
    for copying, scoring, and printing the board.
    """

    def __str__(self):
        """Return string representation by printing the current state.

        Note: This method prints the board and relies on default string
        conversion returning None implicitly. It can be used for quick visual
        inspection.
        """
        self.print_state()

    def reset(self):
        """Reset the environment to a new game state.

        Initializes a fresh 4x4 board with two random tiles and resets the score.

        Returns:
            numpy.ndarray: The initial state after reset.
        """
        self.__init__()
        return self._state

    def execute(self, action):
        """Apply an action to the environment.

        The action is one of [0: left, 1: up, 2: right, 3: down].
        If the game is already over or the action has no effect (i.e., it would
        not change the board), the environment returns with zero reward and the
        state unchanged.

        Args:
            action (int): The action to perform.

        Returns:
            tuple:
                - state (numpy.ndarray): The resulting state after the action (or unchanged if invalid).
                - terminal (bool): True if no further valid moves are available.
                - reward (float): The reward obtained by this action (sum of merged tile values).
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
        """Return the current largest tile value (as power-of-two)."""
        return 2**np.amax(self._state)

    @property
    def states(self):
        """Tensorforce states spec describing the observation space.

        Returns:
            dict: A states specification with shape and dtype information.
        """
        return dict(shape=self._state.shape, type='float')

    @property
    def actions(self):
        """Tensorforce actions spec describing the action space.

        Returns:
            dict: An actions specification with the number of discrete actions.
        """
        return dict(num_actions=len(ACTION_NAMES), type='int')

    def __init__(self, state=None, initial_score=0):
        """Create a new 2048 environment instance.

        Args:
            state (numpy.ndarray, optional): A 4x4 board to initialize from.
                If None, a new board is created with two random tiles.
            initial_score (int, optional): Starting score value.
        """
        self._score = initial_score

        if state is None:
            # Board stores exponents; dtype int is sufficient.
            self._state = np.zeros((4, 4), dtype=np.int)
            self.add_random_tile()
            self.add_random_tile()
        else:
            self._state = state

    def copy(self):
        """Create a deep copy of the environment (state and score).

        Returns:
            Game2048: A new environment instance with copied state and score.
        """
        return Game2048(np.copy(self._state), self._score)

    def game_over(self):
        """Check if no valid actions remain.

        Returns:
            bool: True if the game is over (no moves available), False otherwise.
        """
        for action in range(4):
            if self.is_action_available(action):
                return False
        return True

    def available_actions(self):
        """List all currently available actions.

        Returns:
            list[int]: Actions (0..3) that would change the board if applied.
        """
        return [action for action in range(4) if self.is_action_available(action)]

    def is_action_available(self, action):
        """Determine whether applying an action would change the state.

        Rotates the board so that the action corresponds to a 'left' move,
        then checks if a left action is possible on the rotated board.

        Args:
            action (int): The action to test.

        Returns:
            bool: True if the action would change the board, False otherwise.
        """
        temp_state = np.rot90(self._state, action)
        return self._is_action_available_left(temp_state)

    def _is_action_available_left(self, state):
        """Check if a 'left' move is available on the given board.

        A left move is available if:
        - There exists a non-zero tile that can slide into an empty space left of it, or
        - There exists an adjacent pair of equal non-zero tiles that can merge.

        Args:
            state (numpy.ndarray): 4x4 board oriented for a left move.

        Returns:
            bool: True if any row allows a left move; False otherwise.
        """
        for row in range(4):
            has_empty = False
            for col in range(4):
                # Track if an empty cell was seen so far in the row:
                # a non-zero tile to the right can slide into it.
                has_empty |= state[row, col] == 0
                if state[row, col] != 0 and has_empty:
                    return True
                # Merge opportunity with immediate left neighbor
                if (state[row, col] != 0 and col > 0 and
                        state[row, col] == state[row, col - 1]):
                    return True

        return False

    def do_action(self, action):
        """Apply an action to the environment and spawn a random tile.

        Internally rotates the board to transform the requested direction into
        a left move, performs the left-merge logic, rotates back, updates the
        score, and finally adds a new random tile (2 with prob 0.9, 4 with 0.1).

        Args:
            action (int): The action to perform.

        Returns:
            int: The reward collected from merges during this move.
        """
        temp_state = np.rot90(self._state, action)
        reward = self._do_action_left(temp_state)
        self._state = np.rot90(temp_state, -action)
        self._score += reward

        # Only add a random tile if the board changed (guaranteed here since action availability was checked).
        self.add_random_tile()

        return reward

    def _do_action_left(self, state):
        """Perform the 'left' move on the given board in-place.

        Tiles slide left, merging adjacent equal tiles once per move per position.
        The board stores exponents: merging two tiles of value n produces a tile
        with value n+1, and the reward increment equals 2**(n+1), which matches
        the numerical value of the merged tile.

        Args:
            state (numpy.ndarray): 4x4 board oriented for a left move.

        Returns:
            int: Total reward from all merges in this move.
        """
        reward = 0

        for row in range(4):
            merge_candidate = -1
            # Track which positions in the compacted row have already merged this move.
            merged = np.zeros((4,), dtype=np.bool)

            for col in range(4):
                # Skip empty cells; they don't move themselves.
                if state[row, col] == 0:
                    continue

                # Attempt merge with last placed tile if same value and not merged yet.
                if (merge_candidate != -1 and
                        not merged[merge_candidate] and
                        state[row, merge_candidate] == state[row, col]):
                    state[row, col] = 0
                    merged[merge_candidate] = True
                    state[row, merge_candidate] += 1
                    reward += 2 ** state[row, merge_candidate]

                else:
                    # Move tile to the next compacted position if needed.
                    merge_candidate += 1
                    if col != merge_candidate:
                        state[row, merge_candidate] = state[row, col]
                        state[row, col] = 0

        return reward

    def add_random_tile(self):
        """Add a new random tile to an empty position.

        Chooses a random empty cell and assigns:
        - 1 (tile 2) with probability 0.9
        - 2 (tile 4) with probability 0.1

        Raises:
            AssertionError: If there is no empty cell available.
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
        """Return the current internal state array.

        Returns:
            numpy.ndarray: The 4x4 board of exponents.
        """
        return self._state

    def score(self):
        """Return the current accumulated score.

        Returns:
            int: The sum of all merged tile values obtained so far.
        """
        return self._score
