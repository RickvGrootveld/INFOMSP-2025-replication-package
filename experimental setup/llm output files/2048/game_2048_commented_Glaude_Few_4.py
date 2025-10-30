from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorforce.environments import Environment
import numpy as np

# Action constants for the four possible moves in 2048
ACTION_NAMES = ["left", "up", "right", "down"]
ACTION_LEFT = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3


class Game2048(Environment):
    """Environment implementation for the 2048 game.
    
    This class provides a TensorForce-compatible environment for the 2048 puzzle game.
    The game is played on a 4x4 grid where tiles with powers of 2 can be merged.
    """

    def __str__(self):
        """Returns a string representation of the current game state."""
        self.print_state()

    def reset(self):
        """Resets the game to its initial state.
        
        Returns:
            The initial game state as a numpy array.
        """
        self.__init__()
        return self._state

    def execute(self, action):
        """Executes the given action and returns the resulting state, terminal flag, and reward.
        
        Arguments:
            action:
                The action to execute (0=left, 1=up, 2=right, 3=down).
        
        Returns:
            A tuple of (state, terminal, reward).
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
        """Returns the value of the largest tile on the board."""
        return 2**np.amax(self._state)

    @property
    def states(self):
        """Returns the state space specification for TensorForce."""
        return dict(shape=self._state.shape, type='float')

    @property
    def actions(self):
        """Returns the action space specification for TensorForce."""
        return dict(num_actions=len(ACTION_NAMES), type='int')

    def __init__(self, state=None, initial_score=0):
        """Initializes a new instance of the 2048 game.
        
        Arguments:
            state:
                Optional initial state. If None, creates a new game with two random tiles.
            initial_score:
                The initial score for the game.
        """
        self._score = initial_score

        if state is None:
            self._state = np.zeros((4, 4), dtype=np.int)
            self.add_random_tile()
            self.add_random_tile()
        else:
            self._state = state

    def copy(self):
        """Creates a deep copy of the current game state.
        
        Returns:
            A new Game2048 instance with the same state and score.
        """
        return Game2048(np.copy(self._state), self._score)

    def game_over(self):
        """Checks if the game is over (no valid moves remaining).
        
        Returns:
            True if the game is over, False otherwise.
        """
        for action in range(4):
            if self.is_action_available(action):
                return False
        return True

    def available_actions(self):
        """Returns a list of all currently available actions.
        
        Returns:
            A list of action indices that can be executed.
        """
        return [action for action in range(4) if self.is_action_available(action)]

    def is_action_available(self, action):
        """Checks if the given action would result in a state change.
        
        Arguments:
            action:
                The action to check.
        
        Returns:
            True if the action is available, False otherwise.
        """
        temp_state = np.rot90(self._state, action)
        return self._is_action_available_left(temp_state)

    def _is_action_available_left(self, state):
        """Checks if a left action is available on the given state.
        
        This is an internal helper method that checks if tiles can be moved
        or merged when moving left.
        
        Arguments:
            state:
                The state to check.
        
        Returns:
            True if a left move would change the state, False otherwise.
        """
        for row in range(4):
            has_empty = False
            for col in range(4):
                has_empty |= state[row, col] == 0
                if state[row, col] != 0 and has_empty:
                    return True
                if (state[row, col] != 0 and col > 0 and
                        state[row, col] == state[row, col - 1]):
                    return True

        return False

    def do_action(self, action):
        """Executes the given action and updates the game state.
        
        Arguments:
            action:
                The action to execute.
        
        Returns:
            The reward obtained from executing the action.
        """
        # rotate the state so that the action becomes a left move
        temp_state = np.rot90(self._state, action)
        reward = self._do_action_left(temp_state)
        # rotate back to original orientation
        self._state = np.rot90(temp_state, -action)
        self._score += reward

        self.add_random_tile()

        return reward

    def _do_action_left(self, state):
        """Performs a left move on the given state.
        
        This is an internal helper method that implements the core game logic
        for moving and merging tiles to the left.
        
        Arguments:
            state:
                The state to modify (modified in place).
        
        Returns:
            The reward obtained from merging tiles.
        """
        reward = 0

        for row in range(4):
            merge_candidate = -1
            merged = np.zeros((4,), dtype=np.bool)

            for col in range(4):
                if state[row, col] == 0:
                    continue

                # check if we can merge with the previous tile
                if (merge_candidate != -1 and
                        not merged[merge_candidate] and
                        state[row, merge_candidate] == state[row, col]):
                    state[row, col] = 0
                    merged[merge_candidate] = True
                    state[row, merge_candidate] += 1
                    reward += 2 ** state[row, merge_candidate]

                else:
                    # move tile to the leftmost available position
                    merge_candidate += 1
                    if col != merge_candidate:
                        state[row, merge_candidate] = state[row, col]
                        state[row, col] = 0

        return reward

    def add_random_tile(self):
        """Adds a random tile (2 or 4) to an empty position on the board.
        
        The tile value is 2 with 90% probability and 4 with 10% probability.
        """
        x_pos, y_pos = np.where(self._state == 0)
        assert len(x_pos) != 0
        empty_index = np.random.choice(len(x_pos))
        # value is stored as log2, so 1 represents 2 and 2 represents 4
        value = np.random.choice([1, 2], p=[0.9, 0.1])

        self._state[x_pos[empty_index], y_pos[empty_index]] = value

    def print_state(self):
        """Prints a visual representation of the current game state to the console."""

        def tile_string(value):
            """Formats a tile value for display."""
            if value > 0:
                return '% 5d' % (2 ** value,)
            return "     "

        separator_line = '-' * 25
        print(separator_line)
        for row in range(4):
            print("|" + "|".join([tile_string(v) for v in self._state[row, :]]) + "|")
            print(separator_line)

    def state(self):
        """Returns the current game state.
        
        Returns:
            The current state as a numpy array.
        """
        return self._state

    def score(self):
        """Returns the current game score.
        
        Returns:
            The current score as an integer.
        """
        return self._score
