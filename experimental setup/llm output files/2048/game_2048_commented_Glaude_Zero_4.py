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
    """
    A 2048 game environment compatible with TensorForce.
    
    The game is played on a 4x4 grid where tiles with powers of 2 can be merged.
    The goal is to create larger tiles by merging identical adjacent tiles.
    """

    def __str__(self):
        """Return a string representation of the current game state."""
        self.print_state()

    def reset(self):
        """
        Reset the game to its initial state.
        
        Returns:
            np.ndarray: The initial game state (4x4 grid).
        """
        self.__init__()
        return self._state

    def execute(self, action):
        """
        Execute an action in the game environment.
        
        Args:
            action (int): The action to execute (0=left, 1=up, 2=right, 3=down).
            
        Returns:
            tuple: A tuple containing:
                - state (np.ndarray): The current game state.
                - terminal (bool): Whether the game is over.
                - reward (float): The reward obtained from this action.
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
        Get the value of the largest tile on the board.
        
        Returns:
            int: The value of the largest tile (e.g., 2, 4, 8, 16, ...).
        """
        return 2**np.amax(self._state)

    @property
    def states(self):
        """
        Get the state space specification.
        
        Returns:
            dict: Dictionary describing the state space shape and type.
        """
        return dict(shape=self._state.shape, type='float')

    @property
    def actions(self):
        """
        Get the action space specification.
        
        Returns:
            dict: Dictionary describing the number of actions and type.
        """
        return dict(num_actions=len(ACTION_NAMES), type='int')

    def __init__(self, state=None, initial_score=0):
        """
        Initialize a new 2048 game.
        
        Args:
            state (np.ndarray, optional): An existing game state to initialize with.
                If None, creates a new game with two random tiles.
            initial_score (int, optional): The initial score. Defaults to 0.
        """
        self._score = initial_score

        if state is None:
            # Create a new empty 4x4 board and add two random tiles
            self._state = np.zeros((4, 4), dtype=np.int)
            self.add_random_tile()
            self.add_random_tile()
        else:
            self._state = state

    def copy(self):
        """
        Create a deep copy of the current game state.
        
        Returns:
            Game2048: A new Game2048 instance with copied state and score.
        """
        return Game2048(np.copy(self._state), self._score)

    def game_over(self):
        """
        Check if the game is over (no valid moves available).
        
        Returns:
            bool: True if the game is over, False otherwise.
        """
        for action in range(4):
            if self.is_action_available(action):
                return False
        return True

    def available_actions(self):
        """
        Get a list of all currently available actions.
        
        Returns:
            list: List of action indices that are currently valid.
        """
        return [action for action in range(4) if self.is_action_available(action)]

    def is_action_available(self, action):
        """
        Check if a specific action is available (would change the board state).
        
        Args:
            action (int): The action to check (0=left, 1=up, 2=right, 3=down).
            
        Returns:
            bool: True if the action is available, False otherwise.
        """
        # Rotate the board so we can always check as if moving left
        temp_state = np.rot90(self._state, action)
        return self._is_action_available_left(temp_state)

    def _is_action_available_left(self, state):
        """
        Check if a left move is available on the given state.
        
        A left move is available if any tile can move left or merge with an adjacent tile.
        
        Args:
            state (np.ndarray): The game state to check.
            
        Returns:
            bool: True if a left move would change the state, False otherwise.
        """
        for row in range(4):
            has_empty = False
            for col in range(4):
                has_empty |= state[row, col] == 0
                # A tile can move left if there's an empty space to its left
                if state[row, col] != 0 and has_empty:
                    return True
                # A tile can merge with the tile to its left if they have the same value
                if (state[row, col] != 0 and col > 0 and
                        state[row, col] == state[row, col - 1]):
                    return True

        return False

    def do_action(self, action):
        """
        Perform an action and update the game state.
        
        Args:
            action (int): The action to perform (0=left, 1=up, 2=right, 3=down).
            
        Returns:
            float: The reward obtained from this action (sum of merged tile values).
        """
        # Rotate the board to treat all actions as left moves
        temp_state = np.rot90(self._state, action)
        reward = self._do_action_left(temp_state)
        # Rotate back to original orientation
        self._state = np.rot90(temp_state, -action)
        self._score += reward

        # Add a new random tile after each move
        self.add_random_tile()

        return reward

    def _do_action_left(self, state):
        """
        Perform a left move on the given state, merging tiles and shifting them left.
        
        Args:
            state (np.ndarray): The game state to modify (modified in-place).
            
        Returns:
            float: The reward obtained from merging tiles.
        """
        reward = 0

        for row in range(4):
            merge_candidate = -1
            merged = np.zeros((4,), dtype=np.bool)

            for col in range(4):
                if state[row, col] == 0:
                    continue

                # Check if current tile can merge with the merge candidate
                if (merge_candidate != -1 and
                        not merged[merge_candidate] and
                        state[row, merge_candidate] == state[row, col]):
                    # Merge the tiles
                    state[row, col] = 0
                    merged[merge_candidate] = True
                    state[row, merge_candidate] += 1
                    reward += 2 ** state[row, merge_candidate]

                else:
                    # Move tile to the next available position
                    merge_candidate += 1
                    if col != merge_candidate:
                        state[row, merge_candidate] = state[row, col]
                        state[row, col] = 0

        return reward

    def add_random_tile(self):
        """
        Add a random tile (2 or 4) to an empty position on the board.
        
        The tile value is 2 (represented as 1) with 90% probability,
        or 4 (represented as 2) with 10% probability.
        """
        # Find all empty positions
        x_pos, y_pos = np.where(self._state == 0)
        assert len(x_pos) != 0
        # Choose a random empty position
        empty_index = np.random.choice(len(x_pos))
        # Choose tile value: 1 represents 2, 2 represents 4
        value = np.random.choice([1, 2], p=[0.9, 0.1])

        self._state[x_pos[empty_index], y_pos[empty_index]] = value

    def print_state(self):
        """
        Print a formatted representation of the current game board to the console.
        
        Tiles are displayed with their actual values (2, 4, 8, etc.).
        Empty cells are shown as blank spaces.
        """
        def tile_string(value):
            """Convert a tile value to a formatted string."""
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
        Get the current game state.
        
        Returns:
            np.ndarray: The 4x4 game board where each value represents log2(tile_value).
        """
        return self._state

    def score(self):
        """
        Get the current game score.
        
        Returns:
            int: The cumulative score from all tile merges.
        """
        return self._score
