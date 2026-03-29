import numpy as np

class GameLogic:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.action_size = cols # For Connect 4, actions = columns
        self.board_size = (rows, cols)

    def get_initial_state(self):
        """Returns an empty board (all zeros)."""
        return np.zeros(self.board_size)

    def get_valid_moves(self, state):
        """Returns a binary mask of legal moves (1 for legal, 0 for full)."""
        # A column is valid if the top cell (row 0) is empty (0)
        return (state[0] == 0).astype(int)

    def get_canonical_form(self, state, player):
        """Perspective Shift: Always return board from 'current player' view."""
        return player * state

    def get_next_state(self, state, player, action):
        """Applies a move and returns (new_state, next_player)."""
        next_state = np.copy(state)
        # Connect 4 'Gravity': find the lowest empty row in the column
        for r in reversed(range(self.rows)):
            if next_state[r, action] == 0:
                next_state[r, action] = player
                break
        return next_state, -player # -player switches 1 to -1 or vice versa

    def check_win(self, state, player):
        """Checks if the given player has 4 in a row."""
        # Horizontal
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if all(state[r, c+i] == player for i in range(4)): return True
        # Vertical
        for r in range(self.rows - 3):
            for c in range(self.cols):
                if all(state[r+i, c] == player for i in range(4)): return True
        # Positive Diagonal
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if all(state[r+i, c+i] == player for i in range(4)): return True
        # Negative Diagonal
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if all(state[r-i, c+i] == player for i in range(4)): return True
        return False

    def get_value_and_terminated(self, state, player, action):
        """Returns (value, is_terminated). Value is from current player's view."""
        if self.check_win(state, player):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True # Draw
        return 0, False