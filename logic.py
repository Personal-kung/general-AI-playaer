import numpy as np


class GameLogic:
    """
    General board game engine supporting both gravity (Connect 4)
    and free-placement (Tic-Tac-Toe, Gomoku) mechanics.
    """

    def __init__(self, rows, cols, has_gravity=False, win_streak=4):
        self.rows = rows
        self.cols = cols
        self.has_gravity = has_gravity
        if self.has_gravity:
            self.action_size = self.cols # Should be 7 for Connect 4
        else:
            self.action_size = self.rows * self.cols # Should be 42 for others
        self.win_streak = win_streak
        # Gravity games use columns as actions; others use flat board indices.
        # self.action_size = cols if has_gravity else rows * cols

    def get_initial_state(self):
        return np.zeros((self.rows, self.cols), dtype=np.int8)

    def get_valid_moves(self, state):
        if self.has_gravity:
            # For Connect 4, valid_moves MUST be length 7 (cols)
            return (state[0, :] == 0).astype(np.int8)
        else:
            # For Tic-Tac-Toe, valid_moves MUST be length 42 (rows * cols)
            return (state.flatten() == 0).astype(np.int8)

    def get_next_state(self, state, player, action):
        """Applies move and returns (next_state, next_player)."""
        next_state = state.copy()
        if self.has_gravity:
            for r in reversed(range(self.rows)):
                if next_state[r, action] == 0:
                    next_state[r, action] = player
                    break
        else:
            row, col = divmod(action, self.cols)
            next_state[row, col] = player
        return next_state, -player

    def get_canonical_form(self, state, player):
        return state * player

    def check_win(self, state, player):
        """Standardized win-check using numpy slicing (no warnings)."""
        # Horizontal
        for r in range(self.rows):
            for c in range(self.cols - self.win_streak + 1):
                if np.all(state[r, c : c + self.win_streak] == player):
                    return True
        # Vertical
        for r in range(self.rows - self.win_streak + 1):
            for c in range(self.cols):
                if np.all(state[r : r + self.win_streak, c] == player):
                    return True
        # Diagonals
        for r in range(self.rows - self.win_streak + 1):
            for c in range(self.cols - self.win_streak + 1):
                window = state[r : r + self.win_streak, c : c + self.win_streak]
                if np.all(np.diag(window) == player) or np.all(
                    np.diag(np.fliplr(window)) == player
                ):
                    return True
        return False

    def get_value_and_terminated(self, state, player, action):
        """Returns (value, terminated) where value is from player's perspective."""
        if self.check_win(state, player):
            return 1, True
        if not np.any(self.get_valid_moves(state)):
            return 0, True  # Draw
        return 0, False
