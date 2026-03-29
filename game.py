import numpy as np

class Game:
    def __init__(self):
        # Define global constraints that the AI will use to build its 'Head'
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.state_size = (self.row_count, self.column_count)

    def get_initial_state(self):
        """Returns a starting board (zeros)."""
        return np.zeros(self.state_size)

    def get_next_state(self, state, action, player):
        """Applies a move and returns the new state."""
        # Logic for placing a piece in Connect 4 column 'action'
        pass

    def get_valid_moves(self, state):
        """Returns a binary array (1 for legal, 0 for illegal)."""
        # Essential for 'masking' the AI's policy head so it never picks a full column
        pass

    def check_win(self, state, action):
        """Returns True if the last move resulted in a win."""
        pass

    def get_value_and_terminated(self, state, action):
        """Returns (reward, is_terminated). Reward is 1 for win, 0 for draw/ongoing."""
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False