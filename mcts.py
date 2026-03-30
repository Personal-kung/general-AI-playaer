import math
import torch
import numpy as np


class MCTS:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args  # Expecting {'cpuct': 1.41}
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}

    def search(self, state):
        # 1. Robust Hashing: Ensure state is integer for consistent keys
        if isinstance(state, np.ndarray):
            # 1. Round to handle float precision
            # 2. Convert to int8 to save memory
            state = np.round(state).astype(np.int8)

        s = state.tobytes()

        # 2. EXPANSION / LEAF NODE
        if s not in self.Ps:
            # Prepare tensor [Batch=1, Channels=1, H, W]
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)

            # Use GPU if available (Recommended for laptops with NVIDIA)
            if next(self.model.parameters()).is_cuda:
                state_tensor = state_tensor.cuda()

            self.model.eval()  # Set to eval mode for inference
            with torch.no_grad():
                policy, value = self.model(state_tensor)

            # Policy head usually uses LogSoftmax, so we use exp()
            self.Ps[s] = torch.exp(policy).cpu().numpy().flatten()

            # Apply Masking
            mask = self.game.get_valid_moves(state)
            self.Ps[s] = self.Ps[s] * mask

            sum_ps = np.sum(self.Ps[s])
            if sum_ps > 0:
                self.Ps[s] /= sum_ps
            else:
                # Fallback: If model is totally lost, use uniform distribution
                print(
                    "Warning: Model predicted zero probability for all legal moves. Using uniform."
                )
                self.Ps[s] = mask / np.sum(mask)

            self.Ns[s] = 0
            return -value.item()

        # 3. SELECTION (PUCT Formula)
        best_u = -float("inf")
        best_a = -1

        valid_moves = self.game.get_valid_moves(state)

        for a in range(self.game.cols):
            if valid_moves[a]:
                q = self.Qsa.get((s, a), 0)
                # PUCT: Q + U. We add 1e-8 to avoid division by zero
                u = (
                    self.args["cpuct"]
                    * self.Ps[s][a]
                    * (math.sqrt(self.Ns[s]) / (1 + self.Nsa.get((s, a), 0)))
                )

                if q + u > best_u:
                    best_u = q + u
                    best_a = a

        # 4. RECURSION
        next_state, next_player = self.game.get_next_state(state, 1, best_a)
        next_state = self.game.get_canonical_form(next_state, next_player)

        # This MUST capture the value 'v' from the deeper node
        v = self.search(next_state)

        # 5. BACKPROPAGATION
        # Ensure 's' is the key for the CURRENT state, not the next_state
        if (s, best_a) in self.Qsa:
            self.Qsa[(s, best_a)] = (
                self.Nsa[(s, best_a)] * self.Qsa[(s, best_a)] + v
            ) / (self.Nsa[(s, best_a)] + 1)
            self.Nsa[(s, best_a)] += 1
        else:
            self.Qsa[(s, best_a)] = v
            self.Nsa[(s, best_a)] = 1

        self.Ns[s] += 1
        return -v  # Return negative value for the opponent's perspective
