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

    def search(self, state, depth=0):
        # 1. Terminal State / Depth Check
        if depth > 200:
            return 0

        # Ensure state is consistent for hashing
        state = np.round(state).astype(np.int8)
        s = state.tobytes()

        # 2. EXPANSION / LEAF NODE
        if s not in self.Ps:
            # Prepare tensor [1, 1, Rows, Cols]
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)

            # Check if model is on GPU
            device = next(self.model.parameters()).device
            state_tensor = state_tensor.to(device)

            self.model.eval()
            with torch.no_grad():
                policy, value = self.model(state_tensor)

            # Get flat probabilities [action_size]
            probs = torch.exp(policy).cpu().numpy().flatten()
            mask = self.game.get_valid_moves(state)

            # --- DYNAMIC SHAPE CORRECTION ---
            # If the model gives 7 but logic wants 42 (or vice versa),
            # we must reconcile them based on the 'has_gravity' rule.
            if probs.shape != mask.shape:
                if self.game.has_gravity and probs.shape[0] == self.game.cols:
                    # This is correct for Connect 4 (7 actions)
                    pass
                else:
                    # If they truly don't match, we fallback to the mask
                    # to prevent the broadcast crash.
                    probs = np.ones(mask.shape) / np.sum(mask)

            # MASKING: Ensure shapes match by using game.action_size
            probs = probs * mask
            sum_ps = np.sum(probs)

            if sum_ps > 0:
                self.Ps[s] = probs / sum_ps
            else:
                # Fallback: Uniform distribution over legal moves
                self.Ps[s] = mask / np.sum(mask)

            self.Ns[s] = 0
            return -value.item()

        # 3. SELECTION (PUCT Formula)
        best_u = -float("inf")
        best_a = -1
        valid_moves = self.game.get_valid_moves(state)

        # UNIVERSAL FIX: Iterate through action_size, not just columns
        for a in range(self.game.action_size):
            if valid_moves[a]:
                # Get Q-value or default to 0
                q = self.Qsa.get((s, a), 0)

                # PUCT: Q + U
                # U = C * P(s,a) * sqrt(Sum(N)) / (1 + N(s,a))
                u = (
                    self.args["cpuct"]
                    * self.Ps[s][a]
                    * (math.sqrt(self.Ns[s]) / (1 + self.Nsa.get((s, a), 0)))
                )

                if q + u > best_u:
                    best_u = q + u
                    best_a = a

        # 4. RECURSION
        # best_a is now the winning flat index (or column index for gravity games)
        next_state, next_player = self.game.get_next_state(state, 1, best_a)
        next_state = self.game.get_canonical_form(next_state, next_player)

        v = self.search(next_state, depth=depth + 1)

        # 5. BACKPROPAGATION
        if (s, best_a) in self.Qsa:
            self.Qsa[(s, best_a)] = (
                self.Nsa[(s, best_a)] * self.Qsa[(s, best_a)] + v
            ) / (self.Nsa[(s, best_a)] + 1)
            self.Nsa[(s, best_a)] += 1
        else:
            self.Qsa[(s, best_a)] = v
            self.Nsa[(s, best_a)] = 1

        self.Ns[s] += 1
        return -v
