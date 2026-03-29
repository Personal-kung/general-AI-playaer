import math
import torch
import numpy as np

class MCTS:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args # {'cpuct': 1.41}
        self.Qsa = {}    # stores Q values for (state, action)
        self.Nsa = {}    # stores #times edge (state, action) was visited
        self.Ns = {}     # stores #times state s was visited
        self.Ps = {}     # stores initial policy (returned by neural net)

    def search(self, state):
        """
        Performs one iteration of MCTS (Selection, Expansion, Simulation, Backprop).
        """
        s = state.tobytes() # Use bytes as dictionary key

        # 1. EXPANSION / LEAF NODE
        if s not in self.Ps:
            # First time seeing this state: get Policy (P) and Value (V) from Brain
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
            with torch.no_grad():
                policy, value = self.model(state_tensor)
            
            self.Ps[s] = torch.exp(policy).numpy().flatten()
            
            # Apply Masking: Zero out illegal moves and re-normalize
            mask = self.game.get_valid_moves(state)
            self.Ps[s] = self.Ps[s] * mask
            self.Ps[s] /= np.sum(self.Ps[s]) # Re-normalize
            
            self.Ns[s] = 0
            return -value.item()

        # 2. SELECTION (PUCT Formula)
        best_u = -float('inf')
        best_a = -1
        
        for a in range(self.game.cols):
            if self.game.get_valid_moves(state)[a]:
                # PUCT formula: Q + U
                q = self.Qsa.get((s, a), 0)
                u = self.args['cpuct'] * self.Ps[s][a] * (math.sqrt(self.Ns[s]) / (1 + self.Nsa.get((s, a), 0)))
                
                if q + u > best_u:
                    best_u = q + u
                    best_a = a

        # 3. RECURSION
        next_state, next_player = self.game.get_next_state(state, 1, best_a)
        next_state = self.game.get_canonical_form(next_state, next_player)
        
        v = self.search(next_state)

        # 4. BACKPROPAGATION
        if (s, best_a) in self.Qsa:
            self.Qsa[(s, best_a)] = (self.Nsa[(s, best_a)] * self.Qsa[(s, best_a)] + v) / (self.Nsa[(s, best_a)] + 1)
            self.Nsa[(s, best_a)] += 1
        else:
            self.Qsa[(s, best_a)] = v
            self.Nsa[(s, best_a)] = 1

        self.Ns[s] += 1
        return -v