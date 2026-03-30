import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from mcts import MCTS

def train(model, buffer, batch_size=64, epochs=5):
    """Updates model weights using a batch from the ReplayBuffer."""
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    model.train()

    for _ in range(epochs):
        batch = buffer.sample(batch_size)
        states, target_pis, target_vs = zip(*batch)

        # AlphaZero inputs: [Batch, 1 (Channel), Rows, Cols]
        s_tensor = torch.FloatTensor(np.array(states))
        if s_tensor.ndimension() == 3: # Add channel if missing
            s_tensor = s_tensor.unsqueeze(1)
            
        pi_tensor = torch.FloatTensor(np.array(target_pis))
        v_tensor = torch.FloatTensor(np.array(target_vs)).unsqueeze(1)

        out_pi, out_v = model(s_tensor)

        # Combined Loss: Value (MSE) + Policy (Cross-Entropy)
        value_loss = F.mse_loss(out_v, v_tensor)
        policy_loss = -torch.sum(pi_tensor * out_pi) / pi_tensor.size(0)
        total_loss = (value_loss * 0.5) + policy_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"Training Complete. Loss: {total_loss.item():.4f}")

def execute_episode(game, model, mcts_simulations=50, has_gravity=False):
    """Plays one self-play game and returns augmented training data."""
    train_examples = []
    state = game.get_initial_state()
    current_player = 1
    mcts = MCTS(game, model, args={"cpuct": 1.41})

    while True:
        # Think
        canonical_state = game.get_canonical_form(state, current_player)
        for _ in range(mcts_simulations):
            mcts.search(canonical_state)

        # Get Policy from MCTS visit counts
        s_bytes = canonical_state.astype(np.int8).tobytes()
        counts = [mcts.Nsa.get((s_bytes, a), 0) for a in range(game.action_size)]
        
        total_visits = sum(counts)
        if total_visits == 0:
            policy = game.get_valid_moves(state) / np.sum(game.get_valid_moves(state))
        else:
            policy = np.array(counts) / total_visits

        # Store data before augmenting
        train_examples.append([canonical_state, policy, current_player])

        # Select Action (Stochastic for first 10 moves, then Deterministic)
        action = np.random.choice(len(policy), p=policy) if len(train_examples) < 10 else np.argmax(policy)

        # Move
        state, current_player = game.get_next_state(state, current_player, action)
        reward, terminated = game.get_value_and_terminated(state, -current_player, action)

        if terminated:
            # Backfill rewards correctly
            # winner is -current_player because current_player already flipped
            winner = -current_player
            final_data = []
            for hist_state, hist_policy, hist_player in train_examples:
                p_reward = reward if hist_player == winner else -reward
                # Apply Augmentation here to double/octuple data density
                augmented = get_augmentations(hist_state, hist_policy, game.rows, game.cols, has_gravity)
                for aug_s, aug_p in augmented:
                    final_data.append((aug_s, aug_p, p_reward))
            return final_data

def get_augmentations(state, policy, rows, cols, has_gravity):
    """Applies spatial symmetries to state and policy arrays."""
    # Ensure state has channel dim for flipping logic: [1, Rows, Cols]
    if state.ndim == 2: state = state[np.newaxis, ...]
    
    pi_board = policy.reshape(rows, cols)
    results = [(state, policy)]

    # 1. Horizontal Flip
    h_state = np.flip(state, axis=2).copy()
    h_pi = np.flip(pi_board, axis=1).flatten().copy()
    results.append((h_state, h_pi))

    # 2. Vertical & Rotational (Only if no gravity and board is square)
    if not has_gravity:
        v_state = np.flip(state, axis=1).copy()
        v_pi = np.flip(pi_board, axis=0).flatten().copy()
        results.append((v_state, v_pi))
        
        if rows == cols:
            for k in [1, 2, 3]: # 90, 180, 270 deg
                r_state = np.rot90(state, k, axes=(1, 2)).copy()
                r_pi = np.rot90(pi_board, k).flatten().copy()
                results.append((r_state, r_pi))
    return results