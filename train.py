import torch
import torch.nn.functional as F
import numpy as np
from mcts import MCTS
import torch.optim as optim

# ... your other imports


def train(model, buffer, batch_size=64, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    model.train()

    for epoch in range(epochs):
        batch = buffer.sample(batch_size)
        # Prepare tensors
        states, target_pis, target_vs = zip(*batch)

        # AlphaZero inputs are [Batch, Channels, Rows, Cols]
        s_tensor = torch.FloatTensor(np.array(states)).unsqueeze(1)
        pi_tensor = torch.FloatTensor(np.array(target_pis))
        v_tensor = torch.FloatTensor(np.array(target_vs)).unsqueeze(1)

        # Forward pass
        out_pi, out_v = model(s_tensor)

        # Loss = (v - target_v)^2 - pi * log(out_pi)
        value_loss = F.mse_loss(out_v, v_tensor)
        policy_loss = -torch.sum(pi_tensor * out_pi) / pi_tensor.size(0)
        total_loss = value_loss + policy_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"Training Complete. Loss: {total_loss.item():.4f}")


def execute_episode(game, model, mcts_simulations=50):
    """
    Plays one full game against itself using MCTS.
    Returns a list of (state, mcts_policy, reward) for training.
    """
    train_examples = []
    state = game.get_initial_state()
    current_player = 1

    # Initialize a fresh MCTS for this game
    # 'cpuct' controls exploration (1.41 is the standard 'square root of 2')
    mcts = MCTS(game, model, args={"cpuct": 1.41})

    while True:
        # 1. THOUGHT: Run MCTS simulations from the current state
        for _ in range(mcts_simulations):
            mcts.search(game.get_canonical_form(state, current_player))

        # 2. POLICY: Get move probabilities based on MCTS visit counts (Nsa)
        s = game.get_canonical_form(state, current_player).astype(np.int8).tobytes()
        counts = [mcts.Nsa.get((s, a), 0) for a in range(game.cols)]

        # Normalize counts into a probability distribution (The "Improved Policy")
        total_visits = sum(counts)
        if total_visits == 0:  # Safety fallback
            policy = game.get_valid_moves(state) / np.sum(game.get_valid_moves(state))
        else:
            policy = np.array(counts) / total_visits

        # 3. DATA: Store (Canonical State, Policy, Current Player ID)
        # We store the ID to flip the reward later once we know who won
        train_examples.append(
            [game.get_canonical_form(state, current_player), policy, current_player]
        )

        # 4. ACTION: Pick a move
        # Early game (first 10 moves): Sample from distribution for variety
        # Late game: Pick the absolute best move to finish the game
        if len(train_examples) < 10:
            action = np.random.choice(len(policy), p=policy)
        else:
            action = np.argmax(policy)

        # 5. ADVANCE: Update the board
        state, current_player = game.get_next_state(state, current_player, action)

        # 6. TERMINATION: Check if someone won or it's a draw
        # We check from the perspective of the player who just moved (-current_player)
        reward, terminated = game.get_value_and_terminated(
            state, -current_player, action
        )

        if terminated:
            # Backfill rewards:
            # If reward is 1 (win), give 1 to the winner's moves and -1 to the loser's
            final_data = []
            for hist_state, hist_policy, hist_player in train_examples:
                # If the historical move was made by the winner, reward = reward
                # If by the loser, reward = -reward
                player_reward = reward if hist_player == -current_player else -reward
                final_data.append((hist_state, hist_policy, player_reward))

            return final_data
