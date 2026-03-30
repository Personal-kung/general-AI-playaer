import torch
import numpy as np
from researcher import GameResearcher
from model import AlphaNet
from logic import GameLogic
from mcts import MCTS
from memory import ReplayBuffer
from train import train, execute_episode  # Move execute_episode to train.py or main
import warnings
import sys


# This hides the warning, but Step 1 & 2 fix the actual crash that causes it
warnings.filterwarnings("ignore", category=ResourceWarning)
sys.setrecursionlimit(
    5000
)  # Increase recursion limit for deeper MCTS searches (use with caution)


def initialize_system(game_name, board_image):
    # --- PHASE 1: RESEARCH (From Day 1) ---
    researcher = GameResearcher()
    specs = researcher.analyze_board(board_image, game_name)

    # 1. Initialize Logic
    game = GameLogic(
        rows=specs["rows"], cols=specs["cols"], has_gravity=specs["has_gravity"]
    )

    # --- PHASE 2: ARCHITECTURE ---
    # specs['board_shape'] likely (1, rows, cols)
    model = AlphaNet(
        input_shape=(1, specs["rows"], specs["cols"]), action_size=game.action_size
    )

    # --- PHASE 3: LOGIC & SEARCH ---
    # Extract dimensions for the logic engine
    rows, cols = specs["board_shape"][1], specs["board_shape"][2]
    game = GameLogic(rows=rows, cols=cols)

    return model, game, specs


def start_learning_cycle(model, game, iterations=10):
    buffer = ReplayBuffer(max_size=5000)

    for i in range(iterations):
        print(f"\n--- Iteration {i+1}/{iterations} ---")

        # 1. Self-Play (Generates Data)
        print("Self-playing games...")
        iteration_data = []
        for _ in range(5):  # Play 5 games per iteration
            # execute_episode uses MCTS + Logic + Model
            game_history = execute_episode(game, model, mcts_simulations=25)
            iteration_data.extend(game_history)

        buffer.save(iteration_data)

        # 2. Training (Updates Model)
        if len(buffer.buffer) >= 64:
            print("Training on captured games...")
            train(model, buffer, batch_size=64)

        # 3. Checkpoint
        torch.save(model.state_dict(), "latest_model.pth")
        print(f"Iteration {i+1} complete. Model saved.")


def run_training_cycle(model, game, iterations=50, games_per_iter=10):
    buffer = ReplayBuffer(max_size=10000)  # Keep the last 10k moves
    # Pass 'has_gravity' from the specs we got from the Researcher
    has_gravity = specs.get("has_gravity", False)

    print(f"--- Starting Day 2 Training: {iterations} Iterations ---")

    for i in range(iterations):
        model.eval()  # Set to evaluation mode for self-play
        iteration_examples = []

        print(f"Iteration {i+1}: Self-playing {games_per_iter} games...")
        for g in range(games_per_iter):
            # execute_episode returns a list of (state, policy, reward)
            game_data = execute_episode(
                game, model, mcts_simulations=50, has_gravity=has_gravity
            )
            iteration_examples.extend(game_data)
            print(f"  Game {g+1} complete.")

        # Save new games to the permanent buffer
        buffer.save(iteration_examples)

        # Start training if we have enough data
        if len(buffer.buffer) >= 128:
            print(f"Iteration {i+1}: Training on {len(buffer.buffer)} examples...")
            train(model, buffer, batch_size=64, epochs=3)

            # Save the "Smarter" model
            torch.save(model.state_dict(), "latest_model.pth")
            print(f"  [SAVED] Checkpoint updated: latest_model.pth")
        else:
            print(
                f"  [WAITING] Need more data to start training (Current: {len(buffer.buffer)})"
            )


if __name__ == "__main__":
    # 1. Initialize everything using your successful Day 1 logic
    model, game, specs = initialize_system("Connect 4", "test.png")

    # 2. Kick off the Day 2 loop
    run_training_cycle(model, game)
