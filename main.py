import torch
import numpy as np
from researcher import GameResearcher
from model import AlphaNet
from logic import GameLogic
from mcts import MCTS
from memory import ReplayBuffer
from train import train, execute_episode  # Move execute_episode to train.py or main
import warnings

# This hides the warning, but Step 1 & 2 fix the actual crash that causes it
warnings.filterwarnings("ignore", category=ResourceWarning)


def initialize_system(game_name, board_image):
    # --- PHASE 1: RESEARCH (From Day 1) ---
    researcher = GameResearcher()
    specs = researcher.analyze_board(board_image, game_name)
    # Ensure specs['board_shape'] is exactly what the model expects
    # Force [1, Rows, Cols] regardless of what the LLM thought
    refined_shape = [1, specs["board_shape"][1], specs["board_shape"][2]]

    print(
        f"[2/3] Building model with Shape: {refined_shape} and Actions: {specs['action_size']}"
    )

    # --- PHASE 2: ARCHITECTURE ---
    # specs['board_shape'] likely (1, rows, cols)
    model = AlphaNet(input_shape=refined_shape, action_size=specs["action_size"])

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


if __name__ == "__main__":
    # 1. Setup the board and rules
    model, game, specs = initialize_system("Connect 4", "test.png")

    # 2. Start the Day 2 Learning Loop
    start_learning_cycle(model, game)
