import torch
import torch.optim as optim
import numpy as np
import multiprocessing as mp
import warnings
from model import AlphaNet
from logic import GameLogic
from mcts import MCTS
from train import train
from memory import ReplayBuffer
from researcher import GameResearcher

# Suppress the socket warnings from researcher/ollama
warnings.filterwarnings("ignore", category=ResourceWarning)


def play_single_game_worker(args):
    """
    Worker function: Runs 1 full self-play episode in a separate process.
    Everything must be re-initialized inside the worker for thread-safety.
    """
    (state_dict, rows, cols, has_gravity, action_size, sims) = args

    # 1. Initialize local logic and model
    game = GameLogic(rows=rows, cols=cols, has_gravity=has_gravity)
    model = AlphaNet(input_shape=(1, rows, cols), action_size=action_size)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Setup MCTS for this specific worker
    mcts = MCTS(game, model, args={"cpuct": 1.41})
    game_history = []
    state = game.get_initial_state()
    current_player = 1

    while True:
        canonical_state = game.get_canonical_form(state, current_player)

        # AI Think
        for _ in range(sims):
            mcts.search(canonical_state)

        s_key = canonical_state.astype(np.int8).tobytes()
        counts = [mcts.Nsa.get((s_key, a), 0) for a in range(game.action_size)]

        # Create probability distribution (Policy)
        if sum(counts) > 0:
            probs = np.array(counts) / sum(counts)
        elif sum(counts) == 0:
            mask = game.get_valid_moves(state)
            probs = mask / np.sum(mask)
        else:
            probs = game.get_valid_moves(state) / np.sum(game.get_valid_moves(state))

        game_history.append([canonical_state, probs, None])

        # Choose action based on MCTS policy
        action = np.random.choice(len(probs), p=probs)
        state, next_player = game.get_next_state(state, 1, action)

        if state is None:  # Logic check
            print("Warning: Game Logic returned None state.")
            break

        # Ensure game_history only adds valid data
        if canonical_state is not None and probs is not None:
            game_history.append([canonical_state, probs, None])

        reward, terminated = game.get_value_and_terminated(state, 1, action)

        if terminated:
            # Backfill rewards (-1 for loser, 1 for winner)
            # We return a list of (state, policy, value)
            return [
                (x[0], x[1], reward * ((-1) ** (i % 2)))
                for i, x in enumerate(reversed(game_history))
            ]

        current_player = next_player


def initialize_system(game_name, board_image):
    """PHASE 1 & 2: Dynamic Research and Architecture Initialization."""
    researcher = GameResearcher()
    specs = researcher.analyze_board(board_image, game_name)

    if specs is None:
        print("CRITICAL ERROR: Researcher could not find board specs.")
        # Fallback to default Connect 4 if researcher fails
        specs = {"rows": 6, "cols": 7, "has_gravity": True, "board_shape": (1, 6, 7)}

    rows = specs["rows"]
    cols = specs["cols"]
    has_gravity = specs["has_gravity"]

    # Initialize Logic to get the correct action_size (7 or 42)
    game = GameLogic(rows=rows, cols=cols, has_gravity=has_gravity)

    # Initialize AlphaNet (ResNet version)
    model = AlphaNet(input_shape=(1, rows, cols), action_size=game.action_size)

    return model, game, specs


def run_day3_cycle(model, game, specs, iterations=50, games_per_iter=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Note: Resetting the buffer here means you start from 0 every time you run the script.
    # If you want to keep data across restarts, you'd load a pkl file here.
    buffer = ReplayBuffer(max_size=20000)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=3, factor=0.5
    )

    print(
        f"--- Starting Day 3 Scaling: {game.rows}x{game.cols} (Gravity: {game.has_gravity}) ---"
    )

    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}")

        model.to("cpu")  # Essential for stable multiprocessing
        state_dict = model.state_dict()
        worker_args = (
            state_dict,
            game.rows,
            game.cols,
            game.has_gravity,
            game.action_size,
            60,
        )

        # --- RESTORED PARALLELISM ---
        num_workers = min(mp.cpu_count(), games_per_iter)
        print(f"  Self-playing {games_per_iter} games using {num_workers} workers...")

        with mp.Pool(processes=num_workers) as pool:
            batch_results = pool.map(
                play_single_game_worker, [worker_args] * games_per_iter
            )

        # --- DATA ACCUMULATION ---
        for game_history in batch_results:
            if game_history:  # Ensure worker didn't return None
                for step in game_history:
                    buffer.add(step)

        # --- TRAINING PHASE ---
        model.to(device)
        if len(buffer) >= 128:
            # IMPORTANT: Ensure your train() function in train.py returns the loss value!
            total_loss = train(model, buffer, batch_size=64, epochs=4)

            # Safety check for the scheduler error we saw earlier
            if total_loss is not None:
                scheduler.step(total_loss)
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Training Complete | Loss: {total_loss:.4f} | LR: {current_lr}"
                )
                torch.save(model.state_dict(), "latest_model.pth")
                print(f"  [SAVED] Checkpoint updated.")
        else:
            print(f"  [WAITING] Collecting more data... ({len(buffer)}/128)")


if __name__ == "__main__":
    # REQUIRED for Linux/Ubuntu multiprocessing
    mp.set_start_method("spawn", force=True)

    # 1. Flexible Setup based on image analysis
    model, game, specs = initialize_system("Connect 4", "test.png")    

    # 2. Run the High-Performance Day 3 Loop
    run_day3_cycle(model, game, specs)
