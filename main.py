from researcher import GameResearcher
from model import AlphaNet
import torch

def initialize_smart_program(game_name, board_image):
    print(f"--- Initializing: {game_name} ---")
    researcher = GameResearcher()
    
    # Step 1: Research and Extract
    print("[1/3] Researching rules and analyzing board...")
    specs = researcher.analyze_board(board_image, game_name)
    
    if not specs:
        print("Error: Could not synthesize game parameters.")
        return
    
    print(f"Discovered Specs: {specs}")

    # Step 2: Build Environment/Model
    print("[2/3] Building dynamic neural architecture...")
    model = AlphaNet(input_shape=specs['board_shape'], action_size=specs['action_size'])
    
    # Step 3: Logic-Mirror Test
    print("[3/3] Running architecture validation...")
    sample_input = torch.randn(1, *specs['board_shape'])
    policy, value = model(sample_input)
    
    assert policy.shape == (1, specs['action_size']), "Policy output mismatch!"
    print("Success: System initialized and architecture validated.")
    
    return model, specs

if __name__ == "__main__":
    # Test with a local image of a board game
    initialize_smart_program("Connect 4", "connect4_board.png")