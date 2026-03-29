import ollama
from duckduckgo_search import DDGS
import json
import re

class GameResearcher:
    def __init__(self, vision_model='llava', search_model='llama3'):
        self.vision_model = vision_model
        self.search_model = search_model

    def analyze_board(self, image_path, game_name):
        # 1. Visual Analysis
        prompt = f"Analyze this image of {game_name}. Identify the grid dimensions (rows/cols) and the types of pieces visible."
        response = ollama.chat(
            model=self.vision_model,
            messages=[{'role': 'user', 'content': prompt, 'images': [image_path]}]
        )
        visual_context = response['message']['content']

        # 2. Web Research for Rules
        rules_context = ""
        with DDGS() as ddgs:
            results = ddgs.text(f"{game_name} official rules move set", max_results=3)
            rules_context = "\n".join([r['body'] for r in results])

        # 3. Parameter Synthesis
        synthesis_prompt = f"""
        Based on these inputs, generate a JSON configuration for a game engine.
        Visual: {visual_context}
        Rules: {rules_context}
        
        Return ONLY valid JSON with keys: 
        "board_shape": [channels, rows, cols], 
        "action_size": int, 
        "win_condition": "string"
        """
        
        final_response = ollama.chat(
            model=self.search_model,
            messages=[{'role': 'user', 'content': synthesis_prompt}]
        )
        
        # Extract JSON from response
        match = re.search(r'\{.*\}', final_response['message']['content'], re.DOTALL)
        return json.loads(match.group()) if match else None