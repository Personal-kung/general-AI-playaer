import ollama
from duckduckgo_search import DDGS
import json
import re


class GameResearcher:
    def __init__(self, vision_model="llava", search_model="llama3"):
        self.vision_model = vision_model
        self.search_model = search_model

    def analyze_board(self, image_path, game_name):
        # 1. Visual Analysis
        # Explicitly ask for integers to avoid "6x7" string values
        prompt = f"Analyze this image of {game_name}. Return ONLY the number of rows and columns as integers."
        response = ollama.chat(
            model=self.vision_model,
            messages=[{"role": "user", "content": prompt, "images": [image_path]}],
        )
        visual_context = response["message"]["content"]

        # 2. Web Research
        # Added a timeout to prevent the script from hanging on laptop WiFi
        try:
            with DDGS(timeout=20) as ddgs:
                results = list(
                    ddgs.text(
                        f"{game_name} board dimensions and move count", max_results=2
                    )
                )
                rules_context = "\n".join([r["body"] for r in results])
        except Exception:
            rules_context = "Defaulting to standard rules."

        # 3. Parameter Synthesis
        # We force the 'channels' to 1 and ensure board_shape is a list of ints
        synthesis_prompt = f"""
        Inputs: {visual_context} and {rules_context}
        Synthesize for game: {game_name}
        
        Return ONLY valid JSON:
        {{
            "rows": int, 
            "cols": int,
            "win_condition": "string"
        }}
        """

        final_response = ollama.chat(
            model=self.search_model,
            messages=[{"role": "user", "content": synthesis_prompt}],
        )

        content = final_response["message"]["content"]

        # Robust JSON cleaning: removes markdown and leading/trailing whitespace
        clean_json = re.sub(r"```json|```", "", content).strip()
        match = re.search(r"\{.*\}", clean_json, re.DOTALL)

        if match:
            data = json.loads(match.group())
            rows = data.get("rows", 6)  # Default to 6 if LLM fails
            cols = data.get("cols", 7)  # Default to 7 if LLM fails

            # --- HARD-CODED CORRECTIONS ---
            # 1. Force AlphaZero Standard: [Channels=1, Rows, Cols]
            data["board_shape"] = [1, rows, cols]

            # 2. Calculate Action Size mathematically (Columns for Connect 4)
            # If the LLM didn't provide it, we assume cols (standard for gravity games)
            data["action_size"] = cols

            return data
