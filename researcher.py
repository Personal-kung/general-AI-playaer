import ollama
from duckduckgo_search import DDGS
import json
import re


class GameResearcher:
    """
    Identifies board game parameters via vision and web search.
    Outputs a standardized config for the AlphaZero pipeline.
    """

    def __init__(self, vision_model="llava", search_model="llama3"):
        self.vision_model = vision_model
        self.search_model = search_model

    def analyze_board(self, image_path, game_name):
        """Processes image and rules to return a universal game configuration."""
        # 1. Visual Analysis
        v_prompt = f"Analyze this {game_name} board. Return ONLY the number of rows and columns as integers (e.g., 6, 7)."
        v_resp = ollama.chat(
            model=self.vision_model,
            messages=[{"role": "user", "content": v_prompt, "images": [image_path]}],
        )

        # 2. Web Research
        try:
            with DDGS(timeout=20) as ddgs:
                results = list(
                    ddgs.text(
                        f"{game_name} official board dimensions and win rules",
                        max_results=2,
                    )
                )
                rules_context = "\n".join([r["body"] for r in results])
        except Exception:
            rules_context = "Defaulting to standard rules."

        # 3. Parameter Synthesis
        s_prompt = f"""
        Analyze: Visual({v_resp['message']['content']}) Rules({rules_context})
        Synthesize for '{game_name}'. Return ONLY JSON:
        {{
            "rows": int, 
            "cols": int,
            "has_gravity": bool, 
            "win_condition_length": int
        }}
        """
        s_resp = ollama.chat(
            model=self.search_model, messages=[{"role": "user", "content": s_prompt}]
        )

        # Robust JSON Extraction
        match = re.search(r"\{.*\}", s_resp["message"]["content"], re.DOTALL)
        if match:
            data = json.loads(match.group())

            # Universal Calculation
            # For gravity games (Connect 4), actions = columns.
            # For non-gravity (Tic-Tac-Toe/Go), actions = all cells.
            if data.get("has_gravity", False):
                data["action_size"] = data["cols"]
            else:
                data["action_size"] = data["rows"] * data["cols"]

            data["board_shape"] = [1, data["rows"], data["cols"]]
            return data

        return None
