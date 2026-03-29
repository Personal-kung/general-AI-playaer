import pyautogui
import ollama
from PIL import Image
import os
import time

class LaptopEyes:
    def __init__(self, model='llava'):
        self.model = model
        self.temp_path = "current_board_state.png"

    def capture_and_analyze(self, prompt):
        print(f"--- Taking screenshot in 3 seconds... (Switch to your game window!) ---")
        time.sleep(3)
        
        # 1. Capture screen
        screenshot = pyautogui.screenshot()
        screenshot.save(self.temp_path)
        print(f"Captured: {self.temp_path}")

        # 2. Process with Local Vision Model
        try:
            print(f"Analyzing with {self.model}...")
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user', 
                    'content': prompt, 
                    'images': [self.temp_path]
                }]
            )
            return response['message']['content']
        except Exception as e:
            return f"Vision Error: {e}"
        finally:
            # Optional: Clean up
            if os.path.exists(self.temp_path):
                os.remove(self.temp_path)

if __name__ == "__main__":
    eyes = LaptopEyes()
    # Simple test: Ask it to describe what it sees on your screen
    result = eyes.capture_and_analyze("What game board is visible in this screenshot? Describe the dimensions.")
    print(f"\nAI Analysis:\n{result}")