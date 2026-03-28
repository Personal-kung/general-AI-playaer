This project was built as to learn how to make general AI for ANY game

Based on Alpha family

## Tasklist
To adapt your one-week AlphaZero project to "connect" to an online game and learn its rules through observation, your Day 1 will shift from manual coding to an **Automated Discovery and Environment Reconstruction** phase. 

AlphaZero requires a **perfect simulation** of the game rules to run the thousands of simulations per move needed for training. Because online play is too slow for these simulations, Day 1 must focus on using computer vision and automation to reverse-engineer the game into a local script.

### **Day 1: Connection, Observation, and Local Scenario Building**

Your goal for the first 24 hours is to build an interface that "sees" the online game and translates its mechanics into a Python `Game` class.

#### **1. Screen Perception & Input Tensors**
Instead of manually defining board indices, you must use computer vision to map the online game’s UI to a **multi-channel input tensor**.
*   **Task:** Write a script to capture the game window and identify the board state (e.g., using OpenCV). 
*   **Implementation:** Map visual elements to a matrix (e.g., $6 \times 7 \times 3$ for a Connect 4-style game). Channel 1 should represent your pieces, Channel 2 the opponent's pieces, and Channel 3 the current turn.

#### **2. Input Automation (The "Connection")**
To "learn" the game, your script needs to interact with the online interface.
*   **Task:** Implement mouse and keyboard control (using libraries like `PyAutoGUI`) to click on game coordinates.
*   **Logic:** Create a mapping between your internal "Action Space" (e.g., Column 1-7) and the specific pixel coordinates of the online game's buttons or columns.

#### **3. Rule Deduction & State Transitions**
AlphaZero must know which moves are legal and how the board changes after a move.
*   **Task:** Observe the online game to define `getLegalMoves` and `getNextState`.
*   **Learning Logic:** Use the automation script to attempt moves and observe if the online UI accepts them. Record how the visual state (the board) transitions after an action is confirmed.

#### **4. Reward and Terminal State Identification**
The AI cannot learn without knowing the outcome $z$ ($1$ for win, $-1$ for loss, $0$ for draw).
*   **Task:** Build a "Win-Check" function by identifying the visual cues for a game over (e.g., a "Victory" pop-up or a specific piece alignment).
*   **Output:** This allows your local script to return a terminal value of $1, -1, \text{or } 0$ based on the final board configuration.

#### **5. Final Milestone: The Local Simulator**
By the end of Day 1, you must wrap these observations into a **Local Game Class**. This class must function entirely offline, mimicking the online game's rules perfectly. This is critical because AlphaZero's MCTS needs to explore thousands of "what-if" scenarios every second—a speed impossible to achieve while playing against a live online server.

---

### **Days 2–7: Traditional AlphaZero Pipeline (Unchanged)**

Once your local "game scenario" is running, you proceed with the core AlphaZero architecture using the learned rules as the environment.

*   **Day 2: Neural Network Architecture (PyTorch)**
    *   Build the **Dual-Headed ResNet**. 
    *   **Torso:** Implement ~5–19 residual blocks with batch normalization and skip connections.
    *   **Heads:** Create the **Policy head** (outputs move probabilities $\vec{p}$) and the **Value head** (outputs state evaluation $v$).

*   **Day 3: Monte Carlo Tree Search (MCTS)**
    *   Implement the **PUCT Selection Formula** to balance exploration and exploitation.
    *   Code the four stages: **Selection** (navigating the tree), **Expansion/Evaluation** (using the network to score nodes), and **Backpropagation** (updating value statistics).

*   **Day 4: Self-Play Integration**
    *   Connect the local simulator to the MCTS and Network.
    *   Run the **Self-Play Loop** where the AI plays against its current version to generate training data: tuples of `(state, policy, outcome)`.

*   **Day 5: Optimization & Parallelization**
    *   Implement **Multiprocessing** to run several games simultaneously on your CPU.
    *   Use **Batching** to group network evaluation requests, which can speed up inference by 4x to 10x.

*   **Day 6: Scaled Training**
    *   Begin the heavy computation iteration. Update the network parameters $\theta$ by minimizing the **Composite Loss** (Mean Squared Error for value + Cross-Entropy for policy).
    *   Monitor training using a replay buffer to ensure the agent doesn't "forget" early strategies.

*   **Day 7: Evaluation & Refinement**
    *   **Arena Match:** Pit your trained model against the online game using the automation script you built on Day 1.
    *   Adjust hyperparameters like **Dirichlet noise** (to prevent repetitive play) and the exploration constant ($c_{puct}$).