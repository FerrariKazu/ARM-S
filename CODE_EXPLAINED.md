# 🤖 ARM-S CODEBASE: Presentation Cheat Sheet

This master guide provides a high-level walkthrough of the **Autonomous Robotic Motion System (ARM-S)**. Use this as a reference during the project presentation or when explaining the architecture to new stakeholders.

---

## 🏗️ 1. Architecture: The 4-Layer Brain
The system is built on a 4-layer control hierarchy that allows it to transition from biological muscle signals (EMG) to physical robotic motion.

| Layer | File | Purpose | Analogy |
| :--- | :--- | :--- | :--- |
| **1. Sensing** | `emg_simulator.py` | Captures muscle signals | *The Ears/Nerves* |
| **2. Intent** | `intent_model.py` | Predicts what the user wants to do next | *The Intuition* |
| **3. Planning** | `coordination_controller.py` | Decides how both arms should work together | *The Playbook* |
| **4. Execution**| `pid_controller.py` | Moves the actual motors smoothly | *The Muscles* |

---

## 🧠 2. The 7 "Control Centers"
In `src/robot/controllers/`, we have seven discrete agents that govern specific robotic behaviors:

1.  **PID Controller** (`pid_controller.py`): The "Steering". Ensures the robot follows a target path without jerky movements or vibrations.
2.  **Obstacle Avoidance** (`obstacle_avoidance.py`): The "Safety Shield". Creates an invisible force field around the human's head and body using exponential repulsion math.
3.  **Speed Controller** (`speed_controller.py`): The "Governor". Automatically slows the robot down during 'Surgical' tasks and speeds it up during 'Emergencies'.
4.  **Pattern Recognition** (`pattern_recognition.py`): The "Gesture Reader". Uses math heuristics to detect if the arm is currently REACHING, LIFTING, or SWEEPING.
5.  **Debug Logger** (`debug_logger.py`): The "Black Box". Records every millisecond of telemetry so engineers can replay mistakes.
6.  **EMG Calibration** (`calibration_controller.py`): The "Filter". Normalizes raw electric muscle signals into clean 0-1 values for the AI.
7.  **Coordination Controller** (`coordination_controller.py`): The "Teamwork Coach". Negotiates tasks between the left and right robot arms (e.g., one holds, one reaches).

---

## 📊 3. Data & Training Pipeline
How the AI 'Learns' to predict human intent:

*   **Data Factory** (`training/generate_dataset.py`): A automated recording session. Runs the simulation for hours to capture 10,000+ examples of human movement.
*   **The Librarian** (`src/intent/dataset.py`): Cleans and organizes those 10,000 examples into manageable 'Flashcards' for the neural network.
*   **The Brain** (`src/intent/intent_model.py`): A Transformer architecture (similar to ChatGPT but for motion) that looks back 500ms into the past to predict the future.

---

## 🖥️ 4. Dashboards & Visualization
Modern robotics requires high-fidelity monitoring:

*   **Mission Control** (`dashboard.py`): The main HUD. Visualizes live muscle waves, the system's reaction time (Latency), and the 3D 'Safety Bubble' in real-time.
*   **Technical HUD** (`dashboard_controllers.py`): A deep-dive for engineers. Shows the 'Invisible Forces' (PID error lines, repulsion arrows, and teamwork logic).
*   **3D Simulator** (`launch_viewer.py`): The high-fidelity physics environment (MuJoCo) where you can see the robot interact with the human.

---

## 🧪 5. Verification: How we know it's safe
*   **Logic Inspection** (`tests/test_all_controllers.py`): Runs automated math tests on all 7 controllers to ensure the brain hasn't 'crashed'.
*   **Anatomical Audit** (`tests/verify_kinematics.py`): Forces the robot into 'Extreme Yoga' poses to ensure it never hits the floor or breaks its own limbs.
*   **System Checkup** (`verify_install.py`): Ensures the computer's Graphics Card (GPU) and physics libraries are operating at peak performance.

---

> [!TIP]
> **To launch the full presentation suite:**
> Run `./present.sh` in the root directory. This will open the 3D Viewer, the Main Dashboard, and the Technical Controller HUD simultaneously.
