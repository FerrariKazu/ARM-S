# How to Get Started (For Sandy Antonius)

Hi Sandy! Welcome to the ARM-S project. This repository contains the simulation-first framework for our Supernumerary Robotic Limb system.

## Clone & Setup
Follow these steps to get your development environment ready:

```bash
# 1. Clone the repository
git clone https://github.com/FerrariKazu/ARM-S.git
cd ARM-S

# 2. Run the environment setup script
bash setup_env.sh

# 3. Activate the virtual environment
source srl_env/bin/activate
```

## What To Work On
We are currently entering **Phase 6: ROS 2 Node Integration**. Here are the tasks you can pick up:

- **Implementing `intent_node.py`:** Create the ROS2 node that wraps our Intent Transformer for real-time publishing over DDS.
- **Implementing `planning_node.py`:** Build the coordination logic node that listens to intent and commands the joint controllers.
- **Controller Unit Testing:** Expand scripts in `tests/` to increase coverage for individual controller edge cases.
- **Neural Network Training:** Run the Intent Transformer training pipeline (`training/train_intent.py`) and log the metrics to WandB for hyperparameter tuning.

## Branch Convention
To keep the `main` branch stable, please follow these rules:

- **main:** Stable production-ready code. Always must run flawlessly.
- **feature/sandy-description:** Use this format for any new work (e.g., `feature/sandy-intent-node`).
- **Pull Requests:** Never push directly to `main`. Create a PR and let Mina Magdy review it.

## How To Run Everything
Once set up, you can verify the system and launch the visualization suite:

1.  **Verify Install:** `python verify_install.py`
2.  **Launch Full Demo:** `./present.sh` (Opens the MuJoCo sim + 2 Dashboards)
3.  **Run Tests:** `python tests/test_all_controllers.py`

## Where Everything Lives
- **[CODE_EXPLAINED.md](CODE_EXPLAINED.md):** Read this first! It provides a high-level walkthrough of the architecture and the 7 "brains" of the robot.
- **`src/`:** Contains all the logical layers (Sensing, Intent, Planning, Execution).
- **`models/`:** 3D MuJoCo XML files for the robot and human.

Happy Coding!
— Mina Magdy
