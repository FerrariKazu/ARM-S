# ARM-S: Autonomous Robotic Motion System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python 3.10">
  <img src="https://img.shields.io/badge/ROS2-Humble-orange.svg" alt="ROS2 Humble">
  <img src="https://img.shields.io/badge/MuJoCo-3.1.6-green.svg" alt="MuJoCo 3.1.6">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License MIT">
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg" alt="Status: Active Development">
</p>

## What Is ARM-S?
ARM-S (Autonomous Robotic Motion System) is an advanced research project focused on the development of Supernumerary Robotic Limbs (SRL). It features an AI-driven dual backpack arm system designed to augment human physical capabilities through intuitive interaction and predictive motion planning. By leveraging electromyography (EMG) signals and a sophisticated Transformer-based intent recognition model, ARM-S anticipates human needs and provides proactive assistance in complex tasks.

The system is designed for a variety of high-impact scenarios, from assisting surgeons in the operating room to aiding workers in industrial assembly and providing overhead support during maintenance. **Note: This is a simulation-first project**, utilizing the MuJoCo physics engine to ensure high-fidelity interactions and safety-critical verification before hardware deployment.

## Demo
Run `./present.sh` to launch all 3 visualization components simultaneously.

```bash
# Launch the full presentation suite
./present.sh
```
- **MuJoCo Live Simulation:** The physical world where you watch the robot interact with the human.
- **Main System Dashboard:** Real-time HUD showing live EMG waves, AI confidence, and system latency.
- **Controllers Dashboard:** A deep-dive for engineers showing PID tracking, safety force fields, and coordination logic.

## System Architecture
```text
SENSING [EMG/Pose] → INTENT [Transformer] → PLANNING [Dual-Arm Coach] → EXECUTION [PID/Safety]
```
- **SENSING:** Captures muscle electrical activity and physical limb posture.
- **INTENT:** Predicts what the user is about to do 500ms into the future.
- **PLANNING:** Coordinates teamwork between both robotic arms.
- **EXECUTION:** Moves the physical joints smoothly while maintaining safety "force fields".

## 7 Controller Modules
| Controller | File | Purpose |
| :--- | :--- | :--- |
| **PID Joint Controller** | `pid_controller.py` | Smoothly tracks target angles without vibration. |
| **Obstacle Avoidance** | `obstacle_avoidance.py` | Safety "force field" around the human operator's head. |
| **Speed Controller** | `speed_controller.py` | Hardware-level velocity limits for different task modes. |
| **Pattern Recognition** | `pattern_recognition.py` | Heuristic gesture detection (Reach, Lift, Sweep). |
| **Debug Logger** | `debug_logger.py` | "Black Box" flight recorder for all system telemetry. |
| **EMG Calibration** | `calibration_controller.py` | Normalizes biometric signals to a standard 0-1 range. |
| **Coordination Controller**| `coordination_controller.py`| Manages teamwork protocols (Independent, Symmetric, etc).|

## Quick Start
```bash
# 1. Clone the repo
git clone https://github.com/FerrariKazu/ARM-S.git
cd ARM-S

# 2. Set up the environment
bash setup_env.sh

# 3. Activate the environment
source srl_env/bin/activate

# 4. Verify installation
python verify_install.py

# 5. Launch visual demo
./present.sh
```

## Project Structure
```text
ARM-S/
├── src/
│   ├── perception/      # EMG simulation & biometric capture
│   ├── intent/          # Transformer model & dataset loaders
│   ├── planner/         # Dual-arm coordination playbooks
│   └── robot/           # The 7 physical controllers (PID, Safety, etc.)
├── models/
│   └── mjcf/            # MuJoCo 3D robot & human XML models
├── training/            # Data generation & RL training scripts
├── tests/               # Kinematics & logic verification suites
├── dashboard.py         # Main visual HUD
└── present.sh           # Master launch script
```

## Tech Stack
| Technology | Version | Purpose |
| :--- | :--- | :--- |
| **MuJoCo** | 3.1.6 | High-fidelity physics & interaction simulation |
| **ROS2** | Humble | Distributed node architecture & communication |
| **PyTorch** | 2.x | Neural Network training & real-time inference |
| **Gymnasium** | 0.29 | Reinforcement Learning environment interface |
| **Stable-Baselines3**| 2.1.0 | PPO training for motion planning |
| **Hydra** | 1.3 | Hierarchical configuration management |
| **WandB** | 0.15 | Experiment tracking & visualization |
| **ReportLab**| 4.0 | Automated PDF technical documentation |

## Hardware Requirements
- **Minimum:** 8GB VRAM GPU, 16GB RAM, Ubuntu 22.04
- **Recommended:** NVIDIA RTX 3080+, 32GB RAM
- **Development Tested On:** RTX 4060, i5-12400F, 16GB RAM, WSL2

## Current Status
- [x] Phase 1: Environment Setup
- [x] Phase 2: System Architecture
- [x] Phase 3: Robot Modeling & Simulation
- [x] Phase 4: Intent Transformer (training)
- [x] Phase 5: RL Policy Training (training)
- [ ] Phase 6: ROS 2 Node Integration
- [ ] Phase 7: ONNX Edge Deployment
- [ ] Phase 8: Hardware Prototype
- [ ] Phase 9: Human Subject Testing

## Authors
Mina Magdy (FerrariKazu) and Sandy Antonius

## License
MIT License
