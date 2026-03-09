#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "Starting environment setup for ARM-S..."

# 1. Check Python 3.10
if ! command -v python3.10 &> /dev/null; then
    echo -e "${RED}[FAIL] Python 3.10 is not available in the path.${NC}"
    exit 1
fi
echo -e "${GREEN}[SUCCESS] Python 3.10 is available.${NC}"

# 2. Create venv called srl_env
VENV_NAME="srl_env"
if [ ! -d "$VENV_NAME" ]; then
    python3.10 -m venv "$VENV_NAME"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS] Virtual environment '$VENV_NAME' created.${NC}"
    else
        echo -e "${RED}[FAIL] Failed to create virtual environment.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}[SUCCESS] Virtual environment '$VENV_NAME' already exists.${NC}"
fi

# 3. Upgrade pip and Install packages from requirements.txt
source "$VENV_NAME/bin/activate"
pip install --upgrade pip &> /dev/null

echo "Installing requirements (this may take a few minutes)..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS] All packages from requirements.txt installed.${NC}"
    else
        echo -e "${RED}[FAIL] Failed to install packages from requirements.txt.${NC}"
        exit 1
    fi
else
    echo -e "${RED}[FAIL] requirements.txt not found in current directory.${NC}"
    exit 1
fi

# 4. Verify MuJoCo and PyTorch CUDA installs
echo "Verifying core installations..."

# MuJoCo check
python -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')" &> /dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] MuJoCo installation verified.${NC}"
else
    echo -e "${RED}[FAIL] MuJoCo installation not found or failed.${NC}"
fi

# PyTorch CUDA check
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" &> /dev/null
if [ $? -eq 0 ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo -e "${GREEN}[SUCCESS] PyTorch CUDA verified. GPU: $GPU_NAME${NC}"
else
    echo -e "${RED}[FAIL] PyTorch CUDA not available.${NC}"
fi

echo -e "\nSetup complete! To begin work, activate the environment with:"
echo -e "${GREEN}source $VENV_NAME/bin/activate${NC}"
