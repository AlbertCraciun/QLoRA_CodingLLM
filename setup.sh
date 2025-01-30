#!/bin/bash

echo "Starting Deepseek-R1 QLoRA setup..."

# 1️ Update System Packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# 2️ Install Python & Dependencies
echo "Installing Python, pip, and venv..."
sudo apt install -y python3 python3-pip python3-venv git

# 3️ Create Virtual Environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 4️ Upgrade pip & Install Python Dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 5️ Verify GPU & CUDA Availability
echo "Checking CUDA availability..."
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

echo "Setup complete! Run 'source venv/bin/activate' to activate the environment."
