#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Installing system dependencies..."
sudo apt update
sudo apt install -y build-essential python3 python3-pip python3-venv

echo "Creating virtual environment..."
python3 -m venv .venv --copies

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Ensuring pip is available..."
python3 -m ensurepip --upgrade

echo "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo "Setup complete."
echo "To activate the environment later, run:"
echo "source .venv/bin/activate"
