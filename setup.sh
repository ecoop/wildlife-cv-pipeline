#!/bin/bash
set -e  # Exit on any error

echo "Setting up development environment..."
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete! Run 'source venv/bin/activate' to activate the environment."
