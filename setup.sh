#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# Install system-level dependencies required by scikit-survival
apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    gcc \
    g++

# Install Python packages
pip install -r requirements.txt
