# setup.sh

# Install scikit-survival with its system dependencies
apt-get update && apt-get install -y libgl1 libglib2.0-0

# Install Python packages from requirements.txt
pip install -r requirements.txt
