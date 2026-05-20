# ============================================================
# FILE: bci_pipeline/run_pipeline.sh
# ============================================================
#!/bin/bash
set -e

echo "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "Virtual environment created."
fi

source venv/Scripts/activate || source venv/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt

echo "Creating necessary directories..."
mkdir -p data/raw/fnirs/ds003831
mkdir -p data/raw/seed_iv/eeg_raw_data
mkdir -p data/processed
mkdir -p logs

echo "Running orchestrator..."
python orchestrator.py --subjects 1 2 3 4 5 6 7 8 9 10 --modality multimodal

echo "Pipeline execution completed."
