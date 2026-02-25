#!/bin/bash
# Setup Script for SmolVLA Standalone Package
# Automates the initial setup process

echo "=================================="
echo "SmolVLA Standalone Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "ERROR: Python 3.8+ required"
    exit 1
fi
echo "✓ Python version OK"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo "✓ Virtual environment recreated"
    fi
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "ERROR: Failed to install dependencies"
    exit 1
fi
echo ""

# Check GPU availability
echo "Checking GPU availability..."
gpu_status=$(python3 -c "import torch; print('CUDA' if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU')" 2>/dev/null)
echo "Device: $gpu_status"
if [ "$gpu_status" = "CPU" ]; then
    echo "Note: No GPU detected. Will use CPU (slower)"
    echo "Mac users: See docs/MPS_GUIDE.md for GPU setup"
else
    echo "✓ GPU available"
fi
echo ""

# Check for model and dataset
echo "Checking for model and dataset..."
if [ -d "smolvla_model" ]; then
    echo "✓ Found finetuned model: smolvla_model/"
    has_model=true
else
    echo "! No finetuned model found"
    echo "  You can use --pretrain flag to download from HuggingFace"
    has_model=false
fi

if [ -d "omy_pnp_language" ]; then
    echo "✓ Found dataset: omy_pnp_language/"
    has_dataset=true
else
    echo "! No dataset found"
    echo "  Will use auto-detection when running"
    has_dataset=false
fi
echo ""

# Summary
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Summary:"
echo "  Python: $python_version"
echo "  Device: $gpu_status"
echo "  Virtual Env: venv/"
echo "  Model: $([ "$has_model" = true ] && echo "Found" || echo "Not found (use --pretrain)")"
echo "  Dataset: $([ "$has_dataset" = true ] && echo "Found" || echo "Not found")"
echo ""
echo "Next Steps:"
echo ""
echo "1. Activate virtual environment (if not already active):"
echo "   source venv/bin/activate"
echo ""
echo "2. Read the quick start guide:"
echo "   cat docs/QUICKSTART.md"
echo ""
echo "3. Run a quick test:"
if [ "$has_model" = true ]; then
    echo "   cd scripts && python run_smolvla.py --model ../smolvla_model --episodes 5"
else
    echo "   cd scripts && python run_smolvla.py --pretrain --episodes 5"
fi
echo ""
echo "4. Or try example scripts:"
echo "   cd examples && ./quick_test.sh"
echo ""
echo "For help:"
echo "   cd scripts && python run_smolvla.py --help"
echo "   cat docs/MAIN_SCRIPT_GUIDE.md"
echo ""
echo "=================================="
