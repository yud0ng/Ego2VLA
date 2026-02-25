#!/bin/bash
# QUICKRUN.sh - One-command quick start for SmolVLA
# This script does everything: setup, activation, and run

echo "=================================="
echo "SmolVLA Quick Run"
echo "=================================="
echo ""

# Check if setup is needed
if [ ! -d "venv" ]; then
    echo "First time setup detected..."
    echo "Running setup script..."
    ./setup.sh
    
    if [ $? -ne 0 ]; then
        echo "Setup failed! Please run ./setup.sh manually"
        exit 1
    fi
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment"
    echo "Please run: source venv/bin/activate"
    exit 1
fi

echo "✓ Environment activated"
echo ""

# Check what to run
echo "What would you like to do?"
echo ""
echo "1) Quick test (3 episodes, pretrained model)"
echo "2) Full test (20 episodes, pretrained model)"
echo "3) Use finetuned model (if available)"
echo "4) Custom command (advanced)"
echo "5) Exit"
echo ""

read -p "Choose option (1-5): " option

case $option in
    1)
        echo ""
        echo "Running quick test..."
        cd scripts
        python run_smolvla.py --pretrain --episodes 3 --timeout 30 --seed 42
        ;;
    2)
        echo ""
        echo "Running full test..."
        cd scripts
        python run_smolvla.py --pretrain --episodes 20 --timeout 60 --seed 42
        ;;
    3)
        if [ -d "smolvla_model" ]; then
            echo ""
            echo "Running with finetuned model..."
            cd scripts
            python run_smolvla.py --model ../smolvla_model --episodes 20 --timeout 60 --seed 42
        else
            echo ""
            echo "ERROR: Finetuned model not found!"
            echo "Please copy your model:"
            echo "  cp -r /path/to/smolvla_model ./"
            echo ""
            echo "Or run with pretrained model (option 1 or 2)"
        fi
        ;;
    4)
        echo ""
        echo "Enter custom command (after 'python run_smolvla.py'):"
        echo "Example: --pretrain --task \"Your task\" --episodes 10"
        echo ""
        read -p "Arguments: " args
        cd scripts
        python run_smolvla.py $args
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "Run complete!"
echo "=================================="
echo ""
echo "To run again:"
echo "  ./QUICKRUN.sh"
echo ""
echo "To see all options:"
echo "  cd scripts && python run_smolvla.py --help"
echo ""
echo "To read documentation:"
echo "  cat docs/QUICKSTART.md"
