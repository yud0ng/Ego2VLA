#!/bin/bash
# Custom Task Example
# Demonstrates how to run with custom task instructions

echo "=================================="
echo "SmolVLA Custom Task Example"
echo "=================================="
echo ""
echo "This demonstrates running with custom task instructions"
echo ""

cd ../scripts

# Example 1: Simple pick and place
echo "Example 1: Simple pick and place task"
python run_smolvla.py \
    --pretrain \
    --task "Pick up the red object and place it on the target" \
    --episodes 5 \
    --timeout 60 \
    --seed 42

echo ""
echo "=================================="

# Example 2: More specific instruction
echo "Example 2: Specific object manipulation"
python run_smolvla.py \
    --pretrain \
    --task "Place the red mug on the plate" \
    --episodes 5 \
    --timeout 60 \
    --seed 123

echo ""
echo "=================================="
echo "Custom task examples complete!"
echo "=================================="
echo ""
echo "To run your own custom task:"
echo "  cd scripts"
echo "  python run_smolvla.py --pretrain --task \"Your task here\" --episodes 10"
