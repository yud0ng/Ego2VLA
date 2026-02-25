#!/bin/bash
# Quick Test Script
# Tests basic functionality with pretrained model

echo "=================================="
echo "SmolVLA Quick Test"
echo "=================================="
echo ""
echo "This will run a quick 3-episode test with:"
echo "  - Pretrained model (downloaded from HuggingFace)"
echo "  - 30 second timeout"
echo "  - Default task"
echo ""

cd ../scripts

python run_smolvla.py \
    --pretrain \
    --episodes 3 \
    --timeout 30 \
    --seed 42

echo ""
echo "=================================="
echo "Test Complete!"
echo "=================================="
