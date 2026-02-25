#!/bin/bash
# Use Finetuned Model Example
# Demonstrates how to use a local finetuned model

echo "=================================="
echo "SmolVLA Finetuned Model Example"
echo "=================================="
echo ""

# Check if model exists
if [ ! -d "../smolvla_model" ]; then
    echo "ERROR: Finetuned model not found!"
    echo ""
    echo "Please copy your model to the smolvla_standalone/ directory:"
    echo "  cp -r /path/to/your/smolvla_model ../smolvla_standalone/"
    echo ""
    echo "Or use the pretrained model instead:"
    echo "  ./quick_test.sh"
    exit 1
fi

# Check if dataset exists
if [ ! -d "../omy_pnp_language" ]; then
    echo "WARNING: Dataset not found at ../omy_pnp_language"
    echo "The script will still run but may use incorrect statistics"
    echo ""
fi

echo "Running with finetuned model..."
echo "Model: ../smolvla_model"
echo "Episodes: 20"
echo "Timeout: 60 seconds"
echo ""

cd ../scripts

python run_smolvla.py \
    --model ../smolvla_model \
    --episodes 20 \
    --timeout 60 \
    --hz 20 \
    --seed 42

echo ""
echo "=================================="
echo "Finetuned model test complete!"
echo "=================================="
echo ""
echo "Expected success rate: ~60-80%"
echo ""
echo "To adjust settings:"
echo "  --timeout 120    # Longer timeout"
echo "  --episodes 50    # More episodes"
echo "  --hz 30          # Higher frequency"
