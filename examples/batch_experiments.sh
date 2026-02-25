#!/bin/bash
# Batch Experiments Script
# Runs multiple experiments with different configurations

echo "=================================="
echo "SmolVLA Batch Experiments"
echo "=================================="
echo ""
echo "This will run experiments with:"
echo "  - Different random seeds (3 runs)"
echo "  - Different timeouts (3 configurations)"
echo "  - Results saved to logs/"
echo ""

# Create logs directory
mkdir -p logs

cd ../scripts

# Experiment 1: Different seeds
echo "Experiment 1: Testing reproducibility with different seeds..."
for seed in 42 123 456; do
    echo "  Running with seed ${seed}..."
    python run_smolvla.py \
        --pretrain \
        --seed $seed \
        --episodes 10 \
        --timeout 60 \
        > "../examples/logs/seed_${seed}.log" 2>&1
    
    echo "    Results saved to logs/seed_${seed}.log"
done

echo ""
echo "Experiment 2: Testing different timeout values..."
for timeout in 30 60 90; do
    echo "  Running with ${timeout}s timeout..."
    python run_smolvla.py \
        --pretrain \
        --timeout $timeout \
        --episodes 10 \
        --seed 42 \
        > "../examples/logs/timeout_${timeout}.log" 2>&1
    
    echo "    Results saved to logs/timeout_${timeout}.log"
done

echo ""
echo "=================================="
echo "All experiments complete!"
echo "=================================="
echo ""
echo "Results are in: examples/logs/"
echo ""
echo "Summary of results:"
grep -h "Success" examples/logs/*.log | head -10

echo ""
echo "To view detailed results:"
echo "  cat examples/logs/seed_42.log"
echo "  cat examples/logs/timeout_60.log"
