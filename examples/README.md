# Example Scripts

This directory contains example scripts demonstrating different use cases of SmolVLA.

## 📝 Available Examples

### 1. `quick_test.sh` - Quick Functionality Test
**Purpose**: Verify everything works correctly  
**Duration**: ~2-5 minutes  
**Use case**: First-time setup verification

```bash
./quick_test.sh
```

**What it does**:
- Downloads pretrained model from HuggingFace
- Runs 3 episodes with 30-second timeout
- Tests basic functionality

---

### 2. `batch_experiments.sh` - Batch Experiments
**Purpose**: Run multiple experiments with different configurations  
**Duration**: ~30-60 minutes  
**Use case**: Systematic evaluation

```bash
./batch_experiments.sh
```

**What it does**:
- Tests 3 different random seeds (reproducibility)
- Tests 3 different timeout values (30s, 60s, 90s)
- Saves results to `logs/` directory

**Output**:
```
logs/
├── seed_42.log
├── seed_123.log
├── seed_456.log
├── timeout_30.log
├── timeout_60.log
└── timeout_90.log
```

---

### 3. `custom_task.sh` - Custom Task Instructions
**Purpose**: Demonstrate custom task instructions  
**Duration**: ~10-15 minutes  
**Use case**: Testing with different task descriptions

```bash
./custom_task.sh
```

**What it does**:
- Runs with custom task: "Pick up the red object and place it on the target"
- Runs with specific task: "Place the red mug on the plate"
- Shows how to override default task

---

### 4. `use_finetuned_model.sh` - Finetuned Model
**Purpose**: Run with your finetuned model  
**Duration**: ~15-20 minutes  
**Use case**: Production evaluation

```bash
./use_finetuned_model.sh
```

**What it does**:
- Checks if finetuned model exists
- Runs 20 episodes with 60-second timeout
- Uses local model for better performance

**Requirements**:
- Model must be at `../smolvla_model/`
- Dataset at `../omy_pnp_language/` (optional)

---

## 🚀 Quick Start

### First Time
```bash
# 1. Run quick test to verify setup
./quick_test.sh

# 2. If successful, try custom task
./custom_task.sh

# 3. If you have a finetuned model
./use_finetuned_model.sh

# 4. Run comprehensive batch experiments
./batch_experiments.sh
```

---

## 📊 Understanding Results

### Success Rate Expectations

**Pretrained Model** (`--pretrain`):
- Success Rate: ~5-10%
- Reason: Not trained on specific task
- Use case: Testing only

**Finetuned Model** (local):
- Success Rate: ~60-80%
- Reason: Trained on specific task
- Use case: Production

### Example Output

```
=================================================
Experiment Statistics
=================================================
Total Episodes: 10
Successes: 6 (60.0%)
Timeouts: 3 (30.0%)
Failures: 1
Total Steps: 2341
Average Steps/Episode: 234.1
=================================================
```

---

## 🔧 Customizing Examples

### Modify Existing Scripts

Edit any script to change parameters:

```bash
# Edit quick_test.sh
nano quick_test.sh

# Change this:
--episodes 3 \
--timeout 30

# To this:
--episodes 10 \
--timeout 60
```

### Create Your Own Script

```bash
#!/bin/bash
# my_custom_test.sh

cd ../scripts

python run_smolvla.py \
    --pretrain \
    --task "Your custom task here" \
    --episodes 20 \
    --timeout 90 \
    --hz 30 \
    --seed 1234
```

Make it executable:
```bash
chmod +x my_custom_test.sh
./my_custom_test.sh
```

---

## 📈 Analyzing Results

### View Log Files

```bash
# View specific log
cat logs/seed_42.log

# Search for successes
grep "Success" logs/*.log

# Search for timeouts
grep "Timeout" logs/*.log

# View statistics
grep -A 5 "Experiment Statistics" logs/*.log
```

### Compare Results

```bash
# Compare success rates across seeds
for log in logs/seed_*.log; do
    echo "=== $log ==="
    grep "Successes:" "$log"
done

# Compare timeout effects
for log in logs/timeout_*.log; do
    echo "=== $log ==="
    grep "Successes:" "$log"
    grep "Timeouts:" "$log"
done
```

---

## 🎯 Common Modifications

### Change Number of Episodes

```bash
# In any script, modify:
--episodes 10  # Change to desired number
```

### Change Timeout Duration

```bash
# In any script, modify:
--timeout 60  # Change to desired seconds
```

### Change Control Frequency

```bash
# Add or modify:
--hz 20  # Default
--hz 30  # Higher frequency (smoother)
--hz 10  # Lower frequency (faster)
```

### Use Different Model

```bash
# Pretrained
--pretrain

# Local finetuned
--model ../smolvla_model

# Specific HuggingFace model
--model lerobot/smolvla_base --pretrain
```

---

## 🐛 Troubleshooting

### Script Won't Run

```bash
# Make executable
chmod +x script_name.sh

# Run with bash explicitly
bash script_name.sh
```

### Model Not Found

```bash
# For finetuned model scripts, copy your model:
cp -r /path/to/your/smolvla_model ../

# Or use pretrained instead:
# Edit script and add --pretrain flag
```

### Logs Directory Error

```bash
# Create logs directory manually
mkdir -p logs
```

### Permission Denied

```bash
# Make all scripts executable
chmod +x *.sh
```

---

## 💡 Tips

1. **Start with `quick_test.sh`** - Verifies everything works
2. **Check logs** - Results are saved for later analysis
3. **Use pretrained first** - No model setup needed
4. **Increase timeout** - If many timeouts occur
5. **Multiple runs** - Use different seeds for consistency

---

## 📚 More Information

- Main documentation: `../docs/MAIN_SCRIPT_GUIDE.md`
- Quick start: `../docs/QUICKSTART.md`
- All options: `cd ../scripts && python run_smolvla.py --help`

---

## ✅ Example Workflow

Complete evaluation workflow:

```bash
# 1. Quick test (verify setup)
./quick_test.sh

# 2. Test custom task
./custom_task.sh

# 3. Test with finetuned model (if available)
./use_finetuned_model.sh

# 4. Run comprehensive batch experiments
./batch_experiments.sh

# 5. Analyze results
grep "Successes:" logs/*.log
```

---

**Ready to start?** Run `./quick_test.sh`!
