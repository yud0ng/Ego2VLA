# SmolVLA Standalone Package

A complete, self-contained package for deploying SmolVLA vision-language-action models in MuJoCo simulations.

## 📦 What's Included

This standalone package contains everything you need to run SmolVLA experiments:

```
smolvla_standalone/
├── README.md                    # This file
├── setup.sh                     # Automated setup script
├── requirements.txt             # Python dependencies
├── DIRECTORY_STRUCTURE.txt      # Complete directory tree
│
├── scripts/                     # Executable scripts
│   ├── run_smolvla.py          # Main script (full CLI)
│   ├── run_smolvla_experiment.py
│   └── run_smolvla_with_env.sh
│
├── examples/                    # Example scripts
│   ├── README.md               # Examples documentation
│   ├── quick_test.sh           # Quick functionality test
│   ├── batch_experiments.sh    # Batch evaluation
│   ├── custom_task.sh          # Custom task examples
│   └── use_finetuned_model.sh  # Finetuned model usage
│
├── docs/                        # Documentation
│   ├── QUICKSTART.md           # 5-minute quick start
│   ├── MAIN_SCRIPT_GUIDE.md    # Complete reference
│   ├── MPS_GUIDE.md            # Mac GPU setup
│   ├── PRETRAINED_MODEL_USAGE.md
│   ├── HOW_TO_USE_PRETRAINED.md
│   ├── TIMEOUT_CONTROL_GUIDE.md
│   └── FILE_INDEX.md           # File navigation guide
│
├── config/                      # Configuration files
│   ├── smolvla_omy.yaml        # SmolVLA config
│   └── pi0_omy.yaml            # Pi0 config
│
├── mujoco_env/                  # Environment code
│   ├── y_env2.py               # Main environment
│   ├── ik.py                   # Inverse kinematics
│   ├── utils.py                # Utilities
│   └── ...
│
└── asset/                       # MuJoCo scenes & models
    ├── example_scene_y2.xml    # Default scene
    ├── robotis_omy/            # Robot model
    ├── objaverse/              # Objects
    └── tabletop/               # Table models
```

## 🚀 Quick Start (5 Minutes)

### Step 1: Run Setup

```bash
# Navigate to the standalone directory
cd smolvla_standalone

# Run automated setup
./setup.sh
```

This will:
- ✅ Check Python version (3.8+ required)
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Check GPU availability
- ✅ Verify setup

### Step 2: Activate Environment

```bash
source venv/bin/activate
```

### Step 3: Run Quick Test

```bash
# Option A: Use example script
cd examples
./quick_test.sh

# Option B: Use main script directly
cd scripts
python run_smolvla.py --pretrain --episodes 5 --timeout 30
```

**That's it!** Your first experiment is running.

## 🎯 Common Use Cases

### Use Case 1: Quick Test
```bash
cd examples
./quick_test.sh
```
**Duration**: 2-5 minutes  
**Purpose**: Verify setup works

### Use Case 2: Run with Custom Task
```bash
cd scripts
python run_smolvla.py \
    --pretrain \
    --task "Place the red mug on the plate" \
    --episodes 10 \
    --timeout 60
```
**Purpose**: Test specific task

### Use Case 3: Batch Experiments
```bash
cd examples
./batch_experiments.sh
```
**Duration**: 30-60 minutes  
**Purpose**: Comprehensive evaluation

### Use Case 4: Use Your Finetuned Model
```bash
# First, copy your model to this directory
cp -r /path/to/your/smolvla_model ./

# Then run
cd scripts
python run_smolvla.py \
    --model ../smolvla_model \
    --episodes 20 \
    --timeout 60
```
**Purpose**: Production evaluation

## 🔧 Command Reference

### Main Script Options

```bash
python run_smolvla.py --help
```

**Key Arguments**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrain` | False | Use pretrained model from HuggingFace |
| `--model` | `./smolvla_model` | Path to local model |
| `--task` | None | Custom task instruction |
| `--episodes` | 20 | Number of episodes |
| `--timeout` | 60 | Timeout in seconds |
| `--hz` | 20 | Control frequency |
| `--seed` | 42 | Random seed |

**Examples**:

```bash
# Pretrained model, quick test
python run_smolvla.py --pretrain --episodes 5

# Finetuned model, long timeout
python run_smolvla.py --model ../smolvla_model --timeout 120

# Custom task
python run_smolvla.py --pretrain --task "Your task here"

# High frequency control
python run_smolvla.py --hz 30 --episodes 10
```

## 📊 Understanding Results

### Expected Success Rates

**Pretrained Model** (`--pretrain`):
- Success Rate: **~5-10%**
- Why: Not trained on specific task
- When to use: Testing, exploration

**Finetuned Model** (local):
- Success Rate: **~60-80%**
- Why: Trained on specific task
- When to use: Production, evaluation

### Example Output

```
=================================================
Experiment Statistics
=================================================
Total Episodes: 20
Successes: 15 (75.0%)
Timeouts: 3 (15.0%)
Failures: 2
Total Steps: 4521
Average Steps/Episode: 226.1
=================================================
```

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: macOS, Linux
- **Python**: 3.8+
- **RAM**: 8GB+
- **Storage**: 5GB free

### Recommended
- **OS**: macOS with Apple Silicon, or Linux with NVIDIA GPU
- **Python**: 3.10+
- **RAM**: 16GB+
- **GPU**: Apple M1/M2/M3 or NVIDIA GPU
- **Storage**: 10GB free

### GPU Support
- **CUDA**: NVIDIA GPUs (automatic)
- **MPS**: Apple Silicon Macs (see `docs/MPS_GUIDE.md`)
- **CPU**: Fallback (slower)

## 📁 Optional Files

To use a finetuned model, copy these to the standalone directory:

```bash
# Copy your trained model
cp -r /path/to/smolvla_model ./smolvla_standalone/

# Copy dataset (optional, for correct statistics)
cp -r /path/to/omy_pnp_language ./smolvla_standalone/
```

**Directory structure after copying**:
```
smolvla_standalone/
├── smolvla_model/          # Your finetuned model
│   ├── config.json
│   └── model.safetensors
└── omy_pnp_language/       # Training dataset
    ├── data/
    └── meta/
```

## 🐛 Troubleshooting

### Issue 1: Setup fails

```bash
# Check Python version
python3 --version  # Should be 3.8+

# Try manual installation
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue 2: Import errors

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue 3: MuJoCo viewer closes immediately

```bash
# Check if display is available
echo $DISPLAY

# On macOS, ensure XQuartz is running
# Or run without GUI (if supported)
```

### Issue 4: Low success rate with pretrained model

**This is expected!** Pretrained models have ~5-10% success rate.

**Solutions**:
1. Use a finetuned model (`--model ../smolvla_model`)
2. Increase timeout (`--timeout 120`)
3. Reduce number of episodes for quick tests

### Issue 5: Model download fails

```bash
# Check internet connection
ping huggingface.co

# Try again with explicit model name
python run_smolvla.py --model lerobot/smolvla_base --pretrain

# Or use local model
python run_smolvla.py --model ../smolvla_model
```

## 🔄 Updating

To update the package:

```bash
# Pull latest changes (if using git)
git pull

# Reinstall dependencies
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

### Help Commands
```bash
# Script help
cd scripts && python run_smolvla.py --help

# Setup help
./setup.sh

# Example help
cd examples && cat README.md

# Check GPU
python3 -c "import torch; print('Device:', 'CUDA' if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU')"
```

## 🎯 Next Steps

```bash
# 1. Complete setup
./setup.sh

# 2. Activate environment
source venv/bin/activate

# 3. Read quick start
cat docs/QUICKSTART.md

# 4. Run your first test
cd examples && ./quick_test.sh

# 5. Explore more
cd examples && cat README.md
```

