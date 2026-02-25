#!/bin/bash
# run_smolvla_with_env.sh
# Wrapper script to run SmolVLA with proper environment activation

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}SmolVLA Experiment Runner (with environment)${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d ".venv" ] && [ ! -d "env" ]; then
    echo -e "${RED}Error: No virtual environment found!${NC}"
    echo "Please create and activate a virtual environment first:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Find and activate virtual environment
if [ -d "venv" ]; then
    VENV_PATH="venv"
elif [ -d ".venv" ]; then
    VENV_PATH=".venv"
elif [ -d "env" ]; then
    VENV_PATH="env"
fi

echo -e "${GREEN}Activating virtual environment: $VENV_PATH${NC}"
source "$VENV_PATH/bin/activate"

# Check if required packages are installed
python -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: torch not installed in virtual environment!${NC}"
    echo "Please install requirements:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo -e "${GREEN}Environment activated successfully!${NC}"
echo ""

# Run the script with all arguments passed through
python run_smolvla.py "$@"

# Capture exit code
EXIT_CODE=$?

echo ""
echo -e "${BLUE}==================================================${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Script completed successfully!${NC}"
else
    echo -e "${RED}Script exited with code: $EXIT_CODE${NC}"
fi
echo -e "${BLUE}==================================================${NC}"

exit $EXIT_CODE
