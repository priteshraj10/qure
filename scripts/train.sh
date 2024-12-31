#!/bin/bash

# Colors for status messages
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${RED}Virtual environment not activated. Running activate script...${NC}"
    source scripts/activate.sh
fi

# Check system requirements
echo -e "${GREEN}Checking system requirements...${NC}"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Start training
echo -e "${GREEN}Starting training...${NC}"
python train.py 