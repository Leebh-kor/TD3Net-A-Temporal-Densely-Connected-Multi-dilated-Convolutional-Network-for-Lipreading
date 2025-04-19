#!/bin/bash

# Set PYTHONPATH to include current directory
export PYTHONPATH=$PYTHONPATH:.

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting model test...${NC}"

# Run the model test
if uv run lipreading/model.py "$@"; then
    echo -e "${GREEN}Test completed successfully!${NC}"
else
    echo -e "${RED}Test failed!${NC}"
    exit 1
fi 