#!/bin/bash

# This script runs the automated tests for the fine-tuning logic.

echo "Activating virtual environment..."
source .venv/bin/activate

# The 'discover' command is the standard way to run tests.
# It automatically finds all test files in the 'test' directory.
echo "Running Python unit tests for finetune_llama3.py..."
python3 -m unittest discover -s test -p "test_*.py"

echo "Test run finished."
