#!/bin/bash

# Ensure Julia is in your PATH or provide the full path to it below
JULIA_EXEC="julia"

# Relative path to JSON file with parameters
PARMS_PATH="./parms.json"

# ====================================
# Task 1
# ===================================
SCRIPT_PATH="./task1.jl"
echo "Running Task 1"
$JULIA_EXEC "$SCRIPT_PATH"
echo "Task 1 Complete"

# ====================================
# Task 2
# ===================================
SCRIPT_PATH="./task2.jl"
echo "Running Task 2"
$JULIA_EXEC "$SCRIPT_PATH" "$PARMS_PATH"
echo "Task 2 Complete"

# ====================================
# Task 3-5
# ===================================
SCRIPT_PATH="./task3.jl"
echo "Running Task 3-5"
$JULIA_EXEC "$SCRIPT_PATH" "$PARMS_PATH"
echo "Task 3-5 Complete"

# Exit with the same status code as the last executed command
exit $?
