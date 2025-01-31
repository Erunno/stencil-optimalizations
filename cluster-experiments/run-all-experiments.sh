#!/bin/bash

JOBS_FOLDER="./final-measurements/_scripts"
EXECUTABLE="./internal-scripts/run-one-exp.sh"

for script in "$JOBS_FOLDER"/*.sh; do
    if [ -x "$script" ]; then
        EXECUTABLE=$EXECUTABLE "$script"
    else
        echo "Skipping $script, not executable"
    fi
done
