#!/bin/bash

echo -e "\033[33mCompiling...\033[0m"
./compile.sh

if [ $? -ne 0 ]; then
    echo -e "\033[31mCompilation failed. Exiting...\033[0m"
    exit 1
fi

echo -e "\n\033[33mRunning...\033[0m"

WORK_DIR=$(pwd)/build/src

./gol-run-with-defaults.sh $WORK_DIR
