#!/bin/bash
rm -rf build
cmake -S . -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug
