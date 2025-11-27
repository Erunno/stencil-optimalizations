#!/bin/bash

set -euo pipefail

if [ -d build ]; then
	rm -rf build
fi

cmake -S . -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
