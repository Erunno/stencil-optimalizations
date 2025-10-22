#!/bin/bash
if [ ! -d build ] || [ ! -f build/Makefile ]; then
    bash configure.sh || exit 1
fi

cd build || exit 1
make -j "$(nproc)"
