#!/bin/bash

TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")
NUMPY=$(python -c "import os; import numpy; print(os.path.dirname(numpy.__file__))")

mkdir -p lib/

echo "Compiling CUDA shared library"
nvcc -std=c++11 --shared --gpu-architecture=compute_35 -o lib/libMYTH_cu.so src/MYTH.cu --compiler-options "-fPIC -I src/ -I ${TORCH}/lib/include -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC"

echo "Compiling PyCUDA extension"
python build.py

echo "Compiling Cython extension (if failing, change the include path)"
cython camera_utils.pyx
gcc -c -fPIC -I/home/sdonne/anaconda3/envs/defusr/include/python3.7m/ -I${NUMPY}/core/include/ camera_utils.c -o camera_utils.o
gcc -shared camera_utils.o -o camera_utils.so
rm camera_utils.c camera_utils.o
