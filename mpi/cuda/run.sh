#!/bin/bash
set -eu

# ======== For MVAPICH2 ========
export MV2_SMP_USE_CMA=1
export MV2_USE_CUDA=1


mpiexec -n 4 ./bcast
