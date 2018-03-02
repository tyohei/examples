#!/bin/bash
set -eu

data_count=8192
data_ctype=3


if [[ "$(which mpicc | grep openmpi)" != "" ]]; then
  # ======== For Open MPI ========
  echo "Using Open MPI"
  mpirun -np 4 ./main ${data_count} ${data_ctype}
else
  # ======== For MVAPICH2 ========
  echo "Using MVAPICH2"
  export MV2_SMP_USE_CMA=1
  export MV2_USE_CUDA=1
  export MV2_USE_GPUDIRECT_GDRCOPY=0
  mpiexec -n 4 ./main ${data_count} ${data_ctype}
fi
