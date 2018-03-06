#!/bin/bash
set -eu

mpirun -np 8 -host localhost:8 ./bcast
