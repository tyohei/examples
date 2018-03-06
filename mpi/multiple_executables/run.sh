#!/bin/bash
set -eu

mpirun \
  -np 1 ./hello_master \
  : \
  -np 1 ./hello_slave \
