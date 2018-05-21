#ifndef MY_REDUCE_H
#define MY_REDUCE_H

#include "common.h"

int reduce_h2h(double*, double*, int, MPI_Datatype, int, MPI_Comm);
int reduce_h2d(double*, double*, int, MPI_Datatype, int, MPI_Comm);
int reduce_d2h(double*, double*, int, MPI_Datatype, int, MPI_Comm);
int reduce_d2d(double*, double*, int, MPI_Datatype, int, MPI_Comm);

void reduce_init(const info_t, double**, double**, double**, double**, const int);
void reduce_finalize(const info_t, double*, double*, double*, double*, const int);

#endif  // MY_REDUCE_H
