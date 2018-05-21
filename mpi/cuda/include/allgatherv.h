#ifndef MY_ALLGATHERV_H
#define MY_ALLGATHERV_H

#include "common.h"

int allgatherv_h2h(double*, int, double*, int*, int*, MPI_Datatype, MPI_Comm);
int allgatherv_h2d(double*, int, double*, int*, int*, MPI_Datatype, MPI_Comm);
int allgatherv_d2h(double*, int, double*, int*, int*, MPI_Datatype, MPI_Comm);
int allgatherv_d2d(double*, int, double*, int*, int*, MPI_Datatype, MPI_Comm);

void allgatherv_init(const info_t, double**, double**, double**, double**,
                     int**, int**, const int);
void allgatherv_finalize(const info_t, double*, double*, double*, double*,
                         int*, int*, const int);

#endif  // MY_BCAST_H_
