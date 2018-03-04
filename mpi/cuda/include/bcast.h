#ifndef MY_BCAST_H_
#define MY_BCAST_H_

#include "common.h"

int bcast_h2h(double*, int, MPI_Datatype, int, MPI_Comm);
int bcast_h2d(double*, int, MPI_Datatype, int, MPI_Comm);
int bcast_d2h(double*, int, MPI_Datatype, int, MPI_Comm);
int bcast_d2d(double*, int, MPI_Datatype, int, MPI_Comm);

void bcast_init(const info_t, double**, double**, const int);
void bcast_finalize(const info_t, double*, double*, const int);

#endif  // MY_BCAST_H_
