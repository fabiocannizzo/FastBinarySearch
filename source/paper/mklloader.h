#pragma once

#include <mkl.h>

#define MYCALLBACK __stdcall

typedef int (MYCALLBACK* hello_t)(int);

typedef int (MYCALLBACK* mysum_t)(int x, int y);
typedef int (MYCALLBACK* dfdNewTask1D_t)(DFTaskPtr *, const MKL_INT, const double[], const MKL_INT, const MKL_INT, const double[], const MKL_INT);
typedef int (MYCALLBACK* dfsNewTask1D_t)(DFTaskPtr *, const MKL_INT, const float[], const MKL_INT, const MKL_INT, const float[], const MKL_INT);
typedef int (MYCALLBACK* dfdSearchCells1D_t)(DFTaskPtr, const MKL_INT, const MKL_INT, const double[], const MKL_INT, const double[], MKL_INT[]);
typedef int (MYCALLBACK* dfsSearchCells1D_t)(DFTaskPtr, const MKL_INT, const MKL_INT, const float[], const MKL_INT, const float[], MKL_INT[]);
typedef int (MYCALLBACK* dfDeleteTask_t)(DFTaskPtr *);

extern dfdNewTask1D_t           dfdNewTask1D_ptr;
extern dfsNewTask1D_t           dfsNewTask1D_ptr;
extern dfdSearchCells1D_t       dfdSearchCells1D_ptr;
extern dfsSearchCells1D_t       dfsSearchCells1D_ptr;
extern dfDeleteTask_t           dfDeleteTask_ptr;

void InitMKLWrapper();
void ReleaseMKLWrapper();
