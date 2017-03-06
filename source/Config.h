#pragma once

#include "Type.h"
#include <vector>

// In total we resolve 2*(N-1)*nAvg*nRepeat indices
const size_t   NZ     = 4096;  // size of array Z: must be a multiple of 4
const size_t nAvg     = 50;     // Number of random regenerations of vector X
const size_t nAvgSetup = 100;     // Number of random regenerations of vector X
const size_t nAvgSetupR = 1000000;  // Number of random regenerations of vector X
const size_t nRepeat  = 10000; // Number repetition of the test
const size_t nSetupTargetOps  = 20 * 1048576; // Number repetition of the test
#define INTMIN 1.0
#define INTMAX 5.0

const size_t VecScope[] = {15, 255, 4095, 65535, 1048575};

const Precision PrecScope[] =
    {
        Single,
        Double,
    };

const Algos AlgoScope[] =
    {
        Classic,
  //      ClassicMod,
        ClassicOffset,
        BitSet,
  //      BitSetNoPad,
        Eytzinger,
#ifdef USE_MKL
        MKL,
#endif
        Direct,
        Direct2,
        DirectCache,
    };

const InstrSet InstrScope[] =
    {
        Scalar,
        SSE,
#ifdef USE_AVX
        AVX,
#endif
    };


/*
*  Automatically computed
*/

const size_t nPrec = sizeof(PrecScope)/sizeof(*PrecScope);
const size_t nAlgo = sizeof(AlgoScope)/sizeof(*AlgoScope);
const size_t nInstr = sizeof(InstrScope)/sizeof(*InstrScope);
const size_t nVecs = sizeof(VecScope)/sizeof(*VecScope);

typedef double throughput_t[nPrec][nAvg+1][nInstr][nAlgo];
typedef SetupStats setup_t[nPrec][2];
