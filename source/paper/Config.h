#pragma once

#include "TypesForArticleTest.h"

// Generation of Z
enum ZMode {ConstAlwaysLeft, ConstAlternate, Periodic, Uniform};
const ZMode zGenerationMode = Uniform;
const size_t zRatio = 1;
const size_t periodZ = 2048;            // period of array Z, must be a multiple of 4 (only Periodic mode)
const bool onlyMatchingPoints = false;  // vector Z contains only points belonging to X
const bool useAnywherePoints = false;   // generate points in any poition of [X0,XN) (mainly for sanity check)
const bool sortedZ = false;             // vector Z contains only one number (only Uniform mode)
const size_t NZ = 2048/zRatio;          // size of array Z: must be a multiple of 4


// In total we resolve 2*(N-1)*nAvg*nRepeat indices
const size_t nAvg = 1;                 // Number of random regenerations of vector X
const size_t nAvgSetup = 100;            // Number of random regenerations of vector X
const size_t nAvgSetupR = 1000000;       // Number of random regenerations of vector X
const size_t nRepeat  = 10000*zRatio;    // Number repetition of the test
const size_t nSetupTargetOps  = 20 * 1048576; // Number repetition of the test
#define INTMIN 1.0
#define INTMAX 5.0


const uint32 VecScope[] =
    { 15
    //, 255
    //, 4095
    //, 65535
    //, 1048575
    };

const Precision PrecScope[] =
    {
        Single,
        Double,
    };


const Algos AlgoScope[] =
    {
        DirectCacheFMA,
        DirectFMA,
        Direct2FMA,
        DirectCache,
        Direct,
        Direct2,
        Nonary,
        Pentary,
        Ternary,
        Eytzinger,
        BitSet,
        ClassicOffset,
        MorinOffset,
        BitSetNoPad,
        ClassicMod,
        MorinBranchy,
        Classic,
        LowerBound,
#ifdef USE_MKL
        MKL,
#endif
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
