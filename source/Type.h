#pragma once

#include "Portable.h"
#include <vector>

enum Precision { Single, Double };
enum InstrSet { Scalar, SSE, AVX };
enum Algos { Classic, ClassicMod, ClassicOffset, BitSet, BitSetNoPad, Eytzinger, MKL, Direct, Direct2, DirectCache };

extern const char *PrecNames[];
extern const char *InstrNames[];
extern const char *AlgoNames[];
extern const char *texName[];

template <Precision P> struct PrecTraits;

template <> struct PrecTraits<Single>
{
    typedef float type;
    typedef uint32 itype;
};
template <> struct PrecTraits<Double>
{
    typedef double type;
    typedef uint64 itype;
};

template <InstrSet I>
struct InstrIntTraits;

template <InstrSet I, typename T>
struct InstrFloatTraits;

/*
    Precomputed info
*/

template <Precision P, Algos A>
struct Info;

/*
    Expressions
*/

template <Precision P, Algos A>
struct ExprScalar;

template <Precision P, Algos A, InstrSet I>
struct ExprVector;

/*
Stats
*/

struct SetupStats
{
    std::vector<double> vec;
    double avg, mi, ma, stdev;
    void computeStats()
    {
        double s = 0, s2 = 0;
        size_t n = vec.size();
        mi = ma = vec[0];
        for (size_t i = 0; i < n; ++i) {
            s += vec[i];
            s2 += vec[i] * vec[i];
            if (vec[i]<mi)
                mi = vec[i];
            if (vec[i]>ma)
                ma = vec[i];
        }
        avg = s / n;
        stdev = sqrt(s2 / n - avg*avg);
    };
};