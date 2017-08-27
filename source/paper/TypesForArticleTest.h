#pragma once

#define PAPER_TEST

#include "Type.h"
#include <vector>
#include <map>
#include <string>

using namespace BinSearch;
using std::string;

enum Precision { Single, Double };

template <Precision P> struct PrecTraits;

extern std::map<InstrSet, string> InstrNames;
extern std::map<Algos, string> AlgoNames;
extern std::map<Precision, string> PrecNames;


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
