#pragma once
#include <cstdio>
#include <iostream>
#include <typeinfo>

#include "Type.h"
#include "AAlloc.h"
#include "SIMD.h"
#include "Portable.h"
#include "Config.h"

// generates a random number in single precision between in [range_min, range_max)
template <typename T>
inline T dRand(T range_min, T range_max)
{
    T r = (static_cast<T>(std::rand()) / (static_cast<size_t>(RAND_MAX) + 1)) * (range_max - range_min) + range_min;
    if (r < range_min)
        r = range_min;
    if (r >= range_max)
        r = Details::myprev(range_max);
    return r;
}

template <typename T>
inline bool ok(size_t i, const T *xi, const T& z)
{
    return (z == xi[i]) || (z > xi[i] && z < xi[i + 1]);
}

template <typename T>
inline void error(size_t i, size_t j, const T *xi, const T* zj)
{
    std::cerr << "Error: Z[" << j << "]=" << zj[j]
              << ", x[" << i << "]=" << xi[i]
              << ", x[" << i + 1 << "]=" << xi[i + 1]
              << "\n";
}


// Base class for auxiliary data structure used in the tests
template <typename T>
struct DataWorkspace
{
    uint32 alternateIndex(bool left, uint32 NX)
    {
        uint32 lo = 0;
        uint32 hi = NX-1;
        while (hi - lo > 1) {
            int mid = (hi + lo) >> 1;
            if (left)
                hi = mid;
            else
                lo = mid;
            left = !left;
        }
        return lo;
    }

    // Allocate memory for vector X of size n and initilize it with increments of random size
    // drawn in the range (intMin, intMax)
    // Allocate memory for vector Z and initialize it with the union of the extrema of the intervals in vector X, all mid points
    // and an equal number of extra points extracted randomly in the interval [X_0,X_N)
    // Allocate memory for vector R (same size as Z) containing the indices of the segments [X_i,X_{i+1}) containing the numbers Z
    DataWorkspace( T intMin, T intMax, uint32 NX )
    {
        m_z.resize(NZ);
        m_r.resize(NZ);
        m_x.resize(NX);

        // init x with random gaps drawn in intMin, intMax
        T xold = 0.0;
        T factor = 1.0;
        for (uint32 i = 0; i < NX; ++i) {
            while(true) {
                T incr = dRand(intMin, intMax);
                m_x[i] = xold + factor*incr;
                if(m_x[i] > xold)
                    break;
                else {
                    myassert(false, "non distingushable D_i");
                    factor *= T(1.01);
                }
            };
            xold = m_x[i];
        }

        T x0 = m_x.front();
        T xN = m_x.back();

        size_t nz = NZ;

        myassert(((NZ % 4 == 0) && (NZ>0)), "NZ must be a multiple of 4");
        // init z must belong to [ x[0], x[nx-1] )
        switch(zGenerationMode) {
            case ConstAlwaysLeft:
                for (uint32 j = 0; j < NZ;)
                    m_z[j++] = m_x[0];
                break;
            case ConstAlternate:
                {
                    uint32 i1 = alternateIndex(true,NX);
                    T x = (m_x[i1]+m_x[i1])/T(2);
                    for (uint32 j = 0; j < NZ;)
                        m_z[j++] = x;
                }
                break;
            case Periodic:
                nz = (periodZ < NZ) ? periodZ : NZ;
                myassert(((nz % 4 == 0) && (nz>0)), "periodZ must be a multiple of 4");
#if __cplusplus>=201703L
                [[fallthrough]];
#endif
            case Uniform:
                for (uint32 j = 0; j < nz; ) {
                    size_t i1 = rand() % (NX-1);
                    m_z[j++] = m_x[i1];
                    if (!onlyMatchingPoints) {
                        size_t i2 = rand() % (NX-1);
                        m_z[j++] = (m_x[i2 + 1] + m_x[i2]) / T(2);
                        if (useAnywherePoints) {
                            m_z[j++] = dRand(x0, xN);
                            m_z[j++] = dRand(x0, xN);
                        }
                    }
                }
                for (size_t j = nz; j < NZ; ++j)
                    m_z[j] = m_z[j-nz];
                if(sortedZ)
                    std::sort(m_z.begin(), m_z.end());
                break;
            default:
                std::cerr << "Unknown generation mode\n";
                exit(1);
        };
    }

    void reset()
    {
        size_t nz = m_z.size();
        uint32 nx = static_cast<uint32>(m_x.size());
        for (uint32 j = 0; j < nz; ++j)
            m_r[j] = nx;  // nx is an invalid index
    }

    // check if the indices r, representing the interval in x are containing the numbers in z
    // then reset the indices to invalid values
    void checkAndReset(Algos A, InstrSet I)
    {
        bool first = true;
        size_t nz = m_z.size();
        size_t nerr = 0;
        for (uint32 j = 0; j < nz; ++j) {
            uint32 i = m_r[j];
            T z = m_z[j];
            if ( !(i < m_x.size()) || !ok(i, &m_x[0], z) ) {
                if (first) {
                    std::cerr << "Errors in precision " << typeid(T).name() << " algo " << AlgoNames[A] << " instr set " << InstrNames[I] << std::endl;
                    first = false;
                }
                if (++nerr < 5)
                    error(i, j, &m_x[0], &m_z[0]);
            }
        }
        reset();
    }

    const T *xptr() const { return &m_x.front(); }
    const T *zptr() const { return &m_z.front(); }
    uint32 *rptr() { return &m_r.front(); }

    std::vector<T>  m_x;
    Details::AlignedVec<T> m_z;
    Details::AlignedVec<uint32> m_r;
};
