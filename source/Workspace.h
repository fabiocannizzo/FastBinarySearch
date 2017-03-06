#pragma once
#include <cstdio>
#include <iostream>

#include "Type.h"
#include "AAlloc.h"
#include "SIMD.h"

using namespace std;

// generates a random number in single precision between in [range_min, range_max)
template <typename T>
inline T dRand(T range_min, T range_max)
{
    return (static_cast<T>(std::rand()) / (static_cast<size_t>(RAND_MAX) + 1)) * (range_max - range_min) + range_min;
}

template <typename T>
inline bool ok(size_t i, const T *xi, const T& z)
{
    return z >= xi[i] && z < xi[i + 1];
}

template <typename T>
inline void error(size_t i, size_t j, const T *xi, const T* zj)
{
    cout << "Error: Z[" << j << "]=" << zj[j]
         << ", x[" << i << "]=" << xi[i]
         << ", x[" << i + 1 << "]=" << xi[i + 1]
         << "\n";
}


// Base class for auxiliary data structure used in the tests
template <Precision P>
struct DataWorkspace
{
    typedef typename PrecTraits<P>::type T;

    // Allocate memory for vector X of size n and initilize it with increments of random size
    // drawn in the range (intMin, intMax)
    // Allocate memory for vector Z and initialize it with the union of the extrema of the intervals in vector X, all mid points
    // and an equal number of extra points extracted randomly in the interval [X_0,X_N)
    // Allocate memory for vector R (same size as Z) containing the indices of the segments [X_i,X_{i+1}) containing the numbers Z
    DataWorkspace( T intMin, T intMax, size_t NX )
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
                    assert(false);
                    factor *= T(1.01);
                }
            };
            xold = m_x[i];
        }

        T x0 = m_x.front();
        T xN = m_x.back();

        static_assert(NZ % 4 == 0, "NZ must be a multiple of 4");
        // init z must belong to [ x[0], x[nx-1] )
        for (uint32 j = 0; j < NZ; ) {
            size_t i = rand() % (NX-1);
            m_z[j++] = m_x[i];
            m_z[j++] = (m_x[i + 1] + m_x[i]) / T(2);
            m_z[j++] = dRand(x0, xN);
            m_z[j++] = dRand(x0, xN);
        }
        std::random_shuffle(m_z.begin(), m_z.end());
    }

    // check if the indices r, representing the interval in x are containing the numbers in z
    // then reset the indices to invalid values
    void checkAndReset(Algos A, InstrSet I)
    {
        bool first = true;
        size_t nz = m_z.size();
        uint32 nx = static_cast<uint32>(m_x.size());
        size_t nerr = 0;
        for (uint32 j = 0; j < nz; ++j) {
            uint32 i = m_r[j];
            T z = m_z[j];
            if ( !(i < m_x.size()) || !ok(i, &m_x[0], z) ) {
                if (first) {
                    cout << "Errors in precision " << P << " algo " << A << " instr set " << I << endl;
                    first = false;
                }
                if (++nerr < 5)
                    error(i, j, &m_x[0], &m_z[0]);
            }
            m_r[j] = nx;  // nx is an invalid index
        }
    }

    const T *xptr() const { return &m_x.front(); }
    const T *zptr() const { return &m_z.front(); }
    uint32 *rptr() { return &m_r.front(); }

    std::vector<T>  m_x;
    AlignedVec<T> m_z;
    AlignedVec<uint32> m_r;
};
