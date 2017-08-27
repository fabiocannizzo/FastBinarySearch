#pragma once

#include "Type.h"

namespace BinSearch {
namespace Details {


// Differences with original algorithm
    // 1) use reverse pointers, i.e. *(a-m) instead of *(a+m)  (cost neutral)
    // 2) swap > and < operators (cost neutral)
    // 3) at the end subtract from n-1
template <typename T>
struct AlgoScalarBase<T, MorinBranchy>
{
    AlgoScalarBase(const T* x, const uint32 _n)
        : xi(new T[_n])
        , n(_n)
    {
        // invert array X
        for (uint32 i = 0, e = _n - 1; i < _n; ++i)
            xi[i] = x[e - i];
    }

    FORCE_INLINE
    uint32 scalar(T z) const
    {
        typedef uint32 I;
        
        I lo = 0;
        I hi = n;
        I nm = n - 1;
        const T *a = xi;
        while (lo < hi) {
            I m = (lo + hi) / 2;
            if (z > *(a+m))
                hi = m;
            else if (z < *(a+m))
                lo = m + 1;
            else
                return nm-m;
        }
        return nm-hi;
    }

protected:
    T *xi;
    uint32 n;
};



} // namespace Details
} // namespace BinSearch
