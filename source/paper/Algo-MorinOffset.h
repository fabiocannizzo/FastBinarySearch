#pragma once

#include "Type.h"

namespace BinSearch {
namespace Details {

template <typename T>
struct AlgoScalarBase<T, MorinOffset>
{
    AlgoScalarBase(const T* x, const uint32 _n)
        : xi(new T [_n])
        , xRef(xi + _n - 2)
        , n(_n)
    {
        // invert array X
        for (uint32 i = 0, e = _n - 1; i < _n; ++i)
            xi[i] = x[e - i];
    }

    ~AlgoScalarBase()
    {
        delete[] xi;
    }

    FORCE_INLINE
    uint32 scalar(T z) const
    {
        typedef uint32 I;

        const T *base = xi;
        I n = this->n;
        while (n > 1) {
            const I half = n / 2;
            base = (base[half] > z) ? &base[half] : base;
            n -= half;
        }
        // subtract one instead of subtracting (*base > z), which is always true if z<X[N-1]
        return static_cast<uint32>(xRef - base);
    }

protected:
    T *xi;
    const T *xRef;
    uint32 n;
};

} // namespace Details
} // namespace BinSearch
