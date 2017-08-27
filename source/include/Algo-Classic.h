#pragma once

#include "Type.h"

namespace BinSearch {
namespace Details {

// Auxiliary information specifically used in the classic binary search
template <typename T>
struct AlgoScalarBase<T, Classic>
{
    AlgoScalarBase(const T* x, const uint32 n)
        : xi(x)
        , lastIndex(n - 1)
    {
    }

    FORCE_INLINE
    uint32 scalar(T z) const
    {
        uint32 lo = 0;
        uint32 hi = lastIndex;
        while (hi - lo > 1) {
            int mid = (hi + lo) >> 1;
            if (z < xi[mid])
                hi = mid;
            else
                lo = mid;
        }
        return lo;
    }

private:
    const T *xi;
    uint32 lastIndex;
};


} // namespace Details
} // namespace BinSearch
