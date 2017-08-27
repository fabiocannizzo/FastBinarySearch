#pragma once

#include <algorithm>
#include <functional>
#include <iterator>

#include "Type.h"

namespace BinSearch {
namespace Details {

template <typename T>
struct AlgoScalarBase<T, LowerBound>
{
    AlgoScalarBase(const T* x, const uint32 _n)
        : xi(x)
        , rbegin(x+_n-1)
        , rend(x-1)
    {
    }

    FORCE_INLINE
    uint32 scalar(T z) const
    {
        typedef std::reverse_iterator<const T*> iter;
        iter px = std::lower_bound(iter(rbegin), iter(rend), z, std::greater<T>());
        return static_cast<uint32>(&*px - xi);
    }

protected:
    const T *xi;
    const T *rbegin;
    const T *rend;
};



} // namespace Details
} // namespace BinSearch
