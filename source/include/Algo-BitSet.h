#pragma once

#include "SIMD.h"

namespace BinSearch {
namespace Details {

// Auxiliary information specifically used in the BitSet binary search

template <typename T>
struct AlgoScalarBase<T, BitSet>
{
    AlgoScalarBase(const T* x, const uint32 n)
        : xi(NULL)
    {
        // count bits required to describe the index
        unsigned nbits = 0;
        while ((n - 1) >> nbits)
            ++nbits;

        maxBitIndex = 1 << (nbits - 1);

        // create copy of x extended to the right side to size
        // (1<<nbits) and padded with x[N-1]
        const unsigned nx = 1 << nbits;
        xi = new T[nx];
        std::copy(x, x + n, xi);
        std::fill(xi+n, xi + nx, x[n - 1]);
    }

    //NO_INLINE
    FORCE_INLINE
    uint32 scalar(T z) const
    {
        uint32  i = 0;
        uint32  b = maxBitIndex;

        // the first iteration, when i=0, is simpler
        if (xi[b] <= z)
            i = b;

        while ((b >>= 1)) {
            uint32 r = i | b;
            if (xi[r] <= z)
                i = r;
        };

        return i;
    }

    ~AlgoScalarBase()
    {
        delete [] xi;
    }

protected:
    T *xi;  // duplicate vector x, adding padding to the right
    uint32 maxBitIndex;
};

template <InstrSet I, typename T>
struct AlgoVecBase<I, T, BitSet> : AlgoScalarBase<T, BitSet>
{
    static const uint32 nElem = sizeof(typename InstrFloatTraits<I, T>::vec_t) / sizeof(T);

    typedef AlgoScalarBase<T, BitSet> base_t;

    typedef FVec<I, T> fVec;
    typedef IVec<I, T> iVec;

    struct Constants
    {
        fVec xbv;
        iVec bv;
    };

    AlgoVecBase(const T* x, const uint32 n) : base_t(x, n) {}

    void initConstants(Constants& cst) const
    {
        cst.xbv.setN(base_t::xi[base_t::maxBitIndex]);
        cst.bv.setN(base_t::maxBitIndex);
    }

    //NO_INLINE
    FORCE_INLINE
    void vectorial(uint32 *pr, const T *pz, const Constants& cst) const
    {
        uint32  b = base_t::maxBitIndex;
        fVec zv(pz);

        iVec lbv = cst.bv;

        // the first iteration, when i=0, is simpler
        iVec lev = cst.xbv <= zv;
        iVec iv = lev & lbv;
        while ((b >>= 1)) {
            lbv = lbv >> 1;
            fVec xv;
            iVec rv = iv | lbv;
            xv.setidx(base_t::xi, rv);
            lev = xv <= zv;
            iv.orIf(rv, lev);
        };

        iv.store(pr);
    }
};

} // namespace Details
} // namespace BinSearch
