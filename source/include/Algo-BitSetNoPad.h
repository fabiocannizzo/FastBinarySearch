#pragma once

#ifdef USE_SSE41

#include "SIMD.h"

namespace BinSearch {
namespace Details {

// Auxiliary information specifically used in the modified version of the BitSetNoPad method (the one using min)

template <typename T>
struct AlgoScalarBase<T, BitSetNoPad>
{
    AlgoScalarBase(const T* x, const uint32 n)
        : xi(x)
        , lastIndex(n - 1)
    {
        unsigned nbits = 0;
        while ((n - 1) >> nbits)
            ++nbits;

        maxBitIndex = 1 << (nbits - 1);

        uint32 b = maxBitIndex;
        uint32 i = b;
        nUncheckedIter = 0;
        while ((b >>= 1)) {
            uint32 r = i | b;
            if(r > lastIndex)
                break;
            ++nUncheckedIter;
            i = r;
        };
    }

    //NO_INLINE
    FORCE_INLINE
    uint32 scalar(T z) const
    {
        uint32  i = 0;
        uint32  b = maxBitIndex;
        uint32  u = nUncheckedIter;

        // the first iteration, when i=0, is simpler
        if (xi[b] <= z)
            i = b;

        // unchecked iterations
        while (u--) {
            b >>= 1;
            uint32 r = i | b;
            if (xi[r] <= z)
                i = r;
        };

        // checked iterations
        while ((b >>= 1)) {
            uint32 r = i | b;
            uint32 h = r <= lastIndex ? r : lastIndex;
            if (xi[h] <= z)
                i = r;
        };

        return i;
    }

protected:
    const T *xi;  // duplicate vector x, adding padding to the right
    uint32 maxBitIndex;
    uint32 lastIndex;
    uint32 nUncheckedIter;
};


template <InstrSet I, typename T>
struct AlgoVecBase<I, T, BitSetNoPad> : AlgoScalarBase<T, BitSetNoPad>
{
    static const uint32 nElem = sizeof(typename InstrFloatTraits<I, T>::vec_t) / sizeof(T);

    typedef AlgoScalarBase<T, BitSetNoPad> base_t;

    typedef FVec<I, T> fVec;
    typedef IVec<I, T> iVec;

    struct Constants
    {
        fVec xbv;
        iVec bv;
        iVec xlv;
    };

    AlgoVecBase(const T* x, const uint32 n) : base_t(x, n) {}

    void initConstants(Constants& cst) const
    {
        cst.xbv.setN(base_t::xi[base_t::maxBitIndex]);
        cst.bv.setN(base_t::maxBitIndex);
        cst.xlv.setN(base_t::lastIndex);
    }

    //NO_INLINE
    FORCE_INLINE
        void vectorial(uint32 *pr, const T *pz, const Constants& cst) const
    {
        uint32 b = base_t::maxBitIndex;
        uint32 u = base_t::nUncheckedIter;

        fVec zv(pz);

        iVec lbv = cst.bv;

        // the first iteration, when i=0, is simpler
        iVec lev = cst.xbv <= zv;
        iVec iv = lev & lbv;
        iVec xlv = cst.xlv;

        // unckecked iterations
        while (u--) {
            b >>= 1;
            lbv = lbv >> 1;
            fVec xv;
            iVec rv = iv | lbv;
            xv.setidx(base_t::xi, rv);
            lev = xv <= zv;
            iv.orIf(rv, lev);
        };

        // checked iterations
        while ((b >>= 1)) {
            lbv = lbv >> 1;
            fVec xv;
            iVec rv = iv | lbv;
            iVec hv = min(rv, xlv);
            xv.setidx(base_t::xi, hv);
            lev = xv <= zv;
            iv.orIf(rv, lev);
        };

        iv.store(pr);
    }
};

} // namespace Details
} // namespace BinSearch

#endif
