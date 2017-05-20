#pragma once

#include "SIMD.h"

// Auxiliary information specifically used in the classic binary search modified as suggested by a reviewer
template <typename T>
struct InfoScalar<T,ClassicMod>
{
    typedef T type;

    InfoScalar(const T* x, const size_t n)
        : xi(x)
        , lastIndex(static_cast<uint32>(n - 1))
        , nIter(1 + static_cast<uint32>(log2(n)))
    {
    }

    FORCE_INLINE
        uint32 scalar(T z) const
    {
        uint32 lo = 0;
        uint32 hi = lastIndex;
        uint32 n = nIter;
        while (n--) {
            int mid = (hi + lo) >> 1;
            const bool lt = z < xi[mid];
            // defining this if-else assignment as below cause VS2015
            // to generate two cmov instructions instead of a branch
            if (lt)
                hi = mid;
            if (!lt)
                lo = mid;
        }
        return lo;
    }

protected:
    const T *xi;
    uint32 lastIndex;
    uint32 nIter;
};

// Auxiliary information specifically used in the classic binary search modified as suggested by a reviewer
template <InstrSet I, typename T>
struct Info<I, T, ClassicMod> : InfoScalar<T, ClassicMod>
{
    typedef InfoScalar<T, ClassicMod> base_t;

    typedef FVec<I,T> fVec;
    typedef IVec<I,T> iVec;

    Info(const T* x, const size_t n)
        : base_t(x,n)
    {
        bv.setN(base_t::lastIndex);
        fmaskv.setN(-1);
    }

    FORCE_INLINE
    void vectorial(uint32 *pr, const T *pz) const
    {
        fVec zv(pz);

        iVec lov = iVec::zero();
        iVec fmask = fmaskv;
        iVec hiv = bv;

        // the first iteration, when i=0, is simpler
        uint32 n = base_t::nIter;
        while (n--) {
            iVec midv = (lov + hiv) >> 1;  // this line produces incorrect results on gcc 5.4
            fVec xv;
            xv.setidx( base_t::xi, midv );
            iVec lev = xv <= zv;
            lov.assignIf(midv, lev);
            hiv.assignIf(midv, lev ^ fmask);
        };

        lov.store(pr);
    }

private:
    iVec bv;
    iVec fmaskv;
};

