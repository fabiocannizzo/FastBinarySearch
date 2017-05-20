#pragma once

#include "SIMD.h"

// Auxiliary information specifically used in the modified version of the BitSetNoPad method (the one using min)

template <typename T>
struct InfoScalar<T, BitSetNoPad>
{
    typedef T type;

    InfoScalar(const T* x, const size_t n)
        : xi(x)
        , lastIndex(static_cast<uint32>(n - 1))
    {
        unsigned nbits = 0;
        while ((n - 1) >> nbits)
            ++nbits;

        maxBitIndex = 1 << (nbits - 1);
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

};


template <InstrSet I, typename T>
struct Info<I, T, BitSetNoPad> : InfoScalar<T, BitSetNoPad>
{
    typedef InfoScalar<T, BitSetNoPad> base_t;

    typedef FVec<I, T> fVec;
    typedef IVec<I, T> iVec;

public:
    static const uint32 VecSize = sizeof(fVec) / sizeof(T);

    Info(const T* x, const size_t n)
        : base_t(x, n)
    {
        xbv.setN(base_t::xi[base_t::maxBitIndex]);
        bv.setN(base_t::maxBitIndex);
        xlv.setN(base_t::lastIndex);
    }

    //NO_INLINE
    FORCE_INLINE
        void vectorial(uint32 *pr, const T *pz) const
    {
        uint32  b = base_t::maxBitIndex;
        fVec zv(pz);

        iVec lbv = bv;

        // the first iteration, when i=0, is simpler
        iVec lev = xbv <= zv;
        iVec iv = lev & lbv;
        while ((b >>= 1)) {
            lbv = lbv >> 1;
            fVec xv;
            iVec rv = iv | lbv;
            iVec hv = min(rv, xlv);
            xv.setidx(base_t::xi, hv);
            lev = xv <= zv;
            //iv = iv | ( lev & lbv );
            iv.assignIf(rv, lev);
        };

        iv.store(pr);
    }


private:
    fVec xbv;
    iVec bv;
    iVec xlv;
};

