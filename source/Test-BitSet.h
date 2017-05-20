#pragma once

#include "SIMD.h"

// Auxiliary information specifically used in the BitSet binary search

template <typename T>
struct InfoScalar<T, BitSet>
{
    typedef T type;

    InfoScalar(const T* x, const size_t n)
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

    ~InfoScalar()
    {
        delete [] xi;
    }

protected:
    T *xi;  // duplicate vector x, adding padding to the right
    uint32 maxBitIndex;
};

template <InstrSet I, typename T>
struct Info<I, T, BitSet> : InfoScalar<T, BitSet>
{
    typedef InfoScalar<T, BitSet> base_t;

    typedef FVec<I, T> fVec;
    typedef IVec<I, T> iVec;

public:
    static const uint32 VecSize = sizeof(fVec) / sizeof(T);

    Info(const T* x, const size_t n)
        : base_t(x, n)
    {
        xbv.setN(base_t::xi[base_t::maxBitIndex]);
        bv.setN(base_t::maxBitIndex);
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
            xv.setidx(base_t::xi, rv);
            lev = xv <= zv;
            //iv = iv | ( lev & lbv );
            iv.assignIf(rv, lev);
        };

        iv.store(pr);
    }


private:
    fVec xbv;
    iVec bv;
};

