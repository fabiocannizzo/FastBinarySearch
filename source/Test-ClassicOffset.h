#pragma once

#include "SIMD.h"

// Auxiliary information specifically used in the ClassicOffset binary search
template <typename T>
struct InfoScalar<T, ClassicOffset>
{
    typedef T type;

    InfoScalar(const T* x, const size_t n)
        : xi(x)
        , mid0(static_cast<uint32>(n/2))
        , size1(static_cast<uint32>(n - mid0))
        , nIter(static_cast<uint32>(log2(n)))
    {
    }

    FORCE_INLINE
    uint32 scalar(T z) const
    {
        // variation on original paper: the number of iterations is fixed

        // there is at least one iteration
        uint32 mid = mid0;  // initialised to: size/2
        uint32 i = (z >= xi[mid0]) ? mid : 0;
        uint32 sz = size1;  // initialised to: size - mid
        uint32 n = nIter;   // this is decreaded by 1
        while (n--) {
            uint32 h = sz / 2;
            uint32 mid = i + h;
            if (z >= xi[mid])
                i = mid;
            sz -= h;
        }

        return i;
    }

protected:
    const T *xi;
    uint32 mid0;
    uint32 size1;
    uint32 nIter;
};



template <InstrSet I, typename T>
struct Info<I, T, ClassicOffset> : public InfoScalar<T, ClassicOffset>
{
    typedef InfoScalar<T, ClassicOffset> base_t;

    typedef FVec<I,T> fVec;
    typedef IVec<I,T> iVec;

public:
    static const uint32 VecSize = sizeof(fVec)/sizeof(T);

    Info(const T* x, const size_t n)
        : base_t(x, n)
    {
        xMidV.setN(base_t::xi[base_t::mid0]);
        size1V.setN(base_t::size1);
        mid0V.setN(base_t::mid0);
    }

    //NO_INLINE
    FORCE_INLINE
    void vectorial(uint32 *pr, const T *pz) const
    {
        fVec zV(pz);

        // there is at least one iteration
        iVec midV = mid0V;  // initialised to: size/2
        iVec iV = (zV >= xMidV) & midV;
        iVec szV = size1V;  // initialised to: size - mid
        uint32 n = base_t::nIter;   // this is decreaded by 1
        while(n--) {
            iVec hV = szV >> 1;
            midV = iV + hV;
            fVec xV;
            xV.setidx( base_t::xi, midV );

            iVec geV = zV >= xV;
            iV.assignIf(midV, geV);

            szV = szV - hV;
        }

        iV.store(pr);
    }

private:
    fVec xMidV;
    iVec size1V;
    iVec mid0V;
};

