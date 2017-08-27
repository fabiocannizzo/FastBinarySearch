#pragma once

#include "SIMD.h"

namespace BinSearch {
namespace Details {

// Auxiliary information specifically used in the classic binary search modified as suggested by a reviewer
template <typename T>
struct AlgoScalarBase<T,ClassicMod>
{
    AlgoScalarBase(const T* x, const uint32 n)
        : xi(x)
        , lastIndex(n - 1)
        , nIter(1 + log2(n))
    {
    }

    FORCE_INLINE
    void shrink_interval(const float& z, uint32 mid, uint32& lo, uint32& hi) const
    {
        __asm__ (
            COMISS " %[x], %[z]\n\t"
            "cmovb %[mid],%[hi]\n\t"
            "cmovae %[mid],%[lo]"
            : [hi] "+r"(hi), [lo] "+r"(lo)
            : [mid] "rm"(mid), [x] "x"(xi[mid]), [z] "x"(z)
            : "cc"
        );
    }

    FORCE_INLINE
    void shrink_interval(const double& z, uint32 mid, uint32& lo, uint32& hi) const
    {
        __asm__ (
            COMISD " %[x], %[z]\n\t"
            "cmovb %[mid],%[hi]\n\t"
            "cmovae %[mid],%[lo]"
            : [hi] "+r"(hi), [lo] "+r"(lo)
            : [mid] "rm"(mid), [x] "x"(xi[mid]), [z] "x"(z)
            : "cc"
        );
    }


    FORCE_INLINE
        uint32 scalar(T z) const
    {
        uint32 lo = 0;
        uint32 hi = lastIndex;
        uint32 n = nIter;
        while (n--) {
            int mid = (hi + lo) >> 1;
#if (__GNUC__==6)
            shrink_interval(z, mid, lo, hi);
#else
            hi = (z < xi[mid]) ? mid : hi;
            lo = (z < xi[mid]) ? lo  : mid;
#endif
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
struct AlgoVecBase<I, T, ClassicMod> : AlgoScalarBase<T, ClassicMod>
{
    static const uint32 nElem = sizeof(typename InstrFloatTraits<I, T>::vec_t) / sizeof(T);

    typedef AlgoScalarBase<T, ClassicMod> base_t;

    typedef FVec<I,T> fVec;
    typedef IVec<I,T> iVec;


    struct Constants
    {
        iVec bv;
        iVec fmaskv;
    };

    AlgoVecBase(const T* x, const uint32 n) : base_t(x, n) {}

    void initConstants(Constants& cst) const
    {
        cst.bv.setN(base_t::lastIndex);
        cst.fmaskv.setN(-1);
    }

    FORCE_INLINE
    void vectorial(uint32 *pr, const T *pz, const Constants& cst) const
    {
        fVec zv(pz);

        iVec lov = iVec::zero();
        iVec fmask = cst.fmaskv;
        iVec hiv = cst.bv;

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

} // namespace Details
} // namespace BinSearch
