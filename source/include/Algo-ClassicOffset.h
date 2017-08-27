#pragma once

#include "SIMD.h"

namespace BinSearch {
namespace Details {

// Auxiliary information specifically used in the ClassicOffset binary search
template <typename T>
struct AlgoScalarBase<T, ClassicOffset>
{
    struct Data
    {
        Data() {}
        Data(const T* x, const uint32 n)
            : xi(x)
            , mid0(n/2)
            , size1(n - mid0)
            , nIter(log2(n))
        {
        }
        const T *xi;
        uint32 mid0;
        uint32 size1;
        uint32 nIter;
    };

    AlgoScalarBase(const T* x, const uint32 n)
        : m_data(x,n)
    {
    }

    AlgoScalarBase(const Data& d)
        : m_data(d)
    {
    }

    FORCE_INLINE
    uint32 scalar(T z) const
    {
        const T *xi = m_data.xi;
        // there is at least one iteration
        uint32 mid0 = m_data.mid0;  // initialised to: size/2
        uint32 i = (z >= xi[mid0]) ? mid0 : 0;
        uint32 sz = m_data.size1;  // initialised to: size - mid0
        uint32 n = m_data.nIter;   // this is decreased by 1
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
    Data m_data;
};



template <InstrSet I, typename T>
struct AlgoVecBase<I, T, ClassicOffset> : AlgoScalarBase<T, ClassicOffset>
{
    static const uint32 nElem = sizeof(typename InstrFloatTraits<I, T>::vec_t) / sizeof(T);

    typedef AlgoScalarBase<T, ClassicOffset> base_t;

    typedef FVec<I,T> fVec;
    typedef IVec<I,T> iVec;


    struct Constants
    {
        fVec xMidV;
        iVec size1V;
        iVec mid0V;
    };

    AlgoVecBase(const T* x, const uint32 n) : base_t(x, n) {}

    AlgoVecBase(const typename base_t::Data& d) : base_t(d) {}

    void initConstants(Constants& cst) const
    {
        cst.xMidV.setN(base_t::m_data.xi[base_t::m_data.mid0]);
        cst.size1V.setN(base_t::m_data.size1);
        cst.mid0V.setN(base_t::m_data.mid0);
    }

    //NO_INLINE
    FORCE_INLINE
    void vectorial(uint32 *pr, const T *pz, const Constants& cst) const
    {
        fVec zV(pz);

        const T *xi = base_t::m_data.xi;

        // there is at least one iteration
        iVec midV = cst.mid0V;  // initialised to: size/2
        iVec iV = (zV >= cst.xMidV) & midV;
        iVec szV = cst.size1V;  // initialised to: size - mid
        uint32 n = base_t::m_data.nIter;   // this is decreaded by 1
        while(n--) {
            iVec hV = szV >> 1;
            midV = iV + hV;
            fVec xV;
            xV.setidx( xi, midV );

            iVec geV = zV >= xV;
            iV.assignIf(midV, geV);

            szV = szV - hV;
        }

        iV.store(pr);
    }
};

} // namespace Details
} // namespace BinSearch
