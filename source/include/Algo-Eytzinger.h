#pragma once

#include <algorithm>

#include "AAlloc.h"
#include "SIMD.h"

namespace BinSearch {
namespace Details {

template <typename T>
struct AlgoScalarBase<T,Eytzinger>
{
private:
    struct layoutBuilder
    {
        layoutBuilder(const T *x, AlignedVec<T>& e, size_t size, size_t maxdepth)
            : m_x(x)
            , m_e(e)
            , m_lastIndex(size-1)
            , m_maxdepth(maxdepth)
        {
            std::fill_n(m_index, sizeof(m_index)/sizeof(m_index[0]), -1);
            myassert((maxdepth>0), "interval too small");
            build(0, m_e.size()-1, 0);
        }


        void build(size_t a, size_t b, size_t depth)
        {
            size_t c = (a+b)/2;

	        size_t pos = (1<<depth) + m_index[depth]++;
	        m_e[pos] = m_x[std::min(c,m_lastIndex)];

            if (++depth < m_maxdepth) {
		        build(a,   c, depth);
		        build(c+1, b, depth);
            }
        }

        int m_index[64]; // i-th element contains the position already filled at depth i
        const T *m_x;
        AlignedVec<T>& m_e;
        size_t m_lastIndex, m_maxdepth;
    };


public:
    AlgoScalarBase(const T* x, const uint32 n)
    {
        // count bits required to describe the index
        uint32 maxdepth = 0;
        while ((1u << maxdepth) - 1 < n)
            ++maxdepth;
        m_nLayers = maxdepth;
        m_mask = ~(1 << m_nLayers);

#if 0
        // In order to facilitate debugging of creation of Eytzinger layout
        // we set the elements of the vector x from 1 to n
        for (size_t i = 0; i < n; ++i)
            const_cast<T&>(x[i]) = static_cast<float>(i+1);
#endif

        size_t nl = (1 << maxdepth) - 1;
        m_x.resize(nl);
        std::fill_n(m_x.begin(), nl, x[n-1]);

        layoutBuilder(x, m_x, n, maxdepth);
        xi = &m_x[0];
    }

    FORCE_INLINE
    uint32 one_or_two(const float& z, uint32 p) const
    {
          unsigned char r;
        __asm__ (
            COMISS " %[x],%[z]\n\t"
            "setae %[r]\n\t"
            : [r] "=r"(r)
            : [x] "x"(xi[p]), [z] "x"(z)
            : "cc"
        );
        return r+1;
    }

    uint32 one_or_two(const double& z, uint32 p) const
    {
          unsigned char r;
        __asm__ (
            COMISD " %[x],%[z]\n\t"
            "setae %[r]\n\t"
            : [r] "=r"(r)
            : [x] "x"(xi[p]), [z] "x"(z)
            : "cc"
        );
        return r+1;
    }


    FORCE_INLINE
    uint32 scalar(T z) const
    {
        uint32 d = m_nLayers;

        // the first iteration, when p=0, is simpler
        uint32 p = (z < xi[0]) ? 1 : 2;

        while (--d > 0) {
#if (__GNUC__==7)  // gcc 7 translates the conditional assignment below with a branch
            uint32 o = one_or_two(z, p);
#else
            uint32 o = ((z < xi[p]) ? 1 : 2);
#endif
            p = (p << 1) + o;
        }
        return (p & m_mask);  // clear higher bit
    }

private:
    AlignedVec<T> m_x;  // duplicate vector x, adding padding to the right
protected:
    const T *xi;
    uint32 m_nLayers;
    uint32 m_mask;
};


template <InstrSet I, typename T>
struct AlgoVecBase<I, T, Eytzinger> : AlgoScalarBase<T, Eytzinger>
{
    static const uint32 nElem = sizeof(typename InstrFloatTraits<I, T>::vec_t) / sizeof(T);

    typedef AlgoScalarBase<T, Eytzinger> base_t;

    typedef FVec<I, T> fVec;
    typedef IVec<I, T> iVec;

    struct Constants
    {
        fVec xbv;
        iVec two;
        iVec maskv;
    };

    AlgoVecBase(const T* x, const uint32 n) : base_t(x, n) {}

    void initConstants(Constants& cst) const
    {
        cst.xbv.setN(base_t::xi[0]);
        cst.maskv.setN(base_t::m_mask);
        cst.two.setN(2);
    }
    
    FORCE_INLINE
    void vectorial(uint32 *pr, const T *pz, const Constants& cst) const
    {
        fVec zv(pz);

        uint32 d = base_t::m_nLayers;

        // the first iteration, when p=0, is simpler
        iVec pv = (zv < cst.xbv) + cst.two;

        const iVec two = cst.two;
        while (--d > 0) {
            fVec xv;
            xv.setidx(base_t::xi, pv);
            iVec right = (zv < xv) + two;
            pv = (pv << 1) + right;
        };

        pv = pv & cst.maskv;
        pv.store(pr);
    }
};

} // namespace Details
} // namespace BinSearch
