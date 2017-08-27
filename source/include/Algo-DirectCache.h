#pragma once

#include "Algo-Direct-Common.h"

namespace BinSearch {
namespace Details {

template <typename T, Algos A>
struct AlgoScalarBase<T, A, typename std::enable_if<DirectAux::IsDirectCache<A>::value>::type > : DirectAux::DirectInfo<1,T,A>
{
private:
    typedef DirectAux::DirectInfo<1, T, A> base_t;
    typedef typename IntTraits<T>::itype IT;

public:
    AlgoScalarBase(const typename base_t::Data& d) :  base_t(d) {}
    AlgoScalarBase(const T* px, const uint32 n) :  base_t(px, n) {}

    FORCE_INLINE uint32 scalar(T z) const
    {
        uint32 bidx = base_t::fun_t::f(base_t::data.scaler, base_t::data.cst0, z);
        typename base_t::bucket_t iidx = base_t::data.buckets[bidx];
        IT iidxm = iidx.index() - 1;
        return static_cast<uint32>((iidx.x() <= z) ? iidx.index() : iidxm);
    }
};


template <InstrSet I, typename T, Algos A>
struct AlgoVecBase<I, T, A, typename std::enable_if<DirectAux::IsDirectCache<A>::value>::type> : AlgoScalarBase<T, A>
{
    static const uint32 nElem = sizeof(typename InstrFloatTraits<I, T>::vec_t) / sizeof(T);

private:
    typedef AlgoScalarBase<T, A> base_t;
public:
    typedef typename base_t::bucket_t bucket_t;
private:
    typedef FVec<I, T> fVec;
    typedef IVec<SSE, T> i128;

    FORCE_INLINE
    void resolve(const FVec<SSE, float>& vz, const IVec<SSE, float>& bidx, uint32 *pr) const
    {
        const bucket_t *b = base_t::data.buckets;
        bucket_t b3 = b[bidx.get3()];
        bucket_t b2 = b[bidx.get2()];
        bucket_t b1 = b[bidx.get1()];
        bucket_t b0 = b[bidx.get0()];

        IVec<SSE, float> i(b3.index(), b2.index(), b1.index(), b0.index());
        FVec<SSE, float> vxp(b3.x(), b2.x(), b1.x(), b0.x());
        IVec<SSE, float> le = vz < vxp;
        i = i + le;
        i.store(pr);
    }

    FORCE_INLINE
    void resolve(const FVec<SSE, double>& vz, const IVec<SSE, float>& bidx, uint32 *pr) const
    {
        const bucket_t *b = base_t::data.buckets;
        __m128d b1 = _mm_load_pd(reinterpret_cast<const double *>(b + bidx.get1()));
        __m128d b0 = _mm_load_pd(reinterpret_cast<const double *>(b + bidx.get0()));
        //typename base_t::bucket_t b1 = base_t::buckets[bidx.get1()];
        //typename base_t::bucket_t b0 = base_t::buckets[bidx.get0()];

        FVec<SSE, double> vxp(_mm_shuffle_pd(b0, b1, 0));
        IVec<SSE, double> i(_mm_castpd_si128(_mm_shuffle_pd(b0, b1, 3)));
        IVec<SSE, double> le = (vz < vxp);
        i = i + le;

        union {
            __m128i vec;
            uint32 ui32[4];
        } u;

        u.vec = i;
        pr[0] = u.ui32[0];
        pr[1] = u.ui32[2];

        //_mm_storel_epi64( reinterpret_cast<__m128i*>(base_t::ri+j), i );
    }


#ifdef USE_AVX

    FORCE_INLINE
    void resolve(const FVec<AVX, float>& vz, const IVec<AVX, float>& bidx, uint32 *pr) const
    {
        const bucket_t *b = base_t::data.buckets;
        bucket_t b7 = b[bidx.get7()];
        bucket_t b6 = b[bidx.get6()];
        bucket_t b5 = b[bidx.get5()];
        bucket_t b4 = b[bidx.get4()];
        bucket_t b3 = b[bidx.get3()];
        bucket_t b2 = b[bidx.get2()];
        bucket_t b1 = b[bidx.get1()];
        bucket_t b0 = b[bidx.get0()];

        IVec<AVX, float> i(b7.index(), b6.index(), b5.index(), b4.index(), b3.index(), b2.index(), b1.index(), b0.index());
        FVec<AVX, float> vxp(b7.x(), b6.x(), b5.x(), b4.x(), b3.x(), b2.x(), b1.x(), b0.x());
        IVec<AVX, float> le = vz < vxp;
        i = i + le;
        i.store(pr);
    }

    FORCE_INLINE
    void resolve(const FVec<AVX, double>& vz, const IVec<SSE, float>& bidx, uint32 *pr) const
    {
#if 0
        bucket_t b3 = b[bidx.get3()];
        bucket_t b2 = b[bidx.get2()];
        bucket_t b1 = b[bidx.get1()];
        bucket_t b0 = b[bidx.get0()];

        IVec<AVX, double> i(b3.index(), b2.index(), b1.index(), b0.index());
        FVec<AVX, double> vxp(b3.x(), b2.x(), b1.x(), b0.x());

        IVec<AVX, double> le(vz < vxp);
        i = i + le;

        union {
            __m256i vec;
            uint32 ui32[8];
        } u;
        u.vec = i;
        pr[0] = u.ui32[0];
        pr[1] = u.ui32[2];
        pr[2] = u.ui32[4];
        pr[3] = u.ui32[6];
#else
        resolve(_mm256_castpd256_pd128(vz), bidx, pr);
        __m128d bidpd = _mm_castsi128_pd(bidx);
        FVec<SSE,double> zhi = _mm256_extractf128_pd(vz, 1);
        IVec<SSE,float> bhi = _mm_castpd_si128(_mm_shuffle_pd(bidpd, bidpd, 1));
        resolve(zhi, bhi, pr+2);
#endif
        //        i.store(base_t::ri + j);
    }

#endif

public:

    struct Constants
    {
        fVec vscaler;
        fVec vcst0;
    };

    AlgoVecBase(const typename base_t::Data& d) :  base_t(d) {}
    AlgoVecBase(const T* px, const uint32 n) :  base_t(px, n) {}

    void initConstants(Constants& cst) const
    {
        cst.vscaler.setN(base_t::data.scaler);
        cst.vcst0.setN(base_t::data.cst0);
    }

    FORCE_INLINE
    void vectorial(uint32 *pr, const T *pz, const Constants& cst) const
    {
        fVec vz(pz);
        resolve(vz, base_t::fun_t::f(cst.vscaler, cst.vcst0, vz), pr);
    }
};
} // namespace Details
} // namespace BinSearch
