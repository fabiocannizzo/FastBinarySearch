#pragma once

#include "Test-Direct-Common.h"

template <typename T>
struct InfoScalar<T,DirectCache> : DirectAux::DirectInfo<1,T,DirectCache>
{
private:
    typedef DirectAux::DirectInfo<1, T, DirectCache> base_t;
    typedef DirectAux::BucketElem<DirectCache, T> bucket_t;
    typedef typename IntTraits<T>::itype IT;

public:
    typedef T type;

    InfoScalar(const T* x, const size_t n)
        : base_t(x, n)
        , x0(x[0])
        , buckets(&base_t::buckets[0])
    {
    }

    FORCE_INLINE uint32 scalar(T z) const
    {
        T tmp = (z - x0) * base_t::scaler;
        uint32 bidx = ftoi(FVec1<SSE, T>(tmp));
        bucket_t iidx = buckets[bidx];
        IT iidxm = iidx.index() - 1;
        return static_cast<uint32>((iidx.x() <= z) ? iidx.index() : iidxm);
    }

    const T x0;
    const DirectAux::BucketElem<DirectCache, T>* buckets;
};


template <InstrSet I, typename T>
struct Info<I, T, DirectCache> : InfoScalar<T, DirectCache>
{
private:
    typedef InfoScalar<T, DirectCache> base_t;
    typedef DirectAux::BucketElem<DirectCache, T> bucket_t;

    typedef FVec<I, T> fVec;
    typedef IVec<SSE, T> i128;

    static const uint32 VecSize = sizeof(fVec) / sizeof(T);

    FORCE_INLINE
    void resolve(const FVec<SSE, float>& vz, const IVec<SSE, float>& bidx, uint32 *pr) const
    {
        bucket_t b3 = base_t::buckets[bidx.get3()];
        bucket_t b2 = base_t::buckets[bidx.get2()];
        bucket_t b1 = base_t::buckets[bidx.get1()];
        bucket_t b0 = base_t::buckets[bidx.get0()];

        IVec<SSE, float> i(b3.index(), b2.index(), b1.index(), b0.index());
        FVec<SSE, float> vxp(b3.x(), b2.x(), b1.x(), b0.x());
        IVec<SSE, float> le = vz < vxp;
        i = i + le;
        i.store(pr);
    }

    FORCE_INLINE
    void resolve(const FVec<SSE, double>& vz, const IVec<SSE, float>& bidx, uint32 *pr) const
    {
        __m128d b1 = _mm_load_pd(reinterpret_cast<const double *>(base_t::buckets + bidx.get1()));
        __m128d b0 = _mm_load_pd(reinterpret_cast<const double *>(base_t::buckets + bidx.get0()));
        //bucket_t b1 = base_t::buckets[bidx.get1()];
        //bucket_t b0 = base_t::buckets[bidx.get0()];

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
        bucket_t b7 = base_t::buckets[bidx.get7()];
        bucket_t b6 = base_t::buckets[bidx.get6()];
        bucket_t b5 = base_t::buckets[bidx.get5()];
        bucket_t b4 = base_t::buckets[bidx.get4()];
        bucket_t b3 = base_t::buckets[bidx.get3()];
        bucket_t b2 = base_t::buckets[bidx.get2()];
        bucket_t b1 = base_t::buckets[bidx.get1()];
        bucket_t b0 = base_t::buckets[bidx.get0()];

        IVec<AVX, float> i(b7.index(), b6.index(), b5.index(), b4.index(), b3.index(), b2.index(), b1.index(), b0.index());
        FVec<AVX, float> vxp(b7.x(), b6.x(), b5.x(), b4.x(), b3.x(), b2.x(), b1.x(), b0.x());
        IVec<AVX, float> le = vz < vxp;
        i = i + le;
        i.store(pr);
    }

    FORCE_INLINE
    void resolve(const FVec<AVX, double>& vz, const IVec<SSE, float>& bidx, uint32 *pr) const
    {
        bucket_t b3 = base_t::buckets[bidx.get3()];
        bucket_t b2 = base_t::buckets[bidx.get2()];
        bucket_t b1 = base_t::buckets[bidx.get1()];
        bucket_t b0 = base_t::buckets[bidx.get0()];

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
        //        i.store(base_t::ri + j);
    }

#endif

public:

    Info(const T* x, const size_t n)
        : base_t(x, n)
    {
        vscaler.setN(base_t::scaler);
        vx0.setN(base_t::x0);
    }

    FORCE_INLINE
    void vectorial(uint32 *pr, const T *pz) const
    {
        fVec vz(pz);
        fVec tmp((vz - vx0) * vscaler);
        resolve(vz, ftoi(tmp), pr);
    }

private:
    fVec vscaler;
    fVec vx0;
};
