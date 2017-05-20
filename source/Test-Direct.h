#pragma once

#include "Test-Direct-Common.h"

template <typename T>
struct InfoScalar<T,Direct> : DirectAux::DirectInfo<1,T,Direct>
{
private:
    typedef DirectAux::DirectInfo<1, T, Direct> base_t;

public:
    typedef T type;

    InfoScalar(const T* x, const size_t n)
        : base_t(x,n)
        , xi(x)
        , buckets(reinterpret_cast<const uint32 *>(&base_t::buckets[0]))
    {
    }

    FORCE_INLINE uint32 scalar(T z) const
    {
        T tmp = (z - xi[0]) * base_t::scaler;
        uint32 bidx = ftoi(FVec1<SSE, T>(tmp));
        uint32 iidx = buckets[bidx];
        uint32 iidxm = iidx - 1;
        return (xi[iidx] <= z) ? iidx : iidxm;
    }
protected:
    const T* xi;
    const uint32 *buckets;
};


template <InstrSet I, typename T>
struct Info<I, T, Direct> : InfoScalar<T, Direct>
{
private:
    typedef InfoScalar<T, Direct> base_t;

    typedef FVec<I, T> fVec;
    typedef IVec<SSE, T> i128;

    FORCE_INLINE
        //NO_INLINE
        void resolve(const FVec<SSE, float>& vz, const IVec<SSE, float>& bidx, uint32 *pr) const
    {
        union U {
            __m128i vec;
            uint32 ui32[4];
        } u;
#if 1
        FVec<SSE, float> vxp
        (base_t::xi[(u.ui32[3] = base_t::buckets[bidx.get3()])]
            , base_t::xi[(u.ui32[2] = base_t::buckets[bidx.get2()])]
            , base_t::xi[(u.ui32[1] = base_t::buckets[bidx.get1()])]
            , base_t::xi[(u.ui32[0] = base_t::buckets[bidx.get0()])]
        );
#else
        U b;
        b.vec = bidx;
        FVec<SSE, float> vxp
        (base_t::xi[(u.ui32[3] = base_t::buckets[b.ui32[3]])]
            , base_t::xi[(u.ui32[2] = base_t::buckets[b.ui32[2]])]
            , base_t::xi[(u.ui32[1] = base_t::buckets[b.ui32[1]])]
            , base_t::xi[(u.ui32[0] = base_t::buckets[b.ui32[0]])]
        );
#endif
        IVec<SSE, float> i(u.vec);
        IVec<SSE, float> le = vz < vxp;
        i = i + le;
        i.store(pr);
    }

    FORCE_INLINE
        //NO_INLINE
        void resolve(const FVec<SSE, double>& vz, const IVec<SSE, float>& bidx, uint32 *pr) const
    {
        uint32 b1 = base_t::buckets[bidx.get1()];
        uint32 b0 = base_t::buckets[bidx.get0()];

        FVec<SSE, double> vxp(base_t::xi[b1], base_t::xi[b0]);
        IVec<SSE, double> i(b1, b0);
        IVec<SSE, double> le = (vz < vxp);
        i = i + le;

        union {
            __m128i vec;
            uint32 ui32[4];
        } u;
        u.vec = i;
        pr[0] = u.ui32[0];
        pr[1] = u.ui32[2];
    }

#ifdef USE_AVX

    FORCE_INLINE
        //NO_INLINE
        void resolve(const FVec<AVX, float>& vz, const IVec<AVX, float>& bidx, uint32 *pr) const
    {
#if 1
        union U {
            __m256i vec;
            uint32 ui32[8];
        } u;

        // read indices t

        const float *p7 = &base_t::xi[(u.ui32[7] = base_t::buckets[bidx.get7()])];
        const float *p6 = &base_t::xi[(u.ui32[6] = base_t::buckets[bidx.get6()])];
        const float *p5 = &base_t::xi[(u.ui32[5] = base_t::buckets[bidx.get5()])];
        const float *p4 = &base_t::xi[(u.ui32[4] = base_t::buckets[bidx.get4()])];
        const float *p3 = &base_t::xi[(u.ui32[3] = base_t::buckets[bidx.get3()])];
        const float *p2 = &base_t::xi[(u.ui32[2] = base_t::buckets[bidx.get2()])];
        const float *p1 = &base_t::xi[(u.ui32[1] = base_t::buckets[bidx.get1()])];
        const float *p0 = &base_t::xi[(u.ui32[0] = base_t::buckets[bidx.get0()])];

        FVec<AVX, float> vxp = _mm256_set_ps(*p7, *p6, *p5, *p4, *p3, *p2, *p1, *p0);

        IVec<AVX, float> ip(u.vec);
        IVec<AVX, float> vlep = vz < vxp;
        ip = ip + vlep;

        ip.store(pr);
#elif 0
        IVec<AVX, float> i;
        i.setidx(base_t::buckets, bidx);

        FVec<AVX, float> vxp;
        vxp.setidx(base_t::xi, i);

        IVec<AVX, float> le = vz < vxp;
        i = i + le;

        i.store(pr);
#else
        resolve(vz.lo128(), bidx.lo128(), pr);
        resolve(vz.hi128(), bidx.hi128(), pr + 4);
#endif
    }



    FORCE_INLINE
        //NO_INLINE
        void resolve(const FVec<AVX, double>& vz, const IVec<SSE, float>& bidx, uint32 *pr) const
    {
        FVec<AVX, double> vxp;
#if 0
        IVec<SSE, float> i(_mm_i32gather_epi32(reinterpret_cast<const int32 *>(base_t::buckets), bidx, 4));
        vxp.setidx(base_t::xi, i);
#else
        union {
            __m128i vec;
            uint32 ui32[4];
        } u;
        vxp = _mm256_set_pd
        (base_t::xi[(u.ui32[3] = base_t::buckets[bidx.get3()])]
            , base_t::xi[(u.ui32[2] = base_t::buckets[bidx.get2()])]
            , base_t::xi[(u.ui32[1] = base_t::buckets[bidx.get1()])]
            , base_t::xi[(u.ui32[0] = base_t::buckets[bidx.get0()])]
        );
        IVec<SSE, float> i(u.vec);
#endif

        IVec<SSE, float> le = (vz < vxp).extractLo32s();
        i = i + le;

        i.store(pr);
    }
#endif


public:
    Info(const T* x, const size_t n)
        : base_t(x, n)
    {
        vscaler.setN(base_t::scaler);
        vx0.setN(base_t::xi[0]);
    }

    static const uint32 VecSize = sizeof(fVec) / sizeof(T);

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
