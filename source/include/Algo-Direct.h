#pragma once

#include "Algo-Direct-Common.h"

namespace BinSearch {
namespace Details {

template <typename T, Algos A>
struct AlgoScalarBase<T, A, typename std::enable_if<DirectAux::IsDirect<A>::value>::type> : DirectAux::DirectInfo<1, T, A>
{
private:
    typedef DirectAux::DirectInfo<1, T, A> base_t;

public:
    AlgoScalarBase(const T* x, const uint32 n)
        : base_t(x,n)
        , buckets(reinterpret_cast<const uint32 *>(base_t::data.buckets))
    {
    }

    FORCE_INLINE uint32 scalar(T z) const
    {
        uint32 bidx = base_t::fun_t::f(base_t::data.scaler, base_t::data.cst0, z);
        uint32 iidx = buckets[bidx];
        uint32 iidxm = iidx - 1;
        return (base_t::data.xi[iidx] <= z) ? iidx : iidxm;
    }
protected:
    const uint32 *buckets;
};


template <InstrSet I, typename T, Algos A>
struct AlgoVecBase<I, T, A, typename std::enable_if<DirectAux::IsDirect<A>::value>::type> : AlgoScalarBase<T, A>
{
    static const uint32 nElem = sizeof(typename InstrFloatTraits<I, T>::vec_t) / sizeof(T);

private:
    typedef AlgoScalarBase<T, A> base_t;
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
        (base_t::data.xi[(u.ui32[3] = base_t::buckets[bidx.get3()])]
            , base_t::data.xi[(u.ui32[2] = base_t::buckets[bidx.get2()])]
            , base_t::data.xi[(u.ui32[1] = base_t::buckets[bidx.get1()])]
            , base_t::data.xi[(u.ui32[0] = base_t::buckets[bidx.get0()])]
        );
#else
        U b;
        b.vec = bidx;
        FVec<SSE, float> vxp
        (base_t::data.xi[(u.ui32[3] = base_t::buckets[b.ui32[3]])]
            , base_t::data.xi[(u.ui32[2] = base_t::buckets[b.ui32[2]])]
            , base_t::data.xi[(u.ui32[1] = base_t::buckets[b.ui32[1]])]
            , base_t::data.xi[(u.ui32[0] = base_t::buckets[b.ui32[0]])]
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

        FVec<SSE, double> vxp(base_t::data.xi[b1], base_t::data.xi[b0]);
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

        const float *p7 = &base_t::data.xi[(u.ui32[7] = base_t::buckets[bidx.get7()])];
        const float *p6 = &base_t::data.xi[(u.ui32[6] = base_t::buckets[bidx.get6()])];
        const float *p5 = &base_t::data.xi[(u.ui32[5] = base_t::buckets[bidx.get5()])];
        const float *p4 = &base_t::data.xi[(u.ui32[4] = base_t::buckets[bidx.get4()])];
        const float *p3 = &base_t::data.xi[(u.ui32[3] = base_t::buckets[bidx.get3()])];
        const float *p2 = &base_t::data.xi[(u.ui32[2] = base_t::buckets[bidx.get2()])];
        const float *p1 = &base_t::data.xi[(u.ui32[1] = base_t::buckets[bidx.get1()])];
        const float *p0 = &base_t::data.xi[(u.ui32[0] = base_t::buckets[bidx.get0()])];

        FVec<AVX, float> vxp = _mm256_set_ps(*p7, *p6, *p5, *p4, *p3, *p2, *p1, *p0);

        IVec<AVX, float> ip(u.vec);
        IVec<AVX, float> vlep = vz < vxp;
        ip = ip + vlep;

        ip.store(pr);
#elif 0
        IVec<AVX, float> i;
        i.setidx(base_t::buckets, bidx);

        FVec<AVX, float> vxp;
        vxp.setidx(base_t::data.xi, i);

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
        vxp.setidx(base_t::data.xi, i);
#else
        union {
            __m128i vec;
            uint32 ui32[4];
        } u;
        vxp = _mm256_set_pd
        (base_t::data.xi[(u.ui32[3] = base_t::buckets[bidx.get3()])]
            , base_t::data.xi[(u.ui32[2] = base_t::buckets[bidx.get2()])]
            , base_t::data.xi[(u.ui32[1] = base_t::buckets[bidx.get1()])]
            , base_t::data.xi[(u.ui32[0] = base_t::buckets[bidx.get0()])]
        );
        IVec<SSE, float> i(u.vec);
#endif

        IVec<SSE, float> le = (vz < vxp).extractLo32s();
        i = i + le;

        i.store(pr);
    }
#endif


public:
    struct Constants
    {
        fVec vscaler;
        fVec vcst0;
    };

    AlgoVecBase(const T* x, const uint32 n) : base_t(x, n) {}

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

template <Algos A>
struct AlgoVecBase<SSE, double, A, typename std::enable_if<DirectAux::IsDirect<A>::value>::type> : AlgoScalarBase<double, A>
{
    static const InstrSet I = SSE;
    typedef double T;

    static const uint32 nPipe = 8;
    static const uint32 nElem1 = sizeof(typename InstrFloatTraits<I, T>::vec_t) / sizeof(T);
    static const uint32 nElem = nPipe*nElem1;

    typedef AlgoScalarBase<T, A> base_t;
    typedef FVec<I, T> fVec;
    typedef IVec<SSE, T> i128;

    struct Constants
    {
        fVec vscaler;
        fVec vcst0;
    };

    AlgoVecBase(const T* x, const uint32 n) : base_t(x, n) {}

    void initConstants(Constants& cst) const
    {
        cst.vscaler.setN(base_t::data.scaler);
        cst.vcst0.setN(base_t::data.cst0);
    }

    struct Data
    {
        fVec vz;
        uint32 b[nElem1];
    };

    struct Stage1
    {
        Stage1(const double *_pz, const uint32 *_buckets, Constants _k) : pz(_pz), buckets(_buckets), k(_k) {}
        const double *pz;
        const uint32 *buckets;
        Constants k;

        template <uint32 Iter>
        FORCE_INLINE void run(Data *d) const
        {
            d[Iter].vz = fVec(pz + nElem1*Iter);
            typename FTOITraits<I, T>::vec_t bidx = base_t::fun_t::f(k.vscaler, k.vcst0, d[Iter].vz);
            d[Iter].b[1] = buckets[bidx.get1()];
            d[Iter].b[0] = buckets[bidx.get0()];
        }
    };

    struct Stage2
    {
        Stage2(uint32 *_pr, const T *_xi) : pr(_pr), xi(_xi) {}

        uint32 *pr;
        const T *xi;

        template <uint32 Iter>
        FORCE_INLINE void run(Data *d) const
        {
            FVec<SSE, double> vxp(xi[d[Iter].b[1]], xi[d[Iter].b[0]]);
            IVec<SSE, double> i(d[Iter].b[1], d[Iter].b[0]);
            IVec<SSE, double> le = (d[Iter].vz < vxp);
            i = i + le;

            union {
                __m128i vec;
                uint32 ui32[4];
            } u;
            u.vec = i;
            pr[nElem1*Iter + 0] = u.ui32[0];
            pr[nElem1*Iter + 1] = u.ui32[2];
        }
    };

    FORCE_INLINE
        void vectorial(uint32 *pr, const T *pz, const Constants& cst) const
    {
        Data d[nPipe];
        Pipeliner<nPipe>::go(Stage1(pz, base_t::buckets, cst), d);
        Pipeliner<nPipe>::go(Stage2(pr, base_t::data.xi), d);
    }
};

} // namespace Details
} // namespace BinSearch
