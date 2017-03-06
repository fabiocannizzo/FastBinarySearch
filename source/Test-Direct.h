#include "Test-Direct-Common.h"

template <Precision P>
struct Info<P,Direct> : DirectAux::DirectInfo<1,P,Direct>
{
    typedef typename PrecTraits<P>::type T;
    Info(const std::vector<T>& x) : DirectAux::DirectInfo<1,P,Direct>(x) {}
};


// ***************************************
// Direct Expression
//

template <Precision P>
struct ExprScalar<P,Direct>
{
    typedef typename PrecTraits<P>::type T;

    FORCE_INLINE void init0( DataWorkspace<P>& p, const Info<P,Direct>& info )
    {
        ri = p.rptr();
        zi = p.zptr();
        xi = p.xptr();
        buckets = reinterpret_cast<const uint32*>(&info.buckets.front());
        scaler = info.scaler;
    }

    FORCE_INLINE void scalar( uint32 j ) const
    {
        T z = zi[j];
        T tmp = (z - xi[0]) * scaler;
        uint32 bidx = ftoi(FVec1<SSE,T>(tmp));
        uint32 iidx = buckets[bidx];
        uint32 iidxm = iidx - 1;
        ri[j] = (xi[iidx] <= z)? iidx : iidxm;
    }

protected:
    T scaler;
    uint32 * ri;
    const T* zi;
    const T* xi;
    const uint32* buckets;
};

template <Precision P, InstrSet I>
struct ExprVector<P, Direct,I> : ExprScalar<P,Direct>
{
    typedef ExprScalar<P,Direct> base_t;
    typedef typename PrecTraits<P>::type T;
    typedef FVec<I,T> fVec;
    typedef IVec<SSE,T> i128;

    static const uint32 VecSize = sizeof(fVec)/sizeof(T);

    FORCE_INLINE void initN(DataWorkspace<P>& p, const Info<P,Direct>& info)
    {
        base_t::init0( p, info );
        vscaler.setN( base_t::scaler );
        vx0.setN( base_t::xi[0] );
    }

    FORCE_INLINE
    //NO_INLINE
    void resolve( const FVec<SSE,float>& vz, const IVec<SSE,float>& bidx, uint32 j ) const
    {
        union U{
            __m128i vec;
            uint32 ui32[4];
        } u;
#if 1
        FVec<SSE,float> vxp
            ( base_t::xi[(u.ui32[3] = base_t::buckets[bidx.get3()])]
            , base_t::xi[(u.ui32[2] = base_t::buckets[bidx.get2()])]
            , base_t::xi[(u.ui32[1] = base_t::buckets[bidx.get1()])]
            , base_t::xi[(u.ui32[0] = base_t::buckets[bidx.get0()])]
            );
#else
        U b;
        b.vec = bidx;
        FVec<SSE,float> vxp
            ( base_t::xi[(u.ui32[3] = base_t::buckets[b.ui32[3]])]
            , base_t::xi[(u.ui32[2] = base_t::buckets[b.ui32[2]])]
            , base_t::xi[(u.ui32[1] = base_t::buckets[b.ui32[1]])]
            , base_t::xi[(u.ui32[0] = base_t::buckets[b.ui32[0]])]
            );
#endif
        IVec<SSE,float> i(u.vec);
        IVec<SSE,float> le = vz < vxp;
        i = i + le;
        i.store( base_t::ri+j );
    }

    FORCE_INLINE
    //NO_INLINE
    void resolve( const FVec<SSE,double>& vz, const IVec<SSE,float>& bidx, uint32 j ) const
    {
        uint32 b1 = base_t::buckets[bidx.get1()];
        uint32 b0 = base_t::buckets[bidx.get0()];
        
        FVec<SSE,double> vxp( base_t::xi[b1], base_t::xi[b0] );
        IVec<SSE,double> i( b1, b0 );
        IVec<SSE,double> le = (vz < vxp);
        i = i + le;

        union {
            __m128i vec;
            uint32 ui32[4];
        } u;
        u.vec = i;
        base_t::ri[j] = u.ui32[0];
        base_t::ri[j+1] = u.ui32[2];
    }

#ifdef USE_AVX

    FORCE_INLINE
    //NO_INLINE
    void resolve( const FVec<AVX,float>& vz, const IVec<AVX,float>& bidx, uint32 j ) const
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

        ip.store(base_t::ri + j);
#elif 0
        IVec<AVX,float> i;
        i.setidx(base_t::buckets, bidx);

        FVec<AVX,float> vxp;
        vxp.setidx(base_t::xi, i);

        IVec<AVX,float> le = vz < vxp;
        i = i + le;

        i.store(base_t::ri + j);
#else
        resolve(vz.lo128(), bidx.lo128(), j);
        resolve(vz.hi128(), bidx.hi128(), j+4);
#endif
    }



    FORCE_INLINE
    //NO_INLINE
    void resolve( const FVec<AVX,double>& vz, const IVec<SSE,float>& bidx, uint32 j ) const
    {
        FVec<AVX,double> vxp;
#if 0
        IVec<SSE,float> i(_mm_i32gather_epi32(reinterpret_cast<const int32 *>(base_t::buckets), bidx, 4));
        vxp.setidx(base_t::xi, i);
#else
        union {
            __m128i vec;
            uint32 ui32[4];
        } u;
        vxp = _mm256_set_pd
            ( base_t::xi[(u.ui32[3] = base_t::buckets[bidx.get3()])]
            , base_t::xi[(u.ui32[2] = base_t::buckets[bidx.get2()])]
            , base_t::xi[(u.ui32[1] = base_t::buckets[bidx.get1()])]
            , base_t::xi[(u.ui32[0] = base_t::buckets[bidx.get0()])]
            );
        IVec<SSE,float> i(u.vec);
#endif

        IVec<SSE,float> le = (vz < vxp).extractLo32s();
        i = i + le;

        i.store(base_t::ri + j);
    }
#endif

    FORCE_INLINE void vectorial( uint32 j ) const
    {
        fVec vz( base_t::zi+j );
        fVec tmp( (vz - vx0) * vscaler );
        resolve( vz, ftoi(tmp), j );
    }

private:
    fVec vscaler;
    fVec vx0;
};
