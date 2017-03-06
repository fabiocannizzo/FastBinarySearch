#include "Test-Direct-Common.h"

template <Precision P>
struct Info<P,Direct2> : DirectAux::DirectInfo<2,P,Direct2>
{
    typedef typename PrecTraits<P>::type T;
    Info(const std::vector<T>& x) : DirectAux::DirectInfo<2,P,Direct2>(x) {}
};


// ***************************************
// Direct2 Expression
//

template <Precision P>
struct ExprScalar<P,Direct2>
{
    typedef typename PrecTraits<P>::type T;

    FORCE_INLINE void init0( DataWorkspace<P>& p, const Info<P,Direct2>& info )
    {
        ri = p.rptr();
        zi = p.zptr();
        xi = info.xptr();
        buckets = reinterpret_cast<const uint32*>(&info.buckets.front());
        scaler = info.scaler;
    }

    FORCE_INLINE void scalar( uint32 j ) const
    {
        T z = zi[j];
        T tmp = (z - xi[0]) * scaler;
        uint32 bidx = ftoi(FVec1<SSE, T>(tmp));
        uint32 iidx = buckets[bidx];
        uint32 s1 = 0;
        uint32 s2 = 0;
        const T* p = xi+iidx;
        if (z < *p)
            --iidx;
        if (z < *(p-1))
            --iidx;
        ri[j] = iidx;
    }

protected:
    T scaler;
    uint32 * ri;
    const T* zi;
    const T* xi;
    const uint32* buckets;
};

template <Precision P, InstrSet I>
struct ExprVector<P, Direct2,I> : ExprScalar<P,Direct2>
{
    typedef ExprScalar<P,Direct2> base_t;
    typedef typename PrecTraits<P>::type T;
    typedef FVec<I,T> fVec;
    typedef IVec<SSE,T> i128;

    static const uint32 VecSize = sizeof(fVec)/sizeof(T);

    FORCE_INLINE void initN(DataWorkspace<P>& p, const Info<P,Direct2>& info)
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

        // read indices t
        const float *p3 = &base_t::xi[(u.ui32[3] = base_t::buckets[bidx.get3()])];
        const float *p2 = &base_t::xi[(u.ui32[2] = base_t::buckets[bidx.get2()])];
        const float *p1 = &base_t::xi[(u.ui32[1] = base_t::buckets[bidx.get1()])];
        const float *p0 = &base_t::xi[(u.ui32[0] = base_t::buckets[bidx.get0()])];

        // read pairs ( X(t-1), X(t) )
        __m128 xp3 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(p3-1)));
        __m128 xp2 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(p2-1)));
        __m128 xp1 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(p1-1)));
        __m128 xp0 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(p0-1)));

        // build:
        // { X(t(0)-1), X(t(1)-1), X(t(2)-1), X(t(3)-1) }
        // { X(t(0)),   X(t(1)),   X(t(2)),   X(t(3)) }
        __m128 h13 = _mm_shuffle_ps(xp1, xp3, (1<<2) + (1<<6));
        __m128 h02 = _mm_shuffle_ps(xp0, xp2, (1<<2) + (1<<6));
        __m128 u01 = _mm_unpacklo_ps(h02, h13);
        __m128 u23 = _mm_unpackhi_ps(h02, h13);
        __m128 vxm = _mm_shuffle_ps(u01, u23, (0) + (1<<2) + (0<<4) + (1<<6));
        __m128 vxp = _mm_shuffle_ps(u01, u23, (2) + (3<<2) + (2<<4) + (3<<6));

        IVec<SSE,float> i(u.vec);
        IVec<SSE,float> vlem = vz < vxm;
        IVec<SSE,float> vlep = vz < vxp;
        i = i + vlem + vlep;
        i.store( base_t::ri+j );
    }

    FORCE_INLINE
    //NO_INLINE
    void resolve( const FVec<SSE,double>& vz, const IVec<SSE,float>& bidx, uint32 j ) const
    {
        uint32 b1 = base_t::buckets[bidx.get1()];
        uint32 b0 = base_t::buckets[bidx.get0()];

        const double *p1 = &base_t::xi[b1];
        const double *p0 = &base_t::xi[b0];

        // read pairs ( X(t-1), X(t) )
        __m128d vx1 = _mm_loadu_pd(p1-1);
        __m128d vx0 = _mm_loadu_pd(p0-1);

        // build:
        // { X(t(0)-1), X(t(1)-1) }
        // { X(t(0)),   X(t(1)) }
        __m128d vxm = _mm_shuffle_pd(vx0, vx1, 0);
        __m128d vxp = _mm_shuffle_pd(vx0, vx1, 3);
        
        IVec<SSE,double> i( b1, b0 );
        IVec<SSE,double> vlem = (vz < vxm);
        IVec<SSE,double> vlep = (vz < vxp);
        i = i + vlem + vlep;

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
        union U{
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

        // read pairs ( X(t-1), X(t) )
        __m128 xp7 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(p7-1)));
        __m128 xp6 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(p6-1)));
        __m128 xp5 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(p5-1)));
        __m128 xp4 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(p4-1)));
        __m128 xp3 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(p3-1)));
        __m128 xp2 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(p2-1)));
        __m128 xp1 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(p1-1)));
        __m128 xp0 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(p0-1)));

        // build:
        // { X(t(0)-1), X(t(1)-1), X(t(2)-1), X(t(3)-1) }
        // { X(t(0)),   X(t(1)),   X(t(2)),   X(t(3)) }
        __m128 h57 = _mm_shuffle_ps(xp5, xp7, (1<<2) + (1<<6));  // F- F+ H- H+
        __m128 h46 = _mm_shuffle_ps(xp4, xp6, (1<<2) + (1<<6));  // E- E+ G- G+
        __m128 h13 = _mm_shuffle_ps(xp1, xp3, (1<<2) + (1<<6));  // B- B+ D- D+
        __m128 h02 = _mm_shuffle_ps(xp0, xp2, (1<<2) + (1<<6));  // A- A+ C- C+
        
        __m128 u01 = _mm_unpacklo_ps(h02, h13);  // A- B- A+ B+
        __m128 u23 = _mm_unpackhi_ps(h02, h13);  // C- D- C+ D+
        __m128 u45 = _mm_unpacklo_ps(h46, h57);  // E- F- E+ F+
        __m128 u67 = _mm_unpackhi_ps(h46, h57);  // G- H- G+ H+

        __m128 abcdm = _mm_shuffle_ps(u01, u23, (0) + (1<<2) + (0<<4) + (1<<6));  // A- B- C- D-
        __m128 abcdp = _mm_shuffle_ps(u01, u23, (2) + (3<<2) + (2<<4) + (3<<6));  // A+ B+ C+ D+
        __m128 efghm = _mm_shuffle_ps(u45, u67, (0) + (1<<2) + (0<<4) + (1<<6));  // E- F- G- H-
        __m128 efghp = _mm_shuffle_ps(u45, u67, (2) + (3<<2) + (2<<4) + (3<<6));  // E+ F+ G+ H+

        FVec<AVX,float> vxp = _mm256_insertf128_ps(_mm256_castps128_ps256(abcdm), efghm, 1);
        FVec<AVX,float> vxm = _mm256_insertf128_ps(_mm256_castps128_ps256(abcdp), efghp, 1);

        IVec<AVX,float> ip(u.vec);
        IVec<AVX,float> vlem = vz < vxm;
        IVec<AVX,float> vlep = vz < vxp;
        ip = ip + vlem + vlep;

        ip.store(base_t::ri + j);
    }



    FORCE_INLINE
    //NO_INLINE
    void resolve( const FVec<AVX,double>& vz, const IVec<SSE,float>& bidx, uint32 j ) const
    {
        union {
            __m256i vec;
            uint64 ui64[4];
        } u;

        // read indices t
        const double *p3 = &base_t::xi[(u.ui64[3] = base_t::buckets[bidx.get3()])];
        const double *p2 = &base_t::xi[(u.ui64[2] = base_t::buckets[bidx.get2()])];
        const double *p1 = &base_t::xi[(u.ui64[1] = base_t::buckets[bidx.get1()])];
        const double *p0 = &base_t::xi[(u.ui64[0] = base_t::buckets[bidx.get0()])];

        // read pairs ( X(t-1), X(t) )
        __m128d xp3 = _mm_loadu_pd(p3-1);
        __m128d xp2 = _mm_loadu_pd(p2-1);
        __m128d xp1 = _mm_loadu_pd(p1-1);
        __m128d xp0 = _mm_loadu_pd(p0-1);

        // build:
        // { X(t(0)-1), X(t(1)-1), X(t(2)-1), X(t(3)-1) }
        // { X(t(0)),   X(t(1)),   X(t(2)),   X(t(3)) }
        __m128d h01m = _mm_shuffle_pd(xp0, xp1, 0);
        __m128d h23m = _mm_shuffle_pd(xp2, xp3, 0);
        __m128d h01p = _mm_shuffle_pd(xp0, xp1, 3);
        __m128d h23p = _mm_shuffle_pd(xp2, xp3, 3);
        FVec<AVX,double> vxm = _mm256_insertf128_pd(_mm256_castpd128_pd256(h01m), h23m, 1);
        FVec<AVX,double> vxp = _mm256_insertf128_pd(_mm256_castpd128_pd256(h01p), h23p, 1);

        IVec<AVX,double> i(u.vec);
        IVec<AVX,double> vlem = vz < vxm;
        IVec<AVX,double> vlep = vz < vxp;
        i = i + vlem + vlep;
        i.extractLo32s().store( base_t::ri+j );
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
