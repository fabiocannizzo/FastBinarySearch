#include "Test-Direct-Common.h"

template <Precision P>
struct Info<P,DirectCache> : DirectAux::DirectInfo<1,P,DirectCache>
{
    typedef typename PrecTraits<P>::type T;
    Info(const std::vector<T>& x) : DirectAux::DirectInfo<1,P,DirectCache>(x) {}
};

// ***************************************
// DirectCache Expression
//

template <Precision P>
struct ExprScalar<P,DirectCache>
{
    typedef typename PrecTraits<P>::type T;
    typedef typename PrecTraits<P>::itype IT;
    typedef DirectAux::BucketElem<DirectCache,T> bucket_t;

    FORCE_INLINE void init0( DataWorkspace<P>& p, const Info<P,DirectCache>& info )
    {
        ri = p.rptr();
        zi = p.zptr();
        x0 = p.xptr()[0];
        buckets = &info.buckets.front();
        scaler = info.scaler;
    }

    FORCE_INLINE void scalar( uint32 j ) const
    {
        T z = zi[j];
        T tmp = (z - x0) * scaler;
        uint32 bidx = static_cast<uint32>(tmp);
        bucket_t iidx = buckets[bidx];
        IT iidxm = iidx.index() - 1;
        ri[j] = static_cast<uint32>( (iidx.x() <= z)? iidx.index() : iidxm );
    }

protected:
    T scaler;
    T x0;
    uint32 * ri;
    const T* zi;
    const DirectAux::BucketElem<DirectCache, T>* buckets;
};


template <Precision P, InstrSet I>
struct ExprVector<P, DirectCache, I> : ExprScalar<P,DirectCache>
{
    typedef ExprScalar<P,DirectCache> base_t;
    typedef typename PrecTraits<P>::type T;
    typedef DirectAux::BucketElem<DirectCache,T> bucket_t;
    typedef FVec<I,T> fVec;
    typedef IVec<SSE,T> i128;

    static const uint32 VecSize = sizeof(fVec)/sizeof(T);

    FORCE_INLINE void initN(DataWorkspace<P>& p, const Info<P,DirectCache>& info)
    {
        base_t::init0( p, info );
        vscaler.setN( base_t::scaler );
        vx0.setN( base_t::x0 );
    }

    FORCE_INLINE
    void resolve( const FVec<SSE,float>& vz, const IVec<SSE,float>& bidx, uint32 j ) const
    {
        bucket_t b3 = base_t::buckets[bidx.get3()];
        bucket_t b2 = base_t::buckets[bidx.get2()];
        bucket_t b1 = base_t::buckets[bidx.get1()];
        bucket_t b0 = base_t::buckets[bidx.get0()];

        IVec<SSE,float> i( b3.index(), b2.index(), b1.index(), b0.index() );
        FVec<SSE,float> vxp( b3.x(), b2.x(), b1.x(), b0.x() );
        IVec<SSE,float> le = vz < vxp;
        i = i + le;
        i.store( base_t::ri+j );
    }

    FORCE_INLINE
    void resolve( const FVec<SSE,double>& vz, const IVec<SSE,float>& bidx, uint32 j ) const
    {
        __m128d b1 = _mm_load_pd(reinterpret_cast<const double *>(base_t::buckets+bidx.get1()));
        __m128d b0 = _mm_load_pd(reinterpret_cast<const double *>(base_t::buckets+bidx.get0()));
        //bucket_t b1 = base_t::buckets[bidx.get1()];
        //bucket_t b0 = base_t::buckets[bidx.get0()];

        FVec<SSE,double> vxp( _mm_shuffle_pd(b0, b1, 0) );
        IVec<SSE,double> i(_mm_castpd_si128(_mm_shuffle_pd(b0, b1, 3)));
        IVec<SSE,double> le = (vz < vxp);
        i = i + le;

        union {
            __m128i vec;
            uint32 ui32[4];
        } u;

        u.vec = i;
        base_t::ri[j] = u.ui32[0];
        base_t::ri[j+1] = u.ui32[2];

        //_mm_storel_epi64( reinterpret_cast<__m128i*>(base_t::ri+j), i );
    }


#ifdef USE_AVX

    FORCE_INLINE
    void resolve( const FVec<AVX,float>& vz, const IVec<AVX,float>& bidx, uint32 j ) const
    {
        bucket_t b7 = base_t::buckets[bidx.get7()];
        bucket_t b6 = base_t::buckets[bidx.get6()];
        bucket_t b5 = base_t::buckets[bidx.get5()];
        bucket_t b4 = base_t::buckets[bidx.get4()];
        bucket_t b3 = base_t::buckets[bidx.get3()];
        bucket_t b2 = base_t::buckets[bidx.get2()];
        bucket_t b1 = base_t::buckets[bidx.get1()];
        bucket_t b0 = base_t::buckets[bidx.get0()];

        IVec<AVX,float> i( b7.index(), b6.index(), b5.index(), b4.index(), b3.index(), b2.index(), b1.index(), b0.index() );
        FVec<AVX,float> vxp( b7.x(), b6.x(), b5.x(), b4.x(), b3.x(), b2.x(), b1.x(), b0.x() );
        IVec<AVX,float> le = vz < vxp;
        i = i + le;
        i.store( base_t::ri+j );
    }

    FORCE_INLINE
    void resolve( const FVec<AVX,double>& vz, const IVec<SSE,float>& bidx, uint32 j ) const
    {
        bucket_t b3 = base_t::buckets[bidx.get3()];
        bucket_t b2 = base_t::buckets[bidx.get2()];
        bucket_t b1 = base_t::buckets[bidx.get1()];
        bucket_t b0 = base_t::buckets[bidx.get0()];

        IVec<AVX,double> i(b3.index(), b2.index(), b1.index(), b0.index());
        FVec<AVX,double> vxp(b3.x(), b2.x(), b1.x(), b0.x());

        IVec<AVX,double> le(vz < vxp);
        i = i + le;

        union {
            __m256i vec;
            uint32 ui32[8];
        } u;
        u.vec = i;
        base_t::ri[j] = u.ui32[0];
        base_t::ri[j+1] = u.ui32[2];
        base_t::ri[j+2] = u.ui32[4];
        base_t::ri[j+3] = u.ui32[6];
//        i.store(base_t::ri + j);
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

