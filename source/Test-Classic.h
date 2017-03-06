
// Auxiliary information specifically used in the classic binary search
template <Precision P>
struct Info<P,Classic>
{
    typedef typename PrecTraits<P>::type T;
    Info(const std::vector<T>& x)
        : lastValidIndex(static_cast<uint32>( x.size()-1 ))
    {
    }

    uint32 lastValidIndex;
};


// ***************************************
// Classic Epxression
//

template <Precision P>
struct ExprScalar<P,Classic>
{
    typedef typename PrecTraits<P>::type T;

    FORCE_INLINE
    void init0( DataWorkspace<P>& p, const Info<P,Classic>& info )
    {
        ri = p.rptr();
        zi = p.zptr();
        xi = p.xptr();
        xLast = info.lastValidIndex;
    }

    FORCE_INLINE
    void scalar( uint32 j ) const
    {
        T z = zi[j];
        uint32 lo = 0;
        uint32 hi = xLast;
        while (hi - lo > 1) {
            int mid = (hi + lo) >> 1;
            if (z < xi[mid])
                hi = mid;
            else
                lo = mid;
        }
        ri[j] = lo;
    }

    private:
    uint32 * ri;
    const T* zi;
    const T* xi;
    uint32 xLast;
};

template <Precision P, InstrSet I>
struct ExprVector<P, Classic,I> : ExprScalar<P, Classic>
{
    typedef typename PrecTraits<P>::type T;
    typedef ExprScalar<P,Classic> base_t;

    static const uint32 VecSize = sizeof(typename InstrFloatTraits<I,T>::vec_t)/sizeof(T);

    FORCE_INLINE
    void initN( DataWorkspace<P>& p, const Info<P, Classic>& info )
    {
        base_t::init0( p, info );
    }

    FORCE_INLINE
    void vectorial( uint32 j ) const
    {
        for ( unsigned i = 0; i < VecSize; ++i )
            base_t::scalar( j+i );
    }
};
