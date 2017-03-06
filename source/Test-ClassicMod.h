
// Auxiliary information specifically used in the classic binary search modified as suggested by a reviewer
template <Precision P>
struct Info<P, ClassicMod>
{
    typedef typename PrecTraits<P>::type T;

    Info(const std::vector<T>& x)
        : lastValidIndex(static_cast<uint32>( x.size()-1 ))
        , nIter(1+static_cast<uint32>(log2(x.size())))
    {
    }

    uint32 lastValidIndex;
    uint32 nIter;
};


// ***************************************
// ClassicMod Expression
//

template <Precision P>
struct ExprScalar<P,ClassicMod>
{
    typedef typename PrecTraits<P>::type T;

    FORCE_INLINE
    void init0( DataWorkspace<P>& p, const Info<P,ClassicMod>& info )
    {
        ri = p.rptr();
        zi = p.zptr();
        xi = p.xptr();
        xLast = info.lastValidIndex;
        nIter = info.nIter;
    }

    FORCE_INLINE
    void scalar( uint32 j ) const
    {
        T z = zi[j];
        uint32 lo = 0;
        uint32 hi = xLast;
        uint32 n = nIter;
        while (n--) {
            int mid = (hi + lo) >> 1;
            const bool lt = z < xi[mid];
            // defining this if-else assignment as below cause VS2015
            // to generate two cmov instructions instead of a branch
            if (lt)
                hi = mid;
            if(!lt)
                lo = mid;
        }
        ri[j] = lo;
    }

protected:
    uint32 * ri;
    const T* zi;
    const T* xi;
    uint32 xLast, nIter;
};

template <Precision P, InstrSet I>
struct ExprVector<P,ClassicMod,I> : ExprScalar<P,ClassicMod>
{
    typedef ExprScalar<P,ClassicMod> base_t;
    typedef typename PrecTraits<P>::type T;

    typedef FVec<I,T> fVec;
    typedef IVec<I,T> iVec;

public:
    static const uint32 VecSize = sizeof(fVec)/sizeof(T);

    FORCE_INLINE
    void initN( DataWorkspace<P>& p, const Info<P,ClassicMod>& info )
    {
        base_t::init0( p, info );
        bv.setN(base_t::xLast);
        fmaskv.setN(-1);
    }

    FORCE_INLINE
    void vectorial( uint32 j ) const
    {
        fVec zv( base_t::zi+j );

        iVec lov = iVec::zero();
        iVec hiv = bv;

        // the first iteration, when i=0, is simpler
        uint32 n = base_t::nIter;
        while (n--) {
            iVec midv = (lov + hiv) >> 1;  // this line produces incorrect results on gcc 5.4
            fVec xv;
            xv.setidx( base_t::xi, midv );
            iVec lev = xv <= zv;
            lov.assignIf(midv, lev);
            hiv.assignIf(midv, lev ^ fmaskv);
        };

        lov.store( base_t::ri+j );
    }

private:
    iVec bv;
    iVec fmaskv;
};
