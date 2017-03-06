

// Auxiliary information specifically used in the optimized binary search
template <Precision P>
struct Info<P,BitSet>
{
    typedef typename PrecTraits<P>::type T;
    Info(const std::vector<T>& x)
    {
        // count bits required to describe the index
        size_t sx = x.size();
        unsigned nbits = 0;
        while( (sx-1) >> nbits )
            ++nbits;

        maxBitIndex = 1 << (nbits-1);

        // create copy of x extended to the right side to size
        // (1<<nbits) and padded with x[N-1]
        const unsigned nx = 1 << nbits;
        m_x.reserve( nx );
        m_x.assign( x.begin(), x.end() );
        m_x.resize(nx, x.back());
    }

    FORCE_INLINE const T *xptr() const { return &m_x.front(); }

    uint32 maxBitIndex;
    std::vector<T> m_x;  // duplicate vector x, adding padding to the right
};


// ***************************************
// BitSet Expression
//

template <Precision P>
struct ExprScalar<P,BitSet>
{
    typedef typename PrecTraits<P>::type T;

    FORCE_INLINE void init0( DataWorkspace<P>& p, const Info<P,BitSet>& info )
    {
        ri = p.rptr();
        zi = p.zptr();
        xi = info.xptr();
        maxBitIndex = info.maxBitIndex;
        xb = xi[maxBitIndex];
    }

    //NO_INLINE
    FORCE_INLINE
    void scalar( uint32 j ) const
    {
        uint32  i = 0;
        uint32  b = maxBitIndex;

        T z = zi[j];

        // the first iteration, when i=0, is simpler
        if (xb <= z)
            i = b;

        while ((b >>= 1)) {
            uint32 r = i | b;
            if (xi[r] <= z )
                i = r;
        };

        ri[j] = i;
    }

protected:
    T xb;
    uint32 * ri;
    const T* zi;
    const T* xi;
    uint32 maxBitIndex;
};


template <Precision P, InstrSet I>
struct ExprVector<P, BitSet,I> : ExprScalar<P, BitSet>
{
    typedef ExprScalar<P, BitSet> base_t;
    typedef typename PrecTraits<P>::type T;
    typedef FVec<I,T> fVec;
    typedef IVec<I,T> iVec;
public:

    static const uint32 VecSize = sizeof(fVec)/sizeof(T);

    FORCE_INLINE void initN( DataWorkspace<P>& p, const Info<P,BitSet>& info )
    {
        base_t::init0( p, info );
        xbv.setN(base_t::xb);
        bv.setN(base_t::maxBitIndex);
    }

    //NO_INLINE
    FORCE_INLINE
    void vectorial( uint32 j ) const
    {
        uint32  b = base_t::maxBitIndex;
        fVec zv( base_t::zi+j );

        iVec lbv = bv;

        // the first iteration, when i=0, is simpler
        iVec lev = xbv <= zv;
        iVec iv = lev & lbv;
        while ((b >>= 1)) {
            lbv = lbv >> 1;
            fVec xv;
            iVec rv = iv | lbv;
            xv.setidx( base_t::xi, rv );
            lev = xv <= zv;
            //iv = iv | ( lev & lbv );
            iv.assignIf(rv, lev);
        };

        iv.store( base_t::ri+j );
    }


private:
    fVec xbv;
    iVec bv;
};
