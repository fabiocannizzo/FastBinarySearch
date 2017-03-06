
// Auxiliary information specifically used in the modified version of the BitSet method (the one using min)
template <Precision P>
struct Info<P, BitSetNoPad>
{
    typedef typename PrecTraits<P>::type T;

    Info(const std::vector<T>& x)
        : lastValidIndex(static_cast<uint32>( x.size()-1 ))
    {
        const size_t sx = x.size();
        unsigned nbits = 0;
        while( (sx-1) >> nbits )
            ++nbits;

        maxBitIndex = 1 << (nbits-1);
    }

    uint32 maxBitIndex;
    uint32 lastValidIndex;
};


// ***************************************
// BitSet Expression using min instead of a padded X vector
//
template <Precision P>
struct ExprScalar<P,BitSetNoPad>
{
    typedef typename PrecTraits<P>::type T;

    FORCE_INLINE void init0( DataWorkspace<P>& p, const Info<P,BitSetNoPad>& info )
    {
        ri = p.rptr();
        zi = p.zptr();
        xi = p.xptr();
        maxBitIndex = info.maxBitIndex;
        xLast = info.lastValidIndex;
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
            uint32 h = r<=xLast? r: xLast;
            if (xi[h] <= z )
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
    uint32 xLast;
};

template <Precision P, InstrSet I>
struct ExprVector<P,BitSetNoPad,I> : ExprScalar<P,BitSetNoPad>
{
    typedef ExprScalar<P,BitSetNoPad> base_t;
    typedef typename PrecTraits<P>::type T;
    typedef FVec<I,T> fVec;
    typedef IVec<I,T> iVec;

    static const uint32 VecSize = sizeof(fVec)/sizeof(T);

    FORCE_INLINE void initN( DataWorkspace<P>& p, const Info<P,BitSetNoPad>& info )
    {
        base_t::init0( p, info );
        xbv.setN(base_t::xb);
        bv.setN(base_t::maxBitIndex);
        xlv.setN(base_t::xLast);
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
            iVec hv = min(rv, xlv);
            xv.setidx( base_t::xi, hv );
            lev = xv <= zv;
            //iv = iv | ( lev & lbv );
            iv.assignIf(rv, lev);
        };

        iv.store( base_t::ri+j );
    }

private:
    fVec xbv;
    iVec bv;
    iVec xlv;
};
