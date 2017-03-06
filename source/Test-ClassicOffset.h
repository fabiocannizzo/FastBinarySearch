
// Auxiliary information specifically used in the ClassicOffset binary search
template <Precision P>
struct Info<P,ClassicOffset>
{
    typedef typename PrecTraits<P>::type T;
    Info(const std::vector<T>& x)
        : nIter(static_cast<uint32>(log2(x.size())))
        , size1(static_cast<uint32>(x.size()))
        , mid0(size1/2)
    {
        size1 -= mid0;
    }

    uint32 size1;
    uint32 mid0;
    uint32 nIter;
};


// ***************************************
// ClassicOffset Epxression
//

template <Precision P>
struct ExprScalar<P,ClassicOffset>
{
    typedef typename PrecTraits<P>::type T;

    FORCE_INLINE
    void init0( DataWorkspace<P>& p, const Info<P,ClassicOffset>& info )
    {
        ri = p.rptr();
        zi = p.zptr();
        xi = p.xptr();
        nIter = info.nIter;
        size1 = info.size1;
        mid0 = info.size1;
    }

    FORCE_INLINE
    void scalar( uint32 j ) const
    {
        T z = zi[j];

        // variation on original paper: the number of iterations is fixed

        // there is at least one iteration
        uint32 mid = mid0;  // initialised to: size/2
        uint32 i = (z >= xi[mid0])? mid: 0;
        uint32 sz = size1;  // initialised to: size - mid
        uint32 n = nIter;   // this is decreaded by 1
        while(n--) {
            uint32 h = sz / 2;
            uint32 mid = i+h;
            if (z >= xi[mid])
                i = mid;
            sz -= h;
        }

        ri[j] = i;
    }

protected:
    uint32 * ri;
    const T* zi;
    const T* xi;
    uint32 nIter;
    uint32 size1;
    uint32 mid0;
};


template <Precision P, InstrSet I>
struct ExprVector<P, ClassicOffset,I> : ExprScalar<P, ClassicOffset>
{
    typedef ExprScalar<P, ClassicOffset> base_t;
    typedef typename PrecTraits<P>::type T;
    typedef FVec<I,T> fVec;
    typedef IVec<I,T> iVec;
public:

    static const uint32 VecSize = sizeof(fVec)/sizeof(T);

    FORCE_INLINE void initN( DataWorkspace<P>& p, const Info<P,ClassicOffset>& info )
    {
        base_t::init0( p, info );
        xMidV.setN(base_t::xi[base_t::mid0]);
        size1V.setN(base_t::size1);
        mid0V.setN(base_t::mid0);
    }

    //NO_INLINE
    FORCE_INLINE
    void vectorial( uint32 j ) const
    {
        fVec zV( base_t::zi+j );

        // there is at least one iteration
        iVec midV = mid0V;  // initialised to: size/2
        iVec iV = (zV >= xMidV) & midV;
        iVec szV = size1V;  // initialised to: size - mid
        uint32 n = base_t::nIter;   // this is decreaded by 1
        while(n--) {
            iVec hV = szV >> 1;
            midV = iV + hV;
            fVec xV;
            xV.setidx( base_t::xi, midV );

            iVec geV = zV >= xV;
            iV.assignIf(midV, geV);

            szV = szV - hV;
        }

        iV.store( base_t::ri+j );
    }


private:
    fVec xMidV;
    iVec size1V;
    iVec mid0V;
};

