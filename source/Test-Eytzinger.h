
// Auxiliary information specifically used in the classic binary search
template <Precision P>
struct Info<P,Eytzinger>
{
    typedef typename PrecTraits<P>::type T;

private:
    struct layoutBuilder
    {
        layoutBuilder(const T *x, AlignedVec<T>& e, size_t size, size_t maxdepth)
            : m_x(x)
            , m_e(e)
            , m_lastIndex(size-1)
            , m_maxdepth(maxdepth)
        {
            fill_n(m_index, sizeof(m_index)/sizeof(m_index[0]), -1);
            assert(maxdepth>0);
            build(0, m_e.size()-1, 0);
        }


        void build(size_t a, size_t b, size_t depth)
        {
            size_t c = (a+b)/2;

	        size_t pos = (1<<depth) + m_index[depth]++;
	        m_e[pos] = m_x[min(c,m_lastIndex)];

            if (++depth < m_maxdepth) {
		        build(a,   c, depth);
		        build(c+1, b, depth);
            }
        }

        int m_index[64]; // i-th element contains the position already filled at depth i
        const T *m_x;
        AlignedVec<T>& m_e;
        size_t m_lastIndex, m_maxdepth;
    };


public:
    Info(const std::vector<T>& x)
    {
        // count bits required to describe the index
        size_t sx = x.size();

        uint32 maxdepth = 0;
        while ((1 << maxdepth) - 1 < sx)
            ++maxdepth;
        m_nLayers = maxdepth;
        m_mask = ~(1 << m_nLayers);

#if 0
        // In order to facilitate debugging of creation of Eytzinger layout
        // we set the elements of the vector x from 1 to n
        for (size_t i = 0; i < sx; ++i)
            const_cast<T&>(x[i]) = static_cast<float>(i+1);
#endif

        size_t nl = (1 << maxdepth) - 1;
        m_x.resize(nl);
        fill_n(m_x.begin(), nl, x.back());

        layoutBuilder(&x.front(), m_x, sx, maxdepth);
    }

    FORCE_INLINE const T *xptr() const { return &m_x.front(); }

    AlignedVec<T> m_x;  // duplicate vector x, adding padding to the right
    uint32 m_nLayers;
    uint32 m_mask;
};


// ***************************************
// Eytzinger Expression
//

template <Precision P>
struct ExprScalar<P,Eytzinger>
{
    typedef typename PrecTraits<P>::type T;

    FORCE_INLINE
    void init0( DataWorkspace<P>& p, const Info<P,Eytzinger>& info )
    {
        ri = p.rptr();
        zi = p.zptr();
        xi = info.xptr();
        m_nLayers = info.m_nLayers;
        m_mask = info.m_mask;
    }

    FORCE_INLINE
    void scalar( uint32 j ) const
    {
        T z = zi[j];

        uint32 d = m_nLayers;

        // the first iteration, when p=0, is simpler
        uint32 p = (z < xi[0])? 1: 2;

        while (--d > 0)
            p = (p << 1) + ((z < xi[p])? 1: 2);

        ri[j] = p & m_mask;  // clear higher bit
    }

protected:
    uint32 * ri;
    const T* zi;
    const T* xi;
    uint32 m_nLayers;
    uint32 m_mask;
};

template <Precision P, InstrSet I>
struct ExprVector<P, Eytzinger,I> : ExprScalar<P, Eytzinger>
{
    typedef typename PrecTraits<P>::type T;
    typedef ExprScalar<P,Eytzinger> base_t;
    typedef FVec<I,T> fVec;
    typedef IVec<I,T> iVec;

    static const uint32 VecSize = sizeof(fVec)/sizeof(T);

    FORCE_INLINE
    void initN( DataWorkspace<P>& p, const Info<P, Eytzinger>& info )
    {
        base_t::init0( p, info );
        xbv.setN(base_t::xi[0]);
        maskv.setN(base_t::m_mask);
        two.setN(2);
    }

    FORCE_INLINE
    void vectorial( uint32 j ) const
    {
        fVec zv( base_t::zi+j );

        uint32 d = base_t::m_nLayers;

        // the first iteration, when p=0, is simpler
        iVec pv = (zv < xbv) + two;

        while (--d > 0) {
            fVec xv;
            xv.setidx( base_t::xi, pv );
            iVec right = (zv < xv) + two;
            pv = (pv << 1) + right;
        };

        pv = pv & maskv;
        pv.store( base_t::ri+j );
    }

    fVec xbv;
    iVec two;
    iVec maskv;
};
