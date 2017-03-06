
#if defined(USE_MKL)
#include <mkl.h>
#endif

#if defined(USE_MKL)

template <typename T> struct MKLTraits {};

template <> struct MKLTraits < double >
{
    FORCE_INLINE
        static void search(DFTaskPtr task, MKL_INT n, const double *zp, MKL_INT *rp, const double* datahint)
    {
        dfdSearchCells1D(task, DF_METHOD_STD, n, zp, DF_NO_HINT, datahint, rp);
    }

    static int dfNewTask1D(DFTaskPtr& task, MKL_INT n, double *px)
    {
        return dfdNewTask1D(&task, n, px, DF_QUASI_UNIFORM_PARTITION, 0, 0, 0);
    }

};

template <> struct MKLTraits < float >
{
    FORCE_INLINE
    static void search(DFTaskPtr task, MKL_INT n, const float *zp, MKL_INT *rp, const float* datahint)
    {
        dfsSearchCells1D(task, DF_METHOD_STD, n, zp, DF_NO_HINT, datahint, reinterpret_cast<MKL_INT*>(rp));
    }

    static int dfNewTask1D(DFTaskPtr& task, MKL_INT n, float *px)
    {
        return dfsNewTask1D(&task, n, px, DF_QUASI_UNIFORM_PARTITION, 0, 0, 0);
    }
};

template <Precision P>
struct Info<P,MKL>
{
    typedef typename PrecTraits<P>::type T;
public:
    DFTaskPtr m_task;
    T m_datahint[5];         // additional info about the structure

    ~Info()
    {
        if (m_task)
            dfDeleteTask(&m_task);
    }

    Info(const std::vector<T>& x)
    {
        MKL_INT n = static_cast<MKL_INT>( x.size() );
        static char sizeguard[sizeof(MKL_INT)==sizeof(uint32)?1:-1];
        if (sizeof(MKL_INT) != sizeof(uint32))
            throw;

        m_datahint[0] = 1;
        m_datahint[1] = (T)DF_APRIORI_MOST_LIKELY_CELL;
        m_datahint[2] = 0;
        m_datahint[3] = 1;
        m_datahint[4] = (T)((x.size() / 2) + 1);

        int status = MKLTraits<T>::dfNewTask1D(m_task, n, const_cast<T*>(&x[0]) );
        if (status != DF_STATUS_OK)
            throw;
    }

};

// ***************************************
// MKL Expression
//

template <Precision P>
struct ExprScalar<P,MKL>
{
    typedef typename PrecTraits<P>::type T;

    FORCE_INLINE
    void init0(DataWorkspace<P>& p, const Info<P,MKL>& info)
    {
        ri = p.rptr();
        zi = p.zptr();
        xi = p.xptr();
        m_task = info.m_task;
        m_datahint = info.m_datahint;
    }

    FORCE_INLINE
    void scalar(uint32 j) const
    {
        MKLTraits<T>::search(m_task, 1, zi + j, reinterpret_cast<MKL_INT*>(ri + j), m_datahint);
        ri[j] -= 1;
    }

protected:
    uint32 * ri;
    const T* zi;
    const T* xi;
    DFTaskPtr m_task;
    const T *m_datahint;
};

template <Precision P, InstrSet I>
struct ExprVector<P,MKL,I> : ExprScalar<P,MKL>
{
    typedef typename PrecTraits<P>::type T;
    typedef ExprScalar<P,MKL> base_t;

    static const uint32 VecSize = NZ;

    FORCE_INLINE
    void initN(DataWorkspace<P>& p, const Info<P,MKL>& info)
    {
        base_t::init0(p, info);
    }

    FORCE_INLINE
    void vectorial(uint32) const
    {
        MKLTraits<T>::search(base_t::m_task, NZ, base_t::zi, reinterpret_cast<MKL_INT*>(base_t::ri), base_t::m_datahint);
        for (uint32 i = 0; i < NZ; ++i)
            base_t::ri[i] -= 1;
    }
};


#else

template <Precision P>
struct Info<P,MKL>
{
    typedef typename PrecTraits<P>::type T;
    Info(const std::vector<T>& x) {}
};


#endif
