#pragma once

#ifdef USE_MKL

#ifdef __CYGWIN__
#include "mklloader.h"
#define MYMKLCALL(f) (*f##_ptr)
#else
#define MYMKLCALL(f) f
#endif
#include <mkl.h>

namespace BinSearch {

namespace Details {

template <typename T> struct MKLTraits {};

template <> struct MKLTraits < double >
{
    FORCE_INLINE
        static void search(DFTaskPtr task, MKL_INT n, const double *zp, MKL_INT *rp, const double* datahint)
    {
        MYMKLCALL(dfdSearchCells1D)(task, DF_METHOD_STD, n, zp, DF_NO_HINT, datahint, rp);
    }

    static int dfNewTask1D(DFTaskPtr& task, MKL_INT n, double *px)
    {
        return MYMKLCALL(dfdNewTask1D)(&task, n, px, DF_QUASI_UNIFORM_PARTITION, 0, 0, 0);
    }

};

template <> struct MKLTraits < float >
{
    FORCE_INLINE
    static void search(DFTaskPtr task, MKL_INT n, const float *zp, MKL_INT *rp, const float* datahint)
    {
        MYMKLCALL(dfsSearchCells1D)(task, DF_METHOD_STD, n, zp, DF_NO_HINT, datahint, reinterpret_cast<MKL_INT*>(rp));
    }

    static int dfNewTask1D(DFTaskPtr& task, MKL_INT n, float *px)
    {
        return MYMKLCALL(dfsNewTask1D)(&task, n, px, DF_QUASI_UNIFORM_PARTITION, 0, 0, 0);
    }
};

template <typename T>
struct AlgoScalarBase<T,MKL>
{
    AlgoScalarBase(const T* x, const uint32 n)
    {
        MKL_INT nx = static_cast<MKL_INT>(n);
        static char sizeguard[sizeof(MKL_INT)==sizeof(uint32)?1:-1];
        if (sizeof(MKL_INT) != sizeof(uint32))
            throw;

        m_datahint[0] = 1;
        m_datahint[1] = (T)DF_APRIORI_MOST_LIKELY_CELL;
        m_datahint[2] = 0;
        m_datahint[3] = 1;
        m_datahint[4] = (T)((n / 2) + 1);

        int status = MKLTraits<T>::dfNewTask1D(m_task, nx, const_cast<T*>(x) );
        if (status != DF_STATUS_OK)
            throw;
    }

    FORCE_INLINE
    uint32 scalar(T z) const
    {
        MKL_INT res;
        MKLTraits<T>::search(m_task, 1, &z, &res, m_datahint);
        return (res-1);
    }

    ~AlgoScalarBase()
    {
        if (m_task)
            MYMKLCALL(dfDeleteTask)(&m_task);
    }

protected:
    DFTaskPtr m_task;
    T m_datahint[5];         // additional info about the structure
};


template <typename T>
struct AlgoVecMKL : Details::AlgoScalarBase<T, MKL>
{
    typedef Details::AlgoScalarBase<T, MKL> base_t;

public:
    AlgoVecMKL(const T* x, const size_t n) : base_t(x, n) {}

    FORCE_INLINE
    void vectorial(uint32 *pr, const T *pz, uint32 n) const
    {
        Details::MKLTraits<T>::search(base_t::m_task, n, pz, reinterpret_cast<MKL_INT*>(pr), base_t::m_datahint);
        for (uint32 i = 0; i < NZ; ++i)
            pr[i] -= 1;
    }
};

} // namespace Details


} // namespace BinSearch

#endif
