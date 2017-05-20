#pragma once

// Auxiliary information specifically used in the classic binary search
template <typename T>
struct InfoScalar<T, Classic>
{
    typedef T type;

    InfoScalar(const T* x, const size_t n)
        : xi(x)
        , lastIndex(static_cast<uint32>(n - 1))
    {
    }

    FORCE_INLINE
    uint32 scalar(T z) const
    {
        uint32 lo = 0;
        uint32 hi = lastIndex;
        while (hi - lo > 1) {
            int mid = (hi + lo) >> 1;
            if (z < xi[mid])
                hi = mid;
            else
                lo = mid;
        }
        return lo;
    }

private:
    const T *xi;
    uint32 lastIndex;
};

// Auxiliary information specifically used in the classic binary search
template <InstrSet I, typename T>
struct Info<I, T, Classic> : public InfoScalar<T,Classic>
{
    typedef InfoScalar<T, Classic> base_t;

    Info(const T* x, const size_t n)
        : base_t(x,n)
    {
    }

    FORCE_INLINE
    void vectorial(uint32 *pr, const T *pz) const
    {
        const uint32 VecSize = sizeof(typename InstrFloatTraits<I,T>::vec_t)/sizeof(T);
        for ( unsigned i = 0; i < VecSize; ++i )
            pr[i] = base_t::scalar(pz[i]);
    }
};
