#pragma once

#include <algorithm>

namespace DirectAux {

#define SAFETY_MULTI_PASS true

template <typename T>
inline void checkH(T H, T Dn)
{
    T ifmax = Dn * H;
    if (ifmax >= numeric_limits<uint32>::max()) {
        cout << "Problem unfeasible: index size exceeds uint32 capacity:"
            << " D[N] =" << Dn
            << ", H =" << H
            << ", H D[n] =" << ifmax << "\n";
        exit(0);
    }
}

template <typename T>
struct HResults
{
    HResults(T h, double ratio, size_t n) : H(h), hRatio(ratio), nInc(n) {}
    T H;
    double hRatio;
    size_t nInc;
};

template <int Offset, typename T>
HResults<T> computeH(const T *px, size_t nx)
{
    T x0 = px[0];
    T range = px[nx - 1] - x0;
    myassert((range < numeric_limits<T>::max()), "range too large");

    const T one = T(1.0);

    T Dn = range;

    // check that D_i are strictly increasing and compute minimum value D_{i+Offset}-D_i
    T deltaDMin = std::numeric_limits<T>::max();
    for (size_t i = Offset; i < nx; ++i) {
        T Dnew = px[i] - px[0];
        T Dold = px[i-Offset] - px[0];
        if (Dnew <= Dold) {
            std::cout << "Problem unfeasible: D_i sequence not strictly increasing"
                      << " X[" << 0   << "]=" << px[0]
                      << " X[" << i-Offset << "]=" << px[i-Offset]
                      << " X[" << i   << "]=" << px[i]
                      << "\n";
            exit(0);
        }
        T deltaD = Dnew - Dold;
        if (deltaD < deltaDMin)
            deltaDMin = deltaD;
    }

    // initial guess for H
    T H0 = one / deltaDMin;
    T H = H0;
    checkH(H,Dn);

    // adjust H by trial and error until succeed
    size_t nInc = 0;
    bool modified;
    size_t npasses = 0;
    T step;
    T P = next(H);
    while ((step = P - H) == 0)
        P = next(P);
    do {
        if (npasses++ >= 2) {
            cout << "verification failed\n";
            exit(0);
        }
        modified = false;
        for ( uint32 i = Offset; i < nx; ++i ) {
            T Dnew = px[i] - px[0];
            T Dold = px[i - Offset] - px[0];
            while(1) {
                T ifnew = Dnew * H;
                T ifold = Dold * H;
                uint32 iold = ftoi(FVec1<SSE,T>(ifold));
                uint32 inew = ftoi(FVec1<SSE,T>(ifnew));
                if (inew == iold) {
                    modified = true;
                    H = H+step; //next(max(H+step, H*(H/H0))); // take a bigger H
                    step *= 2;
                    checkH(H,Dn);
                    ++nInc;
                }
                else
                    break;
            }
        }
    } while(SAFETY_MULTI_PASS && modified);

    return HResults<T>(H, (((double)H)/H0)-1.0, nInc);
}


// general definition
template <Algos A, typename T>
struct BucketElem
{
    FORCE_INLINE void set( uint32 b, const T *)
    {
        m_b = b;
    }

    FORCE_INLINE uint32 index() const { return m_b; }

private:
    uint32 m_b;
};

// specialization for DirectCache
template <>
struct BucketElem<DirectCache, float>
{
    void set( uint32 b, const float *xi )
    {
        u.u.x = xi[b];
        u.u.b = b;
    }

    FORCE_INLINE uint32 index() const { return u.u.b; }
    FORCE_INLINE float x() const { return u.u.x; }

private:
    union {
        double dummy;
        struct
        {
            float x;
            uint32 b;
        } u;
    } u;
};

template <>
struct BucketElem<DirectCache, double>
{
    void set( uint32 b, const double *xi )
    {
        u.u.x = xi[b];
        u.u.b = b;
    }

    FORCE_INLINE uint64 index() const { return u.u.b; }
    FORCE_INLINE double x() const { return u.u.x; }

private:
    union {
        __m128i dummy;
        struct
        {
            double x;
            uint64 b;
        } u;
    } u;
};


template <int Offset, typename T, Algos A>
struct DirectInfo
{
    DirectInfo(const T* x, const size_t n)
        : xi(NULL)
    {
        uint32 nx = static_cast<uint32>(n);
        const T *px = &x[0];

        if (Offset>1) {
            xi = new T[nx+Offset-1];
            std::fill_n(xi, Offset-1, x[0]);
            std:copy(x, x+n, xi+(Offset-1));
        }

        HResults<T> res = computeH<Offset>(px, nx);
        scaler = res.H;
        nInc = res.nInc;
        hRatio = res.hRatio;

        uint32 maxIndex = ftoi(FVec1<SSE,T>((px[nx-1] - px[0])*scaler));
        T x0(px[0]);
        buckets.resize(maxIndex + 1); // nb is the max index, hence size is nb+1


        for (uint32 i = nx-1, b = maxIndex, j=nx; ; --i ) {
            T hi = (px[i] - x0) * scaler;
            uint32 idx = ftoi( FVec1<SSE,T>(hi) );
            while (b > idx) {
                buckets[b].set( j, px );
                --b;
            }
            if (b == idx) {
                j = i;
                buckets[b].set(j, px);
                if (b-- == 0)
                    break;
            }

        }

    }

    ~DirectInfo()
    {
        delete [] xi;
    }

    AlignedVec<BucketElem<A,T>>  buckets;
    T *xi;
    T scaler;
    double hRatio;
    size_t nInc;
};


} // namespace DirectAux
