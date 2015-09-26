#include <cstdio>
#include <ctime>
#include <iostream>
#include <limits>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cassert>

#include "AAlloc.h"
#include "SIMD.h"
#include "Unroller.h"

#define USE_MKL
#if defined(USE_MKL)
#include <mkl.h>
#endif

// In total we resolve 2*(N-1)*nAvg*nRepeat indices
const uint32   NX       = 1025;  // Dimension of the vector X. The vector Z has size 2(N-1)
const size_t   NZ       = 4 * (NX - 1);  // size of array Z: must be even
const unsigned nAvg     = 10;     // Number of random regenerations of vector X
const unsigned nRepeat  = 10000; // Number repetition of the test
const float intMin = 0.1f;
const float intMax = 5.0f;

size_t rMin = 1000000, rMax = 0;

// generates a random number in single precision between in [range_min, range_max)
template <class T>
T dRand(T range_min, T range_max)
{
    return (static_cast<T>(std::rand()) / (static_cast<size_t>(RAND_MAX) + 1)) * (range_max - range_min) + range_min;
}

// Base class for auxiliary data structure used in the tests
template <typename T>
struct RawData
{
    // Allocate memory for vector X of size n and initilize it with increments of random size
    // drawn in the range (intMin, intMax)
    // Allocate memory for vector Z and initialize it with the union of the extrema of the intervals in vector X, all mid points 
    // and an equal number of extra points extracted randomly in the interval [X_0,X_N)
    // Allocate memory for vector R (same size as Z) containing the indices of the segments [X_i,X_{i+1}) containing the numbers Z
    RawData( T intMin, T intMax )
    {
        m_z.resize(NZ);
        m_r.resize(NZ);
        m_x.resize(NX);

        // init x with random gaps drawn in intMin, intMax
        T xold = 0.0;
        for (uint32 i = 0; i < NX; ++i)
            m_x[i] = xold += dRand(intMin, intMax);

        // init z must belong to [ x[0], x[nx-1] )
        for (uint32 j = 0; j + 1 < NX; ++j) {
            m_z[4 * j] = m_x[j];
            m_z[4 * j + 1] = (m_x[j + 1] + m_x[j]) / 2;
            m_z[4 * j + 2] = dRand(m_x.front(), m_x.back());
            m_z[4 * j + 3] = dRand(m_x.front(), m_x.back());
        }
        std::random_shuffle(m_z.begin(), m_z.end());
    }

    // check if the indices r, representing the interval in x where containing the numbers in z
    // then reset the indices to invalid values
    void checkAndReset()
    {
        size_t nz = m_z.size();
        uint32 nx = static_cast<uint32>(m_x.size());
        for (uint32 j = 0; j < nz; ++j) {
            uint32 i = m_r[j];
            T z = m_z[j];
            if ( !(i < m_x.size()) || z < m_x[i] || z >= m_x[i + 1])
                std::cout << "Error: Z[" << j << "]=" << z
                << ", x[" << i << "]=" << m_x[i]
                << ", x[" << i + 1 << "]=" << m_x[i + 1]
                << "\n";
            m_r[j] = nx;  // nx is an invalid index
        }
    }

    const T *xptr() const { return &m_x.front(); }
    const T *zptr() const { return &m_z.front(); }
    uint32 *rptr() { return &m_r.front(); }
    
    std::vector<T>  m_x;
    AlignedVec<T, 32> m_z;
    AlignedVec<uint32, 32 > m_r;
};


#if defined(USE_MKL)

template <class T> struct MKLTraits {};

template <> struct MKLTraits < double >
{
    __forceinline
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
    __forceinline
        static void search(DFTaskPtr task, MKL_INT n, const float *zp, MKL_INT *rp, const float* datahint)
    {
        dfsSearchCells1D(task, DF_METHOD_STD, n, zp, DF_NO_HINT, datahint, reinterpret_cast<MKL_INT*>(rp));
    }

    static int dfNewTask1D(DFTaskPtr& task, MKL_INT n, float *px)
    {
        return dfsNewTask1D(&task, n, px, DF_QUASI_UNIFORM_PARTITION, 0, 0, 0);
    }
};

template <class T>
class MKLMethodInfo
{
public:
    DFTaskPtr m_task;
    T m_datahint[5];         // additional info about the structure

    ~MKLMethodInfo()
    {
        if (m_task)
            dfDeleteTask(&m_task);
    }

    MKLMethodInfo(const std::vector<T>& x)
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



#endif


// Auxiliary informatio specifically used in the optimized binary search
template <class T>
struct BitMethodInfo
{
    BitMethodInfo(const std::vector<T>& x)
    {
        // count bits required to describe the index
        size_t sx = x.size();
        unsigned nbits = 0;
        while( (sx-1) >> nbits )
            ++nbits;
            
        maxBitIndex = 1 << (nbits-1);

        // create copy of x extended to the right side to size 
        // (1<<nbits) and padded with x[N-1]
        const size_t nx = 1 << nbits;
        m_x.reserve( nx );
        m_x.assign( x.begin(), x.end() );
        m_x.resize(nx, x.back());
    }

    const T *xptr() const { return &m_x.front(); }

    uint32 maxBitIndex;
    std::vector<T> m_x;  // duplicate vector x, adding padding to the right
};


// Auxiliary information specifically used in the classic binary search
template <class T>
struct NaiveMethodInfo
{
    NaiveMethodInfo(const std::vector<T>& x)
    {
        lastValidIndex = static_cast<uint32>( x.size()-1 );
    }

    uint32 lastValidIndex;
};

template <class T>
struct DirectMethodInfo 
{
    DirectMethodInfo(const std::vector<T>& x)
	{
		uint32 nx = static_cast<uint32>(x.size());
		T hmin = x[1] - x[0];
		for ( uint32 i = 2; i < nx; ++i )
			hmin = std::min( hmin, x[i]-x[i-1] );
		T x0 = x[0];
		T range = x[nx - 1] - x0;
		uint32 nh = 2 + static_cast<uint32>( std::floor( range / hmin ) );
		SIMD0<T> mm_binsize = SIMD0<T>(range) / SIMD0<T>(static_cast<T>(nh));
		binsize = mm_binsize.get0();
		SIMD0<T> mm_x0( x0 );
		buckets.resize( nh );
		for ( long b = static_cast<long>(buckets.size())-1, i = static_cast<long>(x.size()-1); i-- > 0; ) {
			SIMD0<T> xi( x[i] );
			SIMD0<T> hi = (xi - mm_x0) / mm_binsize;
			SIMD0<T> fl = floor(hi);
			I128<T> gt = hi>fl;
			int gti = gt.get0();
			int idx = atoi( hi );
			uint32 ip = i + 1;
			while ( b > idx )
				buckets[b--] = ip;
			if ( !gti )
				buckets[b--] = ip;
		}
        
        if (buckets.size() < rMin)
            rMin = buckets.size();
        if (buckets.size() > rMax)
            rMax = buckets.size();
	}

	std::vector<uint32>  buckets;
	T binsize;
};


// ***************************************
// Naive method
//


#ifdef USE_MKL

template <class T>
class MKLExpr
{
public:
    static const uint32 VecSize = NZ;

    __forceinline
        void init0(RawData<T>& p, const MKLMethodInfo<T>& info)
    {
        ri = p.rptr();
        zi = p.zptr();
        xi = p.xptr();
        m_task = info.m_task;
        m_datahint = info.m_datahint;
    }

    __forceinline
        void initN(RawData<T>& p, const MKLMethodInfo<T>& info)
    {
        init0(p, info);
    }

    __forceinline
    void scalar(uint32 j) const
    {
        MKLTraits<T>::search(m_task, 1, zi + j, reinterpret_cast<MKL_INT*>(ri + j), m_datahint);
        ri[j] -= 1;
    }
        
    __forceinline
    void vectorial(uint32) const
    {
        MKLTraits<T>::search(m_task, NZ, zi, reinterpret_cast<MKL_INT*>(ri), m_datahint);
        for (uint32 i = 0; i < NZ; ++i)
            ri[i] -= 1;
    }

    
protected:
    uint32 * ri;
    const T* zi;
    const T* xi;
    DFTaskPtr m_task;
    const T *m_datahint;
};

#endif

template <class T>
class BinSearchExpr
{
public:
    static const uint32 VecSize = sizeof(__m128d)/sizeof(T);

    __forceinline
    void init0( RawData<T>& p, const NaiveMethodInfo<T>& info )
	{
		ri = p.rptr();
        zi = p.zptr();
        xi = p.xptr();
		xLast = info.lastValidIndex;
	}

    __forceinline
    void initN( RawData<T>& p, const NaiveMethodInfo<T>& info )
	{
		init0( p, info );
	}

	__forceinline
    void scalar( uint32 j ) const
	{
		T z = zi[j];
		uint32 lo = 0;
		uint32 hi = xLast;
		while (hi - lo>1) {
			int mid = (hi + lo) >> 1;
			if (z < xi[mid])
				hi = mid;
			else
				lo = mid;
		}
		ri[j] = lo;
	}

	__forceinline
    void vectorial( uint32 j ) const
    {
		for ( unsigned i = 0; i < VecSize; ++i ) 
			scalar( j+i );
	}

private:
	uint32 * ri;
	const T* zi;
	const T* xi;
	uint32 xLast;
};



// ***************************************
// Bit Method
//


template <class T>
class BitExpr
{
	typedef SIMD<T> f128;
	typedef I128<T> i128;
public:

    static const uint32 VecSize = sizeof(__m128d)/sizeof(T);

    __forceinline void init0( RawData<T>& p, const BitMethodInfo<T>& info )
	{
        ri = p.rptr();
        zi = p.zptr();
        xi = info.xptr();
        maxBitIndex = info.maxBitIndex;
		xb = xi[maxBitIndex];
	}

    __forceinline void initN( RawData<T>& p, const BitMethodInfo<T>& info )
	{
		init0( p, info );
		xbv.setN(xb);
		bv.setN(maxBitIndex);
	}

	//__declspec(noinline)
	__forceinline
	void scalar( uint32 j ) const
	{
		uint32  i = 0;
		uint32  b = maxBitIndex;

		T z = zi[j];

		// the first iteration, when i=0, is simpler
		if (xb <= z)
			i = b;
		
		while ((b >>= 1) > 0) {
			uint32 r = i | b;
			if (xi[r] <= z )
				i = r;
		};

		ri[j] = i;
	}

	//__declspec(noinline)
	__forceinline 
	void vectorial( uint32 j ) const
	{
		//uint32  b = maxBitIndex;
		f128 zv( zi+j );
		
		i128 lbv = bv;

		// the first iteration, when i=0, is simpler
		i128 lev = xbv <= zv;
		i128 iv = lev & lbv;
		while (true) {
			lbv = lbv >> 1;
			if (!lbv.get0())
				break;
			f128 xv;
			i128 rv = iv | lbv;
			xv.setidx( xi, rv );
			lev = xv <= zv;
			//iv = iv | ( lev & lbv );
			iv.assignIf(rv, lev);
		};

		iv.store( ri+j );
	}

	
private:
	f128 xbv;
	i128 bv;
	T xb;
	uint32 * ri;
	const T* zi;
	const T* xi;
	uint32 maxBitIndex;
};


// ***************************************
// Direct method
//

template <class T>
class DirectExpr
{
public:

    static const uint32 VecSize = sizeof(__m128d)/sizeof(T);

    __forceinline void init0( RawData<T>& p, const DirectMethodInfo<T>& info )
	{
        ri = p.rptr();
        zi = p.zptr();
        xi = p.xptr();
		buckets = &info.buckets.front();
		binsize = info.binsize;
	}

    __forceinline void initN(RawData<T>& p, const DirectMethodInfo<T>& info)
	{
		init0( p, info );
		vbinsize.setN( binsize );
		vx0.setN( xi[0] );
	}

	__forceinline void scalar( uint32 j ) const
	{
		T z = zi[j];
		T tmp = (z - xi[0]) / binsize;
		uint32 bidx = static_cast<uint32>(tmp);
		uint32 iidx = buckets[bidx];
		uint32 iidxm = iidx - 1;
		ri[j] = (xi[iidx] <= z)? iidx : iidxm;
	}

	__forceinline
	//__declspec(noinline)
	void resolve( const SIMD<float>& vz, const I128<float>& bidx, uint32 j ) const
	{
		union {
			__m128i vec;
			uint32 ui32[4];
		} i;
		__m128  vxp = _mm_set_ps
			( xi[(i.ui32[3] = buckets[bidx.get3()])]
			, xi[(i.ui32[2] = buckets[bidx.get2()])]
			, xi[(i.ui32[1] = buckets[bidx.get1()])]
			, xi[(i.ui32[0] = buckets[bidx.get0()])]
			);

		__m128i le = vz < vxp;
		i.vec = _mm_add_epi32( i.vec, le );
		I128<float>(i.vec).store( ri+j );
	}

	__forceinline
	//__declspec(noinline) 
	void resolve( const SIMD<double>& vz, const I128<float>& bidx, uint32 j ) const
	{
		__m128d vxp;

		union {
			__m128i vec;
			uint32 ui32[4];
		} i;
		vxp = _mm_set_pd
			( xi[(i.ui32[1] = buckets[bidx.get1()])]
			, xi[(i.ui32[0] = buckets[bidx.get0()])]
			);
		__m128i le = vz < vxp;
		__m128i dec = _mm_shuffle_epi32(le, ((2 << 2) | 0));
		i.vec = _mm_add_epi32( i.vec, dec );

		_mm_storel_epi64( reinterpret_cast<__m128i*>(ri+j), i.vec );
	}

	__forceinline void vectorial( uint32 j ) const
	{
		SIMD<T> vz( zi+j );
		SIMD<T> tmp( (vz - vx0) / vbinsize );
		I128<float> bidx = atoi( tmp );
		resolve( vz, bidx, j );
	}

	
private:
	SIMD<T> vbinsize;
	SIMD<T> vx0;
	T binsize;
	uint32 * ri;
	const T* zi;
	const T* xi;
	const uint32* buckets;
};


// ***************************************
// TEST
//

template < class T
    , template <typename T> class Info
    , template <typename T> class Expr
>
struct ScalarTest
{
    typedef typename Info<T> info_t;
    typedef typename Expr<T> expr_t;

    static double run(RawData<T>& p, const Info<T>& info )
    {
        size_t nz = p.m_z.size();
        clock_t t1 = std::clock();
        for (size_t t = 0; t < nRepeat; ++t) {
            for (uint32 j = 0; j < nz; ++j)
                singlerun(p, info, j);
        }
        clock_t t2 = std::clock();
        p.checkAndReset();
        return (nRepeat * nz) / (static_cast<double>(t2 - t1) / CLOCKS_PER_SEC);
    }

private:
    __declspec(noinline)
    static void singlerun(RawData<T>& p, const Info<T>& info, uint32 j)
    {
        Expr<T> e;
        e.init0(p, info);
        e.scalar(j);
    }
};

template < class T
         , template <typename T> class Info
         , template <typename T> class Expr 
         >
struct VectorTest
{
    typedef typename Info<T> info_t;
    typedef typename Expr<T> expr_t;

    static double run(RawData<T>& p,const Info<T>& info)
    {
        clock_t t1 = std::clock();
        for (size_t t = 0; t < nRepeat; ++t)
            singlerun(p,info);
        clock_t t2 = std::clock();
        p.checkAndReset();
        return (nRepeat * p.m_z.size()) / (static_cast<double>(t2 - t1) / CLOCKS_PER_SEC);
    }
private:
    __declspec(noinline)
    static void singlerun(RawData<T>& p, const Info<T>& info)
    {
        Expr<T> e;
        e.initN(p, info);
        Loop< Expr<T> >::loop(e, static_cast<uint32>( p.m_z.size() ));
    }
};


template < class T, class Tester >
std::pair<std::string, double> runTest(const std::string testname, RawData<T>& d)
{
    double res = 0;

    Tester::info_t info(d.m_x);
    double searchpersec = Tester::run(d, info);
    std::cout << testname << ": " << searchpersec << " search per sec\n";
    return std::make_pair(testname, searchpersec);
}

template <class T>
void testAll(const std::string label)
{
    std::vector<std::pair<std::string, double> > testresults;

    for (size_t i = 0; i < nAvg; ++i) {

        std::cout << "Trial no. " << i << " in " << label << " precision\n";

        RawData<T> d( intMin, intMax );

        testresults.push_back(runTest< T, ScalarTest<T,NaiveMethodInfo,BinSearchExpr> >("naive  scalar      " + label, d));
        testresults.push_back(runTest< T, VectorTest<T,NaiveMethodInfo,BinSearchExpr> >("naive  vector      " + label, d));
        testresults.push_back(runTest< T, ScalarTest<T,BitMethodInfo,  BitExpr>       >("bit    scalar      " + label, d));
        testresults.push_back(runTest< T, VectorTest<T,BitMethodInfo,  BitExpr>       >("bit    vector SSE2 " + label, d));
        testresults.push_back(runTest< T, ScalarTest<T,DirectMethodInfo, DirectExpr>  >("direct scalar      " + label, d));
        testresults.push_back(runTest< T, VectorTest<T, DirectMethodInfo, DirectExpr> >("direct vector SSE2 " + label, d));
#ifdef USE_MKL
        testresults.push_back(runTest< T, ScalarTest<T, MKLMethodInfo, MKLExpr>  >     ("MKL scalar         " + label, d));
        testresults.push_back(runTest< T, VectorTest<T, MKLMethodInfo, MKLExpr> >      ("MKL vector         " + label, d));
#endif
        std::cout << "\n";
    }

    const size_t nt = testresults.size() / nAvg;

    std::cout << "Average results in " << label << " precision\n";
    for (size_t i = 0; i < nt; ++i) {
        
        // average over different experiments
        for (size_t j = 1; j < nAvg; ++j) 
            testresults[i].second += testresults[i + j*nt].second;
        testresults[i].second /= nAvg;
        
        std::cout << testresults[i].first << ": " 
            << testresults[i].second << " search per sec\t " 
            << testresults[i].second / testresults[0].second * 100 << "%\n";
    }
    std::cout << "\n";
}

int main(int argc, char* argv[])
{
    testAll<float>("single");
    testAll<double>("double");

    std::cout << "R-range: " << rMin << "-" << rMax << "\n";

	return 0;
}

