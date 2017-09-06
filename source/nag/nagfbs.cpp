#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <new>

#include "nagfbs.h"

#include "Algo-ClassicOffset.h"
#include "Algo-Direct.h"
#include "Algo-Direct2.h"
#include "Algo-DirectCache.h"

using namespace BinSearch;
using namespace std;

// FIXME: to be defined based on compile macros (0 for C, 1 for Fortran)
#ifdef FORTRAN_LIB
const fbs_uint_t indexOffset = 1;
#else
const fbs_uint_t indexOffset = 0;
#endif


// FIXME: to be defined automatically based on SIMD compilation options
const InstrSet I = SSE;


template <FBS_Code C> struct AlgoMapping;
template <> struct AlgoMapping<FBS_Direct>     { static const Algos A = DirectCache;   };
template <> struct AlgoMapping<FBS_BinSearch>  { static const Algos A = ClassicOffset; };

struct IAlgo
{
    virtual fbs_uint_t scalar(fbs_float_t z) const = 0;
    virtual void vectorial(fbs_uint_t *j, const fbs_float_t *z, fbs_uint_t m) const = 0;
};

template <typename Algo, bool L, bool R>
struct AlgoDispatcher : IAlgo, Algo
{
    AlgoDispatcher(const fbs_float_t *x, uint32 n, const typename Algo::Data& d)
        : Algo(d), x0(x[0]), xN(x[n-1]), N(n)
    {
    }

    virtual fbs_uint_t scalar(fbs_float_t z) const
    {
        fbs_uint_t res;
        if (!L || z >= x0)
            if (!R || z < xN)
                res = indexOffset + Algo::scalar(z);
            else
                res = indexOffset + N;
        else
            res = indexOffset + std::numeric_limits<fbs_uint_t>::max();
        return res;
    }


    virtual void vectorial(fbs_uint_t *pr, const fbs_float_t *pz, fbs_uint_t m) const
    {
        if (!L && !R) {
            while(m) {
                uint32 niter = (uint32) std::min<fbs_uint_t>(numeric_limits<uint32>::max(), m);
                uint32 *pr32 = reinterpret_cast<uint32 *>(pr);
                if (sizeof(fbs_uint_t)==8)
                    pr32 += niter;
                Details::Loop<fbs_float_t,Algo>::loop(*this, pr32, pz, niter);

                for(uint32 j = 0; j < niter; ++j)
                    if (sizeof(fbs_uint_t)==8)  // 64 bits indices
                        pr[j] = indexOffset + pr32[j];
                    else
                        pr[j] += indexOffset;

                pr += niter;
                pz += niter;
                m  -= niter;
            }
        }
        else {
            const uint32 nElem = Algo::nElem;
            const uint32 idealbufsize = 256;
            const uint32 bufsize = nElem * (idealbufsize / nElem + ((idealbufsize % nElem) ? 1 : 0));
            fbs_float_t databuf[bufsize];
            uint32 resbuf[bufsize];
            uint32 indexbuf[bufsize];

            fbs_uint_t vl = indexOffset + std::numeric_limits<fbs_uint_t>::max();
            fbs_uint_t vr = indexOffset + N;

            while(m) {
                uint32 cnt = 0;
                uint32 niter = (uint32) std::min<fbs_uint_t>(bufsize, m);
                for (uint32 j = 0; j < niter; ++j) {
                    fbs_float_t z = pz[j];
                    // FIXME: use SSE2?
                    if (!L || z >= x0)
                        if (!R || z < xN) {
                            databuf[cnt] = z;
                            indexbuf[cnt] = j;
                            ++cnt;
                        }
                        else
                            pr[j] = vr;
                    else
                        pr[j] = vl;
                }

                // FIXME: merge these two loops
                Details::Loop<fbs_float_t,Algo>::loop(*this, resbuf, databuf, cnt);
                for (uint32 j = 0; j < cnt; ++j)
                    pr[indexbuf[j]] = indexOffset + resbuf[j];

                pr += niter;
                pz += niter;
                m -= niter;
            }
        }
    }

    Details::CondData<fbs_float_t,L> x0;
    Details::CondData<fbs_float_t,R> xN;
    Details::CondData<fbs_uint_t,R> N;
};


template <FBS_Code C, bool Left, bool Right>
struct AlgoTraits
{
    typedef Details::BinAlgoBase<I,fbs_float_t,AlgoMapping<C>::A> raw_algo_t;
    typedef AlgoDispatcher<raw_algo_t,Left,Right> algo_t;
    typedef typename raw_algo_t::Data data_t;
};


const char CacheLineSize = 64;

template <bool L, bool R>
int _FBS_getInfo(FBS_Code code, const fbs_float_t *x, fbs_uint_t n, fbs_size_t *mem, FBS_Info *info)
{
  if (n >= numeric_limits<uint32>::max()-2)
    return FBS_X_TOO_LARGE;
  
  info->code = code;
  info->x = x;
  info->n = static_cast<uint32>(n);

  fbs_size_t sz = 0;
  
  switch(code) {
  case FBS_Direct:
    {
      typedef typename AlgoTraits<FBS_Direct,L,R>::algo_t algo_t;
      // FIXME: encapsulate
      fbs_float_t H, k0;
      try {
	H = algo_t::computeH(x, info->n).H;
      } catch(...) {
	return FBS_DIRECT_UNFEASIBLE;
      }
      // FIXME: encapsulate
      k0 = algo_t::fun_t::cst0(H, x[0]);
      uint32 nb = 1 + algo_t::fun_t::f(H, k0, x[n-1]);
      info->direct.scaler = H;
      
      fbs_size_t nb1 = algo_t::bucketvec_t::nBytes(nb);
      fbs_size_t nb2 = sizeof(algo_t) + CacheLineSize;
      sz = nb1 + nb2;
    }
    break;
  case FBS_BinSearch:
    {
      typedef typename AlgoTraits<FBS_BinSearch,L,R>::algo_t algo_t;
      sz = sizeof(algo_t) + CacheLineSize;
    }
    break;
  default:
    return FBS_CODE_UNKNOWN;
  };
  sz++;
  
  *mem = (sz / sizeof(fbs_uint_t)) + ((sz % sizeof(fbs_uint_t))? 1: 0);

  return 0;
}

extern "C"
int FBS_getInfo
	( const FBS_Code *code
    , const fbs_float_t *x
    , const fbs_uint_t *n
    , const bool *outLeft
    , const bool *outRight
    , fbs_size_t *mem
    , FBS_Info *info
    )
{
    info->outLeft = *outLeft;
    info->outRight = *outRight;
    if (*outLeft)
        if (*outRight)
            return _FBS_getInfo<true,true>(*code, x, *n, mem, info);
        else
            return _FBS_getInfo<true,false>(*code, x, *n, mem, info);
    else
        if (*outRight)
            return _FBS_getInfo<false,true>(*code, x, *n, mem, info);
        else
            return _FBS_getInfo<false,false>(*code, x, *n, mem, info);
}

template <typename A>
A *alignAlgoClass(char *workspace)
{
    // we align the algo struct on a cache line and store its offset with respect to workspace in workspace[0],
    // i.e. the class is stored at address (workspace+workspace[0])
    workspace[0] = (CacheLineSize - static_cast<char>(reinterpret_cast<uint64>(workspace+1) % CacheLineSize)) % CacheLineSize;
    return reinterpret_cast<A *>(workspace + workspace[0]);
}

template <bool L, bool R>
int _FBS_setup(const FBS_Info *info, char *workspace)
{
    uint32 n = static_cast<uint32>(info->n);
    const fbs_float_t *x = info->x;

    switch(info->code) {
        case FBS_Direct:
            {
                typedef typename AlgoTraits<FBS_Direct,L,R>::algo_t algo_t;

                // partition workspace
                algo_t *clsptr = alignAlgoClass<algo_t>(workspace);

                // we compute the first address after the end of the class and align and add an additional offset as needed for alignment
                char *bucketws = reinterpret_cast<char *>(clsptr+1);
                bucketws += algo_t::bucketvec_t::shiftAmt(bucketws);
                typedef typename algo_t::bucket_t bucket_t;
                bucket_t *pb = reinterpret_cast<bucket_t *>(bucketws);

                // initialize data
                typename algo_t::Data dinfo(x, n, info->direct.scaler, pb);

                // init algo class
                new (clsptr) algo_t(x, n, dinfo);
            }
            break;
        case FBS_BinSearch:
            {
                typedef typename AlgoTraits<FBS_BinSearch,L,R>::algo_t algo_t;

                // partition workspace
                algo_t *clsptr = alignAlgoClass<algo_t>(workspace);

                // initialize data
                typename algo_t::Data dinfo(x, n);

                // init algo class
                new (clsptr) algo_t(x, n, dinfo);
            }
            break;
        default:
            return FBS_CODE_UNKNOWN;
    };
    return 0;
}

extern "C"
int FBS_setup(const FBS_Info *info, fbs_uint_t *pc)
{
    char *workspace = reinterpret_cast<char*>(pc);
    if (info->outLeft)
        if (info->outRight)
            return _FBS_setup<true,true>(info, workspace);
        else
            return _FBS_setup<true,false>(info, workspace);
    else
        if (info->outRight)
            return _FBS_setup<false,true>(info, workspace);
        else
            return _FBS_setup<false,false>(info, workspace);
}


extern "C"
fbs_uint_t FBS_scalar(const fbs_float_t *z, const fbs_uint_t *info)
{
    const char *pc = reinterpret_cast<const char*>(info);
    return reinterpret_cast<const IAlgo*>(pc+pc[0])->scalar(*z);
}

extern "C"
int FBS_vectorial(fbs_uint_t *j, const fbs_float_t *z, const fbs_uint_t* m, const fbs_uint_t *info)
{
    const char *pc = reinterpret_cast<const char*>(info);
    reinterpret_cast<const IAlgo*>(pc+pc[0])->vectorial(j, z, *m);
    return 0;
}



// Auto rename function names to handle all sorts of Fortran name-mangling
// This uses the same code as in the METIS library
extern "C"
{
#define FRENAME(name, dargs, cargs, name1, name2, name3, name4)   \
  int name1 dargs;                                                \
  int name1 dargs { return name cargs; }                          \
  int name2 dargs;                                                \
  int name2 dargs { return name cargs; }                          \
  int name3 dargs;                                                \
  int name3 dargs { return name cargs; }                          \
  int name4 dargs;                                                \
  int name4 dargs { return name cargs; }


  FRENAME
  (
	  FBS_getInfo,
	  (const FBS_Code *code, const fbs_float_t *x, const fbs_uint_t *n, const bool *outLeft, const bool *outRight, fbs_size_t *mem, FBS_Info *info),
	  (code, x, n, outLeft, outRight, mem, info),
	  FBS_GETINFO,
	  fbs_getinfo,
	  fbs_getinfo_,
	  fbs_getinfo__
   )

  FRENAME
  (
	  FBS_setup,
	  (const FBS_Info *info, fbs_uint_t *algodata),
	  (info, algodata),
	  FBS_SETUP,
	  fbs_setup,
	  fbs_setup_,
	  fbs_setup__
   )

  FRENAME
  (
	  FBS_scalar,
	  (const fbs_float_t *z, const fbs_uint_t *algodata),
	  (z, algodata),
	  FBS_SCALAR,
	  fbs_scalar,
	  fbs_scalar_,
	  fbs_scalar__
   )

  FRENAME
  (
	  FBS_vectorial,
	  (fbs_uint_t *j, const fbs_float_t *z, const fbs_uint_t *m, const fbs_uint_t *algodata),
	  (j, z, m, algodata),
	  FBS_VECTORIAL,
	  fbs_vectorial,
	  fbs_vectorial_,
	  fbs_vectorial__
   )


}

