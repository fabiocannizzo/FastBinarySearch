#pragma once
#include <limits>
#include <cmath>

#ifdef __AVX2__
#define USE_AVX
#endif

#ifndef _MSC_VER
#include <stdint.h>
typedef  int32_t   int32;
typedef uint32_t  uint32;
typedef  int64_t   int64;
typedef uint64_t  uint64;
#else
typedef  __int32   int32;
typedef unsigned __int32  uint32;
typedef  __int64   int64;
typedef unsigned __int64  uint64;
#endif

#define myassert(cond, msg) if (!cond) { cout << "\nassertion failed: " << #cond << ", " << msg << "\n"; exit(1); }

// log2 is not defined in VS2008
#if defined(_MSC_VER) && (_MSC_VER <= 1500)
inline size_t log2 (size_t val) {
    if (val == 1) return 0;
    size_t ret = 0;
    while (val > 1) {
        val >>= 1;
        ret++;
    }
    return ret;
}
#define static_assert(a,b) myassert(a)
#endif

#ifdef _DEBUG
#define DEBUG 
#endif

#ifdef _MSC_VER
#   define FORCE_INLINE __forceinline
#   define NO_INLINE __declspec(noinline)
#else
#   define NO_INLINE __attribute__((noinline))
#   ifdef DEBUG
#       define FORCE_INLINE NO_INLINE
#   else
#       define FORCE_INLINE __attribute__((always_inline)) inline
#   endif
#endif

// nextafter is not defined in VS2008
#if defined(_MSC_VER) && (_MSC_VER <= 1500)
#include <float.h>
inline float mynext(float x)
{
    return _nextafterf(x, std::numeric_limits<float>::max());
}

inline double mynext(double x)
{
    return _nextafter(x, std::numeric_limits<double>::max());
}
inline float myprev(float x)
{
    return _nextafterf(x, -std::numeric_limits<float>::max());
}

inline double myprev(double x)
{
    return _nextafter(x, -std::numeric_limits<double>::max());
}
#else
inline float mynext(float x)
{
    return std::nextafterf(x, std::numeric_limits<float>::max());
}

inline double mynext(double x)
{
    return std::nextafter(x, std::numeric_limits<double>::max());
}
inline float myprev(float x)
{
    return std::nextafterf(x, -std::numeric_limits<float>::max());
}

inline double myprev(double x)
{
    return std::nextafter(x, -std::numeric_limits<double>::max());
}
#endif

template <typename T>
inline T next(T x)
{
    for (int i = 0; i < 4; ++i)
        x = mynext(x);
    return x;
}

template <typename T>
inline T prev(T x)
{
    for (int i = 0; i < 4; ++i)
        x = myprev(x);
    return x;
}