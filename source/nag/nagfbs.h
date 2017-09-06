#pragma once

// algo codes
enum FBS_Code {FBS_Direct, FBS_BinSearch};

// error codes
const int FBS_CODE_UNKNOWN      = 1;
const int FBS_DIRECT_UNFEASIBLE = 2;
const int FBS_X_TOO_LARGE       = 3;
const int FBS_NOT_IMPLEMENTED   = 100;

// types
typedef unsigned long fbs_uint_t;  // we can set this to the same definition as NAGINT

#ifndef FBS_FLOAT
#  define FBS_FLOAT double
#endif
typedef FBS_FLOAT fbs_float_t;

// algo infos
struct DirectInfo
{
    fbs_float_t scaler;
};

typedef unsigned long fbs_size_t;

struct FBS_Info
{
    const fbs_float_t *x;  // array x
    fbs_uint_t n;          // size of array x
    FBS_Code code;         // search algorithm code
    bool outLeft;
    bool outRight;

    DirectInfo direct;
};

extern "C"
{
    int FBS_getInfo
        ( const FBS_Code *code
        , const fbs_float_t *x  // in: array X
        , const fbs_uint_t *n   // in: size of array X
        , const bool *outLeft   // check for Z<X0
        , const bool *outRight  // check for Z>=Xn
        , fbs_size_t *mem       // out: workspace size
        , FBS_Info *info        // out: info
        );

    int FBS_setup
        ( const FBS_Info *info  // in: info
        , fbs_uint_t *algodata  // out: must have 'mem' size
        );

    fbs_uint_t FBS_scalar
        ( const fbs_float_t *z   // in: sought value
        , const fbs_uint_t *algodata // in: algorithm data
        );

    int FBS_vectorial
        ( fbs_uint_t *j          // out: sought indices (size m)
        , const fbs_float_t *z   // in: sought values (size m)
        , const fbs_uint_t *m    // in: size of sought values arrays
        , const fbs_uint_t *algodata   // in: algorithm data
        );

} // extern "C"
