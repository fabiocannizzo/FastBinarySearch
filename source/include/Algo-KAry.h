#pragma once

#include "Type.h"

namespace BinSearch {
namespace Details {

// Auxiliary information specifically used in the Ternary binary search
template <InstrSet I, typename T>
struct AlgoKAry
{
    static const size_t np = sizeof(typename InstrFloatTraits<I, T>::vec_t) / sizeof(T);
    static const size_t k = 1 + np;

    typedef FVec<I, T> fVec;
    typedef IVec<I, T> iVec;

    union Data {
        T x[np];
        fVec v;  // unused: just to achieve correct alignment
    };

private:
    /*
        0  1 2 3  4  5  6  7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
        8 17 2 5 11 14 20 23 0 1  3  4  6  7  9 10 12 13 15 16 18 19 21 22 24 25
    */
    void buildLayout
        ( const T *x       // partition x
        , uint32 xLastInd  // size of x
        , uint32 x_blk_ind // offset of block in x
        , uint32 x_blk_sz  // size of x block
        , uint32 depth     // index of first pivot for this block in vector xi
        , uint32 *z_pos    // total number of node at this layer
        )
    {
        if(x_blk_sz == np) {
            for(uint32 j = 0; j < np; ++j)
                data[z_pos[depth]].x[j]   = x[std::min(x_blk_ind+j,xLastInd)];
            ++z_pos[depth];
        }
        else {
            uint32 x_nxt_blk_sz = x_blk_sz / k;  // size of sub-blocks
            uint32 i[np];
            i[0] = x_blk_ind + x_nxt_blk_sz;
            for(uint32 j = 1; j < np; ++j)
               i[j] = i[j-1] + x_nxt_blk_sz + 1;
            for(uint32 j = 0; j < np; ++j)
                data[z_pos[depth]].x[j] = x[std::min(i[j],xLastInd)];
            ++z_pos[depth];
            buildLayout(x, xLastInd, x_blk_ind, x_nxt_blk_sz, depth+1, z_pos);
            for(uint32 j = 1; j < np; ++j)
                buildLayout(x, xLastInd, i[j-1]+1, x_nxt_blk_sz, depth + 1, z_pos);
            buildLayout(x, xLastInd, i[np-1]+1, x_nxt_blk_sz, depth + 1, z_pos);
        }
    }

public:


    AlgoKAry(const T* x, const uint32 n)
    {
        uint32 d = 0; // depth
        uint32 z_pos[32] = {};  // next index to be populated for each layer in array data
        uint32 p = 1;  // number of nodes at layer d
        uint32 nx = np*p; // number of nodes in an over-dimensioned parfect tree
        while(nx < n) {
            z_pos[d+1] = z_pos[d] + p;
            ++d;
            p *= k;
            nx += np*p;
        }
        data.resize(nx/np);
        di = &data[0];

        max_depth = d;
        grp_size.resize(max_depth+1);
        for (uint32 i = 0, ng = nx/k; i <= max_depth; ++i, ng/=k)
            grp_size[i] = ng+1;
        buildLayout(x, n-1, 0, nx, 0, z_pos);
        /*
        for (size_t i=0; i<n; ++i)
            std::cout << x[i] << ", ";
        std::cout << "\n";
        for (size_t i=0; i<nx/np; ++i)
            for (size_t j=0; j<np; ++j)
                std::cout << di[i].x[j] << ", ";
        std::cout << "\n";
        */
    }


    FORCE_INLINE
    uint32 scalar(T z) const
    {
        fVec vz(z);
        uint32 depth = 0;

        const fVec* pvx = reinterpret_cast<const fVec*>(di);
        fVec vx(pvx[0]);
        iVec ge = vz >= vx;
        int gecnt = ge.countbit();
        uint32 i = grp_size[depth] * gecnt;
        uint32 p = 1 + gecnt;

        while(depth++ < max_depth) {
            vx = pvx[p];
            ge = vz >= vx;
            gecnt = ge.countbit();
            i += grp_size[depth] * gecnt;
            p = 1 + gecnt + k * p;
        }
        return i-1;
    }

private:
    const Data *di;
    AlignedVec<Data> data;
    AlignedVec<uint32> grp_size;
    uint32 max_depth;
};

template <>
struct AlgoScalarBase<double, Ternary> : AlgoKAry<SSE,double>
{
    AlgoScalarBase(const double* x, const uint32 n) : AlgoKAry<SSE,double>(x,n)
    {
    }
};


template <>
struct AlgoScalarBase<float, Pentary> : AlgoKAry<SSE, float>
{
    AlgoScalarBase(const float* x, const uint32 n) : AlgoKAry<SSE, float>(x, n)
    {
    }
};

#ifdef USE_AVX

template <>
struct AlgoScalarBase<double, Pentary> : AlgoKAry<AVX, double>
{
    AlgoScalarBase(const double* x, const uint32 n) : AlgoKAry<AVX, double>(x, n)
    {
    }
};


template <>
struct AlgoScalarBase<float, Nonary> : AlgoKAry<AVX, float>
{
    AlgoScalarBase(const float* x, const uint32 n) : AlgoKAry<AVX, float>(x, n)
    {
    }
};

#endif

} // namespace Details
} // namespace BinSearch
