#pragma once

namespace Details {

template <typename Expr, int N>
struct Body
{
    typedef typename Expr::type T;
    template <uint32 D>
    FORCE_INLINE static void iteration(const Expr& e, uint32 *ri, const T* zi)
    {
        e.vectorial(ri, zi);
        Body<Expr,N-1>::template iteration<D>( e, ri+D, zi+D );
    }

};

template <typename Expr>
struct Body<Expr,0>
{
    typedef typename Expr::type T;
    template <uint32 D>
    FORCE_INLINE static void iteration( const Expr& e, uint32 *ri, const T* zi)
    {
    }
};

} // namespace Details

template <InstrSet I, class Expr>
struct Loop
{
    typedef typename Expr::type T;
    
    FORCE_INLINE static void loop( const Expr& e, uint32 *ri, const T* zi, uint32 n)
    {
        static const uint32 M = 4;
        static const uint32 D = sizeof(typename InstrFloatTraits<I,T>::vec_t)/sizeof(T);

        uint32 j = 0;
        while ( j + (D*M) <= n ) {
            Details::Body<Expr,M>::template iteration<D>( e, ri+j, zi+j );
            j += (D*M);
        }
        while ( j + D <= n ) {
            e.vectorial( ri+j, zi+j );
            j += D;
        }
        while ( j < n ) {
            ri[j] = e.scalar( zi[j] );
            j += 1;
        }
    }
};
