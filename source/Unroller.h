#pragma once

namespace Details {

template <int N>
struct Body
{
    template <class Expr>
    FORCE_INLINE static void iteration(const  Expr& e, uint32 j )
    {
        static const uint32 D = Expr::VecSize;

        e.vectorial( j );
        Body<N-1>::iteration( e, j+D );
    }

};

template <>
struct Body<0>
{
    template <class Expr>
    FORCE_INLINE static void iteration( const Expr& e, uint32 j )
    {
    }
};

} // namespace Details


template <class Expr>
struct Loop
{
    FORCE_INLINE static void loop( const Expr& e, uint32 n )
    {
        static const uint32 M = 4;
        static const uint32 D = Expr::VecSize;

        uint32 j = 0;
        while ( j + (D*M) <= n ) {
            Details::Body<M>::iteration( e, j );
            j += (D*M);
        }
        while ( j + D <= n ) {
            e.vectorial( j );
            j += D;
        }
        while ( j < n ) {
            e.scalar( j );
            j += 1;
        }
    }
};

