// get CPUID capability 



template <class Expr>
class Loop
{
	static const uint32 M = 4;
	static const uint32 D = Expr::VecSize;

	template <int N>
	__forceinline static void body(const  Expr& e, uint32 j )
	{
		e.vectorial( j );
		e.body<N-1>( e, j+D );
	}

	template <>
	__forceinline static void body<0>( const Expr& e, uint32 j )
	{
	}

public:
	__forceinline static void loop( const Expr& e, uint32 n )
	{
		uint32 j = 0;
		while ( j + (D*M) <= n ) {
			body<M>( e, j );
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