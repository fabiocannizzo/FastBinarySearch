#include <emmintrin.h>
#include <smmintrin.h>

#include "Type.h"

#define ATTR __forceinline static

template <class T>
struct SIMD {};

template <class T>
struct I128 {};

template <class T>
struct SIMD0 {};


struct I128Base
{
    I128Base() {}
    I128Base( __m128i v) : vec( v )						{}

	operator __m128i&() { return vec; }
	operator const __m128i&() const { return vec; }

    int32 get0() const { return _mm_cvtsi128_si32( vec ); }

	void assignIf( __m128i& val, __m128i& mask ) { vec = _mm_blendv_epi8(vec, val, mask); }

protected:
    __m128i vec;
};

template <>
struct I128<float> : I128Base
{
    I128() {}
	I128( int32 i )  : I128Base( _mm_set1_epi32( i ) )	{}
    I128( __m128i v) : I128Base( v )						    {}

    void setN( int32 i ) { vec = _mm_set1_epi32( i ); }

	//int32 get1() const { return _mm_cvtsi128_si32( _mm_shuffle_epi32( vec, 1 ) ); }
	//int32 get2() const { return _mm_cvtsi128_si32( _mm_shuffle_epi32( vec, 2 ) ); } 
	//int32 get3() const { return _mm_cvtsi128_si32( _mm_shuffle_epi32( vec, 3 ) ); }
	int32 get1() const { return _mm_extract_epi32(vec, 1); }
	int32 get2() const { return _mm_extract_epi32(vec, 2); }
	int32 get3() const { return _mm_extract_epi32(vec, 3); }

	void store( uint32 *pi ) const { _mm_storeu_si128( reinterpret_cast<__m128i*>(pi), vec ); }
};

template <>
struct I128<double> : I128Base
{
    I128() {}
	I128( int32 i )  : I128Base( _mm_set1_epi64x( i ) )	{}
    I128( __m128i v) : I128Base( v )						    {}

    void setN( int32 i ) { vec = _mm_set1_epi64x( i ); }

	//int32 get1() const { return _mm_cvtsi128_si32( _mm_shuffle_epi32( vec, 2 ) ); };
	int32 get1() const { return _mm_extract_epi32(vec, 2); }

	void store( uint32 *pi ) const
	{ 
		pi[0] = get0();
		pi[1] = get1();
	}
};

__forceinline __m128i operator>> (const I128Base& a, int n) { return _mm_srli_epi32(a, n); }

__forceinline __m128i operator&   (const I128Base& a, const I128Base& b ) { return _mm_and_si128( a, b ); }
__forceinline __m128i operator|   (const I128Base& a, const I128Base& b ) { return _mm_or_si128( a, b ); }



template <>
struct SIMD0<float>
{
    SIMD0() {}
	SIMD0( float f ) : vec( _mm_load_ss( &f ) ) {}
	SIMD0( const __m128& v ): vec( v ) {}

	float get0() const { return _mm_cvtss_f32( vec ); }
	operator __m128&() { return vec; }
	operator const __m128&() const { return vec; }

private:
    __m128 vec;
};

template <>
struct SIMD<float>
{
    SIMD() {}
	SIMD( float f ) : vec( _mm_set1_ps( f ) ) {}
	SIMD( const float *v ) : vec( _mm_loadu_ps( v ) ) {}
    SIMD( const __m128& v) : vec(v) {}

    void set0( float f  ) { vec = _mm_load_ss( &f ); }
    void setN( float f  ) { vec = _mm_set1_ps( f ); }

	void setidx( const float *xi, const I128<float>& idx )
	{ 
		uint32 i0 = idx.get0();
		uint32 i1 = idx.get1();
		uint32 i2 = idx.get2();
		uint32 i3 = idx.get3();
		vec = _mm_set_ps( xi[i3], xi[i2], xi[i1], xi[i0] );
	}

	operator __m128&() { return vec; }
	operator const __m128&() const { return vec; }

	float get0() const { return _mm_cvtss_f32( vec ); }
	float get1() const { return _mm_cvtss_f32( _mm_shuffle_ps( vec, vec, 1 ) ); }
	float get2() const { return _mm_cvtss_f32( _mm_shuffle_ps( vec, vec, 2 ) ); }
	float get3() const { return _mm_cvtss_f32( _mm_shuffle_ps( vec, vec, 3 ) ); }

private:
    __m128 vec;
};

__forceinline __m128 operator+   (const SIMD0<float>& a, const SIMD0<float>& b ) { return _mm_add_ss( a, b ); }
__forceinline __m128 operator-   (const SIMD0<float>& a, const SIMD0<float>& b ) { return _mm_sub_ss( a, b ); }
__forceinline __m128 operator*   (const SIMD0<float>& a, const SIMD0<float>& b ) { return _mm_mul_ss( a, b ); }
__forceinline __m128 operator/   (const SIMD0<float>& a, const SIMD0<float>& b ) { return _mm_div_ss( a, b ); }
__forceinline __m128 floor       (const SIMD0<float>& a ) { return _mm_floor_ss( a, a ); }
__forceinline int atoi           (const SIMD0<float>& a ) { return _mm_cvttss_si32(a); }
__forceinline __m128i operator>  (const SIMD0<float>& a, const SIMD0<float>& b ) { return _mm_castps_si128( _mm_cmpgt_ss( a, b ) ); }

__forceinline __m128 operator-   (const SIMD<float>& a, const SIMD<float>& b ) { return _mm_sub_ps( a, b ); }
__forceinline __m128 operator/   (const SIMD<float>& a, const SIMD<float>& b ) { return _mm_div_ps( a, b ); }
__forceinline __m128i atoi           (const SIMD<float>& a ) { return _mm_cvttps_epi32(a); }
__forceinline __m128i operator<=  (const SIMD<float>& a, const SIMD<float>& b ) { return _mm_castps_si128( _mm_cmple_ps( a, b ) ); }
__forceinline __m128i operator>=  (const SIMD<float>& a, const SIMD<float>& b ) { return _mm_castps_si128( _mm_cmpge_ps( a, b ) ); }
__forceinline __m128i operator<  (const SIMD<float>& a, const SIMD<float>& b) { return _mm_castps_si128(_mm_cmplt_ps(a, b)); }

template <>
struct SIMD0<double>
{
    SIMD0() {}
	SIMD0( double f ) : vec( _mm_load_sd( &f ) ) {}
	SIMD0( const __m128d& v ): vec( v ) {}

	double get0() const { return _mm_cvtsd_f64( vec ); }
	operator __m128d&() { return vec; }
	operator const __m128d&() const { return vec; }

private:
    __m128d vec;
};


template <>
struct SIMD<double>
{
    SIMD() {}
	SIMD( double d )		: vec( _mm_set1_pd( d ) )	{}
	SIMD( const double *v ) : vec( _mm_loadu_pd( v ) )	{}
    SIMD( const __m128d& v)		: vec( v )					{}

    void set0( double f  ) { vec = _mm_load_sd( &f ); }
    void setN( double f  ) { vec = _mm_set1_pd( f ); }

	void setidx( const double *xi, const I128<double>& idx ) { vec = _mm_set_pd( xi[idx.get1()], xi[idx.get0()] ); }

	operator __m128d&() { return vec; }
	operator const __m128d&() const { return vec; }

    double get0() const { return _mm_cvtsd_f64( vec ); }
	double get1() const { return _mm_cvtsd_f64( _mm_shuffle_pd( vec, vec, 1 ) ); }; 

private:
    __m128d vec;
};

__forceinline __m128d operator+   (const SIMD0<double>& a, const SIMD0<double>& b ) { return _mm_add_sd( a, b ); }
__forceinline __m128d operator-   (const SIMD0<double>& a, const SIMD0<double>& b ) { return _mm_sub_sd( a, b ); }
__forceinline __m128d operator*   (const SIMD0<double>& a, const SIMD0<double>& b ) { return _mm_mul_sd( a, b ); }
__forceinline __m128d operator/   (const SIMD0<double>& a, const SIMD0<double>& b ) { return _mm_div_sd( a, b ); }
__forceinline __m128d floor       (const SIMD0<double>& a ) { return _mm_floor_sd( a, a ); }
__forceinline int atoi      (const SIMD0<double>& a ) { return _mm_cvttsd_si32(a); }
__forceinline __m128i operator>  (const SIMD0<double>& a, const SIMD0<double>& b ) { return _mm_castpd_si128( _mm_cmpgt_sd( a, b ) ); }

__forceinline __m128d operator-   (const SIMD<double>& a, const SIMD<double>& b ) { return _mm_sub_pd( a, b ); }
__forceinline __m128d operator/   (const SIMD<double>& a, const SIMD<double>& b ) { return _mm_div_pd( a, b ); }
__forceinline __m128i atoi        (const SIMD<double>& a ) { return _mm_cvttpd_epi32(a); }
__forceinline __m128i operator<=  (const SIMD<double>& a, const SIMD<double>& b ) { return _mm_castpd_si128( _mm_cmple_pd( a, b ) ); }
__forceinline __m128i operator<   (const SIMD<double>& a, const SIMD<double>& b) { return _mm_castpd_si128(_mm_cmplt_pd(a, b)); }
__forceinline __m128i operator>=  (const SIMD<double>& a, const SIMD<double>& b ) { return _mm_castpd_si128( _mm_cmpge_pd( a, b ) ); }

