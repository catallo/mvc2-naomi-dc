/* port_prefix.h: included before every game file in a host build. It stands in for what SHC
   supplied: its built-in routines, and the single-precision float of -fpu=single. */
#ifndef PORT_PREFIX_H
#define PORT_PREFIX_H

#define HOST_PORT 1
#define _STRICT_UNION_ 1
#define __DEV_TYPE_DC__ 1

/* where the host maps the game's main RAM (the decomp's RAM_ADRS) */
#ifndef MVC2_RAM_BASE
#define MVC2_RAM_BASE 0x0C000000
#endif
#define HOST_RAM_BASE MVC2_RAM_BASE

int host_get_imask( void ) ;
void host_set_imask( int mask ) ;
#define _builtin_get_imask()		host_get_imask()
#define _builtin_set_imask( m )		host_set_imask( m )
#define _builtin_get_fpscr()		0x00040001
#define _builtin_set_fpscr( v )		( ( void )( v ) )
#define _builtin_sqrtf( x )		__builtin_sqrtf( x )
#define _builtin_fabsf( x )		__builtin_fabsf( x )

/* SHC's FMAC, c + a * b with one rounding: port/tools/fmac.py writes the sums SHC fused this way. The
   product of two floats is exact in a double; the double is then rounded to a float, as Flycast does it */
#if defined( __psp__ )
float host_fmac( float a, float b, float c ) ;	/* port/sys/fmac_int.c */
#else
static __inline__ float host_fmac( float a, float b, float c )
{
	return ( float )( ( double )a * b + c ) ;
}
#endif
/* its operands in the order SHC's code takes them, as a call's arguments are in no set order */
#define FMAC( a, b, c )		__extension__ ( { float fmac_a = ( a ) ; float fmac_b = ( b ) ; host_fmac( fmac_a, fmac_b, c ) ; } )

/* a float made unsigned as SHC's code makes it: 2^32 off a value of 2^31 or more, then truncated as an
   int, so a negative wraps (ARM's own conversion stops it at 0); port/data/arm_u32.tsv lists the uses */
static __inline__ unsigned sh_u32( float f )
{
	if( !( 2147483648.0f > f ) ) f -= 4294967296.0f ;
	return ( unsigned )( int )f ;
}

#endif
