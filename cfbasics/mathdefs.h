/**
 * some small mathematical conveniences.
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef CHANNELFLOW_MATHDEFS_H
#define CHANNELFLOW_MATHDEFS_H

#include "channelflow/config.h"

#include <arpa/inet.h>

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace chflow {

#ifndef HAVE_DRAND48
#define drand48() rand() / (double(RAND_MAX));
#endif

#ifdef __CYGWIN__
#define drand48() rand() / (double(RAND_MAX));
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __CYGWIN__
#define NAN 0. / 0.
#endif
// Define DEBUG to true if NDEBUG isn't set or if it's set to false
#ifndef DEBUG
#ifndef NDEBUG
#define DEBUG 1
#else
#if NDEBUG
#define DEBUG 0
#else
#define DEBUG 1
#endif /* if NDEBUG */
#endif /* ifndef NDEBUG */
#endif /* ifndef DEBUG */

typedef double Real;
typedef std::complex<double> Complex;

#ifdef HAVE_MPI
typedef ptrdiff_t lint;

#else
typedef int lint;
typedef int MPI_Comm;
static const int MPI_COMM_WORLD = 0;
#endif

// forward declaration
void cferror(const std::string& message);
// typedef unsigned int uint;

enum fieldstate { Physical, Spectral };
enum parity { Even, Odd };

const int REAL_DIGITS = 17;
const int REAL_IOWIDTH = 24;
// SWIG barfs on const Complex (0.0,1.0) for some reason
const Complex I = Complex(0.0, 1.0);
const Real pi = M_PI;

inline int kronecker(int m, int n);
inline int square(int x);
inline int cube(int x);
inline Real square(Real x);
inline Real cube(Real x);
inline Real nr_sign(Real a, Real b);
inline void swap(int& a, int& b);
inline void swap(Real& a, Real& b);
inline void swap(Complex& a, Complex& b);

inline int Greater(int a, int b);
inline int lesser(int a, int b);
inline Real Greater(Real a, Real b);
inline Real lesser(Real a, Real b);

inline int iround(Real x);        // round to closest int
inline int intpow(int n, int p);  // named this way to avoid real,int -> int,int casting
// Real pow(Real n, int p);
// Real pow(Real n, Real p);

// Real fmod(Real x, Real m);   // return y s.t. 0<=y<m & y+N*m==x
inline Real pythag(Real a, Real b);  // sqrt(a^2 + b^2) avoiding over/underflow
inline Real spythag(Real Lx, Real Lxtarg, Real Lz, Real Lztarg, Real phi);
inline bool isPowerOfTwo(int n);

inline Real Re(const Complex& z);  // Real part
inline Real Im(const Complex& z);  // Imag part
// inline Real abs(const Complex& z);// sqrt(a^2 + b^2), provided by std lib
inline Real abs2(const Complex& z);  // a^2 + b^2 for a + b i

inline Real randomReal(Real a = 0, Real b = 1);  // uniform in [a, b]
inline Complex randomComplex();                  // gaussian about zero

inline std::ostream& operator<<(std::ostream& os, Complex z);

inline std::ostream& operator<<(std::ostream& os, fieldstate f);
inline std::istream& operator>>(std::istream& is, fieldstate& f);

inline int kronecker(int m, int n) { return (m == n) ? 1 : 0; }

// inline Real abs(Real x) {return fabs(x);}
inline Real square(Real x) { return x * x; }
inline Real cube(Real x) { return x * x * x; }
inline Real nr_sign(Real a, Real b) { return (b >= 0.0) ? fabs(a) : -fabs(a); }

inline int square(int x) { return x * x; }
inline int cube(int x) { return x * x * x; }
inline void swap(int& a, int& b) {
    int tmp = a;
    a = b;
    b = tmp;
}
inline void swap(Real& a, Real& b) {
    Real tmp = a;
    a = b;
    b = tmp;
}
inline void swap(Complex& a, Complex& b) {
    Complex tmp = a;
    a = b;
    b = tmp;
}

// Inline replacements for >? and <? GNUisms.
inline int Greater(int a, int b) { return (a > b) ? a : b; }
inline int lesser(int a, int b) { return (a < b) ? a : b; }
inline Real Greater(Real a, Real b) { return (a > b) ? a : b; }
inline Real lesser(Real a, Real b) { return (a < b) ? a : b; }
inline Real Re(const Complex& z) { return z.real(); }
inline Real Im(const Complex& z) { return z.imag(); }
inline Real abs2(const Complex& z) {
    return norm(z);  // NOTE std lib norm is misnamed! Returns a^2+b^2 for a+bi.
}

inline Real pythag(Real a, Real b) {
    Real absa = fabs(a);
    Real absb = fabs(b);

    if (absa > absb)
        return absa * (sqrt(1.0 + square(absb / absa)));
    else if (absb > absa)
        return absb * (sqrt(1.0 + square(absa / absb)));
    else
        return 0.0;
}

// distance with a sign
inline Real spythag(Real Lx, Real Lxtarg, Real Lz, Real Lztarg, Real phi) {
    Real ex = cos(phi);
    Real ez = sin(phi);
    Real dLx = Lxtarg - Lx;
    Real dLz = Lztarg - Lz;
    Real dist = pythag(dLx, dLz);
    Real projection = ex * dLx + ez * dLz;
    int sign = 0;
    if (projection > 0)
        sign = 1;
    else if (projection < 0)
        sign = -1;
    return sign * dist;
}

inline bool isPowerOfTwo(int N) {
    int onbits = 0;
    for (unsigned int i = 0; i < int(8 * sizeof(int)); ++i) {
        onbits += N % 2;
        N = N >> 1;
        if (onbits > 1)
            return false;
    }
    return true;
}

// int roundReal(Real x) {
//   int ix = int(x);
//   Real dm = abs(x-(ix-1)); // m==minus:  distance x to (xcast minus 1)
//   Real dc = abs(x-ix);     // c==center: distance x to (xcast)
//   Real dp = abs(x-(ix+1)); // p==plus:   distance x to (xcast plus 1)

//   return ((dm<dp) ? ((dm<dc) ? ix-1 : ix) : ((dc<dp) ? ix : ix+1));
// }

inline int iround(Real x) { return int(x > 0.0 ? x + 0.5 : x - 0.5); }

inline int intpow(int x, int n) {
    if (n < 0)
        cferror("int pow(int, int) : can't do negative exponents, use Real pow(Real,int)");
    switch (n) {
        case 0:
            return 1;
        case 1:
            return x;
        case 2:
            return x * x;
        case 3:
            return x * x * x;
        case 4: {
            int xx = x * x;
            return xx * xx;
        }
        default: {
            // a*x^n is invariant and n is decreasing in this loop
            int a = 1;
            while (n > 0) {
                if (n % 2 == 0) {
                    x *= x;  //     x -> x^2
                    n /= 2;  //     n -> n/2
                }            // a x^n -> a (x^2)^(n/2) = a x^n
                else {
                    a *= x;  //     a -> ax
                    --n;     //     n -> n-1
                }            // a x^n -> (a x) x^(n-1) = a x^n
            }
            return a;
        }
    }
}

/*****************************
Real fmod(Real x, Real modulus) {
  if (modulus < 0) {
    cerr << "error in fmod(x,m) : m should be greater or equal to 0." << endl;
    exit(1);
  }
  if (modulus == 0)
    return 0.0;

  if (x<0)
    x += modulus*(int(fabs(x)/modulus)+1);
  if (x>0)
    x -= modulus*int(x/modulus);

  // Just in case previous algorithm slips up.
  while (x<0)
    x += modulus;
  while (x>=modulus)
    x -= modulus;
  return x;
}
************************************/

inline Real randomReal(Real a, Real b) {
#ifdef HAVE_MPI
    int taskid = 0;
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized)
        MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    srand48((long int)(taskid + 1) * 1000 * drand48());
    return a + (b - a) * drand48();
#else
    return a + (b - a) * drand48();
#endif
}

inline Complex randomComplex() {
    Real a;
    Real b;
    Real r2;
    do {
        a = randomReal(-1, 1);
        b = randomReal(-1, 1);
        r2 = a * a + b * b;
    } while (r2 >= 1.0 || r2 == 0.0);
    return sqrt(-log(r2) / r2) * Complex(a, b);
}

inline std::ostream& operator<<(std::ostream& os, Complex z) {
    os << '(' << Re(z) << ", " << Im(z) << ')';
    return os;
}

inline std::ostream& operator<<(std::ostream& os, fieldstate s) {
    os << ((s == Spectral) ? 'S' : 'P');
    return os;
}

inline std::istream& operator>>(std::istream& is, fieldstate& s) {
    char c = ' ';
    while (c == ' ')
        is >> c;
    switch (c) {
        case 'P':
            s = Physical;
            break;
        case 'S':
            s = Spectral;
            break;
        default:
            std::cerr << "read fieldstate error: unknown fieldstate " << c << std::endl;
            s = Spectral;
            assert(false);
    }
    return is;
}

// inline Complex conjugate(const Complex& z) {
// return Complex(Re(z), -Im(z));
//}

// Beware the standard-library norm(complex) function! It returns the
// squared norm!
// inline Real true_norm(const Complex& z) {
//  return pythag(Re(z), Im(z));
//}
// inline Real norm2(const Complex& z) {
//  return square(Re(z)) + square(Im(z));
//}
//#ifdef WIN32
//#undef assert
//#define assert(expr) (if(!expr) {cout << "Assertion failed. "<<endl; abort();})
//#endif

}  // namespace chflow
#endif /* MATHDEFS_H */
