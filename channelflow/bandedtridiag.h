/**
 * Banded tridiagonal matrix for Chebyshev-Helmholtz eqn.
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef CHANNELFLOW_BANDEDTRIDIAG_H
#define CHANNELFLOW_BANDEDTRIDIAG_H

#include "cfbasics/cfvector.h"
#include "cfbasics/mathdefs.h"

namespace chflow {

typedef double Real;

// Banded trigdiagonal matrix of form

// DxxxxxA
// xxx
//  xxx
//   xxx
//    xxx
//     xxx
//      xx
// row-major ordering. Pointers a_ and d_ point to elems A and D.

class BandedTridiag {
   public:
    BandedTridiag();
    BandedTridiag(int M);
    BandedTridiag(const BandedTridiag& A);
    BandedTridiag(const std::string& filebase);
    ~BandedTridiag();

    BandedTridiag& operator=(const BandedTridiag& A);

    bool operator==(const BandedTridiag& A) const;
    inline int numrows() const;
    // inline int numcols() const;

    inline Real& band(int j);    // A[0,j];
    inline Real& diag(int i);    // A[i,i];
    inline Real& updiag(int i);  // A[i,i+1]
    inline Real& lodiag(int i);  // A[i,i-1];

    inline const Real& band(int i) const;
    inline const Real& diag(int i) const;
    inline const Real& updiag(int i) const;
    inline const Real& lodiag(int i) const;

    Real& elem(int i, int j);
    const Real& elem(int i, int j) const;
    void ULdecomp();  // no pivoting

    void ULsolve(Vector& b) const;
    void multiply(const Vector& x, Vector& b) const;
    void ULsolveStrided(Vector& b, int offset, int stride) const;
    void multiplyStrided(const Vector& x, Vector& b, int offset, int stride) const;

    void print() const;
    void ULprint() const;
    void test() const;
    void save(const std::string& filebase) const;  // each row of output is i j Aij

   private:
    int M_;          // # rows (square matrix)
    int Mbar_;       // M-1
    Real* a_;        // start of data cfarray for elem storage
    Real* d_;        // address of first diagonal elem. d_ = a_ + Mbar_
    Real* invdiag_;  // inverses of diagonal elements
    bool UL_;        // has UL decomp been done?

    int numNonzeros(int M) const;
    int numRows(int nnz) const;
};

inline Real& BandedTridiag::band(int j) {
    assert(j >= 0 && j < M_);
    return a_[Mbar_ - j];
}

inline Real& BandedTridiag::diag(int i) {
    assert(i >= 0 && i < M_);
    return d_[3 * i];
}

inline Real& BandedTridiag::updiag(int i) {
    assert(i >= 0 && i < M_);
    return d_[3 * i - 1];
}

inline Real& BandedTridiag::lodiag(int i) {
    assert(i >= 0 && i < M_);
    return d_[3 * i + 1];
}

inline const Real& BandedTridiag::band(int j) const {
    assert(j >= 0 && j < M_);
    return a_[Mbar_ - j];
}

inline const Real& BandedTridiag::diag(int i) const {
    assert(i >= 0 && i < M_);
    return d_[3 * i];
}

inline const Real& BandedTridiag::updiag(int i) const {
    assert(i >= 0 && i < M_);
    return d_[3 * i - 1];
}

inline const Real& BandedTridiag::lodiag(int i) const {
    assert(i >= 0 && i < M_);
    return d_[3 * i + 1];
}

}  // namespace chflow

#endif
