/**
 * Banded tridiagonal matrix for Chebyshev-Helmholtz eqn.
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "channelflow/bandedtridiag.h"

#include <fstream>
#include <iomanip>

using namespace std;

namespace chflow {

const Real EPSILON = 1e-13;

BandedTridiag::~BandedTridiag() {
    d_ = 0;
    delete[] invdiag_;
    invdiag_ = 0;
    delete[] a_;
    a_ = 0;
}

BandedTridiag::BandedTridiag() : M_(0), Mbar_(-1), a_(0), d_(0), invdiag_(0), UL_(false) {}

BandedTridiag::BandedTridiag(int M)
    : M_(M), Mbar_(M - 1), a_(new Real[4 * M - 2]), d_(a_ + Mbar_), invdiag_(new Real[M]), UL_(false) {
    assert(M >= 0);
    for (int i = 0; i < 4 * M - 2; ++i)
        a_[i] = 0.0;
    for (int i = 0; i < M_; ++i)
        invdiag_[i] = 0.0;
}

BandedTridiag::BandedTridiag(const BandedTridiag& A)
    : M_(A.M_), Mbar_(A.Mbar_), a_(new Real[4 * M_ - 2]), d_(a_ + Mbar_), invdiag_(new Real[M_]), UL_(false) {
    for (int i = 0; i < 4 * M_ - 2; ++i)
        a_[i] = A.a_[i];
    for (int i = 0; i < M_; ++i)
        invdiag_[i] = 0.0;
}

BandedTridiag::BandedTridiag(const string& filebase) : M_(0), Mbar_(-1), a_(0), d_(0), invdiag_(0), UL_(false) {
    ifstream is;
    string filename = ifstreamOpen(is, filebase, ".asc");

    // Read in header. Form is "% N U" for a banded tridiag with 8 non-zero elems,
    // U==0 => UL decomp has not been performed. U==1 indicates it has.
    char c;
    is >> c;
    if (c != '%') {
        string message("BandedTridiag(filebase): bad header in file ");
        message += filename;
        cerr << message << endl;
        assert(false);
    }
    is >> M_;
    int ul;
    is >> ul;
    UL_ = (ul == 0) ? false : true;

    a_ = new Real[4 * M_ - 2];
    int n;  // FOR-SCOPE BUG
    for (n = 0; n < 4 * M_ - 2; ++n)
        a_[n] = 0.0;

    Mbar_ = (M_ - 1);
    d_ = a_ + Mbar_;

    invdiag_ = new Real[M_];
    if (UL_)
        for (int i = 0; i < M_; ++i)
            invdiag_[i] = 1.0 / diag(i);
    else
        for (int i = 0; i < M_; ++i)
            invdiag_[i] = 0.0;

    int nnz;
    switch (M_) {
        case 0:
            nnz = 0;
            break;
        case 1:
            nnz = 1;
            break;
        default:
            nnz = 4 * (M_ - 1);
    }

    int i, j;
    Real x;
    for (n = 0; n < nnz; ++n) {
        is >> i >> j >> x;
        elem(i, j) = x;
    }
}

BandedTridiag& BandedTridiag::operator=(const BandedTridiag& A) {
    if (this != &A) {
        if (M_ != A.M_) {
            delete[] a_;
            M_ = A.M_;
            Mbar_ = A.Mbar_;
            UL_ = A.UL_;
            a_ = new Real[4 * M_ - 2];
            d_ = a_ + Mbar_;
            invdiag_ = new Real[M_];
        }
        for (int i = 0; i < 4 * M_ - 2; ++i)
            a_[i] = A.a_[i];
        for (int i = 0; i < M_; ++i)
            invdiag_[i] = A.invdiag_[i];
    }
    return *this;
}

bool BandedTridiag::operator==(const BandedTridiag& A) const {
    if (M_ != A.M_ || Mbar_ != A.Mbar_ || UL_ != A.UL_)
        return false;

    for (int n = 0; n < 4 * M_ - 2; ++n)
        if (fabs(a_[n] - A.a_[n]) > EPSILON) {
            cout << "BandedTridiag::operator== failed on a[" << n << "]\n";
            cout << setprecision(REAL_DIGITS);
            cout << a_[n] << ' ' << A.a_[n] << endl;
            return false;
        }

    return true;
}

const Real& BandedTridiag::elem(int i, int j) const {
    assert(i == 0 || (i >= 0 && i < M_ && j >= 0 && j < M_ && abs(i - j) <= 1));
    if (i == 0)
        return band(j);
    else if (i == j)
        return diag(i);
    else if (i < j)
        return updiag(i);
    else
        return lodiag(i);
}

Real& BandedTridiag::elem(int i, int j) {
    assert(i == 0 || (i >= 0 && i < M_ && j >= 0 && j < M_ && abs(i - j) <= 1));
    if (i == 0)
        return band(j);
    else if (i == j)
        return diag(i);
    else if (i < j)
        return updiag(i);
    else
        return lodiag(i);
}

void BandedTridiag::print() const {
    cout << "[\n";
    Real eij = 0.0;
    for (int i = 0; i < M_; ++i) {
        for (int j = 0; j < M_; ++j) {
            if (i == 0 || (abs(i - j) <= 1))
                eij = elem(i, j);
            else
                eij = 0.0;
            cout << eij << ' ';
        }
        cout << ";\n";
    }
    cout << "]\n";
}
void BandedTridiag::ULprint() const {
    Real one = 1.0;
    Real zero = 0.0;
    Real eij = 0.0;
    cout << "U = [\n";
    int i;  // MSVC++  FOR-SCOPE BUG
    for (i = 0; i < M_; ++i) {
        for (int j = 0; j < M_; ++j) {
            if (i == 0)
                eij = band(j);
            else if (i == j)
                eij = one;
            else if (i == j - 1)
                eij = updiag(i);
            else
                eij = zero;
            cout << eij << ' ';
        }
        cout << ";\n";
    }
    cout << "]\n";

    cout << "L = [\n";
    for (i = 0; i < M_; ++i) {
        for (int j = 0; j < M_; ++j) {
            if (i == j + 1)
                eij = lodiag(i);
            else if (i == j)
                eij = diag(i);
            else
                eij = zero;
            cout << eij << ' ';
        }
        cout << ";\n";
    }
    cout << "]\n";
}

void BandedTridiag::ULdecomp() {
    int Mb = M_ - 1;
    Real w;
    Real Akk;

    for (int k = Mb; k > 1; --k) {
        Akk = diag(k);
        assert(Akk != 0.0);
        w = lodiag(k);                              // elem(k,k-1);
        diag(k - 1) -= w * (updiag(k - 1) /= Akk);  // elem(k-1,k) /= Akk);
        band(k - 1) -= w * (band(k) /= Akk);
    }
    band(0) -= lodiag(1) * (band(1) /= diag(1));
    for (int i = 0; i < M_; ++i)
        invdiag_[i] = 1.0 / diag(i);

    UL_ = true;
}

// Solve Ax=b given UL=A, via Uy=b, then Lx=y.
void BandedTridiag::ULsolveStrided(Vector& b, int offset, int stride) const {
    assert(UL_ == true);
    int i, j;
    int Mb = M_ - 1;

    if (offset == 0 && stride == 1) {
        // Solve Uy=b by backsubstitution, iterating last row to zeroth.
        // Last row needs no calculation due to sparsity structure.
        // b[Nb] = b[Nb];           // row  M-1
        for (i = Mb - 1; i > 0; --i)  // rows M-2 through 1
            b[i] -= updiag(i) * b[i + 1];
        for (j = i + 1; j < M_; ++j)  // row 0
            b[0] -= band(j) * b[j];

        // Solve Lx=y by forward substitution
        b[0] /= diag(0);  // row  0
        for (i = 1; i < M_; ++i)
            (b[i] -= lodiag(i) * b[i - 1]) *= invdiag_[i];
    } else if (offset == 1 && stride == 1) {
        for (i = Mb - 1; i > 0; --i)  // rows M-2 through 1
            b[1 + i] -= updiag(i) * b[i + 2];
        for (j = i + 1; j < M_; ++j)  // row 0
            b[1] -= band(j) * b[1 + j];
        b[1] /= diag(0);  // row  0
        for (i = 1; i < M_; ++i)
            (b[1 + i] -= lodiag(i) * b[i]) *= invdiag_[i];
    } else if (offset == 0 && stride == 2) {
        for (i = Mb - 1; i > 0; --i)  // rows M-2 through 1
            b[i * 2] -= updiag(i) * b[2 * (i + 1)];
        for (j = i + 1; j < M_; ++j)  // row 0
            b[0] -= band(j) * b[2 * j];
        b[0] /= diag(0);  // row  0
        for (i = 1; i < M_; ++i)
            (b[2 * i] -= lodiag(i) * b[2 * (i - 1)]) *= invdiag_[i];
    } else if (offset == 1 && stride == 2) {
        for (i = Mb - 1; i > 0; --i)  // rows M-2 through 1
            b[1 + i * 2] -= updiag(i) * b[2 * i + 3];
        for (j = i + 1; j < M_; ++j)  // row 0
            b[1] -= band(j) * b[1 + 2 * j];
        b[1] /= diag(0);  // row  0
        for (i = 1; i < M_; ++i)
            (b[1 + 2 * i] -= lodiag(i) * b[2 * i - 1]) *= invdiag_[i];
    } else
        cferror(
            "BandedTridiag::ULsolveStrided(Vector& b, int offset, int stride) : offset must be 0 or 1, stride 1 or 2");
}

// Solve Ax=b given UL=A, via Uy=b, then Lx=y.
void BandedTridiag::ULsolve(Vector& b) const {
    assert(UL_ == true);
    int i, j;
    int Mb = M_ - 1;

    // Solve Uy=b by backsubstitution, iterating last row to zeroth.

    // Last row needs no calculation due to sparsity structure.
    // b[Nb] = b[Nb];           // row  M-1
    for (i = Mb - 1; i > 0; --i)  // rows M-2 through 1
        b[i] -= updiag(i) * b[i + 1];
    for (j = i + 1; j < M_; ++j)  // row 0
        b[0] -= band(j) * b[j];

    // Solve Lx=y by forward substitution
    b[0] /= diag(0);          // row  0
    for (i = 1; i < M_; ++i)  // rows 1 through M-1
        (b[i] -= lodiag(i) * b[i - 1]) /= diag(i);
}

void BandedTridiag::multiply(const Vector& x, Vector& b) const {
    Real sum = 0.0;

    // row 0
    for (int j = 0; j < M_; ++j)
        sum += band(j) * x[j];
    b[0] = sum;

    // rows 1 to (Mbar-1)
    for (int i = 1; i < Mbar_; ++i)
        b[i] = lodiag(i) * x[i - 1] + diag(i) * x[i] + updiag(i) * x[i + 1];

    b[Mbar_] = lodiag(Mbar_) * x[Mbar_ - 1] + diag(Mbar_) * x[Mbar_];
}

void BandedTridiag::multiplyStrided(const Vector& x, Vector& b, int offset, int stride) const {
    assert(offset == 0 || offset == 1);
    assert(stride == 1 || stride == 2);

    Real sum = 0.0;

    // row 0
    for (int j = 0; j < M_; ++j)
        sum += band(j) * x[offset + stride * j];
    b[offset] = sum;

    // rows 1 to (Mbar-1)
    for (int i = 1; i < Mbar_; ++i)
        b[offset + stride * i] = lodiag(i) * x[offset + stride * (i - 1)] + diag(i) * x[offset + stride * i] +
                                 updiag(i) * x[offset + stride * (i + 1)];

    b[offset + stride * Mbar_] =
        lodiag(Mbar_) * x[offset + stride * (Mbar_ - 1)] + diag(Mbar_) * x[offset + stride * Mbar_];
}

//  Guide to the perplexed:
//  b b b b b
//  l d u 0 0
//  0 l d u 0
//  0 0 l d u
//  0 0 0 l d

void BandedTridiag::save(const string& filebase) const {
    string filename(filebase);
    filename += string(".asc");
    ofstream os(filename.c_str());
    os << setprecision(REAL_DIGITS);
    os << "% " << M_ << ' ' << (UL_ ? 1 : 0) << endl;

    // Print the row-0 band elems
    for (int j = 0; j < M_; ++j)
        os << "0 " << j << ' ' << band(j) << '\n';

    // Print the lodiag,diag,updiag triplets in rows 1 through M_-2.
    for (int i = 1; i < M_ - 1; ++i) {
        os << i << ' ' << (i - 1) << ' ' << lodiag(i) << '\n';
        os << i << ' ' << i << ' ' << diag(i) << '\n';
        os << i << ' ' << (i + 1) << ' ' << updiag(i) << '\n';
    }

    // Print the lodiag,diag pair in the last row. 2 elems
    if (M_ > 1) {
        int i = M_ - 1;
        os << i << ' ' << (i - 1) << ' ' << lodiag(i) << '\n';
        os << i << ' ' << i << ' ' << diag(i) << '\n';
    }
    return;
}

}  // namespace chflow
