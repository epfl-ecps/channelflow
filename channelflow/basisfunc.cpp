/**
 * a Complex-Vector-valued Spectral expansion class.
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "channelflow/basisfunc.h"
#include <fstream>
#include <iomanip>
#include "cfbasics/cfbasics.h"
#include "cfbasics/mathdefs.h"
#include "channelflow/symmetry.h"
#include "channelflow/utilfuncs.h"

using namespace std;

namespace chflow {

BasisFunc::BasisFunc() : Nd_(0), Ny_(0), kx_(0), kz_(0), Lx_(0), Lz_(0), a_(0), b_(0), state_(Spectral), u_(0) {}

BasisFunc::BasisFunc(int Ny, const BasisFunc& f)
    : Nd_(f.Nd_),
      Ny_(Ny),
      kx_(f.kx_),
      kz_(f.kz_),
      Lx_(f.Lx_),
      Lz_(f.Lz_),
      a_(f.a_),
      b_(f.b_),
      state_(f.state_),
      u_(Nd_) {
    for (int i = 0; i < Nd_; ++i)
        u_[i].resize(Ny_);
}

BasisFunc::BasisFunc(int Nd, int Ny, int kx, int kz, Real Lx, Real Lz, Real a, Real b, fieldstate s)
    : Nd_(Nd), Ny_(Ny), kx_(kx), kz_(kz), Lx_(Lx), Lz_(Lz), a_(a), b_(b), state_(s), u_(Nd) {
    for (int n = 0; n < Nd_; ++n)
        u_[n] = ComplexChebyCoeff(Ny_, a_, b_, state_);
}

BasisFunc::BasisFunc(const string& filebase)
    : Nd_(0), Ny_(0), kx_(0), kz_(0), Lx_(0), Lz_(0), a_(0), b_(0), state_(Spectral), u_(0) {
    string filename = filebase + string(".asc");
    ifstream is(filename.c_str());
    if (!is.good()) {
        cerr << "BasisFunc::BasisFunc(filebase) : can't open file " << filename << '\n';
        exit(1);
    }
    char c;
    is >> c;
    if (c != '%') {
        cerr << "BasisFunc::BasisFunc(filebase): bad header in file " << filename << endl;
        assert(false);
    }
    is >> Nd_ >> Ny_ >> kx_ >> kz_ >> Lx_ >> Lz_ >> a_ >> b_ >> state_;

    u_.resize(Nd_);
    for (int n = 0; n < Nd_; ++n) {
        u_[n].resize(Ny_);
        u_[n].setBounds(a_, b_);
        u_[n].setState(state_);
    }

    Real r;
    Real i;
    for (int ny = 0; ny < Ny_; ++ny)
        for (int n = 0; n < Nd_; ++n) {
            is >> r >> i;
            u_[n].set(ny, Complex(r, i));
        }
}

BasisFunc::BasisFunc(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v, const ComplexChebyCoeff& w, int kx, int kz,
                     Real Lx, Real Lz)
    : Nd_(3), Ny_(u.numModes()), kx_(kx), kz_(kz), Lx_(Lx), Lz_(Lz), a_(u.a()), b_(u.b()), state_(u.state()), u_(3) {
    u_[0] = u;
    u_[1] = v;
    u_[2] = w;
    for (int n = 1; n < Nd_; ++n) {
        assert(u_[0].congruent(u_[n]));
        ;  // this statement needed for optimized compilation, #define assert(x) ;
    }
}

void BasisFunc::save(const string& filebase, fieldstate savestate) const {
    fieldstate origstate = state_;
    ((BasisFunc&)*this).makeState(savestate);

    string filename(filebase);
    filename += string(".asc");
    ofstream os(filename.c_str());
    os << scientific << setprecision(REAL_DIGITS);
    char sp = ' ';
    os << '%' << sp << Nd_ << sp << Ny_ << sp << kx_ << sp << kz_ << sp << Lx_ << sp << Lz_ << sp << a_ << sp << b_
       << sp << state_ << '\n';

    for (int ny = 0; ny < Ny_; ++ny) {
        for (int n = 0; n < Nd_; ++n)
            os << Re(u_[n][ny]) << sp << Im(u_[n][ny]) << sp;
        os << '\n';
    }
    os.close();

    ((BasisFunc&)*this).makeState(origstate);
}

void BasisFunc::binaryDump(ostream& os) const {
    write(os, Nd_);
    write(os, Ny_);
    write(os, kx_);
    write(os, kz_);
    write(os, Lx_);
    write(os, Lz_);
    write(os, a_);
    write(os, b_);
    write(os, state_);
    for (int n = 0; n < Nd_; ++n)
        u_[n].binaryDump(os);
}

void BasisFunc::binaryLoad(istream& is) {
    if (!is.good()) {
        cerr << "BasisFunc::binaryLoad(istream) : input error" << endl;
        exit(1);
    }
    read(is, Ny_);
    read(is, kx_);
    read(is, kz_);
    read(is, Lx_);
    read(is, Lz_);
    read(is, a_);
    read(is, b_);
    read(is, state_);
    for (int n = 0; n < Nd_; ++n)
        u_[n].binaryLoad(is);
}

void BasisFunc::randomize(Real magn, Real decay, BC aBC, BC bBC) {
    assert(Nd_ == 3);

    setToZero();
    setState(Spectral);

    ComplexChebyCoeff u(Ny_, a_, b_, Spectral);
    ComplexChebyCoeff v(Ny_, a_, b_, Spectral);
    ComplexChebyCoeff w(Ny_, a_, b_, Spectral);
    ComplexChebyCoeff vy(Ny_, a_, b_, Spectral);

    // Make a u,v part
    if (kx_ == 0) {
        u.randomize(magn, decay, aBC, bBC);
        ubcFix(u, aBC, bBC);
    } else {
        v.randomize(magn, decay, aBC, bBC);
        vbcFix(v, aBC, bBC);
        diff(v, u);
        u *= Complex(0.0, Lx_ / (2 * pi * kx_));
    }
    u_[0] += u;
    u_[1] += v;

    u.setToZero();
    v.setToZero();
    w.setToZero();

    // Make a v,w, part
    if (kz_ == 0) {
        w.randomize(magn, decay, aBC, bBC);
        ubcFix(w, aBC, bBC);
    } else {
        v.randomize(magn, decay, aBC, bBC);
        vbcFix(v, aBC, bBC);
        diff(v, w);
        w *= Complex(0.0, Lz_ / (2 * pi * kz_));
    }
    u_[2] += w;
    u_[1] += v;

    if (kx_ == 0 && kz_ == 0) {
        u_[0].im.setToZero();
        u_[1].re.setToZero();  // should already be zero
        u_[2].im.setToZero();
    }
}

void BasisFunc::interpolate(const BasisFunc& phi) {
    assert(phi.state_ == Spectral);
    assert(a_ >= phi.a_ && b_ <= phi.b_);
    assert(Nd_ == phi.Nd_);
    for (int n = 0; n < Nd_; ++n)
        u_[n].interpolate(phi.u_[n]);
    state_ = Physical;
}

void BasisFunc::reflect(const BasisFunc& phi) {
    assert((a_ + b_) / 2 == phi.a() && b_ <= phi.b() && b_ > phi.a());
    assert(phi.state_ == Spectral);
    for (int n = 0; n < Nd_; n += 2)
        u_[n].reflect(phi.u_[n], Odd);
    for (int n = 1; n < Nd_; n += 2)
        u_[n].reflect(phi.u_[n], Even);
    state_ = Physical;
}

void BasisFunc::resize(int Ny, int Nd) {
    if (Nd_ != Nd) {
        Nd_ = Nd;
        u_.resize(Nd_);
        Ny_ = Ny;
        for (int n = 0; n < Nd_; ++n)
            u_[n].resize(Ny_);
    } else if (Ny_ != Ny) {
        Ny_ = Ny;
        for (int n = 0; n < Nd_; ++n)
            u_[n].resize(Ny_);
    }
    setToZero();
}

void BasisFunc::reconfig(const BasisFunc& f) {
    resize(f.Ny(), f.Nd());
    setBounds(f.Lx(), f.Lz(), f.a(), f.b());
    setkxkz(f.kx(), f.kz());
    setState(f.state());
    setToZero();
}

void BasisFunc::setBounds(Real Lx, Real Lz, Real a, Real b) {
    Lx_ = Lx;
    Lz_ = Lz;
    a_ = a;
    b_ = b;
    for (int n = 0; n < Nd_; ++n)
        u_[n].setBounds(a, b);
}
void BasisFunc::setkxkz(int kx, int kz) {
    kx_ = kx;
    kz_ = kz;
}
void BasisFunc::setState(fieldstate s) {
    assert(s == Spectral || s == Physical);
    for (int n = 0; n < Nd_; ++n)
        u_[n].setState(s);
    state_ = s;
}

void BasisFunc::setToZero() {
    for (int n = 0; n < Nd_; ++n)
        u_[n].setToZero();
}

void BasisFunc::fill(const BasisFunc& f) {
    for (int n = 0; n < Nd_; ++n)
        u_[n].fill(f.u_[n]);
}
void BasisFunc::conjugate() {
    for (int n = 0; n < Nd_; ++n)
        u_[n].conjugate();
    kx_ *= -1;
    kz_ *= -1;  // Note that no neg vals of kz appear in FlowFields.
}

bool BasisFunc::geomCongruent(const BasisFunc& Phi) const {
    return (Nd_ == Phi.Nd_ && Ny_ == Phi.Ny_ && Lx_ == Phi.Lx_ && Lz_ == Phi.Lz_ && a_ == Phi.a_ && b_ == Phi.b_);
}

bool BasisFunc::congruent(const BasisFunc& Phi) const {
    return (geomCongruent(Phi) && kx_ == Phi.kx_ && kz_ == Phi.kz_);
}

bool BasisFunc::interoperable(const BasisFunc& Phi) const { return (congruent(Phi) && state_ == Phi.state_); }

void BasisFunc::chebyfft() {
    assert(state_ == Physical);
    ChebyTransform t(Ny_);
    for (int n = 0; n < Nd_; ++n)
        u_[n].chebyfft(t);
    state_ = Spectral;
}

void BasisFunc::ichebyfft() {
    assert(state_ == Spectral);
    ChebyTransform t(Ny_);
    for (int n = 0; n < Nd_; ++n)
        u_[n].ichebyfft(t);
    state_ = Physical;
}
void BasisFunc::makeSpectral() {
    ChebyTransform t(Ny_);
    for (int n = 0; n < Nd_; ++n)
        u_[n].makeSpectral(t);
    state_ = Spectral;
}
void BasisFunc::makePhysical() {
    ChebyTransform t(Ny_);
    for (int n = 0; n < Nd_; ++n)
        u_[n].makePhysical(t);
    state_ = Physical;
}
void BasisFunc::makeState(fieldstate s) {
    ChebyTransform t(Ny_);
    for (int n = 0; n < Nd_; ++n)
        u_[n].makeState(s, t);
    state_ = s;
}

void BasisFunc::chebyfft(const ChebyTransform& t) {
    assert(state_ == Physical);
    for (int n = 0; n < Nd_; ++n)
        u_[n].chebyfft(t);
    state_ = Spectral;
}

void BasisFunc::ichebyfft(const ChebyTransform& t) {
    assert(state_ == Spectral);
    for (int n = 0; n < Nd_; ++n)
        u_[n].ichebyfft(t);
    state_ = Physical;
}
void BasisFunc::makeSpectral(const ChebyTransform& t) {
    for (int n = 0; n < Nd_; ++n)
        u_[n].makeSpectral(t);
    state_ = Spectral;
}
void BasisFunc::makePhysical(const ChebyTransform& t) {
    for (int n = 0; n < Nd_; ++n)
        u_[n].makePhysical(t);
    state_ = Physical;
}
void BasisFunc::makeState(fieldstate s, const ChebyTransform& t) {
    for (int n = 0; n < Nd_; ++n)
        u_[n].makeState(s, t);
    state_ = s;
}

const ComplexChebyCoeff& BasisFunc::u() const { return u_[0]; }
const ComplexChebyCoeff& BasisFunc::v() const { return u_[1]; }
const ComplexChebyCoeff& BasisFunc::w() const { return u_[2]; }
ComplexChebyCoeff& BasisFunc::u() { return u_[0]; }
ComplexChebyCoeff& BasisFunc::v() { return u_[1]; }
ComplexChebyCoeff& BasisFunc::w() { return u_[2]; }

const ComplexChebyCoeff& BasisFunc::operator[](int i) const {
    assert(i >= 0 && i < Nd_);
    return u_[i];
}
ComplexChebyCoeff& BasisFunc::operator[](int i) {
    assert(i >= 0 && i < Nd_);
    return u_[i];
}

bool operator==(const BasisFunc& f, const BasisFunc& g) {
    if (!f.congruent(g))
        return false;
    else {
        for (int i = 0; i < f.Nd(); ++i)
            if (f[i] != g[i])
                return false;
    }
    return true;
}

bool operator!=(const BasisFunc& f, const BasisFunc& g) { return (!(f == g)); }

BasisFunc& BasisFunc::operator*=(const BasisFunc& phi) {
    assert(geomCongruent(phi) && state_ == Physical && phi.state_ == Physical);
    for (int n = 0; n < Nd_; ++n)
        u_[n] *= phi.u_[n];

    // These ops might break POD code.
    kx_ += phi.kx();
    kz_ += phi.kz();
    return *this;
}

BasisFunc& BasisFunc::operator+=(const BasisFunc& phi) {
    assert(interoperable(phi));
    for (int n = 0; n < Nd_; ++n)
        u_[n] += phi.u_[n];
    return *this;
}
BasisFunc& BasisFunc::operator-=(const BasisFunc& phi) {
    assert(interoperable(phi));
    for (int n = 0; n < Nd_; ++n)
        u_[n] -= phi.u_[n];
    return *this;
}
BasisFunc& BasisFunc::operator*=(Real c) {
    for (int n = 0; n < Nd_; ++n)
        u_[n] *= c;
    return *this;
}
BasisFunc& BasisFunc::operator*=(Complex c) {
    for (int n = 0; n < Nd_; ++n)
        u_[n] *= c;
    return *this;
}

BasisFunc conjugate(const BasisFunc& f) {
    BasisFunc g(f);
    g.conjugate();
    return g;
}

// ==================================================================
// L2 norms
Real L2Norm(const BasisFunc& phi, bool normalize) {
    Real rtn2 = 0.0;
    for (int n = 0; n < phi.Nd(); ++n)
        rtn2 += L2Norm2(phi[n], normalize);
    if (!normalize)
        rtn2 *= phi.Lx() * phi.Lz();
    return sqrt(rtn2);
}

Real L2Norm2(const BasisFunc& phi, bool normalize) {
    Real rtn = 0.0;
    for (int n = 0; n < phi.Nd(); ++n)
        rtn += L2Norm2(phi[n], normalize);
    if (!normalize)
        rtn *= phi.Lx() * phi.Lz();
    return rtn;
}

Real L2Dist2(const BasisFunc& f, const BasisFunc& g, bool normalize) {
    assert(f.geomCongruent(g));
    Real rtn = 0.0;
    if (f.kx() == g.kx() && f.kz() == g.kz())
        for (int n = 0; n < f.Nd(); ++n)
            rtn += L2Dist2(f[n], g[n], normalize);
    if (!normalize)
        rtn *= f.Lx() * f.Lz();
    return rtn;
}

Real L2Dist(const BasisFunc& f, const BasisFunc& g, bool normalize) { return sqrt(L2Dist2(f, g, normalize)); }

Complex L2InnerProduct(const BasisFunc& f, const BasisFunc& g, bool normalize) {
    assert(f.geomCongruent(g));
    Complex rtn(0.0, 0.0);
    if (f.kx() == g.kx() && f.kz() == g.kz())
        for (int n = 0; n < f.Nd(); ++n)
            rtn += L2InnerProduct(f[n], g[n], normalize);
    if (!normalize)
        rtn *= f.Lx() * f.Lz();
    return rtn;
}

// ==================================================================
// cheby norms
Real chebyNorm(const BasisFunc& phi, bool normalize) {
    Real rtn2 = 0.0;
    for (int n = 0; n < phi.Nd(); ++n)
        rtn2 += chebyNorm2(phi[n], normalize);
    if (!normalize)
        rtn2 *= phi.Lx() * phi.Lz();
    return sqrt(rtn2);
}

Real chebyNorm2(const BasisFunc& phi, bool normalize) {
    Real rtn = 0.0;
    for (int n = 0; n < phi.Nd(); ++n)
        rtn += chebyNorm2(phi[n], normalize);
    if (!normalize)
        rtn *= phi.Lx() * phi.Lz();
    return rtn;
}

Real chebyDist2(const BasisFunc& f, const BasisFunc& g, bool normalize) {
    assert(f.geomCongruent(g));
    Real rtn = 0.0;
    if (f.kx() == g.kx() && f.kz() == g.kz())
        for (int n = 0; n < f.Nd(); ++n)
            rtn += chebyDist2(f[n], g[n], normalize);
    if (!normalize)
        rtn *= f.Lx() * f.Lz();
    return rtn;
}

Real chebyDist(const BasisFunc& f, const BasisFunc& g, bool normalize) { return sqrt(chebyDist2(f, g, normalize)); }

Complex chebyInnerProduct(const BasisFunc& f, const BasisFunc& g, bool normalize) {
    assert(f.geomCongruent(g));
    Complex rtn(0.0, 0.0);
    if (f.kx() == g.kx() && f.kz() == g.kz())
        for (int n = 0; n < f.Nd(); ++n)
            rtn += chebyInnerProduct(f[n], g[n], normalize);
    if (!normalize)
        rtn *= f.Lx() * f.Lz();
    return rtn;
}

// ==================================================================
// switchable norms
Real norm(const BasisFunc& phi, NormType n, bool normalize) {
    return (n == Uniform) ? L2Norm(phi, normalize) : chebyNorm(phi, normalize);
}

Real norm2(const BasisFunc& phi, NormType n, bool normalize) {
    return (n == Uniform) ? L2Norm2(phi, normalize) : chebyNorm2(phi, normalize);
}

Real dist2(const BasisFunc& f, const BasisFunc& g, NormType n, bool nrmlz) {
    return (n == Uniform) ? L2Dist2(f, g, nrmlz) : chebyDist2(f, g, nrmlz);
}

Real dist(const BasisFunc& f, const BasisFunc& g, NormType n, bool nrmlz) {
    return (n == Uniform) ? L2Dist(f, g, nrmlz) : chebyDist(f, g, nrmlz);
}

Complex innerProduct(const BasisFunc& f, const BasisFunc& g, NormType n, bool nrmlz) {
    return (n == Uniform) ? L2InnerProduct(f, g, nrmlz) : chebyInnerProduct(f, g, nrmlz);
}

// =================================================================
// boundary and div norms

Real divNorm2(const BasisFunc& f, bool normalize) {
    assert(f.state() == Spectral);
    assert(f.Nd() == 3);
    ComplexChebyCoeff div = f.u();
    div *= Complex(0.0, 2 * pi * f.kx() / f.Lx());
    ComplexChebyCoeff tmp = f.w();
    tmp *= Complex(0.0, 2 * pi * f.kz() / f.Lz());
    div += tmp;
    diff(f.v(), tmp);
    div += tmp;
    return L2Norm2(div, normalize);
}

Real divNorm(const BasisFunc& f, bool normalize) { return sqrt(divNorm2(f, normalize)); }

Real divDist2(const BasisFunc& f, const BasisFunc& g, bool normalize) {
    assert(f.state() == Spectral && g.state() == Spectral);
    assert(f.congruent(g));
    assert(f.Nd() == 3);
    ComplexChebyCoeff div = f.u();
    div -= g.u();
    div *= Complex(0.0, 2 * pi * f.kx() / f.Lx());
    ComplexChebyCoeff tmp = f.w();
    tmp -= g.w();
    tmp *= Complex(0.0, 2 * pi * f.kz() / f.Lz());
    div += tmp;
    diff(f.v(), tmp);
    div += tmp;
    diff(g.v(), tmp);
    div -= tmp;
    return L2Norm2(div, normalize);
}

Real divDist(const BasisFunc& f, const BasisFunc& g, bool normalize) { return sqrt(divDist2(f, g, normalize)); }

Real BasisFunc::bcNorm(BC aBC, BC bBC) const {
    Real bcnorm = 0.0;
    if (aBC == Diri)
        for (int n = 0; n < Nd_; ++n)
            bcnorm += abs2(u_[n].eval_a());
    if (bBC == Diri)
        for (int n = 0; n < Nd_; ++n)
            bcnorm += abs2(u_[n].eval_b());
    return sqrt(bcnorm);
}

Real bcNorm2(const BasisFunc& f, bool normalize) {
    Real bc2 = 0.0;
    for (int n = 0; n < f.Nd(); ++n) {
        bc2 += abs2(f[n].eval_a());
        bc2 += abs2(f[n].eval_b());
    }
    if (!normalize)
        bc2 *= f.Lx() * f.Lz();
    return bc2;
}

Real bcNorm(const BasisFunc& f, bool normalize) { return sqrt(bcNorm2(f, normalize)); }

Real bcDist2(const BasisFunc& f, const BasisFunc& g, bool normalize) {
    assert(f.interoperable(g));
    Real bc2 = 0.0;
    for (int n = 0; n < f.Nd(); ++n) {
        bc2 += abs2(f[n].eval_a() - g[n].eval_a());
        bc2 += abs2(f[n].eval_b() - g[n].eval_b());
    }
    if (!normalize)
        bc2 *= f.Lx() * f.Lz();
    return bc2;
}

Real bcDist(const BasisFunc& f, const BasisFunc& g, bool normalize) { return sqrt(bcDist2(f, g, normalize)); }

BasisFunc xdiff(const BasisFunc& f) {
    BasisFunc fx;
    xdiff(f, fx);
    return fx;
}

BasisFunc ydiff(const BasisFunc& f) {
    BasisFunc fy;
    ydiff(f, fy);
    return fy;
}

BasisFunc zdiff(const BasisFunc& f) {
    BasisFunc fz;
    zdiff(f, fz);
    return fz;
}

BasisFunc div(const BasisFunc& f) {
    BasisFunc divf;
    div(f, divf);
    return divf;
}

BasisFunc lapl(const BasisFunc& f) {
    BasisFunc laplf;
    lapl(f, laplf);
    return laplf;
}

BasisFunc grad(const BasisFunc& f) {
    BasisFunc gradf;
    grad(f, gradf);
    return gradf;
}

BasisFunc dot(const BasisFunc& f, const BasisFunc& g) {
    BasisFunc fdotg;
    dot(f, g, fdotg);
    return fdotg;
}

BasisFunc cross(const BasisFunc& f, const BasisFunc& g) {
    BasisFunc fcrossg;
    cross(f, g, fcrossg);
    return fcrossg;
}

BasisFunc curl(const BasisFunc& f) {
    BasisFunc curlf;
    curl(f, curlf);
    return curlf;
}

BasisFunc dotgrad(const BasisFunc& f, const BasisFunc& g) {
    BasisFunc rtn;
    dotgrad(f, g, rtn);
    return rtn;
}

// ===============

void xdiff(const BasisFunc& f, BasisFunc& fx) {
    fx = f;
    fx *= Complex(0.0, 2 * pi * f.kx() / f.Lx());
}

void ydiff(const BasisFunc& f, BasisFunc& fy) {
    assert(f.state() == Spectral);
    fy.reconfig(f);
    for (int n = 0; n < f.Nd(); ++n)
        diff(f[n], fy[n]);
}

void zdiff(const BasisFunc& f, BasisFunc& fz) {
    fz = f;
    fz *= Complex(0.0, 2 * pi * f.kz() / f.Lz());
}

void div(const BasisFunc& f, BasisFunc& divf) {
    assert(f.state() == Spectral);
    assert(f.Nd() == 3);

    divf.resize(f.Ny(), 1);
    divf.setBounds(f.Lx(), f.Lz(), f.a(), f.b());
    divf.setkxkz(f.kx(), f.kz());
    ComplexChebyCoeff tmp(f.Ny(), f.a(), f.b(), Spectral);

    tmp = f[0];
    tmp *= Complex(0.0, 2 * pi * f.kx() / f.Lx());
    divf[0] = tmp;

    diff(f[1], tmp);
    divf[0] += tmp;

    tmp = f[2];
    tmp *= Complex(0.0, 2 * pi * f.kz() / f.Lz());
    divf[0] += tmp;
}

void lapl(const BasisFunc& f, BasisFunc& laplf) {
    assert(f.state() == Spectral);
    laplf = f;
    Real c = -4 * square(pi) * (square(f.kx() / f.Lx()) + square(f.kz() / f.Lz()));
    for (int n = 0; n < f.Nd(); ++n)
        laplf[n] *= c;

    ComplexChebyCoeff tmp;
    for (int n = 0; n < f.Nd(); ++n) {
        diff2(f[n], tmp);
        laplf[n] += tmp;
    }
}

void grad(const BasisFunc& f, BasisFunc& gradf) {
    assert(f.state() == Spectral);
    gradf.resize(f.Ny(), 3 * f.Nd());
    gradf.setBounds(f.Lx(), f.Lz(), f.a(), f.b());
    gradf.setkxkz(f.kx(), f.kz());
    gradf.setState(Spectral);

    int d = f.Nd();
    for (int i = 0; i < f.Nd(); ++i) {
        const ComplexChebyCoeff& f_i = f[i];

        // Do dfi/dx0
        gradf[i * d] = f_i;
        gradf[i * d] *= Complex(0.0, 2 * pi * f.kx() / f.Lx());

        // Do dfi/dx1
        diff(f_i, gradf[i * d + 1]);

        // Do dfi/dx2
        gradf[i * d + 2] = f_i;
        gradf[i * d + 2] *= Complex(0.0, 2 * pi * f.kz() / f.Lz());
    }
}

void dot(const BasisFunc& f_, const BasisFunc& g_, BasisFunc& fdotg) {
    BasisFunc& f = (BasisFunc&)f_;
    BasisFunc& g = (BasisFunc&)g_;
    assert(f.Lx() == g.Lx() && f.Lz() == g.Lz() && f.a() == g.a() && f.Ny() == g.Ny());

    int Ny = f.Ny();
    fieldstate fstate = f.state();
    fieldstate gstate = g.state();
    ChebyTransform t(Ny);
    f.makePhysical(t);
    g.makePhysical(t);

    if (f.Nd() == g.Nd()) {
        fdotg.resize(Ny, 1);
        fdotg.setToZero();
        fdotg.setBounds(f.Lx(), f.Lz(), f.a(), f.b());
        fdotg.setkxkz(f.kx() + g.kx(), f.kz() + g.kz());
        fdotg.setState(Physical);

        int Nd = f.Nd();
        int Ny = f.Ny();
        for (int i = 0; i < Nd; ++i)
            for (int ny = 0; ny < Ny; ++ny)
                fdotg[0].add(ny, f[i][ny] * g[i][ny]);
    } else if (f.Nd() == 3 && g.Nd() == 9) {
        fdotg.resize(Ny, 3);
        fdotg.setToZero();
        fdotg.setBounds(f.Lx(), f.Lz(), f.a(), f.b());
        fdotg.setkxkz(f.kx() + g.kx(), f.kz() + g.kz());
        fdotg.setState(Physical);

        // Nd == 3 : rtn[i] = fj gij
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                ComplexChebyCoeff& fj = f[j];
                ComplexChebyCoeff& gj = g[i3j(i, j)];
                for (int ny = 0; ny < f.Ny(); ++ny)
                    fdotg[i].add(ny, fj[ny] * gj[ny]);
            }
        }
    } else
        cferror("dot(f,g, rtn) on BasisFuncs error : f and g have incompatible dimensions");

    fdotg.save("fdotg");
    f.makeState(fstate, t);
    g.makeState(gstate, t);
    fdotg.makeState(fstate, t);
}

void cross(const BasisFunc& f_, const BasisFunc& g_, BasisFunc& fcg) {
    BasisFunc& f = (BasisFunc&)f_;
    BasisFunc& g = (BasisFunc&)g_;
    assert(f.geomCongruent(g));
    assert(f.Nd() == 3);
    assert(g.Nd() == 3);
    int Ny = f.Ny();

    ChebyTransform t(Ny);
    fieldstate fstate = f.state();
    fieldstate gstate = g.state();
    f.makePhysical(t);
    g.makePhysical(t);
    fcg.reconfig(f);
    fcg.setkxkz(f.kx() + g.kx(), f.kz() + g.kz());
    fcg.setState(Physical);

    for (int ny = 0; ny < Ny; ++ny) {
        Complex fu = f[0][ny];
        Complex fv = f[1][ny];
        Complex fw = f[2][ny];
        Complex gu = g[0][ny];
        Complex gv = g[1][ny];
        Complex gw = g[2][ny];

        fcg[0].set(ny, fv * gw - fw * gv);
        fcg[1].set(ny, fw * gu - fu * gw);
        fcg[2].set(ny, fu * gv - fv * gu);
    }

    f.makeState(fstate, t);
    g.makeState(gstate, t);
    fcg.makeState(fstate, t);
}

void curl(const BasisFunc& f_, BasisFunc& curlf) {
    assert(f_.Nd() == 3);

    BasisFunc& f = (BasisFunc&)f_;
    fieldstate fstate = f.state();
    curlf.reconfig(f);
    curlf.setState(f.state());

    int Ny = f.Ny();
    ChebyTransform t(Ny);

    f.makeSpectral(t);
    BasisFunc fy = ydiff(f);
    f.makeState(fstate, t);

    Complex d_dx = Complex(0.0, 2 * pi * f.kx() / f.Lx());
    Complex d_dz = Complex(0.0, 2 * pi * f.kz() / f.Lz());

    for (int ny = 0; ny < Ny; ++ny) {
        Complex u = f[0][ny];
        Complex v = f[1][ny];
        Complex w = f[2][ny];
        Complex uy = fy[0][ny];
        Complex wy = fy[2][ny];

        curlf[0].set(ny, wy - d_dz * v);
        curlf[1].set(ny, d_dz * u - d_dx * w);
        curlf[2].set(ny, d_dx * v - uy);
    }
}

// rtn = f dot grad g
void dotgrad(const BasisFunc& f, const BasisFunc& g, BasisFunc& rtn) {
    BasisFunc gradg;
    grad(g, gradg);
    dot(f, gradg, rtn);
}

BasisFlags::BasisFlags(BC a, BC b, bool zerodiv, bool orthonorm)
    : aBC(a), bBC(b), zerodivergence(zerodiv), orthonormalize(orthonorm) {}

void ubcFix(ChebyCoeff& u, BC aBC, BC bBC) {
    Real ua = u.a();
    Real ub = u.b();
    u.setBounds(-1, 1);

    if (aBC == Diri && bBC == Diri) {
        u[0] -= 0.5 * (u.eval_b() + u.eval_a());
        u[1] -= 0.5 * (u.eval_b() - u.eval_a());
    } else if (aBC == Diri)
        u[0] -= u.eval_a();
    else if (aBC == Diri)
        u[0] -= u.eval_b();

    u.setBounds(ua, ub);
}

void ubcFix(ComplexChebyCoeff& u, BC aBC, BC bBC) {
    Real ua = u.a();
    Real ub = u.b();
    u.setBounds(-1, 1);

    if (aBC == Diri && bBC == Diri) {
        u.sub(0, 0.5 * (u.eval_b() + u.eval_a()));
        u.sub(1, 0.5 * (u.eval_b() - u.eval_a()));
    } else if (aBC == Diri)
        u.sub(0, u.eval_a());
    else if (aBC == Diri)
        u.sub(0, u.eval_b());

    u.setBounds(ua, ub);
}

void vbcFix(ChebyCoeff& v, BC aBC, BC bBC) {
    Real va = v.a();
    Real vb = v.b();
    v.setBounds(-1, 1);

    if (aBC == Diri && bBC == Diri) {
        // Adjust v to match v(+-1)=v'(+-1)=0 BCs
        ChebyCoeff vy = diff(v);
        Real a = v.eval_a();
        Real b = v.eval_b();
        Real c = vy.eval_a();
        Real d = vy.eval_b();

        v[0] -= 0.5 * (a + b) + 0.125 * (c - d);
        v[1] -= 0.5625 * (b - a) - 0.0625 * (c + d);
        v[2] -= 0.125 * (d - c);
        v[3] -= 0.0625 * (a - b + c + d);
    } else if (aBC == Diri) {
        ChebyCoeff vy = diff(v);
        v[1] -= vy.eval_a();
        v[0] -= v.eval_a();
    } else if (bBC == Diri) {
        ChebyCoeff vy = diff(v);
        v[1] -= vy.eval_b();
        v[0] -= v.eval_b();
    }
    v.setBounds(va, vb);
}

void vbcFix(ComplexChebyCoeff& v, BC aBC, BC bBC) {
    Real va = v.a();
    Real vb = v.b();
    v.setBounds(-1, 1);

    if (aBC == Diri && bBC == Diri) {
        // Adjust v to match v(+-1)=v'(+-1)=0 BCs
        ComplexChebyCoeff vy = diff(v);
        Complex a = v.eval_a();
        Complex b = v.eval_b();
        Complex c = vy.eval_a();
        Complex d = vy.eval_b();

        v.sub(0, 0.5 * (a + b) + 0.125 * (c - d));
        v.sub(1, 0.5625 * (b - a) - 0.0625 * (c + d));
        v.sub(2, 0.125 * (d - c));
        v.sub(3, 0.0625 * (a - b + c + d));
    } else if (aBC == Diri) {
        ComplexChebyCoeff vy = diff(v);
        v.sub(1, vy.eval_a());
        v.sub(0, v.eval_a());
    } else if (bBC == Diri) {
        ComplexChebyCoeff vy = diff(v);
        v.sub(1, vy.eval_b());
        v.sub(0, v.eval_b());
    }
    v.setBounds(va, vb);
}

// Construct
void legendreV(int n, ChebyCoeff& v, BC aBC, BC bBC) {
    int Ny = v.numModes();
    ChebyTransform trans(Ny);
    v.setState(Physical);

    Real piN = pi / (Ny - 1);
    Real n_norm = sqrt(2.0 * n + 1.0);

    for (int q = 0; q < Ny; ++q)
        v[q] = n_norm * legendre(n, cos(q * piN));
    v.makeSpectral(trans);

    Real va = v.a();
    Real vb = v.b();
    v.setBounds(-1, 1);

    if (aBC == Diri && bBC == Diri) {
        // Adjust v to match v(+-1)=v'(+-1)=0 BCs
        ChebyCoeff vy = diff(v);
        Real a = v.eval_a();
        Real b = v.eval_b();
        Real c = vy.eval_a();
        Real d = vy.eval_b();

        v[0] -= 0.5 * (a + b) + 0.125 * (c - d);
        v[1] -= 0.5625 * (b - a) - 0.0625 * (c + d);
        v[2] -= 0.125 * (d - c);
        v[3] -= 0.0625 * (a - b + c + d);
    } else if (aBC == Diri) {
        ChebyCoeff vy = diff(v);
        v[1] -= vy.eval_a();
        v[0] -= v.eval_a();
    } else if (bBC == Diri) {
        ChebyCoeff vy = diff(v);
        v[1] -= vy.eval_b();
        v[0] -= v.eval_b();
    }
    v.setBounds(va, vb);
}

void legendreU(int n, ChebyCoeff& u, BC aBC, BC bBC) {
    int Ny = u.numModes();
    ChebyTransform trans(Ny);
    u.setState(Physical);

    Real piN = pi / (Ny - 1);
    Real n_norm = sqrt(2.0 * n + 1.0);

    for (int q = 0; q < Ny; ++q)
        u[q] = n_norm * legendre(n, cos(q * piN));
    u.makeSpectral(trans);

    Real ua = u.a();
    Real ub = u.b();
    u.setBounds(-1, 1);

    if (aBC == Diri && bBC == Diri) {
        u[0] -= 0.5 * (u.eval_b() + u.eval_a());
        u[1] -= 0.5 * (u.eval_b() - u.eval_a());
    } else if (aBC == Diri)
        u[0] -= u.eval_a();
    else if (bBC == Diri)
        u[0] -= u.eval_b();

    u.setBounds(ua, ub);
}

void legendreV2(int n, ChebyCoeff& v, BC aBC, BC bBC) {
    int Ny = v.numModes();
    ChebyTransform trans(Ny);
    v.setState(Physical);

    //

    Real piN = pi / (Ny - 1);
    Real n_norm = sqrt(2.0 * n + 1.0);

    for (int q = 0; q < Ny; ++q)
        v[q] = n_norm * legendre(n, cos(q * piN));
    v.makeSpectral(trans);

    Real va = v.a();
    Real vb = v.b();
    v.setBounds(-1, 1);

    if (aBC == Diri && bBC == Diri) {
        // Adjust v to match v(+-1)=v'(+-1)=0 BCs
        ChebyCoeff vy = diff(v);
        Real a = v.eval_a();
        Real b = v.eval_b();
        Real c = vy.eval_a();
        Real d = vy.eval_b();

        v[0] -= 0.5 * (a + b) + 0.125 * (c - d);
        v[1] -= 0.5625 * (b - a) - 0.0625 * (c + d);
        v[2] -= 0.125 * (d - c);
        v[3] -= 0.0625 * (a - b + c + d);
    } else if (aBC == Diri) {
        ChebyCoeff vy = diff(v);
        v[1] -= vy.eval_a();
        v[0] -= v.eval_a();
    } else if (bBC == Diri) {
        ChebyCoeff vy = diff(v);
        v[1] -= vy.eval_b();
        v[0] -= v.eval_b();
    }
    v.setBounds(va, vb);
}

// Return nth modal boundary-adapted Legendre polynomial, as described in
// CHQZ06 section 2.3.3. Mon Dec 11 16:48:21 EST 2006
void modalLegendre(int n, ChebyCoeff& u, ChebyTransform& trans) {
    assert(0 <= n && n <= u.N());

    // The model BC-adapted Legendre polynomials are
    // R0(x) = (1-x)/2;
    // R1(x) = (1+x)/2;
    // Rn(x) = (P{n-2}(x)-Pn(x))/sqrt(4n-2); for n>=2

    if (n == 0) {
        u.setState(Spectral);
        u.setToZero();
        u[0] = 0.5;
        u[1] = -0.5;
    } else if (n == 1) {
        u.setState(Spectral);
        u.setToZero();
        u[0] = 0.5;
        u[1] = 0.5;
    } else {
        u.setState(Physical);
        int N = u.N();
        Real piN = pi / (N - 1);
        Real nrm = 1.0 / sqrt(4 * n - 2);
        for (int j = 0; j < N; ++j) {
            Real x = cos(j * piN);
            u[j] = nrm * (legendre(n - 2, x) - legendre(n, x));
        }
        u.makeSpectral(trans);
    }
    return;
}
// void legendre(int n, ChebyCoeff& u, ChebyTransform& trans) {
//   assert(0 <= n && n <= u.N());
//
//   if (n==0) {
//     u.setState(Spectral);
//     u.setToZero();
//     u[0] = 1.0;
//   }
//   else if (n==1) {
//     u.setState(Spectral);
//     u.setToZero();
//     u[1] = 1.0;
//   }
//   else {
//     u.setState(Physical);
//     int N = u.N();
//     Real piN = pi/(N-1);
//     //Real nrm = 1.0/sqrt(4*n-2);
//     for (int j=0; j<N; ++j) {
//       Real x = cos(j*piN);
//       u[j] = legendre(n, x);
//     }
//     u.makeSpectral(trans);
//   }
//   return;
// }

//   v'(y) = -2pi k/L u(y)
//    v(y) = -2pi k/L int_0^y u(y') dy'
void integrateVmode(const ComplexChebyCoeff& u, int k, Real L, ComplexChebyCoeff& v) {
    integrate(u.re, v.im);
    v.im *= -2 * pi * k / L;
    v.im[0] -= v.im.mean();  // set const term to zero
}

//   v'(y) = -2pi (kx/Lx u(y) + kz/Lz w(y))
//    v(y) = -2pi int_0^y (kx/Lx u(y') + kz/Lz w(y')) dy'
void integrateVmode(const ComplexChebyCoeff& u, int kx, Real Lx, const ComplexChebyCoeff& w, int kz, Real Lz,
                    ComplexChebyCoeff& v) {
    integrate(u.re, v.re);  // use v.re as tmp
    v.re *= -2 * pi * kx / Lx;
    integrate(w.re, v.im);
    v.im *= -2 * pi * kz / Lz;
    v.im += v.re;
    v.im[0] -= v.im.mean();  // set const term to zero
    v.re.setToZero();
    return;
}

vector<BasisFunc> complexBasisKxKz(int Ny, int kx, int kz, Real Lx, Real Lz, Real a, Real b, const BasisFlags& flags) {
    BC aBC = flags.aBC;
    BC bBC = flags.bBC;

    vector<BasisFunc> f;  // the return value;

    if (!flags.zerodivergence) {
        ComplexChebyCoeff zero(Ny, a, b, Spectral);
        ComplexChebyCoeff Rn(Ny, a, b, Spectral);
        ChebyTransform trans(Ny);

        int Nbf = 3 * Ny;
        if (aBC == Diri)
            Nbf -= 3;
        if (bBC == Diri)
            Nbf -= 3;

        f.reserve(Nbf);
        //    int bfn = 0;

        // Note: R0(-1) = 1, R1(1) = 1, all other values of Rn(+/-1) = 0

        // Assign Legendre polynomials Pn(x) {ex,ey,ez} to basis set
        if (aBC == Free && bBC == Free) {
            for (int n = 0; n < Ny; ++n) {
                legendre(n, Rn.re, trans);
                f.push_back(BasisFunc(Rn, zero, zero, kx, kz, Lx, Lz));
                f.push_back(BasisFunc(zero, Rn, zero, kx, kz, Lx, Lz));
                f.push_back(BasisFunc(zero, zero, Rn, kx, kz, Lx, Lz));
            }
        }

        // Assign R1(x) {ex,ey,ez} = (1+x)/2 {ex,ey,ez} to basis set
        else if (aBC == Diri && bBC == Free) {
            for (int n = 1; n < Ny; ++n) {
                modalLegendre(n, Rn.re, trans);
                f.push_back(BasisFunc(Rn, zero, zero, kx, kz, Lx, Lz));
                f.push_back(BasisFunc(zero, Rn, zero, kx, kz, Lx, Lz));
                f.push_back(BasisFunc(zero, zero, Rn, kx, kz, Lx, Lz));
            }
        } else if (aBC == Free && bBC == Diri) {
            // Assign R0, {R2,...,R{N-1}} {ex ey ez} to basis set
            modalLegendre(0, Rn.re, trans);
            f.push_back(BasisFunc(Rn, zero, zero, kx, kz, Lx, Lz));
            f.push_back(BasisFunc(zero, Rn, zero, kx, kz, Lx, Lz));
            f.push_back(BasisFunc(zero, zero, Rn, kx, kz, Lx, Lz));
            for (int n = 2; n < Ny; ++n) {
                modalLegendre(n, Rn.re, trans);
                f.push_back(BasisFunc(Rn, zero, zero, kx, kz, Lx, Lz));
                f.push_back(BasisFunc(zero, Rn, zero, kx, kz, Lx, Lz));
                f.push_back(BasisFunc(zero, zero, Rn, kx, kz, Lx, Lz));
            }
        } else if (aBC == Diri && bBC == Diri) {
            // Assign {R2,...,R{N-1}} {ex ey ez} to basis set
            for (int n = 2; n < Ny; ++n) {
                modalLegendre(n, Rn.re, trans);
                f.push_back(BasisFunc(Rn, zero, zero, kx, kz, Lx, Lz));
                f.push_back(BasisFunc(zero, Rn, zero, kx, kz, Lx, Lz));
                f.push_back(BasisFunc(zero, zero, Rn, kx, kz, Lx, Lz));
            }
        }
    }

    // From here on is for zero-div basis sets
    else {
        // Let Rn(x) be the nth modal BC-adapted Legendre function (previous func).
        // Let * in the kx or kz col means non-zero

        // Then the linearly inpdt modes satisfying zero-div and BCs for each
        // kx,kz,aBC,bBC, are

        // kx kz aBC  bBC  type number rootmodes         #modes  case-label
        // 0  0  free free u    N      u=P0-P{N-1}
        //                 v    1      v=const
        //                 w    N      w=P0-P{N-1}       2N+1    Case A

        //       diri free u    N-1    u=R1-R{N-1}
        //                 v    0
        //                 w    N-1    w=R1-R{N-1}       2N-2    Case B

        //       free diri u    N-1    u=R0,R2-R{N-1}
        //                 v    0
        //                 w    N-1    w=R0,R2-R{N-1}    2N-2    Case C

        //       diri diri u    N-2    u=R2-R{N-1}
        //                 v    0
        //                 w    N-2    w=R2-R{N-1}       2N-4    Case D

        // *  0  free free u,v  N-1    u=P0-P{N-2}
        //                 v    1      v=const
        //                 w    N      w=P0-P{N-1}       2N      Case E

        //       diri free u,v  N-2    u=R1-R{N-2}
        //                 v    0
        //                 w    N-1    w=R1-R{N-1}       2N-3    Case F

        //       free diri u,v  N-2    u=R0,R2-R{N-2}
        //                 v    0
        //                 w    N-1    w=R0,R2-R{N-1}    2N-3    Case G

        //       diri diri u,v  N-4    u=R3-R{N-2}
        //                 v    0
        //                 w    N-2    w=R2-R{N-1}       2N-6    Case H

        // 0  *  free free u    N      u=P0-P{N-1}
        //                 v    1      v=const
        //                 w,v  N-1    w=P0-P{N-2}       2N      Case I

        //       diri free u    N-1    u=R1-R{N-1}
        //                 v    0
        //                 w,v  N-2    w=R1-R{N-2}       2N-3    Case J

        //       free diri u    N-1    u=R0,R2-R{N-1}
        //                 v    0
        //                 w,v  N-2    w=R0,R2-R{N-2}    2N-3    Case K

        //       diri diri u    N-2    u=R2-R{N-1}
        //                 v    0
        //                 w,v  N-4    w=R3-R{N-2}       2N-6    Case L

        // *  *  free free u,v  N-1    u=P0-P{N-2}
        //                 v    1      const
        //                 w,v  N-1    w=P0-P{N-2}
        //                 u,w  1      u=R{N-1}          2N      Case M

        //       diri free u,v  N-2    u=R1-R{N-2}
        //                 v    0
        //                 w,v  N-2    w=R1-R{N-2}
        //                 u,w  1      u=R{N-1}          2N-3    Case N

        //       free diri u,v  N-2    u=R0,R2-R{N-2}
        //                 v    0
        //                 w,v  N-2    w=R0,R2-R{N-2}
        //                 u,v  1      u=R{N-1}          2N-3    Case 0

        //       diri diri u,w  1      u=R2
        //                 u,v  N-4    u=R3-R{N-2}
        //                 v    0
        //                 w,v  N-4    w=R3-R{N-2}
        //                 u,w  1      u=R{N-1}          2N-6    Case P

        // In this table, the "type" column indicates how the div-free modes are
        // constructed. Type u has nonzero u component, other modes zero. Type u,v
        // has u and v modes nonzero, with u determined from the rootmodes
        // and v determined from u and the div-free condition.

        const int N = Ny;  // typographical help, compiler will optimize out
        ComplexChebyCoeff u(N, a, b, Spectral);
        ComplexChebyCoeff v(N, a, b, Spectral);
        ComplexChebyCoeff w(N, a, b, Spectral);
        ChebyTransform trans(N);

        if (kx == 0 && kz == 0) {
            // kx kz aBC  bBC  type number rootmodes          #modes  case-label
            // 0  0  free free u    N      u=P0-P{N-1}
            //                 v    1      v=const
            //                 w    N      w=P0-P{N-1}        2N+1    Case A
            if (aBC == Free && bBC == Free) {
                f.reserve(2 * N + 1);

                // Set the single v = const mode
                v.re[0] = 1;  // f[0]
                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                v.re[0] = 0.0;

                // Set the u = R0-R{N-1} modes
                for (int n = 0; n <= N - 1; ++n) {  // f[m] m=1,3,...,2N-1
                    legendre(n, u.re, trans);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                u.setToZero();

                // Set the w = R0-R{N-1} modes
                for (int n = 0; n <= N - 1; ++n) {  // f[m] m=2,4,...,2N
                    legendre(n, w.re, trans);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
            }

            // kx kz aBC  bBC  type number rootmodes         #modes  case-label
            // 0  0  diri free u    N-1    u=R1-R{N-1}
            //                 v    0
            //                 w    N-1    w=R1-R{N-1}       2N-2    Case B
            else if (aBC == Diri && bBC == Free) {
                f.reserve(2 * N - 2);

                // Set the u = R1-R{N-1} modes
                for (int n = 1; n <= N - 1; ++n) {  // f[m] m=0,2,...,2N-4
                    modalLegendre(n, u.re, trans);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                u.setToZero();

                // Set the w = R1-R{N-1} modes
                for (int n = 1; n <= N - 1; ++n) {  // f[m] m=1,3,...,2N-3
                    modalLegendre(n, w.re, trans);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
            }

            // kx kz aBC  bBC  type number modes           #modes  case-label
            // 0  0  free diri u    N-1    u=R0,R2-R{N-1}
            //                 v    0
            //                 w    N-1    w=R0,R2-R{N-1}   2N-2    Case C
            else if (aBC == Free && bBC == Diri) {
                f.reserve(2 * N - 2);

                // Set the u = R0 mode                  // f[0]
                modalLegendre(0, u.re, trans);
                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));

                // Set the u = R2-R{N-1} modes
                for (int n = 2; n <= N - 1; ++n) {  // f[m] m=2,4,...,2N-4
                    modalLegendre(n, u.re, trans);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                u.setToZero();

                // Set the w = R0 mode                  // f[1]
                modalLegendre(0, w.re, trans);
                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));

                // Set the w = R2-R{N-1} modes
                for (int n = 2; n <= N - 1; ++n) {  // f[m] m=3,5,...,2N-3
                    modalLegendre(n, w.re, trans);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
            }

            // kx kz aBC  bBC  type number rootmodes         #modes  case-label
            // 0  0  diri diri u    N-2    u=R2-R{N-1}
            //                 v    0
            //                 w    N-2    w=R2-R{N-1}       2N-4    Case D
            else if (aBC == Diri && bBC == Diri) {
                f.reserve(2 * N - 4);

                // Set the u = R2-R{N-1} modes
                for (int n = 2; n <= N - 1; ++n) {  // f[m] m=0,2,4,...,2N-6
                    modalLegendre(n, u.re, trans);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                u.setToZero();

                // Set the w = R2-R{N-1} modes
                for (int n = 2; n <= N - 1; ++n) {  // f[m] m=1,3,5,...,2N-5
                    modalLegendre(n, w.re, trans);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
            }
        }

        else if (kx != 0 && kz == 0) {
            // kx kz aBC  bBC  type number rootmodes         #modes  case-label
            // *   0 free free u,v  N-1    u=P0-P{N-2}
            //                 v    1      v=const
            //                 w    N      w=P0-P{N-1}       2N      Case E
            if (aBC == Free && bBC == Free) {
                f.reserve(2 * N);

                // Set the single v==const mode
                v.re[0] = 1;
                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));  // f[0]
                v.re[0] = 0.0;

                // Set the u,v modes with u=P0-P{N-2}
                for (int n = 0; n <= N - 2; ++n) {  // f[m] m=2,4,...,2N-2
                    legendre(n, u.re, trans);
                    integrateVmode(u, kx, Lx, v);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                u.setToZero();
                v.setToZero();

                // Set the w = P0-P{N-1} modes
                for (int n = 0; n <= N - 1; ++n) {  // f[m] m=1,3,...,2N-1
                    legendre(n, w.re, trans);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
            }

            // kx kz aBC  bBC  type number rootmodes         #modes  case-label
            // *  0  diri free u,v  N-2    u=R1-R{N-2}
            //                 v    0
            //                 w    N-1    w=R1-R{N-1}       2N-3    Case F
            else if (aBC == Diri && bBC == Free) {
                f.reserve(2 * N - 3);

                // Set the u,v modes with u=R1-R{N-2}
                for (int n = 1; n <= N - 2; ++n) {  // f[m] m=1,3,...,2N-5
                    modalLegendre(n, u.re, trans);
                    integrateVmode(u, kx, Lx, v);
                    v.im[0] -= v.im.eval_a();  // enforce aBC
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                u.setToZero();
                v.setToZero();

                // Set the w = R1-R{N-1} modes
                for (int n = 1; n <= N - 1; ++n) {  // f[m] m=0,2,...,2N-4
                    modalLegendre(n, w.re, trans);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
            }

            // kx kz aBC  bBC  type number rootmodes        #modes  case-label
            // *  0  free diri u,v  N-2    u=R0,R2-R{N-2}
            //                 v    0
            //                 w    N-1    w=R0,R2-R{N-1}   2N-3    Case G
            else if (aBC == Free && bBC == Diri) {
                f.reserve(2 * N - 3);

                // Set the u,v mode with u=R0      // f[1]
                modalLegendre(0, u.re, trans);
                integrateVmode(u, kx, Lx, v);
                v.im[0] -= v.im.eval_b();  // enforce bBC

                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));

                // Set the u,v modes with u=R2-R{N-2}
                for (int n = 2; n <= N - 2; ++n) {  // f[m] m=3,...,2N-5
                    modalLegendre(n, u.re, trans);
                    integrateVmode(u, kx, Lx, v);
                    v.im[0] -= v.im.eval_b();  // enforce aBC
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                u.setToZero();
                v.setToZero();

                // Set the w = R0 mode             // f[0]
                modalLegendre(0, w.re, trans);
                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));

                // Set the w = R2-R{N-1} modes
                for (int n = 2; n <= N - 1; ++n) {  // f[m] m=2,...,2N-4
                    modalLegendre(n, w.re, trans);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
            }

            // kx kz aBC  bBC  type number rootmodes         #modes  case-label
            //  *  0 diri diri u,v  N-4    u=R3-R{N-2}
            //                 v    0
            //                 w    N-2    w=R2-R{N-1}       2N-6    Case H
            else if (aBC == Diri && bBC == Diri) {
                f.reserve(2 * N - 6);

                // Set the w = R2-R{N-2} modes
                for (int n = 2; n <= N - 2; ++n) {  // f[m] m=0,...,2N-8
                    modalLegendre(n, w.re, trans);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                // Set the w = R{N-1} mode         // f[2N-7]
                modalLegendre(N - 1, w.re, trans);
                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                w.setToZero();

                // Set the u,v modes with u=R3-R{N-2}
                for (int n = 3; n <= N - 2; ++n) {  // f[m] m=1,...,2N-9
                    modalLegendre(n, u.re, trans);
                    integrateVmode(u, kx, Lx, v);
                    v.im[0] -= v.im.eval_a();  // enforce aBC and bBC
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
            }
        }

        else if (kx == 0 && kz != 0) {
            // kx kz aBC  bBC  type number rootmodes         #modes  case-label
            // 0  *  free free u    N      u=P0-P{N-1}
            //                 v    1      v=const
            //                 w,v  N-1    w=PdR0-P{N-2}       2N      Case I
            if (aBC == Free && bBC == Free) {
                f.reserve(2 * N);

                // Set the single v==const mode
                v.re[0] = 1;
                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));  // f[0]
                v.re[0] = 0.0;

                // Set the w,v modes with w = P0-P{N-2}
                for (int n = 0; n <= N - 2; ++n) {  // f[m] m=2,4,...,2N-2
                    legendre(n, w.re, trans);
                    integrateVmode(w, kz, Lz, v);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                w.setToZero();
                v.setToZero();

                // Set the u = P0-P{N-1} modes
                for (int n = 0; n <= N - 1; ++n) {  // f[m] m=1,3,...,2N-1
                    legendre(n, u.re, trans);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
            }

            // kx kz aBC  bBC  type number rootmodes         #modes  case-label
            // 0  *  diri free u    N-1    u=R1-R{N-1}
            //                 v    0
            //                 w,v  N-2    w=R1-R{N-2}       2N-3    Case J
            else if (aBC == Diri && bBC == Free) {
                f.reserve(2 * N - 3);

                // Set the w,v modes with w=R1-R{N-2}
                for (int n = 1; n <= N - 2; ++n) {  // f[m] m=1,3,...,2N-5
                    modalLegendre(n, w.re, trans);
                    integrateVmode(w, kz, Lz, v);
                    v.im[0] -= v.im.eval_a();  // enforce aBC
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                w.setToZero();
                v.setToZero();

                // Set the u = R1-R{N-1} modes
                for (int n = 1; n <= N - 1; ++n) {  // f[m] m=0,2,...,2N-4
                    modalLegendre(n, u.re, trans);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
            }

            // kx kz aBC  bBC  type number rootmodes        #modes  case-label
            // 0  *  free diri u    N-1    u=R0,R2-R{N-1}
            //                 v    0
            //                 w,v  N-2    w=R0,R2-R{N-2}   2N-3    Case K
            else if (aBC == Free && bBC == Diri) {
                f.reserve(2 * N - 3);

                // Set the w,v mode with w=R0       // f[1]
                modalLegendre(0, w.re, trans);
                integrateVmode(w, kz, Lz, v);
                v.im[0] -= v.im.eval_b();  // enforce bBC
                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));

                // Set the w,v modes with w=R2-R{N-2}
                for (int n = 2; n <= N - 2; ++n) {  // f[m] m=3,...,2N-5
                    modalLegendre(n, w.re, trans);
                    integrateVmode(w, kz, Lz, v);
                    v.im[0] -= v.im.eval_b();  // enforce bBC
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                w.setToZero();
                v.setToZero();

                // Set the u = R0 mode              // f[0]
                modalLegendre(0, u.re, trans);
                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));

                // Set the u = R2-R{N-1} modes
                for (int n = 2; n <= N - 1; ++n) {  // f[m] m=2,...,2N-4
                    modalLegendre(n, u.re, trans);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
            }

            // kx kz aBC  bBC  type number rootmodes         #modes  case-label
            // 0  *  diri diri u    N-2    u=R2-R{N-1}
            //                 v    0
            //                 w,v  N-4    w=R3-R{N-2}       2N-6    Case L
            else if (aBC == Diri && bBC == Diri) {
                f.reserve(2 * N - 6);

                // Set the u = R2-R{N-2} modes
                for (int n = 2; n <= N - 2; ++n) {  // f[m] m=0,...,2N-8
                    modalLegendre(n, u.re, trans);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }

                // Set the u = R{N-1} mode         // f[2N-7]
                modalLegendre(N - 1, u.re, trans);
                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                u.setToZero();

                // Set the w,v modes with w=R3-R{N-2}
                for (int n = 3; n <= N - 2; ++n) {  // f[m] m=1,...,2N-9
                    modalLegendre(n, w.re, trans);
                    integrateVmode(w, kz, Lz, v);
                    v.im[0] -= v.im.eval_a();  // enforce aBC and bBC
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
            }
        }

        else if (kx != 0 && kz != 0) {
            // kx kz aBC  bBC  type number rootmodes         #modes  case-label
            // *  *  free free u,v  N-1    u=P0-P{N-2}
            //                 v    1      v=const
            //                 w,v  N-1    w=P0-P{N-2}
            //                 u,w  1      u=P{N-1}          2N      Case M
            if (aBC == Free && bBC == Free) {
                f.reserve(2 * N);

                // Set the single v==const mode
                v.re[0] = 1;
                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));  // f[0]
                v.re[0] = 0.0;

                // Set the u,v modes with u==P0-P{N-2}
                for (int n = 0; n <= N - 2; ++n) {  // f[m] m=1,3,...,2N-3
                    legendre(n, u.re, trans);
                    integrateVmode(u, kx, Lx, v);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                u.setToZero();
                v.setToZero();

                // Set the w,v modes with w==P0-P{N-2}
                for (int n = 0; n <= N - 2; ++n) {  // f[m] m=2,4,...,2N-2
                    legendre(n, w.re, trans);
                    integrateVmode(w, kz, Lz, v);
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                w.setToZero();
                v.setToZero();

                // Set the single u,w mode with u=P{N-1} mode
                legendre(N - 1, u.re, trans);  // f[2N-1]
                w.re = u.re;
                w.re *= -kx * Lz / (kz * Lx);
                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
            }

            // kx kz aBC  bBC  type number rootmodes         #modes  case-label
            // *  *  diri free u,v  N-2    u=R1-R{N-2}
            //                 v    0
            //                 w,v  N-2    w=R1-R{N-2}
            //                 u,w  1      u=R{N-1}          2N-3    Case N
            else if (aBC == Diri && bBC == Free) {
                f.reserve(2 * N - 3);

                // Set the u,v modes with u=R1-R{N-2}
                for (int n = 1; n <= N - 2; ++n) {  // f[m] m=0,2,...,2N-6
                    modalLegendre(n, u.re, trans);
                    integrateVmode(u, kx, Lx, v);
                    v.im[0] -= v.im.eval_a();  // enforce aBC
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                u.setToZero();
                v.setToZero();

                // Set the w,v modes with w=R1-R{N-2}
                for (int n = 1; n <= N - 2; ++n) {  // f[m] m=1,3,...,2N-5
                    modalLegendre(n, w.re, trans);
                    integrateVmode(w, kz, Lz, v);
                    v.im[0] -= v.im.eval_a();  // enforce aBC
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                w.setToZero();
                v.setToZero();

                // Set the single u,w mode with uR{N-1}
                modalLegendre(N - 1, u.re, trans);
                w.re = u.re;  // f[2N-4]
                w.re *= -kx * Lz / (kz * Lx);
                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
            }

            // kx kz aBC  bBC  type number rootmodes         #modes  case-label
            // *  *  free diri u,v  N-2    u=R0,R2-R{N-2}
            //                 v    0
            //                 w,v  N-2    w=R0,R2-R{N-2}
            //                 u,w  1      u=R{N-1}          2N-3    Case 0
            else if (aBC == Free && bBC == Diri) {
                f.reserve(2 * N - 3);

                // Set the u,v mode with u=R0
                modalLegendre(0, u.re, trans);  // f[0]
                integrateVmode(u, kx, Lx, v);
                v.im[0] -= v.im.eval_b();  // enforce bBC
                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));

                // Set the u,v modes with u=R2-R{N-2}
                for (int n = 2; n <= N - 2; ++n) {  // f[m] m=2,...,2N-6
                    modalLegendre(n, u.re, trans);
                    integrateVmode(u, kx, Lx, v);
                    v.im[0] -= v.im.eval_b();  // enforce bBC
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                u.setToZero();
                v.setToZero();

                // Set the w,v mode with w=R0
                modalLegendre(0, w.re, trans);  // f[1]
                integrateVmode(w, kz, Lz, v);
                v.im[0] -= v.im.eval_b();  // enforce bBC
                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));

                // Set the w,v mode with w=R2-R{N-1}
                for (int n = 2; n <= N - 2; ++n) {  // f[m] m=3,...,2N-5
                    modalLegendre(n, w.re, trans);
                    integrateVmode(w, kz, Lz, v);
                    v.im[0] -= v.im.eval_b();  // enforce bBC
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                w.setToZero();
                v.setToZero();

                // Set the single u,w mode with u=R{N-1}
                modalLegendre(N - 1, u.re, trans);
                w.re = u.re;  // f[2N-4]
                w.re *= -kx * Lz / (kz * Lx);
                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
            }

            // kx kz aBC  bBC  type   number rootmodes       #modes  case-label
            // *  *  diri diri u,w    1      u=R2
            //                 u,v    N-4    u=R3-R{N-2}
            //                 v      0
            //                 w,v    N-4    w=R2-R{N-2}
            //                 u,w    1      u=R{N-1}          2N-6    Case P
            else if (aBC == Diri && bBC == Diri) {
                f.reserve(2 * N - 6);

                // Set the single mode with u,w based on R2 and v by integration
                modalLegendre(2, u.re, trans);
                w.re = u.re;
                w.re *= -kx * Lz / (kz * Lx);
                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));  // f[0]

                u.setToZero();
                w.setToZero();

                // Set the u,v modes with u=R3-R{N-2}
                for (int n = 3; n <= N - 2; ++n) {  // f[m] m=1,3,...,2N-9
                    modalLegendre(n, u.re, trans);
                    integrateVmode(u, kx, Lx, v);
                    v.im[0] -= v.im.eval_a();  // enforce aBC and bBC
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                u.setToZero();
                v.setToZero();

                // Set the w,v with w=R3-R{N-2}
                for (int n = 3; n <= N - 2; ++n) {  // f[m] m=2,4,...,2N-8
                    modalLegendre(n, w.re, trans);
                    integrateVmode(w, kz, Lz, v);
                    v.im[0] -= v.im.eval_a();  // enforce aBC and bBC
                    f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
                }
                w.setToZero();
                v.setToZero();

                // Set the single u,w mode with u=R{N-1}
                modalLegendre(N - 1, u.re, trans);
                w.re = u.re;  // f[2N-7]
                w.re *= -kx * Lz / (kz * Lx);
                f.push_back(BasisFunc(u, v, w, kx, kz, Lx, Lz));
            }
        }
    }
    if (flags.orthonormalize) {
        orthonormalize(f);
        //    orthonormalize(f); //JH Orthonormalize needs to be called twice?
    }
    return f;
}

// Begun on Wed Dec 13 09:53:17 EST 2006
vector<BasisFunc> complexBasis(int Ny, int kxmax, int kzmax, Real Lx, Real Lz, Real a, Real b,
                               const BasisFlags& flags) {
    // DO NOT CHANGE THE COUNTING OF BASIS FUNCTIONS WITHOUT MODIFIYING
    // THE COUNTING IN realBasis() in realprofile.cpp AS WELL.

    // Calculate the total number of independent modes for 3d velocity fields.
    // This is not easy to understand! It requires careful, tedious counting.
    // Consider the following kx,kz grid of Fourier modes. The letters mark
    // different portions of Fourier modes that have different forms for the
    // polynomials in y.

    //    0 1 2 3 4 5 6 7 kz
    //  0 a b b b b b b .
    //  1 c e e e e e e .
    //  2 c e e e e e e .
    //  3 c e e e e e e .
    //  4 c e e e e e e .
    //  5 . . . . . . . .
    // -4 f g g g g g g .
    // -3 f g g g g g g .
    // -2 f g g g g g g .
    // -1 f g g g g g g .
    // kx

    // For this grid Nx=10, Nz=8, kxmax=4 and kzmax=6. Cosine modes are marked
    // with '.' and are always set to zero (see Trefethen96).

    // The first trick is that modes marked 'f' are complex conjugates of modes
    // marked 'c', so we do not need basis functions for those Fourier modes to
    // determine the linearly independent coefficients in the expansion.

    // If we're constructing a nonzero-div basis, the components u,v,w are indpt,
    // and there are 3*(Ny - #dirichletBCs) polynomials in y per Fourier mode,
    // and [(kzmax+1)*(kxmax+1)+kzmax*kxmax] Fourier modes. Problem solved.
    //         a,b,c,e block      g block

    BC aBC = flags.aBC;
    BC bBC = flags.bBC;
    int Nbf = 0;
    int Nbc = ((aBC == Diri) ? 1 : 0) + ((bBC == Diri) ? 1 : 0);

    // With no divergence constraint between u,v,w, have 3 indpt comps of
    // velocity, times number of polynomials of max degree Ny that match BCs,
    // times (# a,b,c,e Fourier modes + # g Fourier modes)
    if (flags.zerodivergence == false)
        Nbf = 3 * (Ny - Nbc) * ((kzmax + 1) * (kxmax + 1) + kzmax * kxmax);

    // For zero-div basis sets, the counting is more complex. The number of
    // linearly indepedent zero-div polynomials of maximum degree Ny for the
    // above blocks is
    //                        diri,free
    // block      free,free   free,diri   diri,diri
    // a       :    2Ny+1       2Ny-2       2Ny-4
    // b,c,e,g :    2Ny         2Ny-3       2Ny-6
    //
    // The explanation for these numbers can be found in the comments for the
    // complexBasisKxKz(kx,kz,...) function above. The table can be summarized;
    // The 0,0 Fourier mode has 1 more polynomial in y than other Fourier modes.
    // The first Dirichlet BC removes 3 polynomials, the second, 2.
    else {
        // ppfm stands for the number of polynomials per Fourier mode
        int a_ppfm = 0;  // for the 'a' mode: (0,0) Fourier mode
        int b_ppfm = 0;  // for the 'b' and all other modes

        if (Nbc == 0) {
            a_ppfm = 2 * Ny + 1;
            b_ppfm = 2 * Ny;
        } else if (Nbc == 1) {
            a_ppfm = 2 * Ny - 2;
            b_ppfm = 2 * Ny - 3;
        } else if (Nbc == 2) {
            a_ppfm = 2 * Ny - 4;
            b_ppfm = 2 * Ny - 6;
        }

        //    b_ppfm*( (blocks a,b,c,e)  +  (block g)  - (block a))
        //+   a_ppfm*(block a)

        Nbf = b_ppfm * ((kzmax + 1) * (kxmax + 1) + kzmax * kxmax - 1) + a_ppfm * 1;
    }
    // Now fill out them modes!
    // That's 'Git er done!' in the parlance of the south
    vector<BasisFunc> f;
    f.reserve(Nbf);
    //  int j=0;

    // blocks a,b,c,e
    for (int kx = 0; kx <= kxmax; ++kx) {
        for (int kz = 0; kz <= kzmax; ++kz) {
            vector<BasisFunc> fkxkz = complexBasisKxKz(Ny, kx, kz, Lx, Lz, a, b, flags);
            f.insert(f.end(), fkxkz.begin(), fkxkz.end());
            /*      for (int n=0; n<fkxkz.N(); ++n)
                    f[j++] = fkxkz[n];*/
        }
    }
    // block g
    for (int kx = -1; kx >= -kxmax; --kx) {
        for (int kz = 1; kz <= kzmax; ++kz) {
            vector<BasisFunc> fkxkz = complexBasisKxKz(Ny, kx, kz, Lx, Lz, a, b, flags);
            f.insert(f.end(), fkxkz.begin(), fkxkz.end());
            //       for (int n=0; n<fkxkz.N(); ++n)
            // 	f[j++] = fkxkz[n];
        }
    }
    return f;
}

void orthonormalize(vector<BasisFunc>& f) {
    // Modified Gram-Schmidt orthogonalization.
    //  int N=f.size();
    BasisFunc fm_tmp;
    const Real EPSILON = 1e-14;
    vector<BasisFunc>::iterator m;
    for (m = f.begin(); m != f.end(); ++m) {
        BasisFunc& fm = *m;
        Real nrm = L2Norm(fm);
        if (nrm < EPSILON) {
            cerr << "Ran into a basis function with L2Norm < " << EPSILON << " in orthonormalize(vector<BasisFunc>& f)."
                 << endl;
            cerr << "Saving the basis function as error.asc." << endl;
            fm.save("error");
        }
        fm *= 1.0 / nrm;

        int fmkx = fm.kx();
        int fmkz = fm.kz();

        for (vector<BasisFunc>::iterator n = m + 1; n != f.end(); ++n) {
            BasisFunc& fn = *n;

            if (fmkx == fn.kx() && fmkz == fn.kz()) {
                fm_tmp = fm;
                fm_tmp *= L2InnerProduct(fm, fn);
                fn -= fm_tmp;
            }
        }
    }
    return;
}

void checkBasis(const vector<BasisFunc>& e, const BasisFlags& flags, bool orthogcheck) {
    int M = e.size();
    cout << M << " elements in basis set" << endl;
    if (flags.orthonormalize) {
        int bad12 = 0;
        int bad8 = 0;
        int bad4 = 0;
        cout << "\nchecking normality..." << endl;
        int count = 0;
        for (vector<BasisFunc>::const_iterator m = e.begin(); m != e.end(); ++m) {
            Real norm = L2Norm(*m);
            Real err = abs(norm - 1.0);
            if (err > 1e-12) {
                m->save("badnorm" + i2s(count));
                ++bad12;
            }
            if (err > 1e-8)
                ++bad8;
            if (err > 1e-4)
                ++bad4;
            ++count;
        }
        cout << bad12 << " with abs(L2Norm(em)-1) > 1e-12" << endl;
        cout << bad8 << " with abs(L2Norm(em)-1) > 1e-8" << endl;
        cout << bad4 << " with abs(L2Norm(em)-1) > 1e-4" << endl;
    }

    if (flags.zerodivergence) {
        cout << "\nchecking divergence..." << endl;
        int bad12 = 0;
        int bad8 = 0;
        int bad4 = 0;
        int count = 0;
        for (vector<BasisFunc>::const_iterator m = e.begin(); m != e.end(); ++m) {
            Real err = divNorm(*m);
            if (err > 1e-12) {
                m->save("baddiv" + i2s(count));
                ++bad12;
            }
            if (err > 1e-8)
                ++bad8;
            if (err > 1e-4)
                ++bad4;
            ++count;
        }
        cout << bad12 << " with divNorm(em) > 1e-12" << endl;
        cout << bad8 << " with divNorm(em) > 1e-8" << endl;
        cout << bad4 << " with divNorm(em) > 1e-4" << endl;
    }

    if (flags.aBC == Diri || flags.bBC == Diri) {
        cout << "\nchecking boundary conditions..." << endl;
        int bad12 = 0;
        int bad8 = 0;
        int bad4 = 0;
        int count = 0;
        for (vector<BasisFunc>::const_iterator m = e.begin(); m != e.end(); ++m) {
            Real err = m->bcNorm(flags.aBC, flags.bBC);
            if (err > 1e-12) {
                m->save("badbc" + i2s(count));
                ++bad12;
            }
            if (err > 1e-8)
                ++bad8;
            if (err > 1e-4)
                ++bad4;
            ++count;
        }
        cout << bad12 << " with bcNorm(em) > 1e-12" << endl;
        cout << bad8 << " with bcNorm(em) > 1e-8" << endl;
        cout << bad4 << " with bcNorm(em) > 1e-4" << endl;
    }
    if (flags.orthonormalize && orthogcheck) {
        cout << "\nchecking orthogonality..." << endl;
        int bad12 = 0;
        int bad8 = 0;
        int bad4 = 0;
        for (int m = 1; m < M; ++m) {
            for (int n = 0; n < m; ++n) {
                Real ip = abs(L2InnerProduct(e[m], e[n]));
                if (ip > 1e-12)
                    ++bad12;
                if (ip > 1e-8)
                    ++bad8;
                if (ip > 1e-4)
                    ++bad4;
            }
        }
        cout << bad12 << " with abs(L2IP(em,en)) > 1e-12" << endl;
        cout << bad8 << " with abs(L2IP(em,en)) > 1e-8" << endl;
        cout << bad4 << " with abs(L2IP(em,en)) > 1e-4" << endl;
    }
}

BasisFunc& BasisFunc::operator*=(const FieldSymmetry& sigma) {
    assert(Nd_ == 3);

    // Identity escape clause
    if (sigma.sx() == 1 && sigma.sy() == 1 && sigma.sz() == 1 && sigma.ax() == 0.0 && sigma.az() == 0.0)
        return *this;

    // (u,v,w)(x,y,z) -> (su sx u, su sy v, su sz w) (sx x + ax*Lx, sy y, sz z + fz*Lz)
    int s = sigma.s();
    int sx = sigma.sx();
    int sy = sigma.sy();
    int sz = sigma.sz();

    kx_ *= sx;
    kz_ *= sz;

    Real cx = kx_ * 2 * pi * sigma.ax();
    Real cz = kz_ * 2 * pi * sigma.az();

    for (int nd = 0; nd < Nd_; ++nd) {
        u_[nd] *= s;
        u_[nd] *= exp(Complex(0, cx));
        u_[nd] *= exp(Complex(0, cz));
    }

    if (sy == -1) {
        assert(a_ + b_ == 0);
        fieldstate fs = state_;

        if (fs == Physical) {
            for (int nd = 0; nd < Nd_; ++nd) {
                ChebyCoeff Ur = u_[nd].re;
                ChebyCoeff Ui = u_[nd].im;
                for (int ny = 0; ny <= Ny_ / 2; ++ny) {
                    Real tmp = Ur[ny];
                    Ur[ny] = Ur[Ny_ - 1 - ny];
                    Ur[Ny_ - 1 - ny] = tmp;

                    tmp = Ui[ny];
                    Ui[ny] = Ui[Ny_ - 1 - ny];
                    Ui[Ny_ - 1 - ny] = tmp;
                }
            }
        }
        if (fs == Spectral) {
            for (int nd = 0; nd < Nd_; ++nd) {
                ChebyCoeff Ur = u_[nd].re;
                ChebyCoeff Ui = u_[nd].im;
                // 2/3 aliasing
                for (int ny = 0; ny < Ny_ / 3; ++ny) {
                    Ur[2 * ny + 1] *= -1;
                    Ui[2 * ny + 1] *= -1;
                }
            }
        }
    }

    return *this;
}

}  // namespace chflow
