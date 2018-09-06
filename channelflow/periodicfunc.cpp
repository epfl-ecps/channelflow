/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include <fstream>
#include <iomanip>

#include "channelflow/periodicfunc.h"

using namespace std;

namespace chflow {

Vector periodicpoints(int N, Real L) {
    Vector x(N + 1);
    Real dx = L / N;
    for (int n = 0; n <= N; ++n)
        x[n] = n * dx;
    return x;
}

PeriodicFunc::PeriodicFunc(uint N, Real L, fieldstate s, uint fftw_flags)
    : data_handle_(nullptr, fftw_free),
      state_(s),
      fftw_flags_(fftw_flags | FFTW_DESTROY_INPUT),
      forward_plan_(nullptr, fftw_destroy_plan),
      inverse_plan_(nullptr, fftw_destroy_plan) {
    resize(N, L);
}

PeriodicFunc::PeriodicFunc() : PeriodicFunc::PeriodicFunc(0, 0, Spectral, FFTW_ESTIMATE) {}

PeriodicFunc::PeriodicFunc(const PeriodicFunc& f) : PeriodicFunc::PeriodicFunc(f.N_, f.L_, f.state_, f.fftw_flags_) {
    copy(f.rdata_, f.rdata_ + Npad(), rdata_);
}

PeriodicFunc::PeriodicFunc(const string& filebase) : PeriodicFunc::PeriodicFunc() {
    ifstream is;
    string filename = ifstreamOpen(is, filebase, ".asc");
    if (!is) {
        cerr << "PeriodicFunc::PeriodicFunc(filebase) : can't open file " << filebase << " or " << (filebase + ".asc")
             << endl;
        exit(1);
    }

    // Read in header. Form is "% N L s"
    char c;
    int Nval;
    Real Lval;
    is >> c;
    if (c != '%') {
        cerr << "PeriodicFunc(filebase): bad header in file " << filename << endl;
        exit(1);
    }
    is >> Nval >> Lval >> state_;
    resize(Nval, Lval);

    if (state_ == Physical) {
        for (uint n = 0; n < N_; ++n) {
            is >> rdata_[n];
            assert(is.good());
        }
        for (uint n = N_; n < Npad(); ++n)
            rdata_[n] = 0.0;
    } else
        for (uint n = 0; n < Npad(); ++n) {
            is >> rdata_[n];
            assert(is.good());
        }
    is.close();
}

PeriodicFunc& PeriodicFunc::operator=(const PeriodicFunc& f) {
    if (this != &f) {
        PeriodicFunc tmp(f);
        swap(*this, tmp);
    }
    return *this;
}

void PeriodicFunc::resize(uint N, Real L) {
    assert(L > 0);
    N_ = N;
    L_ = L;

    // Reset the memory that is allocated to contain data
    data_handle_.reset(fftw_malloc((Npad()) * sizeof(Real)));

    rdata_ = static_cast<Real*>(data_handle_.get());
    cdata_ = static_cast<Complex*>(data_handle_.get());

    // Recompute FFTW plans once the data buffer has been allocated
    auto fcdata = static_cast<fftw_complex*>(data_handle_.get());

    forward_plan_.reset(static_cast<fftw_plan>(fftw_plan_dft_r2c_1d(N_, rdata_, fcdata, fftw_flags_)));

    inverse_plan_.reset(static_cast<fftw_plan>(fftw_plan_dft_c2r_1d(N_, fcdata, rdata_, fftw_flags_)));

    // Initialize to 0.0
    fill(rdata_, rdata_ + Npad(), 0.0);
}

void PeriodicFunc::fft() {
    assert(state_ == Physical);
    fftw_execute(forward_plan_.get());
    state_ = Spectral;

    // For N_ even, set last cosine mode to zero. See Trefethen96.
    if (N_ % 2 == 0)
        cdata_[Nmodes() - 1] = Complex(0.0, 0.0);

    Real c = 1.0 / N_;
    for (uint n = 0; n < Nmodes(); ++n)
        cdata_[n] *= c;
}

void PeriodicFunc::ifft() {
    assert(state_ == Spectral);
    fftw_execute(inverse_plan_.get());
    state_ = Physical;
}

void PeriodicFunc::makeSpectral() {
    if (state_ == Physical)
        fft();
}
void PeriodicFunc::makePhysical() {
    if (state_ == Spectral)
        ifft();
}

void PeriodicFunc::makeState(fieldstate s) {
    if (state_ != s) {  // need to change state?
        if (state_ == Physical)
            fft();  // state is Physical; do forward fft
        else
            ifft();  // state is Spectral; do inverse fft
    }
}

void PeriodicFunc::randomize(Real magn, Real decay) {
    fieldstate startState = state_;
    state_ = Spectral;
    Real magn_n = magn / decay;

    // The following respects the FFTW data-layout constraints that the
    // first & last mode are Real, and the Channelflow convention that the
    // last mode for N_ even (pure cosine mode) is set to zero. (Trefethen96)
    cdata_[0] = magn_n * Complex(randomReal(-1, 1), 0.0);

    for (uint n = 1; n < Nmodes() - 2; ++n)
        cdata_[n] = (magn_n *= decay) * randomComplex();

    if (N_ % 2 == 0)
        cdata_[Nmodes() - 1] = Complex(0.0, 0.0);

    makeState(startState);
}

void PeriodicFunc::setToZero() {
    for (uint n = 0; n < Npad(); ++n)
        rdata_[n] = 0.0;
}

/**************************************
void PeriodicFunc::fill(const PeriodicFunc& v) {
  assert(v.state_ == Spectral);
  assert(state_ == Spectral);
  int Ncommon = lesser(N_, v.N_);
  for (uint i=0; i<Ncommon; ++i)
    data_[i] = v.data_[i];
  for (uint i=Ncommon; i<N_; ++i)
    data_[i] = 0.0;
}

void PeriodicFunc::interpolate(const PeriodicFunc& v) {
  assert(L_ == v.L_);
  assert(v.state_ == Spectral);
  state_ = Physical;
  for (uint n=0; n<N_; ++n)
    data_[n] = v.eval(x[n]);
  //makeSpectral();
}

void PeriodicFunc::reflect(const PeriodicFunc& v, parity p) {
  assert((a_+b_)/2 == v.a() && b_ <= v.b() && b_ > v.a());
  assert(v.state_ == Spectral);
  state_ = Physical;
  Real piN = pi/(N_-1);
  Real width = (b_-a_)/2;
  Real center = (b_+a_)/2;
  int N2=N_/2;
  int N1=N_-1;
  int sign = (p==Odd) ? -1 : 1;
  for (uint n=0; n<N2; ++n) {
    Real tmp = v.eval(center + width*cos(n*piN));
    data_[n] = tmp;
    data_[N1-n] = sign*tmp;
  }
  FFT1d t(N_);
  makeSpectral(t);
  for (uint n=2*N_/3; n<N_; ++ n)
    data_[n] = 0.0;
  makePhysical(t);
}
*************************/

void PeriodicFunc::setLength(Real L) {
    L_ = L;
    assert(L_ > 0);
}

void PeriodicFunc::setState(fieldstate s) {
    assert(s == Physical || s == Spectral);
    state_ = s;
}

Real PeriodicFunc::eval(Real x) const {
    assert(state_ == Spectral);
    Real alpha = 2 * pi / L_;
    Complex f = cdata_[0];
    for (uint k = 1; k < Nmodes() - 1; ++k)
        f += 2 * Re(cdata_[k] * Complex(cos(alpha * k * x), sin(alpha * k * x)));
    return Re(f);
}

Real PeriodicFunc::operator()(Real x) const {
    assert(state_ == Spectral);
    Real alpha = 2 * pi / L_;
    Complex f = cdata_[0];
    for (uint k = 1; k < Nmodes() - 1; ++k)
        f += 2 * Re(cdata_[k] * Complex(cos(alpha * k * x), sin(alpha * k * x)));
    return Re(f);
}

/********************************************************************
PeriodicFunc PeriodicFunc::eval(const Vector& x)  const {
  PeriodicFunc f(N_, a_, b_, Physical);
  eval(x, f);
  return f;
}

// Numerical Recipes Clenshaw evaluation of Spectral expansion
void PeriodicFunc::eval(const Vector& x, PeriodicFunc& f) const {
  assert(state_ == Spectral);
  int N=x.length();
  if (f.length() != N)
    f.resize(N);
  f.setBounds(a_, b_);
  f.setState(Physical);

  int M=N_;
  for (uint i=0; i<N; ++i) {
    Real y = (2*x[i]-a_-b_)/(b_-a_);
    Real y2 = 2*y;
    Real d=0.0;
    Real dd=0.0;
    for (uint j=M-1; j>0; --j) {
      Real sv=d;
      d = y2*d - dd + data_[j];
      dd=sv;
    }
    f[i] = y*d - dd + data_[0]; // NR has 0.5*c[0], but that gives wrong results!
  }
}
********************************************************************/

Real PeriodicFunc::mean() const {
    Real rtn = 0.0;
    if (state_ == Spectral)
        rtn = Re(cdata_[0]);
    else {
        for (uint n = 0; n < N_; ++n)
            rtn += rdata_[n];
        rtn *= 1.0 / N_;
    }
    return rtn;
}

void PeriodicFunc::save(const string& filebase, fieldstate savestate) const {
    fieldstate origstate = state_;
    ((PeriodicFunc&)*this).makeState(savestate);

    string filename(filebase);
    filename += string(".asc");
    ofstream os(filename.c_str());
    os << scientific << setprecision(REAL_DIGITS);
    os << "% " << N_ << ' ' << L_ << ' ' << state_ << '\n';
    if (N_ != 0) {
        if (savestate == Spectral) {
            for (uint k = 0; k < Nmodes(); ++k)
                os << setw(REAL_IOWIDTH) << Re(cdata_[k]) << ' ' << setw(REAL_IOWIDTH) << Im(cdata_[k]) << '\n';
        } else {
            for (uint n = 0; n < N_; ++n)
                os << setw(REAL_IOWIDTH) << rdata_[n] << '\n';
            os << setw(REAL_IOWIDTH) << rdata_[0] << '\n';
        }
    }
    os.close();
    ((PeriodicFunc&)*this).makeState(origstate);
}

/***********************************************************
void PeriodicFunc::binaryDump(ostream& os) const {
  write(os, N_);
  write(os, L_);
  write(os, state_);
  for (uint i=0; i<N_; ++i)
    write(os, data_[i]);
}
void PeriodicFunc::binaryLoad(istream& is) {
  if (!is.good()) {
    cerr << "PeriodicFunc::binaryLoad(istream& is) : input error\n";
    exit(1);
  }
  int newN_;
  read(is, newN_);
  read(is, L_);
  read(is, state_);
  resize(newN_);
  for (uint i=0; i<N_; ++i) {
    if (!is.good()) {
      cerr << "PeriodicFunc::binaryLoad(istream& is) : input error\n";
      exit(1);
    }
    read(is, data_[i]);
  }
}

void PeriodicFunc::reconfig(const PeriodicFunc& f) {
  resize(f.N());
  setToZero();
  L_ = f.L_;
  state_ = f.state_;
}
***************************************************************/

PeriodicFunc& PeriodicFunc::operator*=(Real c) {
    for (uint n = 0; n < Npad(); ++n)
        rdata_[n] *= c;
    return *this;
}

PeriodicFunc& PeriodicFunc::operator+=(const PeriodicFunc& f) {
    assert(congruent(f));
    for (uint n = 0; n < Npad(); ++n)
        rdata_[n] += f.rdata_[n];
    return *this;
}

PeriodicFunc& PeriodicFunc::operator-=(const PeriodicFunc& f) {
    assert(congruent(f));
    for (uint n = 0; n < Npad(); ++n)
        rdata_[n] -= f.rdata_[n];
    return *this;
}

PeriodicFunc operator*(Real c, const PeriodicFunc& f) {
    PeriodicFunc rtn(f);
    rtn *= c;
    return rtn;
}

PeriodicFunc& PeriodicFunc::operator*=(const PeriodicFunc& f) {
    assert(congruent(f));
    assert(state_ == Physical);
    for (uint n = 0; n < N_; ++n)
        rdata_[n] *= f.rdata_[n];
    return *this;
}

bool PeriodicFunc::congruent(const PeriodicFunc& v) const {
    return (v.N_ == N_ && v.L_ == L_ && v.state_ == state_) ? true : false;
}

PeriodicFunc operator+(const PeriodicFunc& f, const PeriodicFunc& g) {
    PeriodicFunc rtn(f);
    rtn += g;
    return rtn;
}

PeriodicFunc operator-(const PeriodicFunc& f, const PeriodicFunc& g) {
    PeriodicFunc rtn(f);
    rtn -= g;
    return rtn;
}

bool operator==(const PeriodicFunc& f, const PeriodicFunc& g) {
    if (!f.congruent(g))
        return false;
    if (f.state() == Physical)
        for (uint n = 0; n < g.N(); ++n) {
            if (f(n) != g(n))
                return false;
        }
    else
        for (uint k = 0; k < g.Nmodes(); ++k) {
            if (f.cmplx(k) != g.cmplx(k))
                return false;
        }
    return true;
}

bool operator!=(const PeriodicFunc& f, const PeriodicFunc& g) { return !(f == g); }

void swap(PeriodicFunc& f, PeriodicFunc& g) {
    Real rtmp = f.L_;
    f.L_ = g.L_;
    g.L_ = rtmp;

    fieldstate stmp = f.state_;
    f.state_ = g.state_;
    g.state_ = stmp;

    int itmp = f.N_;
    f.N_ = g.N_;
    g.N_ = itmp;

    Real* rdtmp = f.rdata_;
    f.rdata_ = g.rdata_;
    g.rdata_ = rdtmp;

    f.cdata_ = (Complex*)f.rdata_;
    g.cdata_ = (Complex*)g.rdata_;
}

Real newtonSearch(const PeriodicFunc& f, Real xguess, int Nmax, Real eps) {
    assert(f.state() == Spectral);
    PeriodicFunc dfdx = diff(f);

    Real x = xguess;
    for (int n = 0; n < Nmax; ++n) {
        Real fx = f.eval(x);
        if (abs(fx) < eps)
            return x;
        Real df = dfdx.eval(x);
        x -= fx / df;
        // cout << x << " " << fx << endl;
    }
    return x;
}

void integrate(const PeriodicFunc& dfdx, PeriodicFunc& f) {
    assert(dfdx.state() == Spectral);
    if (f.N() != dfdx.N() || f.L() != dfdx.L())
        f.resize(dfdx.N(), dfdx.L());
    f.setState(Spectral);

    f.cmplx(0) = Complex(0.0, 0.0);  // set arb integ const by mean == zero

    // int exp(2pi i k x/L) = L/(2pi i) 1/k exp(2pi i k x/L)
    Real L_2pi = f.L() / (2 * pi);
    for (uint k = 1; k < f.Nmodes(); ++k)
        f.cmplx(k) = Complex(0.0, -L_2pi / k) * dfdx.cmplx(k);
    return;
}

PeriodicFunc integrate(const PeriodicFunc& dfdx) {
    PeriodicFunc u(dfdx.N(), dfdx.L(), Spectral);
    integrate(dfdx, u);
    return u;
}

void diff(const PeriodicFunc& f, PeriodicFunc& dfdx) {
    assert(f.state() == Spectral);
    if (dfdx.N() != f.N() || dfdx.L() != f.L())
        dfdx.resize(f.N(), f.L());
    dfdx.setState(Spectral);

    Real alpha = 2 * pi / f.L();
    for (uint k = 0; k < f.Nmodes(); ++k)
        dfdx.cmplx(k) = Complex(0.0, alpha * k) * f.cmplx(k);
}

void diff2(const PeriodicFunc& f, PeriodicFunc& d2f) {
    assert(f.state() == Spectral);
    if (d2f.N() != f.N() || d2f.L() != f.L())
        d2f.resize(f.N(), f.L());
    d2f.setState(Spectral);

    Real alpha2 = -square(2 * pi / f.L());
    for (uint k = 0; k < f.Nmodes(); ++k)
        d2f.cmplx(k) = (k * k) * alpha2 * f.cmplx(k);
}

void diff(const PeriodicFunc& f, PeriodicFunc& dnf, uint n) {
    assert(f.state() == Spectral);
    if (dnf.N() != f.N() || dnf.L() != f.L())
        dnf.resize(f.N(), f.L());

    Real alpha = 2 * pi / f.L();
    for (uint k = 0; k < f.Nmodes(); ++k)
        dnf.cmplx(k) = pow(Complex(0.0, alpha * k), n) * f.cmplx(k);
}

PeriodicFunc diff(const PeriodicFunc& f) {
    PeriodicFunc df(f.N(), f.L(), Spectral);
    diff(f, df);
    return df;
}

PeriodicFunc diff2(const PeriodicFunc& f) {
    PeriodicFunc d2f(f.N(), f.L(), Spectral);
    diff2(f, d2f);
    return d2f;
}

PeriodicFunc diff(const PeriodicFunc& f, uint n) {
    PeriodicFunc dnf(f.N(), f.L(), Spectral);
    diff(f, dnf, n);
    return dnf;
}

std::ostream& operator<<(ostream& os, const PeriodicFunc& f) {
    os << '{';
    if (f.state() == Physical) {
        for (uint n = 0; n < f.N(); ++n)
            os << setw(REAL_DIGITS) << f(n);
    } else {
        for (uint n = 0; n < f.N(); ++n)
            os << setw(REAL_DIGITS) << Re(f.cmplx(n)) << " + " << setw(REAL_DIGITS) << Im(f.cmplx(n)) << " i ";
    }
    os << '}';
    return os;
}

// ===========================================================
// L2 norms
Real L2Norm2(const PeriodicFunc& f, bool normalize) {
    assert(f.state() == Spectral);
    Real sum = 0.0;
    for (uint k = f.Nmodes() - 1; k > 0; --k)
        sum += 2 * abs2(f.cmplx(k));
    sum += Re(f.cmplx(0));
    if (!normalize)
        sum *= f.L();
    return sum;
}

Real L2Dist2(const PeriodicFunc& f, const PeriodicFunc& g, bool normalize) {
    assert(f.state() == Spectral);
    assert(g.state() == Spectral);
    Real sum = 0.0;
    for (uint k = f.Nmodes() - 1; k > 0; --k)
        sum += 2 * abs2(f.cmplx(k) - g.cmplx(k));
    sum += Re(f.cmplx(0) - g.cmplx(0));
    if (!normalize)
        sum *= f.L();
    return sum;
}

Real L2Norm(const PeriodicFunc& f, bool normalize) { return sqrt(L2Norm2(f, normalize)); }

Real L2Dist(const PeriodicFunc& f, const PeriodicFunc& g, bool normalize) { return sqrt(L2Dist2(f, g, normalize)); }

Real L2IP(const PeriodicFunc& f, const PeriodicFunc& g, bool normalize) {
    assert(f.state() == Spectral);
    assert(g.state() == Spectral);
    Real sum = 0.0;
    for (uint k = f.Nmodes() - 1; k > 0; --k)
        sum += 2 * (Re(f.cmplx(k)) * Re(g.cmplx(k)) + Im(f.cmplx(k)) * Im(g.cmplx(k)));
    sum += Re(f.cmplx(0)) * Re(g.cmplx(0)) + Im(f.cmplx(0)) * Im(g.cmplx(0));
    if (!normalize)
        sum *= f.L();
    return sum;
}

}  // namespace chflow
