/**
 * Real- and Complex-valued Chebyshev expansion classes.
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

// Wordexp is used in fftw_savewisdom and load, to expand possible home
// dir symbol ~. Comment it out if it causes you trouble.

#include "channelflow/chebyshev.h"
#include "cfbasics/mathdefs.h"

#ifdef HAVE_WORDEXP_H
#include <wordexp.h>
#endif

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <memory>

using namespace std;

namespace {
/// @brief Expands an input filename using wordexp, if wordexp
/// is available. If not returns the input path.
///
/// @param[in] input the path to be expanded
///
/// @return the expanded path
inline string expand_filename(const char* input) {
#ifdef HAVE_WORDEXP_H
    auto filename = string((input == nullptr) ? "~/.fftw_wisdom" : input);
    auto w = wordexp_t{};
    auto wordexp_flags = 0;

    w.we_offs = 0;

    auto error = wordexp(filename.c_str(), &w, wordexp_flags);
    if (error) {
        cerr << "Could not expand " << input << "(Error code: " << error << ")" << endl;
        exit(1);
    }
    // Abusing the unique_ptr as if it was a scope guard
    unique_ptr<wordexp_t, void (*)(wordexp_t*)> scope_guard(&w, wordfree);

    if (w.we_wordc > 0) {
        filename = w.we_wordv[0];
    }
#else
    auto filename = string(input);
#endif
    return filename;
}

}  // namespace

namespace chflow {

const Real QUADRATURE_EPSILON = 1e-17;

// Integral_(-1)^1 Tm(y) Tn(y) dy
Real chebyIP(int m, int n) {
    return ((m + n) % 2 == 1) ? 0
                              : (1.0 - m * m - n * n) / ((1.0 + m - n) * (1.0 - m + n) * (1.0 + m + n) * (1.0 - m - n));
}

Real legendre(int n, Real x) {
    Real p = 1.0;
    Real q = 0.0;
    for (int m = 0; m < n; ++m) {
        Real r = q;
        q = p;
        p = ((2 * m + 1) * x * q - m * r) / (m + 1);
    }
    return p;
}

Real chebyshev(int n, Real x) { return cos(n * acos(x)); }

void legendre(int n, ChebyCoeff& u, ChebyTransform& trans, bool normalize) {
    assert(n <= u.N());
    //  ChebyCoeff Pn(N, -1 , 1, Physical);
    const int N = u.N();
    Real piN = pi / (N - 1);
    u.setState(Physical);
    for (int q = 0; q < N; ++q)
        u[q] = legendre(n, cos(q * piN));

    u.makeSpectral(trans);
    if (normalize)
        u *= sqrt((double)(2 * n + 1));
}

ChebyCoeff chebyshev(int N, int n, bool normalize) {
    assert(n < N);
    ChebyCoeff Tn(N, -1, 1, Spectral);
    if (!normalize)
        Tn[n] = 1;
    else
        Tn[n] = sqrt(2 / (pi * cheby_c(n)));
    return Tn;
}

void gaussLegendreQuadrature(int N, Real a, Real b, Vector& x, Vector& w) {
    x = Vector(N);
    w = Vector(N);
    int M = (N + 1) / 2;
    Real middle = 0.5 * (b + a);
    Real radius = 0.5 * (b - a);
    Real pp;
    Real z, z1;

    for (int m = 0; m < M; ++m) {
        z = cos(M_PI * (m + 0.75) / (N + 0.5));
        Real p1, p2, p3;
        int iterations = 0;
        do {
            p1 = 1.0;
            p2 = 0.0;
            for (int n = 0; n < N; ++n) {
                p3 = p2;
                p2 = p1;
                p1 = ((2 * n + 1) * z * p2 - n * p3) / (n + 1);
            }
            pp = N * (z * p1 - p2) / (z * z - 1.0);
            z1 = z;
            z = z1 - p1 / pp;
        } while (fabs(z - z1) > QUADRATURE_EPSILON && ++iterations < 1024);

        x(m) = middle - radius * z;
        x(N - 1 - m) = middle + radius * z;
        w(m) = 2.0 * radius / ((1.0 - z * z) * pp * pp);
        w(N - 1 - m) = w(m);
    }
}

void fftw_loadwisdom(const char* filename_) {
    auto filename = expand_filename(filename_);
    auto rank = mpirank();

    if (rank == 0) {
        unique_ptr<FILE, int (*)(FILE*)> input_file_ptr(fopen(filename.c_str(), "r"), fclose);

        if ((input_file_ptr == nullptr) || (!fftw_import_wisdom_from_file(input_file_ptr.get()))) {
            cerr << "Error reading fftw-wisdom file " << filename << ", proceeding without fftw-wisdom.\n";
        }
    }
#ifdef HAVE_MPI
    fftw_mpi_broadcast_wisdom(MPI_COMM_WORLD);
#endif
}

void fftw_savewisdom(const char* filename_) {
    auto filename = expand_filename(filename_);
    auto rank = mpirank();

#ifdef HAVE_MPI
    fftw_mpi_gather_wisdom(MPI_COMM_WORLD);
#endif

    if (rank == 0) {
        unique_ptr<FILE, int (*)(FILE*)> output_file_ptr(fopen(filename.c_str(), "w"), fclose);
        if (output_file_ptr == nullptr) {
            cerr << "Can't write to fftw-wisdom file " << filename << "\n";
        } else {
            fftw_export_wisdom_to_file(output_file_ptr.get());
        }
    }
}

ChebyCoeff::ChebyCoeff() : Vector(), a_(0), b_(0), state_(Spectral) {}

ChebyCoeff::ChebyCoeff(int N, Real a, Real b, fieldstate s) : Vector(N), a_(a), b_(b), state_(s) { assert(b_ > a_); }

ChebyCoeff::ChebyCoeff(const Vector& v, Real a, Real b, fieldstate s) : Vector(v), a_(a), b_(b), state_(s) {
    assert(b_ > a_);
}

ChebyCoeff::ChebyCoeff(int N, const ChebyCoeff& u) : Vector(N), a_(u.a_), b_(u.b_), state_(u.state_) {
    assert(b_ > a_);
    // assert(state_ == Spectral);
    auto Ncommon = static_cast<unsigned long long>(lesser(N, u.N()));
    for (auto i = 0u; i < Ncommon; ++i)
        data_[i] = u.data_[i];
    for (auto i = Ncommon; i < data_.size(); ++i)
        data_[i] = 0.0;
}

ChebyCoeff::ChebyCoeff(const string& filebase) : Vector(0), a_(0), b_(0), state_(Spectral) {
    ifstream is;
    string filename = ifstreamOpen(is, filebase, ".asc");

    if (!is) {
        cerr << "ChebyCoeff::ChebyCoeff(filebase) : can't open file " << filebase << " or " << (filebase + ".asc")
             << endl;
        exit(1);
    }

    // Read in header. Form is "%N a b s"
    char c;
    int N;
    is >> c;
    if (c != '%') {
        string message("ChebyCoeff(filebase): bad header in file ");
        message += filename;
        cerr << message << endl;
        assert(false);
    }
    is >> N >> a_ >> b_ >> state_;
    resize(N);
    assert(is.good());
    for (int i = 0; i < N; ++i) {
        is >> (*this)[i];
        assert(is.good());
    }
    makeSpectral();
    is.close();
}

/******************
ChebyCoeff::ChebyCoeff(istream& is)
  :
  Vector(0),
  a_(0),
  b_(0),
  state_(Spectral)
{
  binaryLoad(is);
}
******************/

ChebyCoeff::~ChebyCoeff() {}

void ChebyCoeff::chebyfft() {
    ChebyTransform t(data_.size());
    chebyfft(t);
}
void ChebyCoeff::ichebyfft() {
    ChebyTransform t(data_.size());
    ichebyfft(t);
}
void ChebyCoeff::makeSpectral() {
    ChebyTransform t(data_.size());
    makeSpectral(t);
}
void ChebyCoeff::makePhysical() {
    ChebyTransform t(data_.size());
    makePhysical(t);
}
void ChebyCoeff::makeState(fieldstate s) {
    if (state_ != s) {  // need to change state?
        ChebyTransform t(data_.size());
        if (state_ == Physical)  // state is Physical; change to Spectral
            chebyfft(t);
        else
            ichebyfft(t);  // state is Spectral; change to Physical
    }
}

void ChebyCoeff::chebyfft(const ChebyTransform& t) {
    assert(static_cast<unsigned>(t.N()) == data_.size());
    assert(state_ == Physical);
    if (data_.size() < 2) {
        state_ = Spectral;
        return;
    }

    fftw_execute_r2r(t.cosfftw_plan_.get(), data_.data(), data_.data());

    // Factors of 1/2 appear on 0th and (N-1)th coeff because of reln btwn
    // chebyshev and cosine transform. Normalization factor nrm=1/(N-1) is
    // needed because FFTW does unnormalized transforms. After this loop,
    // data_[n] equals the coefficient of T_n.
    auto N_ = data_.size();
    Real nrm = 1.0 / (N_ - 1);
    data_[0] *= 0.5 * nrm;
    for (auto n = 1u; n < N_ - 1; ++n)
        data_[n] *= nrm;
    data_[N_ - 1] *= 0.5 * nrm;

    state_ = Spectral;
}

void ChebyCoeff::ichebyfft(const ChebyTransform& t) {
    assert(static_cast<unsigned>(t.N()) == data_.size());
    assert(state_ == Spectral);
    if (data_.size() < 2) {
        state_ = Physical;
        return;
    }

    // Factors of 2 undo the 1/2's introduced in ::chebyfft.
    auto N_ = data_.size();
    data_[0] *= 2.0;
    data_[N_ - 1] *= 2.0;
    fftw_execute_r2r(t.cosfftw_plan_.get(), data_.data(), data_.data());
    for (auto& item : data_) {
        item *= 0.5;
    }
    state_ = Physical;
}
void ChebyCoeff::makeSpectral(const ChebyTransform& t) {
    if (state_ == Physical)
        chebyfft(t);
}
void ChebyCoeff::makePhysical(const ChebyTransform& t) {
    if (state_ == Spectral)
        ichebyfft(t);
}
void ChebyCoeff::makeState(fieldstate s, const ChebyTransform& t) {
    if (state_ != s) {           // need to change state?
        if (state_ == Physical)  // state is Physical; change to Spectral
            chebyfft(t);
        else
            ichebyfft(t);  // state is Spectral; change to Physical
    }
}

void ChebyCoeff::randomize(Real magn, Real decay, BC aBC, BC bBC) {
    fieldstate startState = state_;
    state_ = Spectral;
    Real magn_n = magn;
    auto N_ = data_.size();
    for (auto n = 0u; n < N_; ++n) {
        data_[n] = magn_n * randomReal(-1, 1);
        magn_n *= decay;
    }
    // Iterating can reduce floating-point errors in BCs
    for (int n = 0; n < 2; ++n) {
        if (N_ == 1 && (aBC == Diri || bBC == Diri))
            data_[0] = 0.0;
        else if (N_ >= 2 && (aBC == Diri && bBC == Diri)) {
            data_[1] -= 0.5 * (eval_b() - eval_a());
            data_[0] -= 0.5 * (eval_b() + eval_a());
        } else if (N_ >= 2 && aBC == Diri)
            data_[0] -= eval_a();
        else if (N_ >= 2 && bBC == Diri)
            data_[0] -= eval_b();
    }
    makeState(startState);
}

void ChebyCoeff::setToZero() { std::fill(data_.begin(), data_.end(), 0.0); }

void ChebyCoeff::fill(const ChebyCoeff& v) {
    assert(v.state_ == Spectral);
    assert(state_ == Spectral);
    int Ncommon = lesser(length(), v.length());
    int i;  // MSVC++ FOR-SCOPE BUG
    for (i = 0; i < Ncommon; ++i)
        data_[i] = v.data_[i];
    for (i = Ncommon; i < length(); ++i)
        data_[i] = 0.0;
}

void ChebyCoeff::interpolate(const ChebyCoeff& v) {
    assert(a_ >= v.a_ && b_ <= v.b_);
    assert(v.state_ == Spectral);
    state_ = Physical;

    Real piN = pi / (data_.size() - 1);
    Real width = (b_ - a_) / 2;
    Real center = (b_ + a_) / 2;
    for (auto n = 0u; n < data_.size(); ++n)
        data_[n] = v.eval(center + width * cos(n * piN));
    makeSpectral();
}

void ChebyCoeff::reflect(const ChebyCoeff& v, parity p) {
    assert((a_ + b_) / 2 == v.a() && b_ <= v.b() && b_ > v.a());
    assert(v.state_ == Spectral);
    state_ = Physical;
    auto N_ = data_.size();
    Real piN = pi / (N_ - 1);
    Real width = (b_ - a_) / 2;
    Real center = (b_ + a_) / 2;
    int N2 = N_ / 2;
    int N1 = N_ - 1;
    int sign = (p == Odd) ? -1 : 1;
    for (int n = 0; n < N2; ++n) {
        Real tmp = v.eval(center + width * cos(n * piN));
        data_[n] = tmp;
        data_[N1 - n] = sign * tmp;
    }
    ChebyTransform t(N_);
    makeSpectral(t);
    for (auto n = 2 * N_ / 3; n < N_; ++n)
        data_[n] = 0.0;
    makePhysical(t);
}

void ChebyCoeff::setBounds(Real a, Real b) {
    a_ = a;
    b_ = b;
    assert(b_ > a_);
}

void ChebyCoeff::setState(fieldstate s) {
    assert(s == Physical || s == Spectral);
    state_ = s;
}

Real ChebyCoeff::eval_b() const {
    if (data_.empty())
        return 0;

    if (state_ == Spectral) {
        Real sum = 0.0;
        for (int n = data_.size() - 1; n >= 0; --n)
            sum += data_[n];
        return sum;
    } else
        return data_[0];
}

Real ChebyCoeff::eval_a() const {
    if (data_.empty())
        return 0;

    if (state_ == Spectral) {
        Real sum = 0.0;
        for (int n = data_.size() - 1; n >= 0; --n)
            sum += data_[n] * ((n % 2 == 0) ? 1 : -1);
        return sum;
    } else {
        return data_[data_.size() - 1];
    }
}

Real ChebyCoeff::slope_a() const {
    Real sum = 0.0;

    auto N_ = data_.size();
    // N=4: 0,1  2,3
    // N=5: 0,1  2,3  4
    for (auto n = 0u; n < N_ - 1; n += 2)
        sum += -n * n * data_[n] + (n + 1) * (n + 1) * data_[n + 1];

    if (N_ % 2 == 1)
        sum -= (N_ - 1) * (N_ - 1) * data_[N_ - 1];

    return 2 * sum / (b_ - a_);
}

Real ChebyCoeff::slope_b() const {
    assert(state_ == Spectral);
    auto N_ = data_.size();
    if (N_ == 0)
        return 0;
    Real sum = 0.0;
    for (int n = N_ - 1; n >= 0; --n)
        sum += n * n * data_[n];
    return 2 * sum / (b_ - a_);
}

Real ChebyCoeff::eval(Real x) const {
    assert(state_ == Spectral);
    auto N_ = data_.size();
    if (N_ == 0)
        return 0;
    Real y = (2 * x - a_ - b_) / (b_ - a_);
    Real y2 = 2 * y;
    Real d = 0.0;
    Real dd = 0.0;
    for (int j = N_ - 1; j > 0; --j) {
        Real sv = d;
        d = y2 * d - dd + data_[j];
        dd = sv;
    }
    return (y * d - dd + data_[0]);  // NR has 0.5*c[0], but that gives wrong results!
}

ChebyCoeff ChebyCoeff::eval(const Vector& x) const {
    ChebyCoeff f(data_.size(), a_, b_, Physical);
    eval(x, f);
    return f;
}

// Numerical Recipes Clenshaw evaluation of Spectral expansion
void ChebyCoeff::eval(const Vector& x, ChebyCoeff& f) const {
    assert(state_ == Spectral);
    int N = x.length();
    if (f.length() != N)
        f.resize(N);
    f.setBounds(a_, b_);
    f.setState(Physical);

    int M = data_.size();
    for (int i = 0; i < N; ++i) {
        Real y = (2 * x[i] - a_ - b_) / (b_ - a_);
        Real y2 = 2 * y;
        Real d = 0.0;
        Real dd = 0.0;
        for (int j = M - 1; j > 0; --j) {
            Real sv = d;
            d = y2 * d - dd + data_[j];
            dd = sv;
        }
        f[i] = y * d - dd + data_[0];  // NR has 0.5*c[0], but that gives wrong results!
    }
}

Real ChebyCoeff::mean() const {
    assert(state_ == Spectral);
    Real sum = data_[0];
    auto N_ = data_.size();
    for (auto n = 2u; n < N_; n += 2)
        sum -= data_[n] / (n * n - 1);  // *2
    return sum;                         // /2
}

// Real ChebyCoeff::energy() const {return L2Norm2(*this, false);}

void ChebyCoeff::save(const string& filebase, fieldstate savestate) const {
    fieldstate origstate = state_;
    ((ChebyCoeff&)*this).makeState(savestate);

    string filename = appendSuffix(filebase, ".asc");
    ofstream os(filename.c_str());
    if (!os.good())
        cferror("ChebyCoeff::save(filebase) :  can't open file " + filename);

    os << scientific << setprecision(REAL_DIGITS);
    os << "% " << data_.size() << ' ' << a_ << ' ' << b_ << ' ' << state_ << '\n';
    for (const auto& item : data_) {
        os << setw(REAL_IOWIDTH) << item << '\n';
    }
    os.close();
    ((ChebyCoeff&)*this).makeState(origstate);
}

void ChebyCoeff::binaryDump(ostream& os) const {
    write(os, static_cast<int>(data_.size()));
    write(os, a_);
    write(os, b_);
    write(os, state_);
    for (const auto& item : data_) {
        write(os, item);
    }
}
void ChebyCoeff::binaryLoad(istream& is) {
    if (!is.good()) {
        cerr << "ChebyCoeff::binaryLoad(istream& is) : input error\n";
        exit(1);
    }
    int newN_;
    read(is, newN_);
    read(is, a_);
    read(is, b_);
    read(is, state_);
    resize(newN_);
    for (auto& item : data_) {
        if (!is.good()) {
            cerr << "ChebyCoeff::binaryLoad(istream& is) : input error\n";
            exit(1);
        }
        read(is, item);
    }
}

void ChebyCoeff::reconfig(const ChebyCoeff& f) {
    resize(f.N());
    setToZero();
    a_ = f.a_;
    b_ = f.b_;
    state_ = f.state_;
}

ChebyCoeff& ChebyCoeff::operator*=(Real c) {
    static_cast<Vector&>(*this) *= c;
    return *this;
}

ChebyCoeff& ChebyCoeff::operator+=(const ChebyCoeff& a) {
    assert(congruent(a));
    static_cast<Vector&>(*this) += a;
    return *this;
}

ChebyCoeff& ChebyCoeff::operator-=(const ChebyCoeff& a) {
    assert(congruent(a));
    static_cast<Vector&>(*this) -= a;
    return *this;
}

ChebyCoeff operator*(Real c, const ChebyCoeff& v) {
    ChebyCoeff rtn(v);
    rtn *= c;
    return rtn;
}

ChebyCoeff& ChebyCoeff::operator*=(const ChebyCoeff& a) {
    assert(congruent(a));
    assert(state_ == Physical);
    for (auto i = 0u; i < data_.size(); ++i)
        data_[i] *= a.data_[i];
    return *this;
}

bool ChebyCoeff::congruent(const ChebyCoeff& v) const {
    return (v.length() == length() && v.a_ == a_ && v.b_ == b_ && v.state_ == state_) ? true : false;
}

ChebyCoeff operator+(const ChebyCoeff& u, const ChebyCoeff& v) {
    ChebyCoeff rtn(u);
    rtn += v;
    return rtn;
}

ChebyCoeff operator-(const ChebyCoeff& u, const ChebyCoeff& v) {
    ChebyCoeff rtn(u);
    rtn -= v;
    return rtn;
}

bool operator==(const ChebyCoeff& u, const ChebyCoeff& v) {
    if (!u.congruent(v))
        return false;
    for (int i = 0; i < u.numModes(); ++i)
        if (u[i] != v[i])
            return false;
    return true;
}

bool operator!=(const ChebyCoeff& u, const ChebyCoeff& v) { return !(u == v); }

void swap(ChebyCoeff& f, ChebyCoeff& g) {
    using std::swap;
    swap(f.a_, g.a_);
    swap(f.b_, g.b_);
    swap(f.state_, g.state_);
    swap(f.data_, g.data_);
}

void integrate(const ChebyCoeff& dudy, ChebyCoeff& u) {
    assert(dudy.state() == Spectral);
    int N = dudy.numModes();
    if (u.numModes() != N)
        u.resize(N);
    u.setBounds(u.a(), u.b());
    u.setState(Spectral);

    Real h2 = (dudy.b() - dudy.a()) / 2;
    switch (N) {
        case 0:
            break;
        case 1:
            u[0] = 0.0;
            break;
        case 2:
            u[0] = 0;
            u[1] = h2 * dudy[0];
            break;
        default:
            u[1] = h2 * (dudy[0] - dudy[2] / 2);
            for (int n = 2; n < N - 1; ++n)
                u[n] = h2 * (dudy[n - 1] - dudy[n + 1]) / (2 * n);
            u[N - 1] = h2 * dudy[N - 2] / (2 * (N - 1));
            u[0] -= u.mean();  // const term is arbitrary, set to zero.
    }
    return;
}

ChebyCoeff integrate(const ChebyCoeff& dudy) {
    ChebyCoeff u(dudy.length(), dudy.a(), dudy.b(), Spectral);
    integrate(dudy, u);
    return u;
}

void diff(const ChebyCoeff& u, ChebyCoeff& dudy) {
    assert(u.state() == Spectral);
    if (dudy.numModes() != u.numModes())
        dudy.resize(u.numModes());
    dudy.setBounds(u.a(), u.b());
    dudy.setState(Spectral);

    int Nb = u.numModes() - 1;
    if (Nb == -1)
        return;
    if (Nb == 0) {
        dudy[0] = 0.0;
        return;
    }
    // See eqn 2.5.25 Canuto Hussaini Quateroni Zhang
    // Spectral methods, fundamentals in Single Domains
    // Springer 2006
    Real scale = 4.0 / u.L();
    dudy[Nb] = 0.0;
    dudy[Nb - 1] = scale * Nb * u[Nb];
    for (int n = Nb - 2; n >= 0; --n)
        dudy[n] = dudy[n + 2] + scale * (n + 1) * u[n + 1];
    dudy[0] *= 0.5;

    // dudy *= 2.0/u.L();
}

void diff2(const ChebyCoeff& u, ChebyCoeff& d2udy2) {
    ChebyCoeff dudy(u.length(), u.a(), u.b(), Spectral);
    diff(u, dudy);
    diff(dudy, d2udy2);
}

void diff2(const ChebyCoeff& u, ChebyCoeff& d2udy2, ChebyCoeff& tmp) {
    diff(u, tmp);
    diff(tmp, d2udy2);
}

void diff(const ChebyCoeff& f, ChebyCoeff& df, int n) {
    assert(n >= 0);
    assert(f.state() == Spectral);
    df = f;
    ChebyCoeff tmp;
    for (int k = 0; k < n; ++k) {
        diff(df, tmp);
        swap(df, tmp);
    }
    return;
}

ChebyCoeff diff(const ChebyCoeff& u) {
    ChebyCoeff du(u.length(), u.a(), u.b(), Spectral);
    diff(u, du);
    return du;
}

ChebyCoeff diff2(const ChebyCoeff& u) {
    ChebyCoeff du2(u.length(), u.a(), u.b(), Spectral);
    diff2(u, du2);
    return du2;
}

ChebyCoeff diff(const ChebyCoeff& f, int n) {
    assert(n >= 0);
    ChebyCoeff g = f;
    ChebyCoeff gy(f.N(), f.a(), f.b(), Spectral);

    for (int n_ = 0; n_ < n; ++n_) {
        diff(g, gy);
        swap(g, gy);
    }
    return g;
}

Vector chebypoints(int N, Real a, Real b) {
    Vector xcheb(N);
    Real piN = pi / (N - 1);
    Real radius = (b - a) / 2;
    Real center = (b + a) / 2;
    for (int j = 0; j < N; ++j)
        xcheb[j] = center + radius * cos(j * piN);
    return xcheb;
}

// ===========================================================
// L2 norms
Real L2Norm2(const ChebyCoeff& u, bool normalize) {
    assert(u.state() == Spectral);
    const int N = u.numModes();
    const int e = 1;
    Real sum = 0.0;
    for (int m = N - 1; m >= 0; --m) {
        // Real um = u[m];
        Real psum = 0.0;
        for (int n = m % 2; n < N; n += 2)
            psum += (u[n] * (e - m * m - n * n)) / ((e + m - n) * (e - m + n) * (e + m + n) * (e - m - n));
        sum += u[m] * psum;
    }
    if (!normalize)
        sum *= u.b() - u.a();
    return sum;
}

Real L2Dist2(const ChebyCoeff& u, const ChebyCoeff& v, bool normalize) {
    ChebyCoeff tmp(u);
    tmp -= v;
    return L2Norm2(tmp, normalize);
}

Real L2Norm(const ChebyCoeff& u, bool normalize) { return sqrt(L2Norm2(u, normalize)); }

Real L2Dist(const ChebyCoeff& u, const ChebyCoeff& v, bool normalize) { return sqrt(L2Dist2(u, v, normalize)); }

Real L2InnerProduct(const ChebyCoeff& u, const ChebyCoeff& v, bool normalize) {
    assert(u.state() == Spectral && v.state() == Spectral);
    assert(u.a() == v.a() && u.b() == v.b());
    assert(u.numModes() == v.numModes());
    int N = u.numModes();
    Real sum = 0.0;
    Real e = 1.0;
    for (int m = N - 1; m >= 0; --m) {
        Real um = u[m];
        Real psum = 0.0;
        for (int n = m % 2; n < N; n += 2)
            psum += um * v[n] * (e - m * m - n * n) / ((e + m - n) * (e - m + n) * (e + m + n) * (e - m - n));
        sum += psum;
    }
    if (!normalize)
        sum *= u.b() - u.a();
    return sum;
}

// ===========================================================
// cheby norms
Real chebyNorm2(const ChebyCoeff& u, bool normalize) {
    assert(u.state() == Spectral);
    int N = u.numModes();
    Real sum = 0.0;
    for (int m = N - 1; m > 0; --m)
        sum += square(u[m]);
    if (N > 0)
        sum += 2 * square(u[0]);  // coeff of T_0(x) has prefactor of 2 in norm
    if (!normalize)
        sum *= u.b() - u.a();
    sum *= pi / 2;
    return sum;
}

Real chebyDist2(const ChebyCoeff& u, const ChebyCoeff& v, bool normalize) {
    assert(u.state() == Spectral && v.state() == Spectral);
    assert(u.a() == v.a() && u.b() == v.b());
    assert(u.numModes() == v.numModes());
    int N = u.numModes();
    Real sum = 0.0;
    for (int m = N - 1; m > 0; --m)
        sum += square(u[m] - v[m]);
    if (N > 0)
        sum += 2 * square(u[0] - v[0]);  // coeff of T_0(x) has prefactor of 2 in norm
    if (!normalize)
        sum *= u.b() - u.a();
    return sum * pi / 2;
}

Real chebyNorm(const ChebyCoeff& u, bool normalize) { return sqrt(chebyNorm2(u, normalize)); }

Real chebyDist(const ChebyCoeff& u, const ChebyCoeff& v, bool normalize) { return sqrt(chebyDist2(u, v, normalize)); }

Real chebyInnerProduct(const ChebyCoeff& u, const ChebyCoeff& v, bool normalize) {
    assert(u.state() == Spectral && v.state() == Spectral);
    assert(u.a() == v.a() && u.b() == v.b());
    assert(u.numModes() == v.numModes());
    int N = u.numModes();
    Real sum = 0.0;
    for (int m = N - 1; m > 0; --m)
        sum += u[m] * v[m];
    if (N > 0)
        sum += 2 * u[0] * v[0];  // coeff of T_0(x) has prefactor of 2 in norm
    if (!normalize)
        sum *= u.b() - u.a();
    return sum * pi / 2;
}

Real LinfNorm(const ChebyCoeff& f) {
    fieldstate s = f.state();
    ((ChebyCoeff&)f).makePhysical();

    Real rtn = 0.0;
    for (int m = 0; m < f.N(); ++m)
        rtn = Greater(abs(f[m]), rtn);

    ((ChebyCoeff&)f).makeState(s);

    return rtn;
}

Real LinfDist(const ChebyCoeff& f, const ChebyCoeff& g) {
    assert(f.congruent(g));
    fieldstate fs = f.state();
    fieldstate gs = f.state();
    ((ChebyCoeff&)f).makePhysical();
    ((ChebyCoeff&)g).makePhysical();

    Real rtn = 0.0;
    for (int m = 0; m < f.N(); ++m)
        rtn = Greater(rtn, abs(f[m] - g[m]));

    ((ChebyCoeff&)f).makeState(fs);
    ((ChebyCoeff&)g).makeState(gs);
    return rtn;
}

// Probably INEFFICIENT
Real L1Norm(const ChebyCoeff& f, bool normalize) {
    ChebyCoeff g(f);
    g.makePhysical();
    for (int n = 0; n < g.N(); ++n)
        g[n] = abs(g[n]);
    g.makeSpectral();

    ChebyCoeff G(g);
    integrate(g, G);

    Real rtn = G.eval_b() - G.eval_a();

    if (normalize)
        rtn /= g.b() - g.a();

    return rtn;
}

Real L1Dist(const ChebyCoeff& f, const ChebyCoeff& g, bool normalize) {
    assert(f.congruent(g));
    fieldstate fs = f.state();
    fieldstate gs = f.state();
    ((ChebyCoeff&)f).makePhysical();
    ((ChebyCoeff&)g).makePhysical();

    ChebyCoeff d(f);
    for (int n = 0; n < g.N(); ++n)
        d[n] = abs(f[n] - g[n]);
    d.makeSpectral();

    ChebyCoeff D(g);
    integrate(d, D);

    Real rtn = D.eval_b() - D.eval_a();

    if (normalize)
        rtn /= d.b() - d.a();

    ((ChebyCoeff&)f).makeState(fs);
    ((ChebyCoeff&)g).makeState(gs);

    return rtn;
}

// ===========================================================
// switchable norms
Real norm2(const ChebyCoeff& u, NormType n, bool normalize) {
    return (n == Uniform) ? L2Norm2(u, normalize) : chebyNorm2(u, normalize);
}

Real dist2(const ChebyCoeff& u, const ChebyCoeff& v, NormType n, bool normalize) {
    return (n == Uniform) ? L2Dist2(u, v, normalize) : chebyDist2(u, v, normalize);
}

Real norm(const ChebyCoeff& u, NormType n, bool normalize) {
    return (n == Uniform) ? L2Norm(u, normalize) : chebyNorm(u, normalize);
}
Real dist(const ChebyCoeff& u, const ChebyCoeff& v, NormType n, bool normalize) {
    return (n == Uniform) ? L2Dist(u, v, normalize) : chebyDist(u, v, normalize);
}
Real innerProduct(const ChebyCoeff& u, const ChebyCoeff& v, NormType n, bool normalize) {
    return (n == Uniform) ? L2InnerProduct(u, v, normalize) : chebyInnerProduct(u, v, normalize);
}

//=======================================================================
// ComplexChebyCoeff

ComplexChebyCoeff::ComplexChebyCoeff() : re(), im() {}

ComplexChebyCoeff::ComplexChebyCoeff(int N, Real a, Real b, fieldstate s) : re(N, a, b, s), im(N, a, b, s) {}
ComplexChebyCoeff::ComplexChebyCoeff(int N, const ComplexChebyCoeff& u) : re(N, u.re), im(N, u.im) {}

ComplexChebyCoeff::ComplexChebyCoeff(const ChebyCoeff& r, const ChebyCoeff& i) : re(r), im(i) {}

ComplexChebyCoeff::ComplexChebyCoeff(const string& filebase) : re(), im() {
    ifstream is;
    string filename = ifstreamOpen(is, filebase, ".asc");
    if (!is) {
        cerr << "ComplexChebyCoeff::ComplexChebyCoeff(filebase) : "
             << "can't open file " << filebase << " or " << (filebase + ".asc") << endl;
        exit(1);
    }

    // Read in header. Form is "%N a b s"
    char c;
    int N;
    is >> c;
    if (c != '%') {
        string message("ComplexChebyCoeff(filebase): bad header in file ");
        message += filename;
        cerr << message << endl;
        assert(false);
    }
    Real a, b;
    fieldstate s;
    is >> N >> a >> b >> s;
    assert(is.good());

    re.resize(N);
    im.resize(N);
    re.setBounds(a, b);
    im.setBounds(a, b);
    re.setState(s);
    im.setState(s);

    for (int i = 0; i < N; ++i) {
        is >> re[i] >> im[i];
        assert(is.good());
    }
    is.close();
}

/****************
ComplexChebyCoeff::ComplexChebyCoeff(istream& is)
:
re(is),
im(is)
{
if (!is.good()) {
cerr << "ComplexChebyCoeff::ComplexChebyCoeff(istream& is) : input error\n";
exit(1);
}}
*****************/
void ComplexChebyCoeff::resize(int N) {
    re.resize(N);
    im.resize(N);
}

void ComplexChebyCoeff::reconfig(const ComplexChebyCoeff& f) {
    re.reconfig(f.re);
    im.reconfig(f.im);
}

void ComplexChebyCoeff::chebyfft() {
    ChebyTransform t(re.numModes());
    chebyfft(t);
}
void ComplexChebyCoeff::ichebyfft() {
    ChebyTransform t(re.numModes());
    ichebyfft(t);
}
void ComplexChebyCoeff::makeSpectral() {
    ChebyTransform t(re.numModes());
    makeSpectral(t);
}
void ComplexChebyCoeff::makePhysical() {
    ChebyTransform t(re.numModes());
    makePhysical(t);
}
void ComplexChebyCoeff::makeState(fieldstate s) {
    ChebyTransform t(re.numModes());
    re.makeState(s, t);
    im.makeState(s, t);
}

void ComplexChebyCoeff::chebyfft(const ChebyTransform& t) {
    re.chebyfft(t);
    im.chebyfft(t);
}
void ComplexChebyCoeff::ichebyfft(const ChebyTransform& t) {
    re.ichebyfft(t);
    im.ichebyfft(t);
}
void ComplexChebyCoeff::makeSpectral(const ChebyTransform& t) {
    re.makeSpectral(t);
    im.makeSpectral(t);
}
void ComplexChebyCoeff::makePhysical(const ChebyTransform& t) {
    re.makePhysical(t);
    im.makePhysical(t);
}
void ComplexChebyCoeff::makeState(fieldstate s, const ChebyTransform& t) {
    re.makeState(s, t);
    im.makeState(s, t);
}

void ComplexChebyCoeff::randomize(Real magn, Real decay, BC aBC, BC bBC) {
    re.randomize(magn, decay, aBC, bBC);
    im.randomize(magn, decay, aBC, bBC);
}
void ComplexChebyCoeff::setToZero() {
    re.setToZero();
    im.setToZero();
}
void ComplexChebyCoeff::fill(const ComplexChebyCoeff& v) {
    re.fill(v.re);
    im.fill(v.im);
}
void ComplexChebyCoeff::interpolate(const ComplexChebyCoeff& v) {
    re.interpolate(v.re);
    im.interpolate(v.im);
}
void ComplexChebyCoeff::reflect(const ComplexChebyCoeff& v, parity p) {
    re.reflect(v.re, p);
    im.reflect(v.im, p);
}

void ComplexChebyCoeff::setState(fieldstate s) {
    re.setState(s);
    im.setState(s);
}
void ComplexChebyCoeff::setBounds(Real a, Real b) {
    re.setBounds(a, b);
    im.setBounds(a, b);
}

Complex ComplexChebyCoeff::eval_a() const { return Complex(re.eval_a(), im.eval_a()); }
Complex ComplexChebyCoeff::eval_b() const { return Complex(re.eval_b(), im.eval_b()); }
Complex ComplexChebyCoeff::slope_a() const { return Complex(re.slope_a(), im.slope_a()); }
Complex ComplexChebyCoeff::slope_b() const { return Complex(re.slope_b(), im.slope_b()); }
Complex ComplexChebyCoeff::eval(Real x) const { return Complex(re.eval(x), im.eval(x)); }
ComplexChebyCoeff& ComplexChebyCoeff::operator+=(const ComplexChebyCoeff& u) {
    re += u.re;
    im += u.im;
    return *this;
}
ComplexChebyCoeff& ComplexChebyCoeff::operator-=(const ComplexChebyCoeff& u) {
    re -= u.re;
    im -= u.im;
    return *this;
}

ComplexChebyCoeff& ComplexChebyCoeff::operator*=(Real c) {
    re *= c;
    im *= c;
    return *this;
}

ComplexChebyCoeff& ComplexChebyCoeff::operator*=(const ComplexChebyCoeff& u) {
    assert(congruent(u));
    assert(re.state() == Physical);
    assert(im.state() == Physical);
    Real r;
    Real i;
    for (int ny = 0; ny < im.numModes(); ++ny) {
        r = re[ny] * u.re[ny] - im[ny] * u.im[ny];
        i = re[ny] * u.im[ny] + im[ny] * u.re[ny];
        re[ny] = r;
        im[ny] = i;
    }
    return *this;
}
void ComplexChebyCoeff::conjugate() { im *= -1.0; }

void ComplexChebyCoeff::save(const string& filebase, fieldstate savestate) const {
    fieldstate origstate = re.state();
    ((ChebyCoeff&)*this).makeState(savestate);

    string filename = appendSuffix(filebase, ".asc");
    ofstream os(filename.c_str());
    if (!os.good())
        cferror("ComplexChebyCoeff::save(filebase) :  can't open file " + filename);

    os << scientific << setprecision(REAL_DIGITS);
    os << "% " << re.length() << ' ' << re.a() << ' ' << re.b() << ' ' << re.state() << '\n';
    for (int n = 0; n < re.length(); ++n)
        os << setw(REAL_IOWIDTH) << re[n] << ' ' << setw(REAL_IOWIDTH) << im[n] << '\n';
    os.close();

    ((ChebyCoeff&)*this).makeState(origstate);
}

void ComplexChebyCoeff::binaryDump(ostream& os) const {
    re.binaryDump(os);
    im.binaryDump(os);
}
void ComplexChebyCoeff::binaryLoad(istream& is) {
    re.binaryLoad(is);
    im.binaryLoad(is);
}

Complex ComplexChebyCoeff::mean() const { return Complex(re.mean(), im.mean()); }

// Real ComplexChebyCoeff::energy() const {
// return re.energy() + im.energy();
//}

ComplexChebyCoeff& ComplexChebyCoeff::operator*=(Complex c) {
    Real cr = Re(c);
    Real ci = Im(c);
    Real ur;
    Real ui;
    for (int n = 0; n < re.numModes(); ++n) {
        ur = re[n];
        ui = im[n];
        re[n] = ur * cr - ui * ci;
        im[n] = ur * ci + ui * cr;
    }
    return *this;
}

bool ComplexChebyCoeff::congruent(const ComplexChebyCoeff& v) const {
    assert(re.congruent(im));
    return (re.congruent(v.re) && im.congruent(v.im));
}

bool operator==(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v) { return u.re == v.re && u.im == v.im; }
bool operator!=(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v) { return !(u == v); }

void swap(ComplexChebyCoeff& f, ComplexChebyCoeff& g) {
    swap(f.re, g.re);
    swap(f.im, g.im);
}

ComplexChebyCoeff operator*(Real c, const ComplexChebyCoeff& v) {
    ComplexChebyCoeff rtn(v);
    rtn *= c;
    return rtn;
}

ComplexChebyCoeff operator+(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v) {
    ComplexChebyCoeff rtn(u);
    rtn += v;
    return rtn;
}

ComplexChebyCoeff operator-(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v) {
    ComplexChebyCoeff rtn(u);
    rtn -= v;
    return rtn;
}

void diff(const ComplexChebyCoeff& u, ComplexChebyCoeff& du) {
    diff(u.re, du.re);
    diff(u.im, du.im);
}

void diff2(const ComplexChebyCoeff& u, ComplexChebyCoeff& d2u) {
    diff2(u.re, d2u.re);
    diff2(u.im, d2u.im);
}
void diff2(const ComplexChebyCoeff& u, ComplexChebyCoeff& d2u, ComplexChebyCoeff& tmp) {
    diff2(u.re, d2u.re, tmp.re);
    diff2(u.im, d2u.im, tmp.im);
}

void diff(const ComplexChebyCoeff& u, ComplexChebyCoeff& du, int n) {
    diff(u.re, du.re, n);
    diff(u.im, du.im, n);
}
ComplexChebyCoeff diff(const ComplexChebyCoeff& u) {
    ComplexChebyCoeff du(u.numModes(), u.a(), u.b(), Spectral);
    diff(u.re, du.re);
    diff(u.im, du.im);
    return du;
}
ComplexChebyCoeff diff2(const ComplexChebyCoeff& u) {
    ComplexChebyCoeff du2(u.numModes(), u.a(), u.b(), Spectral);
    diff2(u.re, du2.re);
    diff2(u.im, du2.im);
    return du2;
}
ComplexChebyCoeff diff(const ComplexChebyCoeff& u, int n) {
    ComplexChebyCoeff du(u.numModes(), u.a(), u.b(), Spectral);
    diff(u.re, du.re, n);
    diff(u.im, du.im, n);
    return du;
}

void integrate(const ComplexChebyCoeff& du, ComplexChebyCoeff& u) {
    integrate(du.re, u.re);
    integrate(du.im, u.im);
}

ComplexChebyCoeff integrate(const ComplexChebyCoeff& du) {
    ComplexChebyCoeff u(du.numModes(), du.a(), du.b(), Spectral);
    integrate(du.re, u.re);
    integrate(du.im, u.im);
    return u;
}

// ====================================================================
ChebyTransform::ChebyTransform(int N, uint flags)
    : N_(N), flags_(flags | FFTW_DESTROY_INPUT), cosfftw_plan_(nullptr, fftw_destroy_plan) {
    assert(N_ > 0);

    // Build an FFTW plan that can be applied to a number of different cfarrays,
    // using the FFTW guru interface. See FFTW guru documentation.
    unique_ptr<Real, void (*)(void*)> tmp(static_cast<Real*>(fftw_malloc(N_ * sizeof(Real))), fftw_free);
    cosfftw_plan_.reset(fftw_plan_r2r_1d(N, tmp.get(), tmp.get(), FFTW_REDFT00, flags_));
}

ChebyTransform::ChebyTransform(const ChebyTransform& t) : ChebyTransform::ChebyTransform(t.N_, t.flags_) {}

ChebyTransform& ChebyTransform::operator=(const ChebyTransform& t) {
    using std::swap;

    // Copy and swap only if it's not the same object
    if (&t != this) {
        ChebyTransform tmp(t);
        swap(*this, tmp);
    }

    return *this;
}

ostream& operator<<(ostream& os, const ComplexChebyCoeff& u) {
    for (int i = 0; i < u.length(); ++i)
        os << '(' << u.re[i] << ", " << u.im[i] << ")\n";
    return os;
}

// =======================================================================
// L2 norms
Real L2Norm2(const ComplexChebyCoeff& u, bool normalize) { return L2Norm2(u.re, normalize) + L2Norm2(u.im, normalize); }

Real L2Dist2(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v, bool normalize) {
    return L2Dist2(u.re, v.re, normalize) + L2Dist2(u.im, v.im, normalize);
}

Real L2Norm(const ComplexChebyCoeff& u, bool normalize) {
    return sqrt(L2Norm2(u.re, normalize) + L2Norm2(u.im, normalize));
}

Real L2Dist(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v, bool normalize) {
    return sqrt(L2Dist2(u.re, v.re, normalize) + L2Dist2(u.im, v.im, normalize));
}

Complex L2InnerProduct(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v, bool normalize) {
    assert(u.state() == Spectral && v.state() == Spectral);
    assert(u.numModes() == v.numModes());
    int N = u.numModes();

    // Unroll the complex operations, for efficiency.
    // Profiling an L2InnerProduct-intensive program reveals that
    // complex * double is expensive
    int e = 1;
    Real sum_r = 0.0;
    Real sum_i = 0.0;
    Real psum_r;
    Real psum_i;
    Real um_r;
    Real um_i;
    Real vn_r;
    Real vn_i;
    Real k;
    for (int m = N - 1; m >= 0; --m) {
        um_r = u.re[m];
        um_i = u.im[m];
        psum_r = 0.0;
        psum_i = 0.0;
        for (int n = m % 2; n < N; n += 2) {
            k = Real(e - m * m - n * n) / Real((e + m - n) * (e - m + n) * (e + m + n) * (e - m - n));
            vn_r = v.re[n];
            vn_i = v.im[n];

            psum_r += k * (um_r * vn_r + um_i * vn_i);
            psum_i += k * (um_i * vn_r - um_r * vn_i);
        }
        sum_r += psum_r;
        sum_i += psum_i;
    }
    if (!normalize) {
        sum_r *= u.b() - u.a();
        sum_i *= u.b() - u.a();
    }
    return Complex(sum_r, sum_i);
}

Real L1Norm(const ComplexChebyCoeff& u, bool normalize) { return L1Norm(u.re, normalize) + L1Norm(u.im, normalize); }

Real L1Dist(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v, bool normalize) {
    return L1Dist(u.re, v.re, normalize) + L1Dist(u.im, v.im, normalize);
}

Real LinfNorm(const ComplexChebyCoeff& u) { return LinfNorm(u.re) + LinfNorm(u.im); }

Real LinfDist(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v) {
    return LinfDist(u.re, v.re) + LinfDist(u.im, v.im);
}

// =======================================================================
// cheby norms
Real chebyNorm2(const ComplexChebyCoeff& u, bool normalize) {
    return chebyNorm2(u.re, normalize) + chebyNorm2(u.im, normalize);
}

Real chebyDist2(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v, bool normalize) {
    return chebyDist2(u.re, v.re, normalize) + chebyDist2(u.im, v.im, normalize);
}

Real chebyNorm(const ComplexChebyCoeff& u, bool normalize) {
    return sqrt(chebyNorm2(u.re, normalize) + chebyNorm2(u.im, normalize));
}

Real chebyDist(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v, bool normalize) {
    return sqrt(chebyDist2(u.re, v.re, normalize) + chebyDist2(u.im, v.im, normalize));
}

Complex chebyInnerProduct(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v, bool normalize) {
    assert(u.state() == Spectral && v.state() == Spectral);
    assert(u.numModes() == v.numModes());
    int N = u.numModes();

    // Unroll the complex operations, for efficiency.
    // Profiling an L2InnerProduct-intensive program reveals that
    // complex * double is expensive
    Real sum_r = 0.0;
    Real sum_i = 0.0;
    for (int m = N - 1; m > 0; --m) {
        sum_r += u.re[m] * v.re[m] + u.im[m] * v.im[m];
        sum_i += u.im[m] * v.re[m] - u.re[m] * v.im[m];
    }
    if (N > 0) {
        sum_r += 2 * (u.re[0] * v.re[0] + u.im[0] * v.im[0]);
        sum_i += 2 * (u.im[0] * v.re[0] - u.re[0] * v.im[0]);
    }
    if (!normalize) {
        sum_r *= u.b() - u.a();
        sum_i *= u.b() - u.a();
    }
    return Complex(sum_r * pi / 2, sum_i * pi / 2);
}

// ===========================================================
// switchable norms
Real norm2(const ComplexChebyCoeff& u, NormType n, bool normalize) {
    return (n == Uniform) ? L2Norm2(u, normalize) : chebyNorm2(u, normalize);
}

Real dist2(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v, NormType n, bool normalize) {
    return (n == Uniform) ? L2Dist2(u, v, normalize) : chebyDist2(u, v, normalize);
}

Real norm(const ComplexChebyCoeff& u, NormType n, bool normalize) {
    return (n == Uniform) ? L2Norm(u, normalize) : chebyNorm(u, normalize);
}

Real dist(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v, NormType n, bool normalize) {
    return (n == Uniform) ? L2Dist(u, v, normalize) : chebyDist(u, v, normalize);
}
Complex innerProduct(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v, NormType n, bool normalize) {
    return (n == Uniform) ? L2InnerProduct(u, v, normalize) : chebyInnerProduct(u, v, normalize);
}

}  // namespace chflow
