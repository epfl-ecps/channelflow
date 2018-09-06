/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <string.h>  // for strdupa
#include <fstream>
#include <iomanip>

#include "channelflow/flowfield.h"
#include "channelflow/symmetry.h"
#include "channelflow/utilfuncs.h"

using namespace std;

namespace chflow {

FieldSymmetry::FieldSymmetry() : s_(1), sx_(1), sy_(1), sz_(1), ax_(0), az_(0) {}

FieldSymmetry::FieldSymmetry(int s) : s_(s), sx_(1), sy_(1), sz_(1), ax_(0.0), az_(0.0) {}

FieldSymmetry::FieldSymmetry(Real ax, Real az) : s_(1), sx_(1), sy_(1), sz_(1), ax_(ax), az_(az) {}

FieldSymmetry::FieldSymmetry(int sx, int sy, int sz, Real ax, Real az, int s)
    : s_(s), sx_(sx), sy_(sy), sz_(sz), ax_(ax), az_(az) {
    if (abs(s) != 1 || abs(sx) != 1 || abs(sy) != 1 || abs(sz) != 1) {
        cerr << "error in FieldSymmetry::FieldSymmetry(sx, sy, sz, ax, az, s) :" << endl;
        cerr << "s,sx,sy,sz should be +/-1, but they are " << s << ',' << sx << ',' << sx << ',' << sz << endl;
        exit(1);
    }
}

FieldSymmetry::FieldSymmetry(bool sx, bool sy, bool sz, Real ax, Real az, bool s)
    : s_(s ? -1 : 1), sx_(sx ? -1 : 1), sy_(sy ? -1 : 1), sz_(sz ? -1 : 1), ax_(ax), az_(az) {}

FieldSymmetry::FieldSymmetry(const string& filebase) : s_(1), sx_(1), sy_(1), sz_(1), ax_(0.0), az_(0.0) {
    ifstream is;
    string filename = ifstreamOpen(is, filebase, ".asc");
    if (!is) {
        cerr << "FieldSymmetry::FieldSymmetry(filebase) : can't open file " << filebase << " or " << (filebase + ".asc")
             << endl;
        exit(1);
    }

    // Read in header. Form is "%N a b s"
    string comment;
    while (is.peek() == '%')
        getline(is, comment);
    is >> s_ >> sx_ >> sy_ >> sz_ >> ax_ >> az_;
    if (!is.good())
        cerr << "warning: bad istream in reading FieldSymmetry from file " << filename << endl;
    is.close();
}

int FieldSymmetry::s(int i) const {
    assert(i >= 0 && i < 3);
    int rtn = 0;
    switch (i) {
        case 0:
            rtn = sx_;
            break;
        case 1:
            rtn = sy_;
            break;
        case 2:
            rtn = sz_;
            break;
        default:
            cerr << "error in FieldSymmetry::s(int i) : i == " << i << " != 0,1, or 2" << endl;
            exit(1);
    }
    return rtn;
}

int FieldSymmetry::sign(int i) const { return s(i); }

void FieldSymmetry::save(const std::string& filebase, ios::openmode openflag) const {
    string filename = appendSuffix(filebase, ".asc");
    ofstream os(filename.c_str(), openflag);
    char s = ' ';
    os << setprecision(17);
    // os << "% (s sx sy sz ax az) [u,v,w](x,y,z) = s [sx u, sy v, sz w](sx x + ax Lx, sy y, sz z + az Lz)\n";
    // os << "% s sx sy sz ax az\n";

    os << s_ << s << sx_ << s << sy_ << s << sz_ << s << ax_ << s << az_ << endl;
}

bool FieldSymmetry::isIdentity() const {
    bool is_identity = false;
    if (sx_ == 1 && sy_ == 1 && sz_ == 1 && ax_ == 0.0 && az_ == 0.0 && s_ == 1) {
        is_identity = true;
    }
    return is_identity;
}

FlowField FieldSymmetry::operator()(const FlowField& u) const {
    FlowField v(u);
    v *= (*this);
    return v;
}

RealProfile FieldSymmetry::operator()(const RealProfile& u) const {
    RealProfile v(u);
    v *= (*this);
    return v;
}

FieldSymmetry& FieldSymmetry::operator*=(Real c) {
    if (s_ != 1 || sx_ != 1 || sy_ != 1 || sz_ != 1)
        cerr << "warning: FieldSymmetry *= Real should only be applied to pure translations" << endl;
    // ax_ = fmod(c*ax_+0.5, 1.0) - 0.5;
    // az_ = fmod(c*az_+0.5, 1.0) - 0.5;
    ax_ = c * ax_;
    az_ = c * az_;
    return *this;
}

FieldSymmetry& FieldSymmetry::operator*=(const FieldSymmetry& p) {
    s_ *= p.s();
    sx_ *= p.sx();
    sy_ *= p.sy();
    sz_ *= p.sz();
    // ax_ = fmod(p.ax() + p.sx()*ax_ + 0.5, 1.0) - 0.5;
    // az_ = fmod(p.az() + p.sz()*az_ + 0.5, 1.0) - 0.5;
    ax_ = p.ax() + p.sx() * ax_;
    az_ = p.az() + p.sz() * az_;
    return *this;
}

FieldSymmetry operator*(const FieldSymmetry& p, const FieldSymmetry& q) {
    int s = p.s() * q.s();
    int sx = p.sx() * q.sx();
    int sy = p.sy() * q.sy();
    int sz = p.sz() * q.sz();
    // Real ax = fmod(p.ax() + p.sx()*q.ax()+0.5, 1.0) - 0.5;
    // Real az = fmod(p.az() + p.sz()*q.az()+0.5, 1.0) - 0.5;
    Real ax = p.ax() + p.sx() * q.ax();
    Real az = p.az() + p.sz() * q.az();
    return FieldSymmetry(sx, sy, sz, ax, az, s);
}

FieldSymmetry inverse(const FieldSymmetry& s) { return FieldSymmetry(s.sx(), s.sy(), s.sz(), -s.ax(), -s.az(), s.s()); }

FlowField operator*(const FieldSymmetry& s, const FlowField& u) { return s(u); }

bool operator==(const FieldSymmetry& p, const FieldSymmetry& q) {
    return (p.s() == q.s() && p.sx() == q.sx() && p.sy() == q.sy() && p.sz() == q.sz() && p.ax() == q.ax() &&
            p.az() == q.az());
}

bool operator!=(const FieldSymmetry& p, const FieldSymmetry& q) { return !(p == q); }

FieldSymmetry quadraticInterpolate(const cfarray<FieldSymmetry>& sn, const cfarray<Real>& mun, Real mu) {
    if (sn.length() != 3 || mun.length() != 3) {
        cerr << "error in quadraticInterpolate(cfarray<FlowField>& un, cfarray<Real>& mun, Real mu, Real eps)\n";
        cerr << "sn.length() != 3 || mun.length() !=3\n";
        cerr << "sn.length()  == " << sn.length() << '\n';
        cerr << "mun.length() == " << mun.length() << '\n';
        cerr << "exiting" << endl;
        exit(1);
    }

    const int sx = sn[0].sx();
    const int sy = sn[0].sy();
    const int sz = sn[0].sz();
    const int s = sn[0].s();

    if (sn[1].sx() != sx || sn[2].sx() != sx || sn[1].sy() != sy || sn[2].sy() != sy || sn[1].sz() != sz ||
        sn[2].sz() != sz || sn[1].s() != s || sn[2].s() != s) {
        cerr << "error in quadraticInterpolate(cfarray<FieldSymmetry>& sn, cfarray<Real>& mun, Real mu, Real eps)\n";
        cerr << "Incompatible symmetries for continuous extrapolation. Exiting" << endl;
        exit(1);
    }

    cfarray<Real> fn(3);

    for (int i = 0; i < 3; ++i)
        fn[i] = sn[i].ax();
    const Real ax = isconst(fn) ? fn[0] : quadraticInterpolate(fn, mun, mu);

    for (int i = 0; i < 3; ++i)
        fn[i] = sn[i].az();
    const Real az = isconst(fn) ? fn[0] : quadraticInterpolate(fn, mun, mu);

    return FieldSymmetry(sx, sy, sz, ax, az, s);
}

FieldSymmetry polynomialInterpolate(const cfarray<FieldSymmetry>& sn, const cfarray<Real>& mun, Real mu) {
    if (sn.length() != mun.length() || sn.length() < 1) {
        cerr << "error in quadraticInterpolate(cfarray<FlowField>& un, cfarray<Real>& mun, Real mu, Real eps)\n";
        cerr << "sn.length() != mun.length() or sn.length < 1\n";
        cerr << "sn.length()  == " << sn.length() << '\n';
        cerr << "mun.length() == " << mun.length() << '\n';
        cerr << "exiting" << endl;
        exit(1);
    }

    const int sx = sn[0].sx();
    const int sy = sn[0].sy();
    const int sz = sn[0].sz();
    const int s = sn[0].s();

    if (sn[1].sx() != sx || sn[2].sx() != sx || sn[1].sy() != sy || sn[2].sy() != sy || sn[1].sz() != sz ||
        sn[2].sz() != sz || sn[1].s() != s || sn[2].s() != s) {
        cerr << "error in quadraticInterpolate(cfarray<FieldSymmetry>& sn, cfarray<Real>& mun, Real mu, Real eps)\n";
        cerr << "Incompatible symmetries for continuous extrapolation. Exiting" << endl;
        exit(1);
    }

    const int N = sn.length();
    cfarray<Real> fn(N);

    for (int i = 0; i < N; ++i)
        fn[i] = sn[i].ax();
    const Real ax = polynomialInterpolate(fn, mun, mu);

    for (int i = 0; i < N; ++i)
        fn[i] = sn[i].az();
    const Real az = polynomialInterpolate(fn, mun, mu);

    return FieldSymmetry(sx, sy, sz, ax, az, s);
}

ostream& operator<<(ostream& os, const FieldSymmetry& s) {
    // streamsize p = os.precision();
    os << setw(3) << setiosflags(ios::right) << s.s() << setw(3) << setiosflags(ios::right) << s.sx() << setw(3)
       << setiosflags(ios::right) << s.sy() << setw(3) << setiosflags(ios::right) << s.sz()
       << resetiosflags(ios::adjustfield) << '\t' << s.ax() << '\t' << s.az();
    /***********
    char sp = ' ';
    os << '('
       << s.s() << sp
       << s.sx() << sp
       << s.sy() << sp
       << s.sz() << sp
       << s.ax() << sp
       << s.az()
       << ')';
    ***************/
    return os;
}

istream& operator>>(istream& is, FieldSymmetry& sigma) {
    // char c;
    /// is >> c;
    // if (c !='(' ) {
    // cerr << "error in istream >> FieldSymmetry : expected '(' as opening character" << endl;
    // cerr << "got '" << c << "' instead" << endl;
    //}
    int s, sx, sy, sz;
    Real ax, az;
    is >> s >> sx >> sy >> sz >> ax >> az;
    // if (c !=')' )
    // cferror("error in istream >> FieldSymmetry : expected ')' as closing character");

    sigma = FieldSymmetry(sx, sy, sz, ax, az, s);
    return is;
}

void project(const FieldSymmetry& s, const FlowField& u, FlowField& Pu) {
    if (&Pu == &u) {
        FlowField tmp = u;
        tmp *= s;
        tmp += u;
        tmp *= 0.5;
        Pu = tmp;
    } else {
        Pu = u;
        Pu *= s;
        Pu += u;
        Pu *= 0.5;
    }
}

FlowField project(const FieldSymmetry& s, FlowField& u) {
    FlowField Pu;
    project(s, u, Pu);
    return Pu;
}

void project(const cfarray<FieldSymmetry>& s, const FlowField& u, FlowField& Pu) {
    //   cout << "Entering project3" << flush;
    //   MPI_Barrier(MPI_COMM_WORLD);
    //   cout << "..." << endl;
    //
    FlowField tmp;
    Pu = u;
    for (int n = 0; n < s.length(); ++n) {
        //     cout << "Projecting "  << n << flush;
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     cout << "..." << endl;

        tmp = Pu;
        tmp *= s[n];
        Pu += tmp;
    }
    Pu *= pow(0.5, s.length());
}

FlowField project(const cfarray<FieldSymmetry>& s, FlowField& u) {
    //   cout << "Entering project2" << flush;
    //   MPI_Barrier(MPI_COMM_WORLD);
    //   cout << "..." << endl;
    //
    FlowField Pu;
    project(s, u, Pu);
    return Pu;
}

void project(const cfarray<FieldSymmetry>& s, FlowField& u, const string& label, ostream& os, Real eps) {
    //   cout << "Entering project" << flush;
    //   MPI_Barrier(MPI_COMM_WORLD);
    //   cout << "..." << endl;
    if (s.length() > 0) {
        //     cout << "Entering if clause in project" << flush;
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     cout << "..." << endl;

        printout("Projecting " + label + " onto invariant symmetric subspace ...", os);
        FlowField Pu = project(s, u);
        Real Perr = L2Dist(u, Pu);
        if (Perr > eps) {
            printout("WARNING: Total projection error is " + r2s(Perr) + " > " + r2s(eps), os);
            printout("WARNING: Are you sure you have specified the generators for the symmetry group correctly?", os);
        }
        u = Pu;
    }
}

SymmetryList::SymmetryList() {}

SymmetryList::SymmetryList(int N) : cfarray<FieldSymmetry>(N) {}

SymmetryList::SymmetryList(const string& filebase) {
    uint N;
    if (mpirank() == 0) {
        ifstream is;
        ifstreamOpen(is, filebase, ".asc");
        if (!is) {
            cerr << "SymmetryList::SymmetryList(filebase) : can't open file " << filebase << " or "
                 << (filebase + ".asc") << endl;
            exit(1);
        }
        char s;
        is >> s;
        if (s != '%') {
            cerr << "SymmetryList(filebase) error : first line of file should be" << endl;
            cerr << "% N" << endl;
            cerr << "where N is an integer spcifying how many symmetries are listed" << endl;
            exit(1);
        }
        is >> N;
        resize(N);
        for (uint n = 0; n < N; ++n)
            is >> (*this)[n];
    }

#ifdef HAVE_MPI
    MPI_Bcast(&N, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (mpirank() != 0)
        resize(N);
    for (uint n = 0; n < N; ++n) {
        int ss[4];
        Real a[2];
        if (mpirank() == 0) {
            FieldSymmetry s = (*this)[n];
            ss[0] = s.sx();
            ss[1] = s.sy();
            ss[2] = s.sz();
            ss[3] = s.s();
            a[0] = s.ax();
            a[1] = s.az();
        }
        MPI_Bcast(ss, 4, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(a, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (mpirank() != 0)
            (*this)[n] = FieldSymmetry(ss[0], ss[1], ss[2], a[0], a[1], ss[3]);
    }
#endif
}

void SymmetryList::save(const string& filebase) const {
    if (mpirank() == 0) {
        string filename = appendSuffix(filebase, ".asc");
        ofstream os(filename.c_str());

        int N = this->length();
        os << "% " << N << '\n';
        os << setprecision(17);
        for (int n = 0; n < N; ++n)
            os << (*this)[n] << endl;
    }
}

ostream& operator<<(ostream& os, const SymmetryList& s) {
    for (int n = 0; n < s.length(); ++n)
        os << s[n] << endl;
    return os;
}

}  // namespace chflow
