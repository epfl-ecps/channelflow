/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include "channelflow/realprofileng.h"
#include <algorithm>
#include <iostream>

using namespace std;

namespace chflow {

RealProfileNG::RealProfileNG(const int jx, const int jz, const int Nd, const int Ny, const Real Lx, const Real Lz,
                             const Real a, const Real b, const fieldstate state)
    : state_(state), jx_(jx), jz_(jz), Nd_(Nd), Ny_(Ny), Lx_(Lx), Lz_(Lz), a_(a), b_(b) {
    assert(Nd > 0);
    u_.reserve(Nd);
    ChebyCoeff u(Ny, a, b, Spectral);
    u_.push_back(u);
    for (int i = 1; i < Nd; ++i) {
        ChebyCoeff v = u;
        u_.push_back(v);
    }
}

RealProfileNG::RealProfileNG(const vector<ChebyCoeff> u, const int jx, const int jz, const Real Lx, const Real Lz)
    : state_(u[0].state()),
      jx_(jx),
      jz_(jz),
      Nd_(u.size()),
      Ny_(u[0].N()),
      Lx_(Lx),
      Lz_(Lz),
      a_(u[0].a()),
      b_(u[0].b()),
      u_(u) {
    for (vector<ChebyCoeff>::const_iterator it = u.begin(); it != u.end(); ++it)
        assert((*it).state() == state_);
}

RealProfileNG::RealProfileNG() : state_(Spectral), jx_(0), jz_(0), Nd_(1), Ny_(1), Lx_(0), Lz_(0), a_(-1), b_(1) {
    u_.reserve(Nd());
    ChebyCoeff u(Ny(), a(), b(), Spectral);
    u_.push_back(u);
}

RealProfileNG::RealProfileNG(const RealProfileNG& e)
    : state_(e.state()),
      jx_(e.jx()),
      jz_(e.jz()),
      Nd_(e.Nd()),
      Ny_(e.Ny()),
      Lx_(e.Lx()),
      Lz_(e.Lz()),
      a_(e.a()),
      b_(e.b()),
      u_(e.u_) {}

RealProfileNG& RealProfileNG::operator=(const RealProfileNG& e) {
    jx_ = e.jx_;
    jz_ = e.jz_;
    Lx_ = e.Lx_;
    Lz_ = e.Lz_;
    a_ = e.a_;
    b_ = e.b_;
    u_ = e.u_;
    Nd_ = e.Nd_;
    Ny_ = e.Ny_;
    return *this;
}

Real L2InnerProduct(const RealProfileNG& e1, const RealProfileNG& e2, const bool normalize) {
    assert(e1.congruent(e2));
    if ((e1.jx() != e2.jx()) || (e1.jz() != e2.jz()))
        return 0;

    Real l2ip = 0;
    for (int i = 0; i < e1.Nd(); ++i)
        l2ip += L2InnerProduct(e1.u_[i], e2.u_[i]);

    if (e1.jx() != 0) {
        l2ip /= 2;
    }
    if (e1.jz() != 0) {
        l2ip /= 2;
    }

    if (!normalize)
        l2ip *= e1.Lx() * e1.Lz();

    return l2ip;
}

RealProfileNG& RealProfileNG::operator*=(const Real c) {
    for (vector<ChebyCoeff>::iterator d = u_.begin(); d != u_.end(); ++d)
        (*d) *= c;
    return *this;
}

RealProfileNG& RealProfileNG::operator+=(const RealProfileNG& e) {
    assert(compatible(e));
    vector<ChebyCoeff>::const_iterator eit = e.u_.begin();
    for (vector<ChebyCoeff>::iterator d = u_.begin(); d != u_.end(); ++d) {
        (*d) += (*eit);
        ++eit;
    }
    return *this;
}

RealProfileNG& RealProfileNG::operator-=(const RealProfileNG& e) {
    assert(compatible(e));
    vector<ChebyCoeff>::const_iterator eit = e.u_.begin();
    for (vector<ChebyCoeff>::iterator d = u_.begin(); d != u_.end(); ++d) {
        (*d) -= (*eit);
        ++eit;
    }
    return *this;
}

RealProfileNG& RealProfileNG::operator*=(const FieldSymmetry& sigma) {
    assert(Nd() == 3);

    int multiplier = sigma.s();
    if (sigma.ax() == 0.5)
        jx() % 2 != 0 ? multiplier *= -1 : multiplier *= 1;
    else if (sigma.ax() != 0)
        cferror("Shifts other than 0 and 1/2 are not implemented!");

    if (sigma.az() == 0.5)
        jz() % 2 != 0 ? multiplier *= -1 : multiplier *= 1;
    else if (sigma.az() != 0)
        cferror("Shifts other than 0 and 1/2 are not implemented!");

    int umultiplier = sigma.sx() * multiplier;
    int vmultiplier = sigma.sy() * multiplier;
    int wmultiplier = sigma.sz() * multiplier;
    if (sigma.sx() == -1) {
        if (jx() < 0)
            umultiplier *= -1;
        else if (jx() > 0) {
            vmultiplier *= -1;
            wmultiplier *= -1;
        }
    }

    if (sigma.sz() == -1) {
        if (jz() < 0)
            wmultiplier *= -1;
        else if (jz() > 0) {
            umultiplier *= -1;
            vmultiplier *= -1;
        }
    }

    if (sigma.sy() == -1) {
        for (vector<ChebyCoeff>::iterator d = u_.begin(); d != u_.end(); ++d) {
            if (d->state() == Spectral) {
                for (int i = 1; i < Ny(); i += 2)
                    (*d)[i] *= -1;
            } else {
                assert(d->state() == Physical);
                for (int i = 0; i < Ny() / 2; ++i) {
                    Real tmp = (*d)[i];
                    (*d)[i] = (*d)[Ny() - i - 1];
                    (*d)[Ny() - i - 1] = tmp;
                }
            }
        }
    }

    u_[0] *= umultiplier;
    u_[1] *= vmultiplier;
    u_[2] *= wmultiplier;

    return (*this);
}

bool RealProfileNG::compatible(const RealProfileNG& e) const {
    if (e.Nd() != Nd())
        return false;
    if (e.Ny() != Ny())
        return false;
    if (e.a() != a())
        return false;
    if (e.b() != b())
        return false;
    if (e.Lx() != Lx())
        return false;
    if (e.Lz() != Lz())
        return false;
    if (e.state() != state())
        return false;
    if (e.jx() != jx())
        return false;
    if (e.jz() != jz())
        return false;

    return true;
}

bool RealProfileNG::congruent(const RealProfileNG& e) const {
    if (e.Nd() != Nd())
        return false;
    if (e.Ny() != Ny())
        return false;
    if (e.a() != a())
        return false;
    if (e.b() != b())
        return false;
    if (e.Lx() != Lx())
        return false;
    if (e.Lz() != Lz())
        return false;
    if (e.state() != state())
        return false;

    return true;
}

Complex RealProfileNG::normalization_p(const int d) const {
    int jjx = jx();
    int jjz = jz();
    if (d == 0) {
        jjz *= -1;
    } else if (d == 1) {
        jjx *= -1;
        jjz *= -1;
    } else if (d == 2) {
        jjx *= -1;
    }

    Complex Z(1.0, 0.0);
    if (jjz > 0) {
        Z /= 2;
    }
    if (jjz < 0) {
        Z /= 2.0 * I;
    }
    if (jjx > 0) {
        Z /= 2;
    }
    if (jjx < 0) {
        Z /= 2.0 * I;
    }
    return Z;
}

Complex RealProfileNG::normalization_m(const int d) const {
    int jjx = jx();
    int jjz = jz();
    if (d == 0)
        jjz *= -1;
    else if (d == 1) {
        jjx *= -1;
        jjz *= -1;
    } else if (d == 2) {
        jjx *= -1;
    }

    if (jjx == 0)
        return 0;

    Complex Z(1.0, 0.0);
    if (jjz > 0) {
        Z /= 2;
    }
    if (jjz < 0) {
        Z /= 2.0 * I;
    }
    if (jjx > 0) {
        Z /= 2;
    }
    if (jjx < 0) {
        Z /= -2.0 * I;
    }
    return Z;
}

Real L2Norm2(const RealProfileNG& e, bool normalize) {
    Real norm = 0;
    for (vector<ChebyCoeff>::const_iterator d = e.u_.begin(); d != e.u_.end(); ++d) {
        norm += L2Norm2(*d, normalize);
    }
    if (e.jx() != 0)
        norm /= 2;
    if (e.jz() != 0)
        norm /= 2;
    if (!normalize)
        norm *= e.Lx() * e.Lz();
    return norm;
}

vector<RealProfileNG> realBasisNG(const int Ny, const int kxmax, const int kzmax, const Real Lx, const Real Lz,
                                  const Real a, const Real b) {
    // Build y polynomials
    vector<ChebyCoeff> P;
    P.reserve(Ny + 2);
    ChebyTransform trans(Ny);
    for (int ny = 0; ny <= Ny; ++ny) {
        ChebyCoeff u(Ny, a, b, Spectral);
        legendre(ny, u, trans, false);
        P.push_back(u);
    }

    ChebyCoeff tmp(Ny, a, b, Spectral);
    ChebyCoeff tmp2(Ny, a, b, Spectral);

    vector<ChebyCoeff> R;
    R.reserve(Ny + 1);

    {
        ChebyCoeff& u = tmp;
        u.setToZero();
        u[0] = 0.5;
        u[1] = -0.5;
        R.push_back(u);

        u.setToZero();
        u[0] = 0.5;
        u[1] = 0.5;
        R.push_back(u);
    }

    for (int ny = 2; ny <= Ny; ++ny) {
        tmp = P[ny];
        tmp -= P[ny - 2];
        R.push_back(tmp);
    }

    vector<ChebyCoeff> Q;
    Q.reserve(Ny);

    // Wasteful, but handy for having indices match up
    for (int i = 0; i < 3; ++i)
        Q.push_back(tmp);

    for (int ny = 3; ny < Ny - 1; ++ny) {
        tmp = R[ny + 1];
        tmp *= 1.0 / (2 * ny + 1);
        tmp2 = R[ny - 1];
        tmp2 *= 1.0 / (2 * ny - 3);
        tmp -= tmp2;
        Q.push_back(tmp);
    }

    vector<ChebyCoeff> u;
    u.push_back(tmp);
    u.push_back(tmp);
    u.push_back(tmp);
    vector<RealProfileNG> basis;
    {
        const int J = kxmax - 1;
        const int K = kzmax - 1;
        const int L = Ny - 1;
        basis.reserve(4 * (J + 2 * J * K + K) * (L - 2) + 2 * (L - 1));
    }

    u[0].setToZero();
    u[1].setToZero();
    u[2].setToZero();

    const Real alpha = 2 * M_PI / Lx;
    const Real gamma = 2 * M_PI / Lz;

    // sigma = 0, jx=0, jz=0
    for (int ny = 2; ny < Ny; ++ny) {
        u[0] = R[ny];
        RealProfileNG e(u, 0, 0, Lx, Lz);
        basis.push_back(e);
    }

    // sigma = 1, jx=0, jz=0
    u[0].setToZero();
    for (int ny = 2; ny < Ny; ++ny) {
        u[2] = R[ny];
        RealProfileNG e(u, 0, 0, Lx, Lz);
        basis.push_back(e);
    }

    // sigma = 0, jx!=0, jz=0
    u[0].setToZero();
    u[1].setToZero();
    for (int ny = 2; ny < Ny; ++ny) {
        u[2] = R[ny];
        for (int jx = 1; jx < kxmax; ++jx) {
            RealProfileNG e(u, jx, 0, Lx, Lz);
            basis.push_back(e);
            e.setJx(-jx);
            basis.push_back(e);
        }
    }

    // sigma = 1, jx!=0, jz=0
    u[2].setToZero();
    for (int ny = 3; ny < Ny - 1; ++ny) {
        u[0] = R[ny];
        for (int jx = 1; jx < kxmax; ++jx) {
            u[1] = Q[ny];

            u[1] *= alpha * jx;
            RealProfileNG e(u, jx, 0, Lx, Lz);
            basis.push_back(e);

            u[1] *= -1;
            RealProfileNG e2(u, -jx, 0, Lx, Lz);
            basis.push_back(e2);
        }
    }

    // sigma = 0, jx = 0, jz !=0
    u[1].setToZero();
    u[2].setToZero();
    for (int ny = 2; ny < Ny; ++ny) {
        u[0] = R[ny];
        for (int jz = 1; jz < kzmax; ++jz) {
            RealProfileNG e(u, 0, jz, Lx, Lz);
            basis.push_back(e);
            e.setJz(-jz);
            basis.push_back(e);
        }
    }

    // sigma = 1, jx = 0, jz != 0
    u[0].setToZero();
    for (int ny = 3; ny < Ny - 1; ++ny) {
        u[2] = R[ny];
        for (int jz = 1; jz < kzmax; ++jz) {
            u[1] = Q[ny];

            u[1] *= gamma * jz;
            RealProfileNG e(u, 0, jz, Lx, Lz);
            basis.push_back(e);

            u[1] *= -1;
            RealProfileNG e2(u, 0, -jz, Lx, Lz);
            basis.push_back(e2);
        }
    }

    // sigma = 0, jx!=0, jz != 0
    u[1].setToZero();
    for (int ny = 2; ny < Ny; ++ny) {
        for (int jx = 1; jx < kxmax; ++jx) {
            u[2] = R[ny];
            u[2] *= -alpha * jx;
            for (int jz = 1; jz < kzmax; ++jz) {
                u[0] = R[ny];
                u[0] *= gamma * jz;

                //+,+
                RealProfileNG e(u, jx, jz, Lx, Lz);
                basis.push_back(e);

                //-,+
                e.u_[2] *= -1;
                e.setJx(-jx);
                basis.push_back(e);

                //-,-
                e.u_[0] *= -1;
                e.setJz(-jz);
                basis.push_back(e);

                //+,-
                e.u_[2] *= -1;
                e.setJx(jx);
                basis.push_back(e);
            }
        }
    }

    // sigma = 1, jx != 0, jz!=0
    for (int ny = 3; ny < Ny - 1; ++ny) {
        for (int jx = 1; jx < kxmax; ++jx) {
            u[2] = R[ny];
            u[2] *= alpha * jx;
            for (int jz = 1; jz < kzmax; ++jz) {
                u[0] = R[ny];
                u[0] *= gamma * jz;

                u[1] = Q[ny];
                u[1] *= 2.0 * alpha * gamma * jx * jz;

                //+,+
                RealProfileNG e(u, jx, jz, Lx, Lz);
                basis.push_back(e);

                //-,-
                e.u_[0] *= -1;
                e.u_[2] *= -1;
                e.setJx(-jx);
                e.setJz(-jz);
                basis.push_back(e);

                //-,+
                e.u_[1] *= -1;
                e.u_[0] *= -1;
                e.setJz(jz);
                basis.push_back(e);

                //+,-
                e.u_[2] *= -1;
                e.u_[0] *= -1;
                e.setJx(jx);
                e.setJz(-jz);
                basis.push_back(e);
            }
        }
    }
    return basis;
}

void orthonormalize(vector<RealProfileNG>& basis) {
    RealProfileNG tmp(basis[0]);
    for (vector<RealProfileNG>::iterator it1 = basis.begin(); it1 != basis.end(); ++it1) {
        for (vector<RealProfileNG>::iterator it2 = basis.begin(); it2 != it1; ++it2) {
            if (it1->jx() == it2->jx() && it1->jz() == it2->jz()) {
                Real l2ip = L2InnerProduct(*it1, *it2);
                tmp = *it2;
                tmp *= l2ip;
                *it1 -= tmp;
            } else
                continue;
        }
        (*it1) *= 1.0 / L2Norm(*it1);
    }
}

struct SymmetryChecker : public unary_function<const RealProfileNG&, bool> {
    vector<FieldSymmetry> s;
    RealProfileNG work;
    Real tol;
    SymmetryChecker(const vector<FieldSymmetry>& ss, const Real tolerence) : s(ss), tol(tolerence) { ; }

    // True if e is not symmetric wrt any one of the given symmetries
    bool operator()(const RealProfileNG& e) {
        for (vector<FieldSymmetry>::const_iterator si = s.begin(); si != s.end(); ++si) {
            work = e;
            work *= (*si);
            work -= e;
            if (L2Norm(work) > tol)
                return true;
        }
        return false;
    }
};

void selectSymmetries(vector<RealProfileNG>& basis, const vector<FieldSymmetry>& s, const Real tolerance) {
    SymmetryChecker chk(s, tolerance);
    basis.erase(remove_if(basis.begin(), basis.end(), chk), basis.end());
}

void RealProfileNG::makeSpectral() {
    if (state_ == Physical) {
        for (vector<ChebyCoeff>::iterator ui = u_.begin(); ui != u_.end(); ++ui)
            ui->makeSpectral();
        state_ = Spectral;
    }
}

void RealProfileNG::makePhysical() {
    if (state_ == Spectral) {
        for (vector<ChebyCoeff>::iterator ui = u_.begin(); ui != u_.end(); ++ui)
            ui->makePhysical();
        state_ = Physical;
    }
}

void RealProfileNG::makeState(const fieldstate s) {
    if (s == Physical)
        makePhysical();
    else if (s == Spectral)
        makeSpectral();
    else
        cferror("RealProfileNG::makeState(fieldstate) : Recieved state which was neither Spectral nor Physical");
}

void RealProfileNG::makeSpectral(const ChebyTransform& t) {
    if (state_ == Physical) {
        for (vector<ChebyCoeff>::iterator ui = u_.begin(); ui != u_.end(); ++ui)
            ui->makeSpectral(t);
        state_ = Spectral;
    }
}

void RealProfileNG::makePhysical(const ChebyTransform& t) {
    if (state_ == Spectral) {
        for (vector<ChebyCoeff>::iterator ui = u_.begin(); ui != u_.end(); ++ui)
            ui->makePhysical(t);
        state_ = Physical;
    }
}

void RealProfileNG::makeState(const fieldstate s, const ChebyTransform& t) {
    if (s == Physical)
        makePhysical(t);
    else if (s == Spectral)
        makeSpectral(t);
    else
        cferror(
            "RealProfileNG::makeState(const fieldstate, const ChebyTransform&) : Recieved state which was neither "
            "Spectral nor Physical");
}

}  // namespace chflow
