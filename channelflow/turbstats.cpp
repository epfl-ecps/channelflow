/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <fstream>
#include <iomanip>

#include "channelflow/turbstats.h"

#include "cfbasics/mathdefs.h"
using namespace std;

namespace chflow {

TurbStats::TurbStats() : count_(0), nu_(0) {}

TurbStats::TurbStats(const ChebyCoeff& Ubase, Real nu)
    : count_(0),
      nu_(nu),
      Ubase_(Ubase),
      ubase_(Ubase.numModes(), Ubase.a(), Ubase.b(), Physical),
      U_(Ubase.numModes(), Ubase.a(), Ubase.b(), Physical),
      uu_(Ubase.numModes(), Ubase.a(), Ubase.b(), Physical),
      uv_(Ubase.numModes(), Ubase.a(), Ubase.b(), Physical),
      uw_(Ubase.numModes(), Ubase.a(), Ubase.b(), Physical),
      vv_(Ubase.numModes(), Ubase.a(), Ubase.b(), Physical),
      vw_(Ubase.numModes(), Ubase.a(), Ubase.b(), Physical),
      ww_(Ubase.numModes(), Ubase.a(), Ubase.b(), Physical) {
    if (Ubase_.state() == Spectral) {
        ChebyTransform t(Ubase_.numModes());
        Ubase_.makePhysical(t);
    }
}

void TurbStats::reset() {
    count_ = 0;
    Ubase_.setToZero();
    ubase_.setToZero();
    U_.setToZero();
    uu_.setToZero();
    uv_.setToZero();
    uw_.setToZero();
    vv_.setToZero();
    vw_.setToZero();
    ww_.setToZero();
}

void TurbStats::addData(FlowField& un, FlowField& tmp) {
    fieldstate xzstate = un.xzstate();
    fieldstate ystate = un.ystate();
    int Nx = un.numXgridpts();
    int Ny = un.numYgridpts();
    int Nz = un.numZgridpts();

    // Add 0,0 profile to mean velocity
    un.makeSpectral_xz();
    un.makePhysical_y();

    for (int ny = 0; ny < Ny; ++ny) {
        ubase_[ny] += Re(un.cmplx(0, ny, 0, 0));
        U_[ny] += Re(un.cmplx(0, ny, 0, 0)) + Ubase_[ny];
    }

    // Compute uu(y).
    un.makePhysical_xz();
    tmp.setToZero();
    tmp.setState(Physical, Physical);

    // tmp[0]=uu, tmp[1]=vv, tmp[2]=ww
    for (int nx = 0; nx < Nx; ++nx)
        for (int nz = 0; nz < Nz; ++nz)
            for (int ny = 0; ny < Ny; ++ny) {
                // cache thrash
                Real u = un(nx, ny, nz, 0) + Ubase_[ny];
                Real v = un(nx, ny, nz, 1);
                Real w = un(nx, ny, nz, 2);
                tmp(nx, ny, nz, 0) = u * u;
                tmp(nx, ny, nz, 1) = u * v;
                tmp(nx, ny, nz, 2) = u * w;
                tmp(nx, ny, nz, 3) = v * v;
                tmp(nx, ny, nz, 4) = v * w;
                tmp(nx, ny, nz, 5) = w * w;
            }
    tmp.makeSpectral_xz();
    for (int ny = 0; ny < Ny; ++ny) {
        uu_[ny] += Re(tmp.cmplx(0, ny, 0, 0));
        uv_[ny] += Re(tmp.cmplx(0, ny, 0, 1));
        uw_[ny] += Re(tmp.cmplx(0, ny, 0, 2));
        vv_[ny] += Re(tmp.cmplx(0, ny, 0, 3));
        vw_[ny] += Re(tmp.cmplx(0, ny, 0, 4));
        ww_[ny] += Re(tmp.cmplx(0, ny, 0, 5));
    }
    ++count_;
    un.makeState(xzstate, ystate);
}

Real TurbStats::ustar() const {
    ChebyCoeff U = U_;
    U *= 1.0 / count_;
    ChebyTransform trans(U.N());
    U.makeSpectral(trans);
    ChebyCoeff dUdy = diff(U);
    dUdy.makePhysical(trans);
    return sqrt(nu_ / 2 * (abs(dUdy.eval_a() + abs(dUdy.eval_b()))));
}

Real TurbStats::bulkReynolds() const {
    // calculate bulk velocity
    ChebyCoeff U = U_;
    U *= 1.0 / count_;
    ChebyTransform trans(U.N());
    U.makeSpectral(trans);
    Real Ubulk = U.mean();
    return 0.5 * (U.b() - U.a()) * Ubulk / nu_;
}

Real TurbStats::parabolicReynolds() const { return 1.5 * bulkReynolds(); }

Real TurbStats::centerlineReynolds() const {
    ChebyCoeff U = U_;
    U *= 1.0 / count_;
    ChebyTransform trans(U.N());
    U.makeSpectral(trans);
    Real center = 0.5 * (U.a() + U.b());
    Real Ucenter = U.eval(center);
    return 0.5 * (U.b() - U.a()) * Ucenter / nu_;
}

Real TurbStats::hplus() const { return ustar() / nu_ * (U_.b() - U_.a()) / 2; }

Vector TurbStats::yplus() const {
    int Ny = U_.numModes();
    Real a = U_.a();
    Real b = U_.b();
    Real c = ustar() / nu_;

    Vector y = chebypoints(Ny, a, b);
    Vector yp(Ny);
    for (int ny = 0; ny < Ny; ++ny)
        yp[ny] = c * (y[ny] - a);
    return yp;
}

ChebyCoeff TurbStats::U() const {
    ChebyCoeff rtn(U_);
    rtn *= 1.0 / count_;
    return rtn;
}

ChebyCoeff TurbStats::ubase() const {
    ChebyCoeff rtn(ubase_);
    rtn *= 1.0 / count_;
    return rtn;
}

ChebyCoeff TurbStats::uu() const {
    int Ny = uu_.numModes();
    ChebyCoeff rtn(Ny, uu_.a(), uu_.b(), Physical);
    Real c = 1.0 / count_;
    for (int ny = 0; ny < Ny; ++ny)
        rtn[ny] = c * uu_[ny] - square(c * U_[ny]);

    return rtn;
}

ChebyCoeff TurbStats::uv() const {
    ChebyCoeff rtn = uv_;
    rtn *= 1.0 / count_;
    return rtn;
}

ChebyCoeff TurbStats::uw() const {
    ChebyCoeff rtn = uw_;
    rtn *= 1.0 / count_;
    return rtn;
}
ChebyCoeff TurbStats::vv() const {
    ChebyCoeff rtn = vv_;
    rtn *= 1.0 / count_;
    return rtn;
}
ChebyCoeff TurbStats::vw() const {
    ChebyCoeff rtn = vw_;
    rtn *= 1.0 / count_;
    return rtn;
}
ChebyCoeff TurbStats::ww() const {
    ChebyCoeff rtn = ww_;
    rtn *= 1.0 / count_;
    return rtn;
}

void TurbStats::msave(const string& filebase, bool wallunits) const {
    string filename(filebase);
    filename += string(".asc");
    ofstream os(filename.c_str());
    os << setprecision(REAL_DIGITS);
    char s = ' ';
    int Ny = uu_.numModes();
    Real u_star = ustar();

    os << "% Ny a b nu datacount wallunits? ustar\n";

    os << "% " << Ny << s << uu_.a() << s << uu_.b() << s << nu_ << s << count_ << s << wallunits << s << u_star
       << '\n';

    os << "% <uu> <uv> <uw> <vv> <vw> <ww> <utot> <utot-Ubase> Ubase\n";

    Real c = (wallunits) ? 1.0 / square(u_star) : 1.0;
    Real d = (wallunits) ? 1.0 / u_star : 1.0;
    c /= count_;
    d /= count_;
    for (int ny = 0; ny < Ny; ++ny) {
        os << c * (uu_[ny] - square(U_[ny]) / count_) << s << c * uv_[ny] << s << c * uw_[ny] << s << c * vv_[ny] << s
           << c * vw_[ny] << s << c * ww_[ny] << s << d * U_[ny] << s << d * ubase_[ny] << s << d * Ubase_[ny] << '\n';
    }
}

}  // namespace chflow
