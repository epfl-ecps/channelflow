/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */

#include "nsolver/multiShootingDSI.h"
#include "cfbasics/cfbasics.h"

using namespace std;
using namespace Eigen;

namespace chflow {

MultishootingDSI::MultishootingDSI() {}

MultishootingDSI::MultishootingDSI(int nShot, bool TSearch, bool xrelative, bool zrelative, Real TRef, Real axRef,
                                   Real azRef, bool fix_tphase)
    : Tsearch_(TSearch),
      xrelative_(xrelative),
      zrelative_(zrelative),
      fixtphase_(fix_tphase),
      nShot_(nShot),
      axRef_(axRef),
      azRef_(azRef),
      TRef_(TRef) {}

void MultishootingDSI::setDSI(DSI& dsi, int vec_size, bool isVecLong) {
    assert(Tsearch_ == false || Tsearch_ == true);
    assert(xrelative_ == false || xrelative_ == true);
    assert(zrelative_ == false || zrelative_ == true);

    const int taskid = mpirank();

    if (isVecLong) {
        Nxtot_ = vec_size;
        Nx_ = (taskid == 0) ? Nxtot_ - Tsearch_ - xrelative_ - zrelative_ : Nxtot_;
        Ny_ = Nx_ / nShot_;
        Nytot_ = (taskid == 0) ? Ny_ + Tsearch_ + xrelative_ + zrelative_ : Ny_;
    } else {
        Nytot_ = vec_size;
        Ny_ = (taskid == 0) ? Nytot_ - Tsearch_ - xrelative_ - zrelative_ : Nytot_;
        Nx_ = Ny_ * nShot_;
        Nxtot_ = (taskid == 0) ? Nx_ + Tsearch_ + xrelative_ + zrelative_ : Nx_;
    }

    NT_x_ = Nx_ + Tsearch_ - 1;
    Nax_x_ = Nx_ + Tsearch_ + xrelative_ - 1;
    Naz_x_ = Nx_ + Tsearch_ + xrelative_ + zrelative_ - 1;

    NT_y_ = Ny_ + Tsearch_ - 1;
    Nax_y_ = Ny_ + Tsearch_ + xrelative_ - 1;
    Naz_y_ = Ny_ + Tsearch_ + xrelative_ + zrelative_ - 1;

    dsi_ = &dsi;

    isDSIset_ = true;
}

bool MultishootingDSI::isDSIset() { return isDSIset_; }

bool MultishootingDSI::isVecMS(int Vec_size, bool isAC) {
    const int taskid = mpirank();
    Real length = (isAC && taskid == 0) ? Vec_size - 1 : Vec_size;
    bool isVecMS_ = (length == Nxtot_) ? true : false;
    return isVecMS_;
}

void MultishootingDSI::updateMu(Real mu) { dsi_->updateMu(mu); }

VectorXd MultishootingDSI::eval(const VectorXd& x) {
    VectorXd Gx(Nxtot_);
    setToZero(Gx);

    MatrixXd y(Nytot_, nShot_);
    y = extractVectors(x);

    MatrixXd Gy(Nytot_, nShot_);
    if (nShot_ > 1) {
        // multi-shoot, applying symmetry only on last shot
        for (int i = 0; i < nShot_ - 1; i++)
            Gy.col(i) = (*dsi_).eval(y.col(i), y.col(i + 1), false);
        Gy.col(nShot_ - 1) = (*dsi_).eval(y.col(nShot_ - 1), y.col(0), true);
    } else {
        // for one shot, use single-shot eval
        Gy.col(0) = (*dsi_).eval(y.col(0));
    }

    Gx = toVector(Gy);

    const int taskid = mpirank();

    if (fixtphase_) {
        Real diffobs = fixtphase(x);
        if (taskid == 0)
            Gx(NT_x_) = diffobs;
    }

    return Gx;
}

VectorXd MultishootingDSI::Jacobian(const VectorXd& x, const VectorXd& dx, const VectorXd& Gx, const Real& epsDx,
                                    bool centdiff, int& fcount) {
    VectorXd DG_dx;

    if (nShot_ == 1 && !(fixtphase_)) {
        MatrixXd y(Nytot_, nShot_);
        y = extractVectors(x);
        DG_dx = (*dsi_).Jacobian(y.col(0), dx, Gx, epsDx, centdiff, fcount);
    } else {
        Real step_magn = L2Norm(dx);
        Real eps = (step_magn < epsDx) ? 1 : epsDx / step_magn;
        if (centdiff) {
            Real eps2 = 0.5 * eps;
            VectorXd x_epsdx = x + eps2 * dx;
            VectorXd Gx_epsdxplus = eval(x_epsdx);
            ++fcount;
            x_epsdx = x - eps2 * dx;
            VectorXd Gx_epsdxminus = eval(x_epsdx);
            ++fcount;
            DG_dx = 1 / eps * (Gx_epsdxplus - Gx_epsdxminus);
        } else {
            VectorXd x_epsdx = x + eps * dx;
            VectorXd Gx_epsdx = eval(x_epsdx);
            ++fcount;
            DG_dx = 1 / eps * (Gx_epsdx - Gx);
        }
    }

    return DG_dx;
}

VectorXd MultishootingDSI::toVector(const MatrixXd& y) {
    VectorXd x(Nxtot_);

    for (int i = 0; i < nShot_; i++) {
        VectorXd yvec = y.col(i);

        for (int j = 0; j < Ny_; j++)
            x(i * Ny_ + j) = yvec(j);
    }

    const int taskid = mpirank();

    if (taskid == 0 && Tsearch_)
        x(NT_x_) = y(NT_y_, nShot_ - 1) * nShot_ / TRef_;
    if (taskid == 0 && xrelative_)
        x(Nax_x_) = y(Nax_y_, nShot_ - 1) / axRef_;
    if (taskid == 0 && zrelative_)
        x(Naz_x_) = y(Naz_y_, nShot_ - 1) / azRef_;

    return x;
}

MatrixXd MultishootingDSI::extractVectors(const VectorXd& x) {
    const int taskid = mpirank();

    Real T = 0, ax = 0, az = 0;

    if (taskid == 0) {
        T = Tsearch_ ? x(NT_x_) * TRef_ : 0;
        ax = xrelative_ ? x(Nax_x_) * axRef_ : 0;
        az = zrelative_ ? x(Naz_x_) * azRef_ : 0;
    }
    Real Tms = T / nShot_;

    MatrixXd y(Nytot_, nShot_);
    setToZero(y);

    VectorXd yvec(y.rows());
    setToZero(yvec);

    for (int i = 0; i < nShot_; i++) {
        for (int j = 0; j < Ny_; j++)
            yvec(j) = x(i * Ny_ + j);

        if (taskid == 0 && Tsearch_)
            yvec(NT_y_) = Tms;

        y.col(i) = yvec;
    }
    if (taskid == 0 && xrelative_)
        y(Nax_y_, nShot_ - 1) = ax;
    if (taskid == 0 && zrelative_)
        y(Naz_y_, nShot_ - 1) = az;

    return y;
}

VectorXd MultishootingDSI::makeMSVector(const VectorXd& yvec) {
    Real T = 0, ax = 0, az = 0;

    const int taskid = mpirank();

    if (taskid == 0) {
        T = Tsearch_ ? yvec(NT_y_) : 0;
        ax = xrelative_ ? yvec(Nax_y_) : 0;
        az = zrelative_ ? yvec(Naz_y_) : 0;
    }
    Real Tms = T / nShot_;

    MatrixXd y(Nytot_, nShot_);
    setToZero(y);
    y.col(0) = yvec;

    VectorXd Gy(yvec.rows());
    setToZero(Gy);

    for (int i = 1; i < nShot_; i++) {
        if (taskid == 0 && Tsearch_)
            y(NT_y_, i - 1) = Tms;
        if (taskid == 0 && xrelative_)
            y(Nax_y_, i - 1) = 0;
        if (taskid == 0 && zrelative_)
            y(Naz_y_, i - 1) = 0;

        Gy = (*dsi_).eval(y.col(i - 1), VectorXd::Zero(Nytot_), false);
        y.col(i) = Gy;
    }

    if (taskid == 0 && Tsearch_)
        y(NT_y_, nShot_ - 1) = Tms;
    if (taskid == 0 && xrelative_)
        y(Nax_y_, nShot_ - 1) = ax;
    if (taskid == 0 && zrelative_)
        y(Naz_y_, nShot_ - 1) = az;

    VectorXd x(Nxtot_);
    x = toVector(y);

    return x;
}

VectorXd MultishootingDSI::xdiff(const VectorXd& x) {
    MatrixXd y;
    y = extractVectors(x);

    VectorXd diffyvec;
    diffyvec = (*dsi_).xdiff(y.col(0));

    MatrixXd diffy(Nytot_, nShot_);
    setToZero(diffy);
    diffy.col(0) = diffyvec;

    VectorXd xdiff(x.rows());
    xdiff.setZero();
    xdiff = toVector(diffy);

    return xdiff;
}

VectorXd MultishootingDSI::zdiff(const VectorXd& x) {
    MatrixXd y;
    y = extractVectors(x);

    VectorXd diffyvec;
    diffyvec = (*dsi_).zdiff(y.col(0));

    MatrixXd diffy(Nytot_, nShot_);
    setToZero(diffy);
    diffy.col(0) = diffyvec;

    VectorXd zdiff(x.rows());
    zdiff.setZero();
    zdiff = toVector(diffy);

    return zdiff;
}

VectorXd MultishootingDSI::tdiff(const VectorXd& x, Real epsDt) {
    MatrixXd y;
    y = extractVectors(x);

    VectorXd diffyvec;
    diffyvec = (*dsi_).tdiff(y.col(0), epsDt);

    MatrixXd diffy(Nytot_, nShot_);
    setToZero(diffy);
    diffy.col(0) = diffyvec;

    VectorXd tdiff(x.rows());
    tdiff.setZero();
    tdiff = toVector(diffy);

    return tdiff;
}

Real MultishootingDSI::extractT(const VectorXd& x) {
    MatrixXd y;
    y = extractVectors(x);
    Real Tms = (*dsi_).extractT(y.col(nShot_ - 1));
    if (Tsearch_)
        Tms *= nShot_;
    return Tms;
}

Real MultishootingDSI::extractXshift(const VectorXd& x) {
    MatrixXd y;
    y = extractVectors(x);
    Real axms = (*dsi_).extractXshift(y.col(nShot_ - 1));
    return axms;
}

Real MultishootingDSI::extractZshift(const VectorXd& x) {
    MatrixXd y;
    y = extractVectors(x);
    Real azms = (*dsi_).extractZshift(y.col(nShot_ - 1));
    return azms;
}

Real MultishootingDSI::observable(VectorXd& x) {
    MatrixXd y;
    y = extractVectors(x);

    VectorXd yvec;

    Real obs = 0;
    for (int i = 0; i < nShot_; i++) {
        yvec = y.col(i);
        obs += (*dsi_).observable(yvec);
    }
    obs = obs / nShot_;
    return obs;
}

void MultishootingDSI::phaseShift(VectorXd& x, bool isAC) {
    const int taskid = mpirank();
    Real mu = 0.0;
    int N = x.size();
    if (isAC && taskid == 0)
        mu = x(N - 1);

    MatrixXd y;
    y = extractVectors(x);
    (*dsi_).phaseShift(y);
    x = toVector(y);

    if (isAC && taskid == 0) {
        x.conservativeResize(N);
        x(N - 1) = mu;
    }
}

Real MultishootingDSI::fixtphase(const VectorXd& x) {
    if (!Tsearch_)
        throw runtime_error("Using tphasehack is relevant only for searching for orbits !!!");

    MatrixXd y;
    y = extractVectors(x);

    VectorXd yvec;
    yvec = y.col(0);

    const int taskid = mpirank();
    Real Tint = extractT(x);
    Tint *= 1e-5;
    if (taskid == 0) {
        yvec(NT_y_) = Tint;
        if (xrelative_)
            yvec(Nax_y_) = 0.0;
        if (zrelative_)
            yvec(Naz_y_) = 0.0;
    }

    VectorXd yvec_dt = (*dsi_).eval(yvec, VectorXd::Zero(Nytot_), false);
    Real obsdiff = ((*dsi_).tph_observable(yvec_dt) - (*dsi_).tph_observable(yvec)) / Tint;

    return obsdiff;
}

bool MultishootingDSI::tph() { return (Tsearch_ && fixtphase_); }

Real MultishootingDSI::DSIL2Norm(const VectorXd& x) {
    MatrixXd y;
    y = extractVectors(x);

    Real norm;
    norm = (*dsi_).DSIL2Norm(y.col(0));

    return norm;
}

void MultishootingDSI::save(const VectorXd& x, const string filebase, const string outdir) {
    MatrixXd y;
    y = extractVectors(x);

    VectorXd yvec = y.col(0);

    const int taskid = mpirank();
    if (taskid == 0) {
        yvec(NT_y_) = y(NT_y_, nShot_ - 1) * nShot_;
        if (xrelative_)
            yvec(Nax_y_) = y(Nax_y_, nShot_ - 1);
        if (zrelative_)
            yvec(Naz_y_) = y(Naz_y_, nShot_ - 1);
    }

    (*dsi_).save(yvec, filebase, outdir);

    if (nShot_ > 1) {
        string shotdir = outdir + "Multishooting/";
        mkdir(shotdir);
        for (int i = 1; i < nShot_; i++) {
            string filebase_ms = filebase + i2s(i);
            (*dsi_).save(y.col(i), filebase_ms, shotdir, true);
        }
    }
}

string MultishootingDSI::stats(const VectorXd& x) {
    MatrixXd y;
    y = extractVectors(x);

    return (*dsi_).stats(y.col(0));
}

pair<string, string> MultishootingDSI::stats_minmax(const VectorXd& x) {
    const int taskid = mpirank();
    MatrixXd y;
    y = extractVectors(x);

    if (taskid == 0 && Tsearch_)
        y(NT_y_, 0) *= nShot_;  // give full time period to stats_minmax via first shot

    return (*dsi_).stats_minmax(y.col(0));
}

int MultishootingDSI::nShot() { return nShot_; }

}  // namespace chflow
