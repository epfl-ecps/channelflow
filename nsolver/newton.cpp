/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */
#include "nsolver/config.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <iostream>
#include <stdexcept>
#include "nsolver/bicgstab.h"
#include "nsolver/gmres.h"
#include "nsolver/newton.h"

using namespace std;

using namespace Eigen;

namespace chflow {

ArclengthConstraint::ArclengthConstraint() : yLast_(VectorXd(1)), muRef_(0), ds_(0) {}

ArclengthConstraint::ArclengthConstraint(const VectorXd& yLast, Real ds, Real muRef)
    : use_(true), yLast_(yLast), muRef_(muRef), ds_(ds) {}

Real ArclengthConstraint::ds() { return ds_; }

void ArclengthConstraint::setDs(Real newDs) { ds_ = newDs; }

void ArclengthConstraint::setYLast(const VectorXd& yLast) { yLast_ = yLast; }

VectorXd ArclengthConstraint::yLast() { return yLast_; }

bool ArclengthConstraint::use() { return use_; }

void ArclengthConstraint::notUse() { use_ = false; }

Real ArclengthConstraint::arclength2(const VectorXd& y) {
    int N = y.size();
    if (mpirank() == 0)
        N--;
    Real sum = 0;
    for (int i = 0; i < N; ++i)
        sum += pow(y(i) - yLast_(i), 2);
    if (mpirank() == 0)
        sum += pow((y(N) - yLast_(N)) / muRef_, 2);
#ifdef HAVE_MPI
    Real sum_local = sum;
    MPI_Allreduce(&sum_local, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    return sum;
}

Real ArclengthConstraint::arclength(const VectorXd& y) { return sqrt(arclength2(y)); }

Real ArclengthConstraint::arclengthDiff(const VectorXd& y) { return arclength(y) - ds_; }

Real ArclengthConstraint::muFromVector(const VectorXd& y) {
    Real mu = 0;
    if (mpirank() == 0)
        mu = y(y.size() - 1);
#ifdef HAVE_MPI
    MPI_Bcast(&mu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    return mu;
}

VectorXd ArclengthConstraint::makeVector(const VectorXd& x, Real mu) {
    VectorXd y(x);
    if (use_ && mpirank() == 0) {
        y.conservativeResize(y.size() + 1);
        y(y.size() - 1) = mu;
    }
    return y;
}

VectorXd ArclengthConstraint::extractVector(const VectorXd& y) {
    VectorXd x(y);
    if (use_ && mpirank() == 0) {
        x.conservativeResize(y.size() - 1);
    }
    return x;
}

Newton::Newton(ostream* logstream_, string outdir_, Real epsSearch)
    : logstream(logstream_),
      outdir(outdir_),
      epsSearch_(epsSearch),
      AC(new ArclengthConstraint()),
      msDSI_(new MultishootingDSI()) {}

VectorXd Newton::evalWithAC(const VectorXd& y, int& fcount) {
    ++fcount;
    if (AC->use()) {
        // Evaluate f(x) and store arclength difference in last component of f(y)
        Real mu = AC->muFromVector(y);
        msDSI_->updateMu(mu);
        //     cout << mpirank() << " mu = " << mu << endl;
        VectorXd x = AC->extractVector(y);
        VectorXd fy = msDSI_->eval(x);
        if (fy.size() != x.size())
            throw runtime_error(
                "NewtonAlgorithm::evalWithAC(): vector returned by dsi.eval(x) has not the same size as x: " +
                r2s(fy.size()) + " != " + r2s(y.size()));

        Real dsd = AC->arclengthDiff(y);

        fy.conservativeResize(y.size());

        if (mpirank() == 0)
            fy(fy.size() - 1) = 0;

        //     cout << "L2Norm(fy) = " << L2Norm(fy) << "\t" << flush;
        if (mpirank() == 0)
            fy(fy.size() - 1) = dsd;
        //     cout << "dsd = " << dsd << "\t" << "L2Norm(fy) = " << L2Norm(fy) << "\t" << endl;
        return fy;
    } else {
        return msDSI_->eval(y);
    }
}

VectorXd Newton::jacobianWithAC(const VectorXd& y, const VectorXd& dy, const VectorXd& Gy, const Real& epsDx,
                                bool centdiff, int& fcount) {
    if (AC->use()) {
        Real step_magn = L2Norm(dy);
        Real eps = (step_magn < epsDx) ? 1 : epsDx / step_magn;
        VectorXd DG_dy;

        if (centdiff) {
            Real eps2 = 0.5 * eps;
            VectorXd y_epsdy = y + eps2 * dy;
            VectorXd Gy_epsdyplus = evalWithAC(y_epsdy, fcount);
            y_epsdy = y - eps2 * dy;
            VectorXd Gy_epsdyminus = evalWithAC(y_epsdy, fcount);
            DG_dy = 1 / eps * (Gy_epsdyplus - Gy_epsdyminus);
        } else {
            VectorXd y_epsdy = y + eps * dy;
            VectorXd Gy_epsdy = evalWithAC(y_epsdy, fcount);
            DG_dy = 1 / eps * (Gy_epsdy - Gy);
        }

        return DG_dy;

    } else {
        return msDSI_->Jacobian(y, dy, Gy, epsDx, centdiff, fcount);
    }
}

void Newton::setArclengthConstraint(ArclengthConstraint* newAC) {
    delete AC;
    isACset = true;
    AC = newAC;
}

MultishootingDSI* Newton::getMultishootingDSI() { return msDSI_; }

}  // namespace chflow
