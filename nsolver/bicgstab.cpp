/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */
#include "nsolver/bicgstab.h"
#include <iostream>
using namespace std;
using namespace Eigen;

namespace chflow {

// BiCGStab::BiCGStab () {};

BiCGStab::BiCGStab(VectorXd rhs) {
    r = rhs;
    r0 = r;

    r0_sqnorm = r0.squaredNorm();
    rhs_sqnorm = rhs.squaredNorm();
    residual_ = (r.squaredNorm() / rhs_sqnorm);

    rho = 1;
    alpha = 1;
    w = 1;

    rho_old = 0;
    beta = 0;
    int n = rhs.size();
    v = VectorXd::Zero(n);
    p = VectorXd::Zero(n);
    //   y = VectorXd::Zero(n);
    //   z = VectorXd::Zero(n);
    s = VectorXd::Zero(n);
    t = VectorXd::Zero(n);
    x = VectorXd::Zero(n);
    solution_ = VectorXd::Zero(n);
}

VectorXd BiCGStab::step1() {
    rho_old = rho;
    rho = r0.dot(r);
    if (abs(rho) < 1e-16 * r0_sqnorm) {
        // The new residual vector became too orthogonal to the arbitrarily choosen direction r0
        // Let's restart with a new r0:
        r0 = r;
        rho = r0_sqnorm = r.squaredNorm();
    }
    beta = (rho / rho_old) * (alpha / w);
    p = r + beta * (p - w * v);
    // Here, we need A*p, so we return p and ask the user to supply us with A*p in step2
    return p;
}

VectorXd BiCGStab::step2(VectorXd& Ap) {
    v = Ap;
    alpha = rho / r0.dot(v);
    s = r - alpha * v;
    // Here, we need A*s, so we return s and ask the user to supply us with A*s in step3
    return s;
}

VectorXd BiCGStab::step3(VectorXd& As) {
    t = As;
    Real tmp = t.squaredNorm();
    if (tmp > Real(0))
        w = t.dot(s) / tmp;
    else
        w = Real(0);
    r = s - w * t;

    x += alpha * p + w * s;

    Real currentResidual = sqrt(r.squaredNorm() / rhs_sqnorm);
    if (currentResidual < residual_) {
        solution_ = x;
        residual_ = currentResidual;
    }
    return x;
}

Real BiCGStab::residual() { return residual_; }

VectorXd BiCGStab::solution() { return solution_; }

}  // namespace chflow
