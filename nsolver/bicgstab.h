/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */

#ifndef NSOLVER_BICGSTAB_H
#define NSOLVER_BICGSTAB_H

#include "cfbasics/cfbasics.h"

namespace chflow {

class BiCGStab {
    // usage to solve A*x = b
    //   BiCGStab bicgstab(b);
    //   for (int i=0; i<maxiter; i++) {
    //     VectorXd p = bicgstab.step1();
    //     VectorXd Ap = A*p;
    //     VectorXd s = bicstab.step2(Ap);
    //     VectorXd As = A*s;
    //     bicstab.step3(As);
    //     if (bicgstab.residual() < eps)
    //       break;
    //   }
    //   return bicgstab.solution();

    // The residual decreases monotonically with i since always the
    // best solution is returned by residual() and solution().
    // step3() returns the current progress of the iteration, which may
    // differ from solution() if the last step led to an increase in the
    // residual.

   public:
    BiCGStab(Eigen::VectorXd b);
    Eigen::VectorXd step1();
    Eigen::VectorXd step2(Eigen::VectorXd& Ap);
    Eigen::VectorXd step3(Eigen::VectorXd& As);

    Eigen::VectorXd solution();
    Real residual();

   private:
    Eigen::VectorXd r, r0;
    Real r0_sqnorm, rhs_sqnorm;
    Real rho, alpha, w, rho_old, beta;
    Eigen::VectorXd v, p, kt, ks, s, t, x, solution_;
    Real residual_;
};

}  // namespace chflow

#endif
