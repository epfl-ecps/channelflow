/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */

#ifndef NSOLVER_FGMRES_H
#define NSOLVER_FGMRES_H

#include "cfbasics/cfbasics.h"

namespace chflow {

/*==================================================================================*/
/*            Class FGMRES                                                           */
/*==================================================================================*/

class FGMRES {
    // Flexible GMRES

   public:
    FGMRES();
    FGMRES(const Eigen::VectorXd& b, int Niterations, Real minCondition = 1e-13);

    const Eigen::VectorXd& testVector() const;
    void iterate(const Eigen::VectorXd& testvec, const Eigen::VectorXd& A_testvec);

    const Eigen::VectorXd& solution() const;  // current best approx to soln x of Ax=b
    const Eigen::VectorXd& guess() const;
    Real residual() const;  // |Ax-b|/|b|

    int n() const;
    int Niter() const;

    Eigen::MatrixXd Hn() const;  // Hn  = (n+1) x n submatrix of H
    Eigen::MatrixXd Zn() const;  // Qn  = M x n     submatrix of Q
    Eigen::MatrixXd AZn() const;
    Eigen::MatrixXd Vn() const;        // Qn1 = M x (n+1) submatrix of Q
    void resetV();                     // reset Qn to size(b)
    const Eigen::MatrixXd& V() const;  // Q is large, so avoid copying it

    Eigen::VectorXd solve(const Eigen::VectorXd& bprime, Real& residual);
    Eigen::VectorXd b();

   private:
    int M_;      // dimension of vector space
    int Niter_;  // max number of iterations
    int n_;      // current iteration number
    Real condition_;

    Eigen::MatrixXd H_;   // Hessenberg Matrix
    Eigen::MatrixXd V_;   // Orthogonalaized basis
    Eigen::MatrixXd Z_;   // Prefined vectors
    Eigen::MatrixXd AZ_;  // AZ_(:,j) = A*Z_(:,j)
    Eigen::VectorXd x0_;  // guess;
    Eigen::VectorXd v_;
    Eigen::VectorXd qn_;
    Eigen::VectorXd xn_;
    Real bnorm_;
    Real residual_;
};
}  // namespace chflow
#endif
