/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */

#ifndef NSOLVER_GMRES_H
#define NSOLVER_GMRES_H

#include "cfbasics/cfbasics.h"

namespace chflow {

/*==================================================================================*/
/*            Class GMRES                                                           */
/*==================================================================================*/

// Iterative GMRES solution to Ax=b.
// Usage for classic iterative GMRES solution of Ax=b:
// GMRES gmres(b, N);
// for (int n=0; n<N; ++n) {
//   VectorXd q  = gmres.testVector();
//   VectorXd Aq = A*q; // or however else you calculate A*q
//   gmres.iterate(Aq);
//   VectorXd x = gmres.solution(); // current estimate of soln
//   cout << "krylov residual == " << gmres.residual() << endl;
// }
//
// Additional functionality:
// Find approx solution x' of Ax'=b' projected into current Krylov subspace.
// Real residual;
// Vector xprime = grmes.solve(bprime, residual);

class GMRES {
   public:
    GMRES();
    GMRES(const Eigen::VectorXd& b, int Niterations, Real minCondition = 1e-13);

    const Eigen::VectorXd& testVector() const;
    void iterate(const Eigen::VectorXd& A_testvec);

    const Eigen::VectorXd& solution() const;  // current best approx to soln x of Ax=b
    const Eigen::VectorXd& guess() const;
    Real residual() const;  // |Ax-b|/|b|

    int n() const;
    int Niter() const;

    Eigen::MatrixXd Hn() const;        // Hn  = (n+1) x n submatrix of H
    Eigen::MatrixXd Qn() const;        // Qn  = M x n     submatrix of Q
    Eigen::MatrixXd Qn1() const;       // Qn1 = M x (n+1) submatrix of Q
    void resetQ();                     // reset Qn to size(b)
    const Eigen::MatrixXd& Q() const;  // Q is large, so avoid copying it

    Eigen::VectorXd solve(const Eigen::VectorXd& bprime, Real& residual);

   private:
    int M_;      // dimension of vector space
    int Niter_;  // max number of iterations
    int n_;      // current iteration number
    Real condition_;

    Eigen::MatrixXd H_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd Vn_;
    Eigen::VectorXd x0_;  // guess;
    Eigen::VectorXd v_;
    Eigen::VectorXd qn_;
    Eigen::VectorXd xn_;
    Real bnorm_;
    Real residual_;
};
}  // namespace chflow
#endif
