/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */
#ifndef NSOLVER_ARNOLDI_H
#define NSOLVER_ARNOLDI_H

#include "cfbasics/cfbasics.h"

namespace chflow {

/*==================================================================================*/
/*            Class Arnoldi                                                         */
/*==================================================================================*/

// Arnoldi iteration to estimate eigenvalues of matrix A. Usage:
// Arnoldi arnoldi(N, b);
// for (int n=0; n<N; ++n) {
//   VectorXd q  = arnoldi.testVector();
//   VectorXd Aq = A*q; // or however else you calculate A*q
//   arnoldi.iterate(Aq);
//   VectorXcd ew = arnoldi.ew(); // current estimate of eigenvalues
// }

class Arnoldi {
   public:
    Arnoldi();
    Arnoldi(const Eigen::VectorXd& b, int Niterations, Real minCondition = 1e-13);

    const Eigen::VectorXd& testVector() const;        // get test vector q
    virtual void iterate(const Eigen::VectorXd& Aq);  // tell Arnoldi the value of Aq

    void orthocheck();  // save Q' Q into file QtQ.asc, should be I.

    int n() const;      // current iteration number
    int Niter() const;  // total number iterations

    const Eigen::VectorXcd& ew();  // current estimate of eigenvals
    const Eigen::MatrixXcd& ev();  // current estimate of eigenvecs
    const Eigen::VectorXd& rd();   // current estimate of eigenvecs residual

    virtual ~Arnoldi(){};

   protected:
    int M_;      // dimension of linear operator (A is M x M)
    int Niter_;  // number of Arnoldi iterations
    int n_;      // current iteration number
    Real condition_;

    Eigen::MatrixXd H_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd Vn_;
    Eigen::VectorXd qn_;
    Eigen::VectorXd v_;

    Eigen::VectorXcd ew_;
    Eigen::MatrixXcd ev_;
    Eigen::VectorXd rd_;

    virtual void eigencalc();
};

}  // namespace chflow
#endif
