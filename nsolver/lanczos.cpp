/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */
#include "lanczos.h"
#include "cfbasics/cfbasics.h"
#include "nsolver/config.h"  // for interfacing from other programs

#ifdef HAVE_MPI
#include "mpi.h"
#endif

using namespace std;
using namespace Eigen;

namespace chflow {

/*==================================================================================*/
/*            Class Lanczos                                                         */
/*==================================================================================*/
Lanczos::Lanczos() : Arnoldi() {}

Lanczos::Lanczos(const VectorXd& b, int Niterations, Real minCondition)
    : Arnoldi(b, Niterations, minCondition), T_(Niter_ + 1, Niter_ + 1) {
    setToZero(T_);
}

void Lanczos::iterate(const VectorXd& Aq) {
    // #ifdef HAVE_MPI
    //   if (usempi()) MPI_Comm_rank ( MPI_COMM_WORLD, &taskid );
    // #endif
    if (n_ == Niter_) {
        cerr << "warning : Lanczos::iterate(Aq) : \n"
             << "reached maximum number of iterations. doing nothing.\n";
    }
    v_ = Aq;

    // Lanczos algorithm
    if (n_ != 0) {
        VectorXd Qj_1 = Q_.col(n_ - 1);
        v_ -= T_(n_ - 1, n_) * Qj_1;
    }
    VectorXd Qj = Q_.col(n_);
    T_(n_, n_) = L2IP(Qj.transpose(), v_);
    // Orthogonalize v for numerical stability
    for (int j = 0; j <= n_; ++j) {
        Qj = Q_.col(j);
        v_ -= L2IP(Qj.transpose(), v_) * Qj;
    }

    Real vnorm = L2Norm(v_);

    if (abs(vnorm) < condition_) {
        T_(n_ + 1, n_) = 0.0;
        v_.setZero();
        cerr << "Lanczos breakdown. Exiting\n";
        exit(1);
    } else {
        T_(n_ + 1, n_) = vnorm;
        T_(n_, n_ + 1) = vnorm;
        v_ = v_ * 1.0 / vnorm;
    }

    Q_.col(n_ + 1) = v_;
    qn_ = Q_.col(n_ + 1);
    ++n_;
}

void Lanczos::eigencalc() {
    // do eigenvalue calculation inside a scope to kill off
    MatrixXd Tn = T_.block(0, 0, n_, n_);
    SelfAdjointEigenSolver<MatrixXd> eignTn;
    eignTn.compute(Tn);
    ew_ = eignTn.eigenvalues().cast<std::complex<double>>();
    MatrixXd W = eignTn.eigenvectors();
    sort_by_abs(ew_, W);

    // MatrixXd Qn  = Q_.block ( 0,0,M_,n_ );
    // ev_ = Qn*W;
    // unroll this mult to avoid taking submatrix
    // ev(i,j) = Qn(i,k) * W(k,j);
    ev_ = MatrixXd(M_, n_).cast<std::complex<double>>();
    rd_ = VectorXd(n_);
    Real sum1 = 0.0;
    sum1 = Q_.col(n_).norm();
    for (int i = 0; i < M_; ++i)
        for (int j = 0; j < n_; ++j) {
            Real sum = 0.0;
            for (int k = 0; k < n_; ++k)
                sum += Q_(i, k) * W(k, j);
            ev_(i, j) = sum;
            rd_(j) = sum1 * abs(W(n_ - 1, j));
        }
}

}  // namespace chflow
