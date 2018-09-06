/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */
#include "arnoldi.h"

#include "cfbasics/cfbasics.h"
#include "nsolver/config.h"  // for interfacing from other programs

#ifdef HAVE_MPI
#include "mpi.h"
#endif

using namespace std;

using namespace Eigen;

namespace chflow {

/*==================================================================================*/
/*            Class Arnoldi                                                         */
/*==================================================================================*/
Arnoldi::Arnoldi() : M_(0), Niter_(0), n_(0), condition_(0) {}

Arnoldi::Arnoldi(const VectorXd& b, int Niterations, Real minCondition)
    : M_(b.size()),
      Niter_(Niterations),
      n_(0),
      condition_(minCondition),
      H_(Niter_ + 1, Niter_),
      Q_(M_, Niter_ + 1),
      qn_(b),
      v_(M_) {
    setToZero(H_);
    setToZero(Q_);
    qn_ = qn_ * 1.0 / L2Norm(qn_);
    qn_ = qn_ * 1.0 / L2Norm(qn_);
    Q_.col(0) = qn_;
}

void Arnoldi::orthocheck() {
    ofstream os("QtQ.asc");

    for (int i = 0; i < n_; ++i) {
        RowVectorXd Qit = Q_.col(i).transpose();
        for (int j = 0; j < n_; ++j)
            os << L2IP(Qit, Q_.col(j)) << ' ';
        os << endl;
    }
}

void Arnoldi::iterate(const VectorXd& Aq) {
    // #ifdef HAVE_MPI
    //   if (usempi()) MPI_Comm_rank ( MPI_COMM_WORLD, &taskid );
    // #endif
    if (n_ == Niter_) {
        cerr << "warning : Arnoldi::iterate(Aq) : \n"
             << "reached maximum number of iterations. doing nothing.\n";
    }
    v_ = Aq;

    // Orthogonalize v and insert it in Q matrix
    // cout << "calculating arnoldi iterate..." << endl;

    for (int j = 0; j <= n_; ++j) {
        // cout << j << ' ' << flush;
        VectorXd Qj = Q_.col(j);
        H_(j, n_) = L2IP(Qj.transpose(), v_);
        v_ -= H_(j, n_) * Qj;
    }
    // cout << endl;
    Real vnorm = L2Norm(v_);

    if (abs(vnorm) < condition_) {
        H_(n_ + 1, n_) = 0.0;
        v_.setZero();
        cerr << "Arnoldi breakdown. Exiting\n";
        exit(1);
    } else {
        H_(n_ + 1, n_) = vnorm;
        v_ = v_ * 1.0 / vnorm;
    }
    Q_.col(n_ + 1) = v_;
    qn_ = Q_.col(n_ + 1);
    ++n_;
}

const VectorXd& Arnoldi::testVector() const { return qn_; }

int Arnoldi::n() const { return n_; }
int Arnoldi::Niter() const { return Niter_; }

const VectorXcd& Arnoldi::ew() {
    if (ew_.rows() != n_)
        eigencalc();
    return ew_;
}

const VectorXd& Arnoldi::rd() { return rd_; }

void Arnoldi::eigencalc() {
    // do eigenvalue calculation inside a scope to kill off
    MatrixXd Hn = H_.block(0, 0, n_, n_);
    EigenSolver<MatrixXd> eignHn(Hn, /*compute_eigenvectors=*/true);
    ew_ = eignHn.eigenvalues();
    MatrixXcd W = eignHn.eigenvectors();
    sort_by_abs(ew_, W);

    // MatrixXd Qn  = Q_.block ( 0,0,M_,n_ );
    // ev_ = Qn*W;
    // unroll this mult to avoid taking submatrix
    // ev(i,j) = Qn(i,k) * W(k,j);
    ev_ = MatrixXcd(M_, n_);
    rd_ = VectorXd(n_);
    Real sum1 = 0.0;
    sum1 = Q_.col(n_).norm();
    for (int i = 0; i < M_; ++i)
        for (int j = 0; j < n_; ++j) {
            Complex sum = 0.0;
            for (int k = 0; k < n_; ++k)
                sum += Q_(i, k) * W(k, j);
            ev_(i, j) = sum;
            rd_(j) = sum1 * abs(W(n_ - 1, j));  // Residual calculation from Athanasios C. Antoulas
                                                // Approximation of Large-Scale Dynamical System
        }
}

const MatrixXcd& Arnoldi::ev() {
    if (ev_.rows() != n_)
        eigencalc();
    return ev_;
}
}  // namespace chflow
