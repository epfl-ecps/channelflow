/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */
#include "fgmres.h"

using namespace std;

using namespace Eigen;

namespace chflow {

/*==================================================================================*/
/*            Class FGMRES                                                           */
/*==================================================================================*/

FGMRES::FGMRES() : M_(0), Niter_(0), n_(0), condition_(0) {}

FGMRES::FGMRES(const VectorXd& b, int Niterations, Real minCondition)
    : M_(b.size()),
      Niter_(Niterations),
      n_(0),
      condition_(minCondition),
      H_(Niter_ + 1, Niter_),
      V_(M_, 1),
      Z_(M_, 1),
      AZ_(M_, 1),
      qn_(b),
      xn_(M_),
      bnorm_(L2Norm(b)),
      residual_(bnorm_) {
    setToZero(H_);
    setToZero(V_);
    setToZero(Z_);
    setToZero(AZ_);
    qn_ = qn_ * 1.0 / L2Norm(qn_);
    V_.col(0) = qn_;
}

// Refer to Trefethen & Bau chapter 35 for algorithm and notation
void FGMRES::iterate(const VectorXd& q, const VectorXd& Aq) {
    if (n_ == Niter_) {
        cerr << "warning : GMRES::iterate(Ab) : \n"
             << "reached maximum number of iterations. doing nothing\n";
        return;
    }

    Z_.col(n_) = q;
    AZ_.col(n_) = Aq;

    v_ = Aq;
    // Orthogonalize v and insert it in Q matrix
    // cout << "calculating arnoldi iterate..." << endl;
    for (int j = 0; j <= n_; ++j) {
        VectorXd Vj = V_.col(j);
        H_(j, n_) = L2IP(Vj.transpose(), v_);
        v_ -= H_(j, n_) * Vj;
    }
    Real vnorm = L2Norm(v_);

    // If v is within subspace spanned by Q, replace it with a random vector.
    // There's a measure 0 chance that any particular random vector will also
    // spanned by Q, so allow for a limited number of retries.
    int maxretries = 10;
    int retries = 0;
    while (abs(vnorm) < condition_ && retries++ < maxretries) {
        cerr << "GMRES benign breakdown: \n"
             << "  Using random vector instead of linearly dependent Krylov vector.\n";

        for (int i = 0; i < v_.size(); ++i)
            v_(i) = randomReal();

        // Gram-Schmidt orthogonalize v w.r.t Q
        for (int j = 0; j <= n_; ++j) {
            VectorXd Vj = V_.col(j);
            H_(j, n_) = Vj.transpose() * v_;
            v_ -= H_(j, n_) * Vj;
        }
        vnorm = L2Norm(v_);
    }

    H_(n_ + 1, n_) = vnorm;
    v_ = v_ * 1.0 / vnorm;
    // TobiasHack: keep a reasonable equilibrium between allocating all storage at once
    // and requiring new memory at every iteration
    if (V_.cols() <= n_ + 2) {
        int newsize = std::min((int)(V_.cols() + 100), Niter_ + 2);
        V_.conservativeResize(NoChange, newsize);
        Z_.conservativeResize(NoChange, newsize);
        AZ_.conservativeResize(NoChange, newsize);
    }
    V_.col(n_ + 1) = v_;
    qn_ = V_.col(n_ + 1);

    MatrixXd Hn = H_.block(0, 0, n_ + 2, n_ + 1);
    VectorXd bk(n_ + 2);
    setToZero(bk);
    bk(0) = bnorm_;

    VectorXd y = Hn.householderQr().solve(bk);
    residual_ = L2Norm((VectorXd)(Hn * y - bk)) / L2Norm(bk);
    // Unroll this matrix mult to eliminate need for submatrix extraction & copy
    // xn_ = Qn*y; // M x n+1 times n+1 x 1 == M x 1
    for (int i = 0; i < M_; ++i) {
        Real sum = 0.0;
        for (int j = 0; j < n_ + 1; ++j)
            sum += Z_(i, j) * y(j);
        xn_(i) = sum;
    }

    ++n_;  // increment
}

VectorXd FGMRES::b() { return V_.col(0); }

int FGMRES::n() const { return n_; }
int FGMRES::Niter() const { return Niter_; }

const VectorXd& FGMRES::testVector() const { return qn_; }
const VectorXd& FGMRES::solution() const { return xn_; }
Real FGMRES::residual() const { return residual_; }
MatrixXd FGMRES::Hn() const { return H_.block(0, 0, n_ + 1, n_); }
MatrixXd FGMRES::Zn() const { return Z_.block(0, 0, M_, n_); }
MatrixXd FGMRES::AZn() const { return AZ_.block(0, 0, M_, n_); }
MatrixXd FGMRES::Vn() const { return V_.block(0, 0, M_, n_ + 1); }
const MatrixXd& FGMRES::V() const { return V_; }
void FGMRES::resetV() { V_.conservativeResize(NoChange, 1); }
}  // namespace chflow
