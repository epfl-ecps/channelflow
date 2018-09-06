/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */
#include "gmres.h"

using namespace std;

using namespace Eigen;

namespace chflow {

/*==================================================================================*/
/*            Class GMRES                                                           */
/*==================================================================================*/

GMRES::GMRES() : M_(0), Niter_(0), n_(0), condition_(0) {}

GMRES::GMRES(const VectorXd& b, int Niterations, Real minCondition)
    : M_(b.size()),
      Niter_(Niterations),
      n_(0),
      condition_(minCondition),
      H_(Niter_ + 1, Niter_),
      Q_(M_, 1),
      qn_(b),
      xn_(M_),
      bnorm_(L2Norm(b)),
      residual_(bnorm_) {
    setToZero(H_);
    setToZero(Q_);
    qn_ = qn_ * 1.0 / L2Norm(qn_);
    Q_.col(0) = qn_;
}

// Refer to Trefethen & Bau chapter 35 for algorithm and notation
void GMRES::iterate(const VectorXd& Aq) {
    if (n_ == Niter_) {
        cerr << "warning : GMRES::iterate(Ab) : \n"
             << "reached maximum number of iterations. doing nothing\n";
        return;
    }
    v_ = Aq;
    // Orthogonalize v and insert it in Q matrix
    // cout << "calculating arnoldi iterate..." << endl;
    for (int j = 0; j <= n_; ++j) {
        VectorXd Qj = Q_.col(j);
        H_(j, n_) = L2IP(Qj.transpose(), v_);
        v_ -= H_(j, n_) * Qj;
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
            VectorXd Qj = Q_.col(j);
            H_(j, n_) = Qj.transpose() * v_;
            v_ -= H_(j, n_) * Qj;
        }
        vnorm = L2Norm(v_);
    }

    H_(n_ + 1, n_) = vnorm;
    v_ = v_ * 1.0 / vnorm;
    // TobiasHack: keep a reasonable equilibrium between allocating all storage at once
    // and requiring new memory at every iteration
    if (Q_.cols() <= n_ + 2) {
        int newsize = std::min((int)(Q_.cols() + 100), Niter_ + 2);
        Q_.conservativeResize(NoChange, newsize);
    }
    Q_.col(n_ + 1) = v_;
    qn_ = Q_.col(n_ + 1);

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
            sum += Q_(i, j) * y(j);
        xn_(i) = sum;
    }

    ++n_;  // increment
}

// GMRES provides a partial Hessenberg factorization Qn* A Qn = Hn.
// So Ax=b projected into nth Krylov subspace is Qn* (Ax-b) = 0.
// Restrict x to nth Krylov subspace x = Qn y, giving
// Qn* A Qn y = Qn* b
// Hn y = Qn* b
// Solve that last one w least squares. Assuming A is full rank, the residual
// of Ax-b will be given by the magnitiude of the part of b that sticks out
// of the nth Krylov subspace: r = |b - Q (Q*b)|/|b|
VectorXd GMRES::solve(const VectorXd& bprime, Real& resid) {
    assert(bprime.size() == M_);

    MatrixXd Hn = H_.block(0, 0, n_, n_);  // n x n principal submatrix
    // Matrix Qn = Q_.extract(0,0,M_-1,n_-1); // M x n principal submatrix

    // unroll
    // VectorXd Qnb = (bprime.transpose()*Qn).transpose(); // avoid transposing Qn
    //
    // Qntb = (bprime' Qn)' = Qn' b has shape n x M times M x 1 == n x 1

    VectorXd Qntb(n_);
    for (int i = 0; i < n_; ++i) {
        Real sum = 0.0;
        for (int j = 0; j < M_; ++j)
            sum += Q_(j, i) * bprime(j);
        Qntb(i) = sum;
    }

    VectorXd y = Hn.householderQr().solve(Qntb);
    // leastsquares(Hn, Qntb, y, resid);

    // unroll
    // VectorXd error = Qn * Qntb; // M x n times n x 1 == M x 1
    VectorXd error(M_);
    for (int i = 0; i < M_; ++i) {
        Real sum = 0.0;
        for (int j = 0; j < n_; ++j)
            sum += Q_(i, j) * Qntb(j);
        error(i) = sum;
    }
    error -= bprime;
    resid = L2Norm(error) / L2Norm(bprime);

    // unroll
    // VectorXd xprime = Qn*y; // M x n times n x 1 == M x 1
    VectorXd xprime(M_);
    for (int i = 0; i < M_; ++i) {
        Real sum = 0.0;
        for (int j = 0; j < n_; ++j)
            sum += Q_(i, j) * y(j);
        xprime(i) = sum;
    }
    return xprime;
}

int GMRES::n() const { return n_; }
int GMRES::Niter() const { return Niter_; }

const VectorXd& GMRES::testVector() const { return qn_; }
const VectorXd& GMRES::solution() const { return xn_; }
Real GMRES::residual() const { return residual_; }
MatrixXd GMRES::Hn() const { return H_.block(0, 0, n_ + 1, n_); }
MatrixXd GMRES::Qn() const { return Q_.block(0, 0, M_, n_); }
MatrixXd GMRES::Qn1() const { return Q_.block(0, 0, M_, n_ + 1); }
const MatrixXd& GMRES::Q() const { return Q_; }
void GMRES::resetQ() { Q_.conservativeResize(NoChange, 1); }
}  // namespace chflow
