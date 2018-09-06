/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */

#ifndef NSOLVER_BICGSTABL_H
#define NSOLVER_BICGSTABL_H

#include "cfbasics/cfbasics.h"

// typedef VectorXd (*Rn2Rnfunc)(const VectorXd& x);

namespace chflow {

template <class vec>
class BiCGStabL {
   public:
    BiCGStabL(std::function<vec(const vec&)> A, const vec& b, int l, int maxIter);
    void iterate();
    const vec& solve(Real tol = 1e-8);
    const vec& solution() { return xsoln; }
    Real residual() { return residual_; }
    int l() { return L; }

   private:
    vec rtilde;
    Real rho, alpha, omega, beta;
    int L, l_, Lmax = 6;
    int N;
    std::vector<vec> u, r;
    vec x, xsoln;
    Eigen::MatrixXd tau;
    Eigen::VectorXd gamma, gammap, gammapp, sigma;
    std::function<vec(const vec&)> A_;
    Real residual_;
    int nIter_, maxIter_;
    Real rhsnorm;
    int nNoDecrease = 0;
};

template <class vec>
inline BiCGStabL<vec>::BiCGStabL(std::function<vec(const vec&)> A, const vec& b, int l, int maxIter)
    : L(l), l_(l), A_(A), nIter_(0), maxIter_(maxIter) {
    vec nullvec = b;
    nullvec *= 0;

    if (l_ == 0)
        L = Lmax;

    // Memory allocation
    x = nullvec;
    u.resize(L + 1);
    r.resize(L + 1);
    for (int j = 0; j <= L; ++j) {
        u[j] = nullvec;
        r[j] = nullvec;
    }

    gamma = gammap = gammapp = sigma = Eigen::VectorXd::Zero(L + 1);
    tau = Eigen::MatrixXd(L + 1, L + 1);

    // Initialization

    r[0] = b;  // - A_(x0);
    rtilde = r[0];

    rho = 1;
    alpha = 0;
    beta = 0;
    omega = 1;

    // u[0] = 0

    rhsnorm = L2Norm(b);
    residual_ = L2Norm(r[0]) / rhsnorm;
}

template <class vec>
inline void BiCGStabL<vec>::iterate() {
    if (l_ == 0) {
        if (omega < 1e-6 && L < Lmax)
            L++;
    }

    rho = -omega * rho;

    // Bi-CG part
    for (int j = 0; j < L; ++j) {
        Real rho1 = L2IP(r[j], rtilde);
        beta = alpha * rho1 / rho;
        rho = rho1;
        for (int i = 0; i <= j; ++i) {
            u[i] = r[i] - beta * u[i];
        }

        u[j + 1] = A_(u[j]);

        //     gamma =
        alpha = rho / L2IP(u[j + 1], rtilde);

        for (int i = 0; i <= j; ++i) {
            r[i] -= alpha * u[i + 1];
        }
        r[j + 1] = A_(r[j]);
        x += alpha * u[0];
    }

    // (mod. GS) MR part
    for (int j = 1; j <= L; ++j) {
        for (int i = 1; i < j; ++i) {
            tau(i, j) = L2IP(r[j], r[i]) / sigma(i);
            r[j] -= tau(i, j) * r[i];
        }
        sigma(j) = L2IP(r[j], r[j]);
        gammap(j) = L2IP(r[0], r[j]) / sigma(j);
    }

    omega = gamma(L) = gammap(L);

    // gamma = T^{-1} gamma'
    for (int j = L - 1; j >= 1; --j) {
        gamma(j) = gammap(j);
        for (int i = j + 1; i <= L; ++i) {
            gamma(j) -= tau(j, i) * gamma(i);
        }
    }
    // gamma'' = T S gamma
    for (int j = 1; j < L; ++j) {
        gammapp(j) = gamma(j + 1);
        for (int i = j + 1; i < L; ++i) {
            gammapp(j) += tau(j, i) * gamma(i + 1);
        }
    }

    // update
    x += gamma(1) * r[0];
    r[0] -= gammap(L) * r[L];

    u[0] -= gamma(L) * u[L];

    for (int j = 1; j < L; ++j) {
        u[0] -= gamma(j) * u[j];
        x += gammapp(j) * r[j];
        r[0] -= gammap(j) * r[j];
    }

    Real rtmp = L2Norm(r[0]) / rhsnorm;
    if (rtmp < residual_) {
        residual_ = rtmp;
        xsoln = x;
        nNoDecrease = 0;
        if (l_ == 0)
            L = 1;
    }

    nIter_++;
}

template <class vec>
inline const vec& BiCGStabL<vec>::solve(Real tol) {
    while (nIter_ < maxIter_ && residual_ > tol)
        iterate();
    return x;
}

}  // namespace chflow

#endif
