/**
 * Solution of Helmholtz eqn in Chebyshev expansions
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef CHANNELFLOW_HELMHOLTZ_H
#define CHANNELFLOW_HELMHOLTZ_H

#include "cfbasics/mathdefs.h"
#include "channelflow/bandedtridiag.h"
#include "channelflow/chebyshev.h"

namespace chflow {

// HelmholtzSolver solves the 1d Helmholtz eqn:  u'' - lambda u = f on [a,b]
// with Dirichlet or Neuman boundary conditions at a and b
// with spectral chebyshev "tau" algorithm. u is unknown, f and lambda
// are known.  The linear algebra problem looks like A u_hat = B f_hat,
// where u_hat = chebyfft(u). Algorithm is from Canuto and Hussaini
// "Spectral Methods in Fluid Dynamics" section 5.1.2.
// For convenience I added a nu argument that defaults to 1.0
// Change it to solve nu u'' - lambda u = f

class HelmholtzSolver {
   public:
    HelmholtzSolver();
    HelmholtzSolver(int numberModes, Real a, Real b, Real lambda, Real nu = 1.0);

    // a and b are dirichlet BCs for u: ua = u(a), ub = u(b)
    void solve(ChebyCoeff& u, const ChebyCoeff& f, Real ua, Real ub) const;
    void verify(const ChebyCoeff& u, const ChebyCoeff& f, Real ua, Real ub, bool verbose = false) const;
    Real residual(const ChebyCoeff& u, const ChebyCoeff& f, Real ua, Real ub) const;

    // Solve nu u''(y) - lambda u(y) - mu = f(y), mean(u) = umean, u(-+1)=a,b
    // for u and mu, given nu, lambda, f, ua, ub, and umean.
    void solve(ChebyCoeff& u, Real& mu, const ChebyCoeff& f, Real umean, Real ua, Real ub) const;
    void verify(ChebyCoeff& u, Real& mu, const ChebyCoeff& f, Real umean, Real ua, Real ub) const;
    Real residual(const ChebyCoeff& u, Real mu, const ChebyCoeff& f, Real umean, Real ua, Real ub) const;

    Real lambda() const;

   private:
    int N_;           // Canuto & Hussain's N, section 5.1.2
    int nModes_;      // N_+1
    int nEvenModes_;  // number of even modes
    int nOddModes_;   // number of odd modes
    Real a_;
    Real b_;
    Real lambda_;
    Real nu_;
    inline int beta(int n) const;
    inline int c(int n) const;

    // Matrix form of Helmholtz eqn is A uhat = B fhat, even/odd modes decouple
    BandedTridiag Ae_;  // LHS Tridiag for even modes: Ae uhat_even = Be fhat_even
    BandedTridiag Ao_;  // LHS Tridiag for odd  modes: Ao uhat_odd  = Bo fhat_odd
    BandedTridiag Be_;  // RHS Tridiag for even modes
    BandedTridiag Bo_;  // RHS Tridiag for odd modes
};

// Canuto & Hussaini pg 67 and 130
// inline int HelmholtzSolver::beta(int n) const {return (n > N_-2) ? 0 : 1;}
inline int HelmholtzSolver::beta(int n) const { return (n > N_ - 2) ? 0 : 1; }
inline int HelmholtzSolver::c(int n) const { return (n == 0 || n == N_) ? 2 : 1; }

}  // namespace chflow
#endif
