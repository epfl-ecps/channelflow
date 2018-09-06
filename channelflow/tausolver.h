/**
 * Solves vector "tau" eqns (Canuto & Hussaini eqn 7.3.18-20)
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef CHANNELFLOW_TAUSOLVER_H
#define CHANNELFLOW_TAUSOLVER_H

#include "cfbasics/cfvector.h"
#include "cfbasics/mathdefs.h"
#include "channelflow/chebyshev.h"
#include "channelflow/helmholtz.h"

namespace chflow {

// Class for solving 7.3.18-7.3.20 of Canuto & Hussaini
//      nu u''_jk(y) - lambda u_jk(y) - grad P_jk = -R_jk,
//                                       div u_jk = 0
//                                      u_jk(+-1) = 0
//
// where u_jk(y) is the vector-valued jkth xz-Fourier coeff of u(x,y,z)
//          P(y) is in R
// and the vector operators are interpreted accordingingly (diff in x
// equal multiplication by 2 pi k).
Real divcheck(std::string& label, int kx, int kz, Real Lx, Real Lz, const ComplexChebyCoeff& u,
              const ComplexChebyCoeff& v, const ComplexChebyCoeff& w, bool verbose = false);

class TauSolver {
   public:
    TauSolver();
    TauSolver(int kx, int kz, Real Lx, Real Lz, Real a, Real b, Real lambda, Real nu, int Ny,
              bool tauCorrection = true);
    // TauSolver(int kx, int kz, Real Lx, Real Lz, Real a, Real b, Real lambda,
    // Real nu, int nChebyModes, bool dx_on=true, bool dz_on=true,
    // bool tauCorrection=true);

    void solve(ComplexChebyCoeff& u, ComplexChebyCoeff& v, ComplexChebyCoeff& w, ComplexChebyCoeff& P,
               const ComplexChebyCoeff& Rx, const ComplexChebyCoeff& Ry, const ComplexChebyCoeff& Rz) const;

    Real verify(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v, const ComplexChebyCoeff& w,
                const ComplexChebyCoeff& P, const ComplexChebyCoeff& Rx, const ComplexChebyCoeff& Ry,
                const ComplexChebyCoeff& Rz, bool verbose = false) const;

    // Solve tau eqns with additional unknown, time-varying, -dPdx on LHS.
    // and additional constraint mean(u) = umean.
    void solve(ComplexChebyCoeff& u, ComplexChebyCoeff& v, ComplexChebyCoeff& w, ComplexChebyCoeff& P, Real& dPdx,
               Real& dPdz, const ComplexChebyCoeff& Rx, const ComplexChebyCoeff& Ry, const ComplexChebyCoeff& Rz,
               Real umean, Real wmean) const;

    Real verify(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v, const ComplexChebyCoeff& w,
                const ComplexChebyCoeff& P, Real dPdx, Real dPdz, const ComplexChebyCoeff& Rx,
                const ComplexChebyCoeff& Ry, const ComplexChebyCoeff& Rz, Real umean, Real wmean,
                bool verbose = false) const;

    // Enforce v'(+-1)==0 with influence matrix method, leaving v(+-1)==0 and RHS unchanged.
    void influenceCorrection(ChebyCoeff& P, ChebyCoeff& v) const;

    // These are helper functions for the above, not meant to be used alone,
    // but provided as public for testing purposes. r == div(R).
    void solve_P_and_v(ChebyCoeff& P, ChebyCoeff& v, const ChebyCoeff& r, const ChebyCoeff& Ry, Real& sigmaNb1,
                       Real& sigmaNb) const;

    Real verify_P_and_v(const ChebyCoeff& P, const ChebyCoeff& v, const ChebyCoeff& r, const ChebyCoeff& Ry,
                        Real sigmaNb1, Real sigmaNb, bool verbose = false) const;

    int kx() const { return kx_; }
    int kz() const { return kz_; }
    // Real kxLx() const{return kxLx_;}
    // Real kzLz() const{return kzLz_;}
    Real lambda() const { return lambda_; }
    Real nu() const { return nu_; }

   private:
    int N_;             // number of Chebyshev modes
    int Nb_;            // N-1. Convenience variable, corresponds to C&H's N.
    int kx_;            // x wave number
    int kz_;            // z wave number
    Real two_pi_kxLx_;  // 2 pi kx/Lx
    Real two_pi_kzLz_;  // 2 pi kz/Lz
    Real kappa2_;       // 4 pi^2 [(kx/Lx)^2 + (kz/Lz)^2]
    Real a_;
    Real b_;
    Real lambda_;
    Real nu_;             // viscosity
    bool tauCorrection_;  // Try to eliminate tau errors in (P,v) solutions

    HelmholtzSolver pressureHelmholtz_;
    HelmholtzSolver velocityHelmholtz_;

    // These quantities are constant in time in the tau algorithm.
    // So they're initialized in the constructor and kept fixed.
    ChebyCoeff P_0_;
    ChebyCoeff v_0_;
    ChebyCoeff P_plus_;
    ChebyCoeff v_plus_;
    ChebyCoeff P_minus_;
    ChebyCoeff v_minus_;

    // Element of the inverse of the influence matrix.
    Real i00_;
    Real i01_;
    Real i10_;
    Real i11_;

    Real sigma0_Nb1_;  // tau correction factor for mode Nb-1
    Real sigma0_Nb_;   // tau correction factor for mode Nb

    // L1 norm of first N-2 components of u.
    Real tauNorm(const ChebyCoeff& u) const;
    Real tauNorm(const ComplexChebyCoeff& u) const;
    Real tauDist(const ChebyCoeff& u, const ChebyCoeff& v) const;
    Real tauDist(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v) const;
};

}  // namespace chflow
#endif
