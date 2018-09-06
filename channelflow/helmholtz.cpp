/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "channelflow/helmholtz.h"
#include "channelflow/chebyshev.h"

using namespace std;

namespace chflow {

const Real EPSILON = 1.0;

HelmholtzSolver::HelmholtzSolver()
    : N_(0), nModes_(0), nEvenModes_(0), nOddModes_(0), a_(0), b_(0), lambda_(0), nu_(0), Ae_(), Ao_(), Be_(), Bo_() {}

HelmholtzSolver::HelmholtzSolver(int numberModes, Real a, Real b, Real lambda, Real nu)
    : N_(numberModes - 1),
      nModes_(numberModes),
      nEvenModes_(N_ / 2 + 1),
      nOddModes_(N_ / 2),
      a_(a),
      b_(b),
      lambda_(lambda),
      nu_(nu),
      Ae_(nEvenModes_),
      Ao_(nOddModes_),
      Be_(nEvenModes_),
      Bo_(nOddModes_) {
    assert(nModes_ % 2 == 1);  // Is this required or assumed in C&H?
    assert(nModes_ > 2);

    Real nuscaled = nu / square((b_ - a_) / 2);

    // Canuto & Hussaini eqn 5.1.24.
    // Even mode matrices:
    // Assign zeroth row of Ae_ and Be_
    Be_.diag(0) = 1.0;
    int i;
    for (i = 0; i < nEvenModes_; ++i)
        Ae_.band(i) = 1.0;
    // Assign first through rows of Ae_ and Be_
    for (i = 1; i < nEvenModes_; ++i) {
        int n = 2 * i;
        int nn = n * n;
        Ae_.lodiag(i) = -(c(n - 2) * lambda) / (4 * n * (n - 1));
        Ae_.diag(i) = (nuscaled + (beta(n) * lambda) / (2 * (nn - 1)));
        if (beta(n + 2))
            Ae_.updiag(i) = -lambda / (4 * n * (n + 1));

        Be_.lodiag(i) = ((Real)c(n - 2)) / (4 * n * (n - 1));
        Be_.diag(i) = -((Real)beta(n)) / (2 * (nn - 1));
        if (beta(n + 2))
            Be_.updiag(i) = 1.0 / (4 * n * (n + 1));
    }

    // Odd mode matrices
    Bo_.diag(0) = 1.0;
    for (i = 0; i < nOddModes_; ++i)
        Ao_.band(i) = 1.0;
    for (i = 1; i < nOddModes_; ++i) {
        int n = 2 * i + 1;
        int nn = n * n;
        Ao_.lodiag(i) = -(c(n - 2) * lambda) / (4 * n * (n - 1));
        Ao_.diag(i) = (nuscaled + (beta(n) * lambda) / (2 * (nn - 1)));
        if (beta(n + 2))
            Ao_.updiag(i) = -lambda / (4 * n * (n + 1));

        Bo_.lodiag(i) = ((Real)c(n - 2)) / (4 * n * (n - 1));
        Bo_.diag(i) = -((Real)beta(n)) / (2 * (nn - 1));
        if (beta(n + 2))
            Bo_.updiag(i) = 1.0 / (4 * n * (n + 1));
    }
    Ae_.ULdecomp();
    Ao_.ULdecomp();
}

void HelmholtzSolver::solve(ChebyCoeff& u, const ChebyCoeff& f, Real ua, Real ub) const {
    assert(f.state() == Spectral);
    Be_.multiplyStrided(f, u, 0, 2);
    Bo_.multiplyStrided(f, u, 1, 2);

    u[0] = (ub + ua) / 2;
    u[1] = (ub - ua) / 2;

    // Solve for A u = g
    Ae_.ULsolveStrided(u, 0, 2);
    Ao_.ULsolveStrided(u, 1, 2);

    //#ifdef DEBUG
    // verify(u, f, ua, ub);
    //#endif
    u.setState(Spectral);
}

Real HelmholtzSolver::residual(const ChebyCoeff& u, const ChebyCoeff& f, Real ua, Real ub) const {
    ChebyTransform trans(nModes_);
    assert(nModes_ == u.length());

    ChebyCoeff u_yy = diff2(u);

    Real errorTau = 0.0;
    for (int i = 0; i < nModes_ - 2; ++i)
        errorTau += fabs(nu_ * u_yy[i] - lambda_ * u[i] - f[i]);

    return errorTau;

    // calculate || A*u - B*f ||
    // g = B*f
    ChebyCoeff g(nModes_, a_, b_, Spectral);
    Be_.multiplyStrided(f, g, 0, 2);
    Bo_.multiplyStrided(f, g, 1, 2);
    g[0] = (ub + ua) / 2;
    g[1] = (ub - ua) / 2;

    // h = A*u
    ChebyCoeff h(nModes_, a_, b_, Spectral);
    Ae_.multiplyStrided(u, h, 0, 2);
    Ao_.multiplyStrided(u, h, 1, 2);

    return L1Dist(g, h);
}

void HelmholtzSolver::verify(const ChebyCoeff& u, const ChebyCoeff& f, Real ua, Real ub, bool verbose) const {
    ChebyTransform trans(nModes_);
    assert(nModes_ == u.length());

    ChebyCoeff u_yy = diff2(u);

    Real errorTau = 0.0;
    Real errorL1 = 0.0;
    int i;  // MSVC++ FOR-SCOPE BUG
    for (i = 0; i < nModes_ - 2; ++i)
        errorTau += fabs(nu_ * u_yy[i] - lambda_ * u[i] - f[i]);
    errorL1 = errorTau;
    for (i = nModes_ - 2; i < nModes_; ++i)
        errorL1 += fabs(nu_ * u_yy[i] - lambda_ * u[i] - f[i]);
    Real norm = Greater(L1Norm(u), L1Norm(f));
    norm = (norm > EPSILON) ? norm : 1.0;
    Real uas = u.eval_a();
    Real ubs = u.eval_b();

    if (verbose) {
        cerr << "Helmholtz::verify() { " << endl;
        cerr << "N nu lambda == " << N_ << ' ' << nu_ << ' ' << lambda_ << endl;
        cerr << "tauNorm(nu*uyy - lambda*u - f) == " << errorTau << endl;
        cerr << " L1Norm(nu*uyy - lambda*u - f) == " << errorL1 << endl;
        cerr << "fabs(uas - ua)      == " << fabs(uas - ua) << endl;
        cerr << "fabs(ubs - ub)      == " << fabs(ubs - ub) << endl;
        cerr << "} Helmholtz::verify()" << endl;
    }
    assert(fabs(errorTau / norm) < EPSILON);
    assert(fabs((uas - ua) / norm) < EPSILON);
    assert(fabs((ubs - ub) / norm) < EPSILON);
}

void HelmholtzSolver::solve(ChebyCoeff& u, Real& mu, const ChebyCoeff& f, Real umean, Real ua, Real ub) const {
    // Solve by breaking u into u = uf + um, where
    // nu uf'' - lambda uf = f  and uf(-+1) = ua,ub
    // nu um'' - lambda um = mu and um(-+1) = 0,0
    /**************
    ChebyCoeff ua(nModes_);
    solve(ua, f, a, b);
    real uamean = ua.mean();

    ChebyCoeff uc(nModes_);
    ChebyCoeff rhs(nModes_);
    rhs[0] = nu_;
    solve(uc, rhs, 0.0, 0.0);  // rhs == nu
    real ucmean = uc.mean();

    mu = nu_*(umean - uamean)/ucmean;
    rhs = f;
    rhs[0] += mu;
    solve(u, rhs, a, b);       // rhs == f + mu
    *******************/

    // Solve nu u'' - lambda u == f + mu, mean(u) == umean, u(+-1) == a,b.
    // for v and mu.

    // Solve by breaking u into u = ua + ub, where
    // eqn1. nu ua'' - lambda ua = f  and ua(-+1) = a,b. Solve
    // eqn2. nu ub'' - lambda ub = mu and ub(-+1) = 0,0  mu is O(1)

    // Rescale eqn2, via uc = mu/nu ub. Results in nu on RHS, nu is O(1/Re),
    // so eqn3 is better conditioned than eqn2.
    // eqn3. nu uc'' - lambda uc = nu and uc(-+1) = 0,0. Solve

    // Then mu = nu*(umean - mean(ua))/mean(uc).

    // Solve eqn1.
    int N = nModes_;
    ChebyCoeff utmp(N, a_, b_, Spectral);  // utmp serves as uf
    ChebyCoeff rtmp(f);
    solve(utmp, rtmp, ua, ub);
    Real uamean = utmp.mean();
    // cout << "eqn1 residual " << residual(utmp, rtmp, a,b) << endl;

    // Solve eqn3.
    rtmp.setToZero();  // rtmp serves as rhs of eqn3.
    rtmp[0] = nu_;
    solve(utmp, rtmp, 0.0, 0.0);  // utmp serves as uc
    // cout << "eqn3 residual " << residual(utmp,rtmp,0.0,0.0) << endl;
    Real ucmean = utmp.mean();

    // Solve eqn0 using known value of mu.
    mu = nu_ * (umean - uamean) / ucmean;
    rtmp = f;
    rtmp[0] += mu;  // rtmp serves as f + mu
    solve(u, rtmp, ua, ub);
    // cout << "eqn0 residual " << residual(u,rtmp,a,b) << endl;
}

void HelmholtzSolver::verify(ChebyCoeff& u, Real& mu, const ChebyCoeff& f, Real umean, Real ua, Real ub) const {
    cerr << "Helmholtz::verify(u,f,a,b,mu,umean) {" << endl;
    ChebyCoeff rhs(f);
    rhs[0] += mu;
    verify(u, f, ua, ub);
    cerr << "umean - mean(u) === " << umean - u.mean() << endl;
    cerr << "} Helmholtz::verify(u,f,ua,ub,mu,umean)" << endl;
}

Real HelmholtzSolver::residual(const ChebyCoeff& u, Real mu, const ChebyCoeff& f, Real umean, Real ua, Real ub) const {
    ChebyCoeff rhs(f);
    rhs[0] += mu;
    Real res = residual(u, rhs, ua, ub);
    res += abs(umean - u.mean());
    return res;
}

Real HelmholtzSolver::lambda() const { return lambda_; }

}  // namespace chflow
