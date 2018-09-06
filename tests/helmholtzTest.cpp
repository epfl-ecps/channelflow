/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <iostream>
#include "channelflow/helmholtz.h"

using namespace std;

using namespace chflow;

Real tauDist(const ChebyCoeff& u, const ChebyCoeff& v) {
    assert(u.length() == v.length());
    Real sum = 0.0;
    for (int i = 0; i < u.length() - 1; ++i)
        sum += abs(u[i] - v[i]);
    return sum;
}

ChebyCoeff interpolate(const ChebyCoeff& u, int N) {
    assert(u.state() == Spectral);
    ChebyCoeff uinterp(N, u.a(), u.b(), Physical);
    Vector x = chebypoints(N, u.a(), u.b());
    for (int i = 0; i < N; ++i)
        uinterp[i] = u.eval(x[i]);

    ChebyTransform t(N);
    uinterp.makeSpectral(t);
    return uinterp;
}

int main() {
    bool save = false;
    bool verbose = true;

    int N = 49;
    int nTests = 32;
    int failures = 0;

    const Real EPSILON = 1e-11;
    const Real magn = 1.0;
    const Real Lmax = 10;

    ChebyTransform trans(N);

    cerr << "helmholtzTest: " << flush;
    if (verbose) {
        cout << "\n====================================================" << endl;
        cout << "helmholtzTest\n\n";
    }

    Vector y = chebypoints(N, -1.0, 1.0);
    if (save)
        y.save("y");

    for (int test = 0; test < nTests; ++test) {
        Real nu = 1;
        Real a = randomReal(-Lmax / 2, 0);
        Real b = randomReal(0, Lmax / 2);
        Real lambda = 1;
        Real decay = 0.5;

        if (verbose) {
            cout << "Test #" << test << endl;
            cout << "lambda == " << lambda << endl;
            cout << "nu     == " << nu << endl;
            cout << "a,b    == " << a << ' ' << b << endl;
            cout << "spectral decay == " << decay << endl;
        }

        // Construct a "true" solution to nu v'' - lambda v = r, v(a)=va, v(b)=vb
        // using a higher-order expansion

        int N2 = 2 * (N - 1) + 1;
        ChebyCoeff vt(N2, a, b, Spectral);
        vt.randomize(magn, decay, Free, Free);
        Real va = vt.eval_a();
        Real vb = vt.eval_b();
        if (verbose)
            cout << "va vb == " << va << ' ' << vb << endl;

        ChebyCoeff nu_vtyy = diff2(vt);
        nu_vtyy *= nu;

        ChebyCoeff rt(vt);
        rt *= -lambda;
        rt += nu_vtyy;

        // Now interpolate the true v(y) onto a coarser expansion.
        ChebyCoeff v = interpolate(vt, N);
        ChebyCoeff r = interpolate(rt, N);

        // Use Helmholtz solver to solve above eqn for vs ("vsolve").
        HelmholtzSolver helmholtz(N, a, b, lambda, nu);

        ChebyCoeff vs(N, a, b, Spectral);
        helmholtz.solve(vs, r, va, vb);

        Real vas = vs.eval_a();
        Real vbs = vs.eval_b();

        if (save) {
            v.save("v");
            vs.save("vs");
        }

        Real errL1 = L1Dist(v, vs) / L1Norm(v);
        Real errTau = tauDist(v, vs) / L1Norm(v);
        Real aerr = fabs(va - vas);
        Real berr = fabs(vb - vbs);
        if (verbose)
            cout << "problem 1: nu v'' - lambda v = f, v(a),v(b) = va,vb" << endl;
        if (errL1 > EPSILON || aerr > EPSILON || berr > EPSILON) {
            cout << "ERROR : " << endl;
            ++failures;
        }

        if (verbose) {
            cout << "tauDist(v,vs)/L1Norm(v) == " << errTau << endl;
            cout << " L1Dist(v,vs)/L1Norm(v) == " << errL1 << endl;
            cout << "fabs(va-vas)            == " << aerr << endl;
            cout << "fabs(vb-vbs)            == " << berr << endl;
            cout << "residual on vs == " << helmholtz.residual(vs, r, va, vb) << endl;
        }

        // Also, construct, solve, and verify a problem of the form
        // nu v'' - lambda v = r + mu, mean(v) = vmean, v(+-1) = a,b.

        if (verbose)
            cout << "problem 2: nu v'' - lambda v = f + mu, mean(v) = vmean, v(a),v(b) = va,vb" << endl;

        Real mu = drand48();

        ChebyCoeff rhs(r);
        rhs[0] -= mu;
        Real vmean = v.mean();

        Real mus;
        helmholtz.solve(vs, mus, rhs, vmean, va, vb);

        errL1 = L1Dist(v, vs) / L1Norm(v);
        errTau = tauDist(v, vs) / L1Norm(v);
        aerr = fabs(va - vas);
        berr = fabs(vb - vbs);
        Real meanerr = fabs(vmean - v.mean());
        Real muerr = fabs(mus - mu);

        if (errL1 > EPSILON || aerr > EPSILON || berr > EPSILON || meanerr > EPSILON || muerr > EPSILON) {
            ++failures;
            if (verbose)
                cout << "ERROR : " << endl;
        }
        if (verbose) {
            cout << "tauDist(v,vs)/L1Norm(v) == " << errTau << endl;
            cout << " L1Dist(v,vs)/L1Norm(v) == " << errL1 << endl;
            cout << "fabs(va-vas)            == " << aerr << endl;
            cout << "fabs(vb-vbs)            == " << berr << endl;
            cout << "fabs(mu-ms)             == " << muerr << endl;
            cout << "fabs(mean-means)        == " << meanerr << endl;
        }
    }

    if (failures == 0) {
        cerr << "\t   pass   " << endl;
        cout << "\t   pass   " << endl;
        return 0;
    } else {
        cerr << "\t** FAIL **" << endl;
        cout << "\t** FAIL **" << endl;
        return 1;
    }
}
