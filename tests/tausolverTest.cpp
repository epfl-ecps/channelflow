/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <fstream>
#include <iomanip>
#include <iostream>
#include "channelflow/tausolver.h"

using namespace std;

using namespace chflow;

void randomVprofile(ComplexChebyCoeff& v, Real decay);
void randomUprofile(ComplexChebyCoeff& u, Real decay);
void randomProfile(ComplexChebyCoeff& u, ComplexChebyCoeff& v, ComplexChebyCoeff& w, ComplexChebyCoeff& P, int kx,
                   int kz, Real Lx, Real Lz, Real decay);

void uniformsave(const ComplexChebyCoeff& u, const string& filebase);

const int Nunif = 100;

int main(int argc, char* argv[]) {
    int failures = 0;
    int nTests = 100;
    // 	#ifdef HAVE_MPI
    cfMPI_Init(&argc, &argv);
    {
        // 	#endif

        bool save = true;
        bool verbose = true;

        bool taucorrect = true;
        int N = 49;  // Chebyshev expansion length;

        Real epsilon = 1e-9;

        cerr << "tausolverTest: " << flush;
        if (verbose) {
            cout << "\n====================================================" << endl;
            cout << "tausolverTest\n\n";
        }

        ChebyTransform trans(N);

        for (int test = 0; test < nTests; ++test) {
            bool failure = false;
            Real Lx = 2 * pi * (1 + drand48());
            Real Lz = 2 * pi * (1 + drand48());
            Real a = 1.0 + 0.1 * drand48();
            Real b = a + (2 + drand48());
            int kx = rand() % 32;
            int kz = rand() % 32;
            Real dt = 0.02;
            Real nu = 1.0 / 1000.0;
            Real lambda = 2.0 / dt + 4 * pi * pi * nu * (square(kx / Lx) + square(kz / Lz));
            Real decay = 0.5;

            if (verbose) {
                cout << "Tausolver test #" << test << endl;
                cout << "a b    == " << a << ' ' << b << endl;
                cout << "Lx Lz  == " << Lx << ' ' << Lz << endl;
                cout << "kx Lz  == " << kx << ' ' << kz << endl;
                cout << "lambda == " << lambda << endl;
                cout << "nu     == " << nu << endl;
                cout << "decay  == " << decay << endl;
            }

            // Construct a div-free field (u,v,w) with zero BCS at +/-1 and a
            // random zero-mean pressure field P. From those, calculate
            // R = (nu del^2 u'' - lambda u - grad P). Then use TauSolver to solve
            // nu del^2 u - lambda u - grad P = R for u and P from R, and compare.

            ComplexChebyCoeff P(N, a, b, Spectral);
            ComplexChebyCoeff u(N, a, b, Spectral);
            ComplexChebyCoeff v(N, a, b, Spectral);
            ComplexChebyCoeff w(N, a, b, Spectral);

            randomProfile(u, v, w, P, kx, kz, Lx, Lz, decay);

            ComplexChebyCoeff vy = diff(v);
            // Calculate R = nu lapl u'' - lambda u - grad P
            ComplexChebyCoeff Rx(u);
            ComplexChebyCoeff Ry(v);
            ComplexChebyCoeff Rz(w);
            Rx *= lambda;
            Ry *= lambda;
            Rz *= lambda;

            ComplexChebyCoeff nu_uyy(N, a, b, Spectral);
            ComplexChebyCoeff nu_vyy(N, a, b, Spectral);
            ComplexChebyCoeff nu_wyy(N, a, b, Spectral);
            diff2(u, nu_uyy);
            diff2(v, nu_vyy);
            diff2(w, nu_wyy);
            nu_uyy *= -nu;
            nu_vyy *= -nu;
            nu_wyy *= -nu;

            Rx += nu_uyy;
            Ry += nu_vyy;
            Rz += nu_wyy;

            ComplexChebyCoeff Px(P);
            ComplexChebyCoeff Pz(P);
            Px *= Complex(0.0, 2 * pi * (kx / Lx));
            Pz *= Complex(0.0, 2 * pi * (kz / Lz));
            ComplexChebyCoeff Py(N, a, b, Spectral);
            diff(P, Py);

            Rx += Px;
            Ry += Py;
            Rz += Pz;

            ComplexChebyCoeff usolve(N, a, b, Spectral);
            ComplexChebyCoeff vsolve(N, a, b, Spectral);
            ComplexChebyCoeff vysolve(N, a, b, Spectral);
            ComplexChebyCoeff wsolve(N, a, b, Spectral);
            ComplexChebyCoeff Psolve(N, a, b, Spectral);
            // Real sigma0 = 0;
            // Real sigma1 = 0;
            if (verbose)
                cout << "Constructing TauSolver {" << endl;
            TauSolver tausolver(kx, kz, Lx, Lz, a, b, lambda, nu, N, taucorrect);
            if (verbose) {
                cout << "} done ctoring TauSolver" << endl;
                cout << "----------------------------------------------------" << endl;
            }

            if (verbose)
                cout << "Verifying analytic Tau solution {" << endl;
            tausolver.verify(u, v, w, P, Rx, Ry, Rz);
            if (verbose) {
                cout << "} done verifying analytic Tau problem" << endl;
                cout << "----------------------------------------------------" << endl;
                cout << "Solving Tau problem numerically {" << endl;
            }
            tausolver.solve(usolve, vsolve, wsolve, Psolve, Rx, Ry, Rz);
            if (verbose) {
                cout << "} done solving Tau problem numerically" << endl;
                cout << "----------------------------------------------------" << endl;
                cout << "Verifying numerical Tau solution {" << endl;
            }
            if (tausolver.verify(usolve, vsolve, wsolve, Psolve, Rx, Ry, Rz, verbose) > epsilon)
                failure = true;

            if (verbose) {
                cout << "} done verifying numerical Tau problem" << endl;
                cout << "----------------------------------------------------" << endl;
            }

            // Compute divergence of u and us
            if (verbose) {
                cout << "L2Dist(u,us)/L2Norm(u) == " << L2Dist(u, usolve) / L2Norm(u) << endl;
                cout << "L2Dist(v,vs)/L2Norm(v) == " << L2Dist(v, vsolve) / L2Norm(v) << endl;
                cout << "L2Dist(w,ws)/L2Norm(w) == " << L2Dist(w, wsolve) / L2Norm(w) << endl;
                cout << "L2Dist(P,Ps)/L2Norm(P) == " << L2Dist(P, Psolve) / L2Norm(P) << endl;
            }

            diff(vsolve, vysolve);

            ComplexChebyCoeff tmp;
            ComplexChebyCoeff udiv = vy;
            tmp = u;
            tmp *= (2 * pi * kx / Lx) * I;
            udiv += tmp;
            tmp = w;
            tmp *= (2 * pi * kz / Lz) * I;
            udiv += tmp;

            ComplexChebyCoeff udivs = vysolve;
            tmp = usolve;
            tmp *= (2 * pi * kx / Lx) * I;
            udivs += tmp;
            tmp = wsolve;
            tmp *= (2 * pi * kz / Lz) * I;
            udivs += tmp;

            if (L2Norm(udiv) > epsilon)
                failure = true;

            if (verbose) {
                cout << "L2Norm(udiv)  == " << L2Norm(udiv) << endl;
                cout << "L2Norm(udivs) == " << L2Norm(udivs) << endl;
                cout << "L2Dist(udiv,udivs)  == " << L2Dist(udiv, udivs) << endl;
            }

            if (save) {
                ComplexChebyCoeff diverge(N, a, b, Spectral);
                diff(vsolve, diverge);
                ComplexChebyCoeff tmp(N, a, b, Spectral);
                tmp = usolve;
                tmp *= 2 * pi * I * (kx / Lx);
                diverge += tmp;
                tmp = wsolve;
                tmp *= 2 * pi * I * (kz / Lz);
                diverge += tmp;

                Vector y = chebypoints(N, a, b);
                y.save("y");
                u.save("u");
                v.save("v");
                w.save("w");
                P.save("P");
                vy.save("vy");
                Py.save("Py");
                usolve.save("us");
                vsolve.save("vs");
                wsolve.save("ws");
                Psolve.save("Ps");
                vysolve.save("vys");
                Rx.save("Rx");
                Ry.save("Ry");
                Rz.save("Rz");
                diverge.save("divs");

                Vector yf(Nunif);
                Real dy = (b - a) / Nunif;
                for (int n = 0; n < Nunif; ++n)
                    yf[n] = a + n * dy;

                yf.save("yf");
                uniformsave(u, "uf");
                uniformsave(v, "vf");
                uniformsave(w, "wf");
                uniformsave(P, "Pf");
                uniformsave(Py, "Pyf");
                uniformsave(vy, "vyf");
                uniformsave(usolve, "usf");
                uniformsave(vsolve, "vsf");
                uniformsave(wsolve, "wsf");
                uniformsave(Psolve, "Psf");
                uniformsave(vysolve, "vysf");
                uniformsave(Rx, "Rxf");
                uniformsave(Ry, "Ryf");
                uniformsave(Rz, "Rzf");
                uniformsave(diverge, "divsf");

                ComplexChebyCoeff uyy = diff2(u);
                ComplexChebyCoeff vyy = diff2(v);
                ComplexChebyCoeff wyy = diff2(w);
                uyy.save("uyy");
                vyy.save("vyy");
                wyy.save("wyy");
                ComplexChebyCoeff uyys = diff2(usolve);
                ComplexChebyCoeff vyys = diff2(vsolve);
                ComplexChebyCoeff wyys = diff2(wsolve);
                ComplexChebyCoeff Pys = diff(Psolve);
                uyys.save("uyys");
                vyys.save("vyys");
                wyys.save("wyys");
                Pys.save("Pys");
                vysolve.save("vys");

                usolve.ichebyfft(trans);
                vsolve.ichebyfft(trans);
                wsolve.ichebyfft(trans);
                Psolve.ichebyfft(trans);
                vysolve.ichebyfft(trans);

                // Everybody is physical
                if (verbose) {
                    cout << " us(+/-1) == " << usolve[0] << ' ' << usolve[N - 1] << endl;
                    cout << " vs(+/-1) == " << vsolve[0] << ' ' << vsolve[N - 1] << endl;
                    cout << "vsy(+/-1) == " << vysolve[0] << ' ' << vysolve[N - 1] << endl;
                    cout << " ws(+/-1) == " << wsolve[0] << ' ' << wsolve[N - 1] << endl;
                    cout << " Ps(+/-1) == " << Psolve[0] << ' ' << Psolve[N - 1] << endl;
                }
            }

            if (failure)
                ++failures;
        }
        //   #ifdef HAVE_MPI
    }
    cfMPI_Finalize();
    // #endif
    if (failures == 0) {
        cerr << "\t   pass   " << endl;
        cout << "\t   pass   " << endl;
        return 0;
    } else {
        cerr << "\t** #failures/#tests = " << Real(failures) / Real(nTests) << endl;
        cerr << "\t** FAIL **" << endl;
        cout << "\t** FAIL **" << endl;
        return 1;
    }
}

void randomUprofile(ComplexChebyCoeff& u, Real decay) {
    // Set a random u(y)
    Real mag = 1.0;
    int N = u.length();
    for (int n = 0; n < N; ++n) {
        u.set(n, mag * randomComplex());
        if (n > 2)
            mag *= decay;
    }

    // Adjust u(y) so that u(+-1) == 0
    Complex u0 = (u.eval_b() + u.eval_a()) / 2.0;
    Complex u1 = (u.eval_b() - u.eval_a()) / 2.0;
    u.sub(0, u0);
    u.sub(1, u1);
}

void randomVprofile(ComplexChebyCoeff& v, Real decay) {
    int N = v.length();
    // Assign a random v(y).
    Real mag = 1.0;
    for (int n = 0; n < N; ++n) {
        v.set(n, mag * randomComplex());
        if (n > 2)
            mag *= decay;
    }
    // Temporarily set bounds to [-1,1] to ease BC adjustment
    // Adjust v so that v(+-1) == v'(+/-1) == 0, by subtracting off
    // s0 T0(x) + s1 T1(x) + s2 T2(x) + s3 T3(x), with s's chosen to
    // have same BCs as v.
    Real a = v.a();
    Real b = v.b();
    v.setBounds(-1, 1);
    ComplexChebyCoeff vy = diff(v);

    Complex A = v.eval_a();
    Complex B = v.eval_b();
    Complex C = vy.eval_a();
    Complex D = vy.eval_b();

    // The numercial coeffs are inverse of the matrix (values found with Maple)
    // T0(1)   T1(1)   T2(1)   T3(1)
    // T0(-1)  T1(-1)  T2(-1)  T3(-1)
    // T0'(1)  T1'(1)  T2'(1)  T3'(1)
    // T0'(-1) T1'(-1) T2'(-1) T3'(-1)
    Complex s0 = 0.5 * (A + B) + 0.125 * (C - D);
    Complex s1 = 0.5625 * (B - A) - 0.0625 * (C + D);
    Complex s2 = 0.125 * (D - C);
    Complex s3 = 0.0625 * (A - B + C + D);

    // Subtract off the coeffs
    v.sub(0, s0);
    v.sub(1, s1);
    v.sub(2, s2);
    v.sub(3, s3);

    v.setBounds(a, b);
}

void randomProfile(ComplexChebyCoeff& u, ComplexChebyCoeff& v, ComplexChebyCoeff& w, ComplexChebyCoeff& P, int kx,
                   int kz, Real Lx, Real Lz, Real decay) {
    int N = u.length();
    ChebyTransform trans(N);
    Real magn = 1.0;

    P.randomize(magn, decay, Diri, Diri);

    if (kx == 0 && kz == 0) {
        // Assign an odd perturbation to u, so as not to change mean(U).
        // Just set even modes to zero.
        w.re.setToZero();  // randomUprofile(w, mag, decay);
        w.im.setToZero();

        randomUprofile(u, decay);
        for (int n = 0; n < N; n += 2)
            u.re[n] = 0.0;
        u.im.setToZero();

        v.setToZero();

        return;
    }

    // Other kx,kz cases are based on a random v(y).
    randomVprofile(v, decay);
    ComplexChebyCoeff vy = diff(v);

    if (kx == 0) {
        u.setToZero();
        w = vy;
        w *= -Lz / (2.0 * pi * I * Real(kz));
    } else if (kz == 0) {
        w.setToZero();
        u = vy;
        u *= -Lx / (2.0 * pi * I * Real(kx));
    } else {
        // Finally, the general case, where kx, kz != 0 and u,v,w are nonzero
        // Set a random u(y)
        randomUprofile(u, decay);

        // Calculate w from div u == ux + vy + wz == 0.
        ComplexChebyCoeff ux(u);
        ux *= 2.0 * pi * I * (kx / Lx);

        // Set w = -Lz/(2*pi*I*kz) * (ux + vy);
        w = vy;
        w += ux;
        w *= -Lz / (2.0 * pi * I * Real(kz));
    }

    // Check divergence
    ComplexChebyCoeff ux(u);
    ux *= 2 * pi * I * (kx / Lx);
    ComplexChebyCoeff wz(w);
    wz *= 2 * pi * I * (kz / Lz);

    ComplexChebyCoeff div(ux);
    div += vy;
    div += wz;
}

void uniformsave(const ComplexChebyCoeff& f, const string& filebase) {
    string filename = filebase + ".asc";
    ofstream os(filename.c_str());

    ComplexChebyCoeff g(f);
    g.makeSpectral();
    Real a = g.a();
    Real b = g.b();
    Real dy = (b - a) / (Nunif - 1);
    for (int n = 0; n < Nunif; ++n) {
        Real y = a + n * dy;
        os << g.re.eval(y) << ' ' << g.im.eval(y) << '\n';
    }
    os.close();
}
