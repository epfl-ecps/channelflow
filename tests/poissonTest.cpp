/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <iomanip>
#include <iostream>
#include "channelflow/diffops.h"
#include "channelflow/flowfield.h"
#include "channelflow/poissonsolver.h"

// Test correctness of Poisson solver.

using namespace std;

using namespace chflow;

int main(int argc, char* argv[]) {
    Real maxerr = 1e-13;
    Real err;

    cfMPI_Init(&argc, &argv);
    {
        CfMPI* cfmpi = &CfMPI::getInstance();
        const bool save = false;
        const bool verbose = true;

        // Define gridsize
        const int Nx = 24;
        const int Ny = 49;
        const int Nz = 24;
        const int Nd = 3;

        // Define box size
        const Real Lx = 1.75 * pi;
        const Real a = -1.0;
        const Real b = 1.0;
        const Real Lz = 1.2 * pi;

        // Define size and smoothness of initial disturbance
        const Real decay = 0.6;
        const Real magn = 0.01;
        const int kxmax = 7;
        const int kzmax = 7;

        cerr << "poissonTest: " << flush;

        if (verbose) {
            cout << "\n====================================================" << endl;
            cout << "poissonTest\n\n";
        }

        ComplexChebyCoeff uprof(Ny, a, b, Spectral);

        FlowField u(Nx, Ny, Nz, Nd, Lx, Lz, a, b, cfmpi);
        for (int i = 0; i < Nd; ++i)
            for (int kx = -kxmax; kx <= kxmax; ++kx) {
                int mx = u.mx(kx);
                for (int kz = 0; kz <= kzmax; ++kz) {
                    int mz = u.mz(kz);
                    if (mx >= u.mxlocmin() && mx < u.mxlocmin() + u.Mxloc() && mz >= u.mzlocmin() &&
                        mz < u.mzlocmin() + u.Mzloc()) {
                        randomUprofile(uprof, magn, decay);
                        Real c = pow(decay, abs(kx) + abs(kz));
                        for (int my = 0; my < Ny; ++my)
                            u.cmplx(mx, my, mz, i) = c * uprof[my];
                    }
                }
            }

        // FlowField u(Nx,Ny,Nz,3,Lx,Lz,a,b);
        // u.addPerturbations(kxmax, kzmax, magn, decay);

        Vector x = u.xgridpts();
        Vector y = u.ygridpts();
        Vector z = u.zgridpts();
        if (save) {
            x.save("x");
            y.save("y");
            z.save("z");
        }

        FlowField f;
        lapl(u, f);

        FlowField v(Nx, Ny, Nz, Nd, Lx, Lz, a, b, cfmpi);

        PoissonSolver poisson(v);
        poisson.solve(v, f);
        // poisson.verify(v, f);

        FlowField g;
        lapl(v, g);

        if (save)
            for (int mx = 0; mx <= 3; ++mx) {
                for (int mz = 0; mz <= 3; ++mz) {
                    string lbl = i2s(mx) + i2s(mz);
                    u.saveProfile(mx, mz, "u" + lbl);
                    v.saveProfile(mx, mz, "v" + lbl);
                    f.saveProfile(mx, mz, "lapl_u" + lbl);
                    g.saveProfile(mx, mz, "lapl_v" + lbl);
                }
            }

        if (verbose) {
            cout << "Given u with u=0 on boundary, " << endl;
            cout << "Let f = lapl u," << endl;
            cout << "solve lapl v = f for v, and " << endl;
            cout << "let g = lapl v" << endl;
            cout << "Thus v approximates u" << endl;
            cout << "And  g approximates f." << endl;
            cout << "Results: " << endl;
            cout << "L2Norm(u)   == " << L2Norm(u) << endl;
            cout << "L2Norm(v)   == " << L2Norm(v) << endl;
            cout << "L2DIst(u,v) == " << L2Dist(u, v) << endl;
        }

        FlowField w(u);
        w -= v;

        if (save) {
            u.saveSpectrum("uspec");
            v.saveSpectrum("vspec");
            w.saveSpectrum("wspec");
        }

        FlowField h(f);
        h -= g;

        if (verbose) {
            cout << "L2Norm(f)   == " << L2Norm(f) << endl;
            cout << "L2Norm(g)   == " << L2Norm(g) << endl;
            cout << "L2Dist(f,g) == " << L2Dist(f, g) << endl;
        }

        if (save) {
            f.saveSpectrum("fspec");
            g.saveSpectrum("gspec");
            h.saveSpectrum("hspec");
        }

        err = L2Dist(f, g) / L2Norm(f) + L2Dist(u, v) / L2Norm(u);

        // #ifdef HAVE_MPI
    }
    cfMPI_Finalize();
    // #endif
    if (err < maxerr) {
        cerr << "\t   pass   " << endl;
        cout << "\t   pass   " << endl;
        return 0;
    } else {
        cerr << "\t** FAIL **" << endl;
        cout << "\t** FAIL **" << endl;
        cout << "   err == " << err << endl;
        cout << "maxerr == " << maxerr << endl;
        return 1;
    }
}
