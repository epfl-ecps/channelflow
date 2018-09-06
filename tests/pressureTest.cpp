/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <iomanip>
#include <iostream>
#include "channelflow/diffops.h"
#include "channelflow/flowfield.h"
#include "channelflow/poissonsolver.h"

// Compare skew-symmetric and rotational calculations of FlowField
// nonlinearity methods.

using namespace std;

using namespace chflow;

int main(int argc, char* argv[]) {
    const bool save = false;
    const bool verbose = true;
    const Real maxerr = 1e-7;
    Real err = 0.0;
#ifdef HAVE_MPI
    cfMPI_Init(&argc, &argv);
    {
#endif

        // Define gridsize
        const int Nx = 24;
        const int Ny = 49;
        const int Nz = 24;

        // Define box size
        const Real Lx = 2 * pi;
        const Real a = -1.0;
        const Real b = 1.0;
        const Real Lz = 2 * pi;

        // Define flow parameters
        const Real Reynolds = 400.0;
        const Real nu = 1.0 / Reynolds;

        // Define size and smoothness of initial disturbance
        const Real decay = 0.5;
        const Real magnitude = 0.01;
        const int kxmax = 3;
        const int kzmax = 3;

        NonlinearMethod nonl_method = Convection;

        cerr << "pressureTest: " << flush;
        if (verbose) {
            cout << "\n====================================================" << endl;
            cout << "pressureTest\n\n";
        }

        // Construct data fields: 3d velocity and 1d pressure
        FlowField u(Nx, Ny, Nz, 3, Lx, Lz, a, b);
        u.addPerturbations(kxmax, kzmax, magnitude, decay);

        // Construct base flow for plane Couette: U(y) = y
        ChebyCoeff U(Ny, a, b, Physical);
        ChebyCoeff W(Ny, a, b, Physical);
        Vector x = u.xgridpts();
        Vector y = u.ygridpts();
        Vector z = u.zgridpts();
        for (int ny = 0; ny < Ny; ++ny)
            U[ny] = y[ny];

        if (save) {
            x.save("x");
            y.save("y");
            z.save("z");
            U.save("U");
            y.save("y");
        }

        U.makeSpectral();
        ChebyCoeff Uy = diff(U);
        ChebyCoeff UyT(U);
        U.makeSpectral();
        Uy.makePhysical();

        W.setToZero();
        W.makeSpectral();

        FlowField p(Nx, Ny, Nz, 1, Lx, Lz, a, b);

        // rotationalNL(u,U,frot,tmp);
        // alternatingNL(u,U,falt,tmp, true);
        // convectionNL(u,U,fcnv,tmp);
        // divergenceNL(u,U,fdiv,tmp);
        // skewsymmetricNL(u,U,fsym,tmp);
        // linearizedNL(u,U,flin);

        PressureSolver psolver(u, U, W, nu, 0, nonl_method);

        psolver.solve(p, u);
        err += psolver.verify(p, u);

        // Check that grad_p and div_nonl cancel out each other's divergence
        FlowField grad_p;
        grad(p, grad_p);

        FlowField nonl(Nx, Ny, Nz, 3, Lx, Lz, a, b);
        FlowField tmp(Nx, Ny, Nz, 9, Lx, Lz, a, b);
        FlowField div_nonl(Nx, Ny, Nz, 3, Lx, Lz, a, b);

        DNSFlags flags;
        flags.nonlinearity = nonl_method;
        navierstokesNL(u, U, W, nonl, tmp, flags);
        div(nonl, div_nonl);

        FlowField diff(nonl);
        diff += grad_p;

        if (verbose) {
            cout << "L2Norm(u)       == " << L2Norm(u) << endl;
            cout << "L2Norm(p)       == " << L2Norm(p) << endl;
            cout << "L2Norm(nonl)    == " << L2Norm(nonl) << endl;
            cout << "L2Norm(grad_p)  == " << L2Norm(grad_p) << endl;
            cout << "L2Norm(diff)    == " << L2Norm(diff) << endl;
            cout << "divNorm(nonl)   == " << divNorm(nonl) << endl;
            cout << "divNorm(grad_p) == " << divNorm(grad_p) << endl;
            cout << "divNorm(diff)   == " << divNorm(diff) << endl;
        }
#ifdef HAVE_MPI
    }
    cfMPI_Finalize();
#endif
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

    /**************************************************
    navierstokesNL(u, U, nonl, tmp, nonl_method);
    div(nonl, div_nonl);

    PoissonSolver psolver(Nx,Ny,Nz,1,Lx,Lz,a,b);
    psolver.solve(p, div_nonl);
    psolver.verify(p, div_nonl);

    FlowField lapl_p = lapl(p);

    for (int mx=0; mx<=3; ++mx) {
      for (int mz=0; mz<=3; ++mz) {
        string lbl = i2s(mx)+i2s(mz);
        p.saveProfile(mx,mz, "p"+lbl);
        div_nonl.saveProfile(mx,mz, "div_nonl"+lbl);
        lapl_p.saveProfile(mx,mz, "lapl_p"+lbl);
      }
    }
    p.saveSpectrum("pspec");
    div_nonl.saveSpectrum("div_nonlspec");
    lapl_p.saveSpectrum("lapl_pspec");

    cout << "L2Norm(u)       == " << L2Norm(u) << endl;
    cout << "L2Norm(p)       == " << L2Norm(p) << endl;
    cout << "L2Norm(div_nonl)== " << L2Norm(div_nonl) << endl;
    cout << "L2Norm(lapl_p)  == " << L2Norm(lapl_p) << endl;
    cout << "L2Dist(lapl_p, div_nonl)  == " << L2Dist(lapl_p, div_nonl) << endl;
    cout << "bcNorm(u)       == " << bcNorm(u) << endl;
    cout << "bcNorm(p)       == " << bcNorm(p) << endl;
    cout << "bcNorm(div_nonl)== " << bcNorm(div_nonl) << endl;
    *************************************************/
}
