/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <iostream>

#include "channelflow/diffops.h"
#include "channelflow/flowfield.h"
#include "channelflow/utilfuncs.h"

using namespace std;
using namespace chflow;

int main(int argc, char** argv) {
    cfMPI_Init(&argc, &argv);
    {
        string purpose = "Quadratic extrapolation of FlowField u(mu) as function of parameter mu";

        ArgList args(argc, argv, purpose);

        const bool fixdiv = args.getflag("-dv", "--divergence", "project field onto div-free subspace");
        const Real eps = args.getreal("-e", "--epsilson", 1e-13, "don't interpolate Lx,Lz,a,b if |diffs| < eps");

        cfarray<Real> mu(3);    // continuation paramter
        cfarray<string> un(3);  // filenames for continuation fields

        mu[0] = args.getreal(8, "mu1", "continuation parameter for u1");
        un[0] = args.getstr(7, "<flowfield>", "input field u1");
        mu[1] = args.getreal(6, "mu2", "continuation parameter for u2");
        un[1] = args.getstr(5, "<flowfield>", "input field u2");
        mu[2] = args.getreal(4, "mu3", "continuation parameter for u3");
        un[2] = args.getstr(3, "<flowfield>", "input field u3");

        const Real mug = args.getreal(2, "mu", "continuation parameter for output field");
        const string outname = args.getstr(1, "<flowfield>", "output field");

        args.check();
        args.save("./");

        // Require mu0<mu1<mu2 or mu0>mu1>mu2 or
        if (!((mu[0] < mu[1] && mu[1] < mu[2]) || (mu[0] > mu[1] && mu[1] > mu[2]))) {
            cerr << "error: mu[n] must be in ascending or descending order\n";
            for (int n = 0; n < 3; ++n)
                cerr << "mu[" << n << "] == " << mu[n] << endl;
            exit(1);
        }

        cfarray<FlowField> u(3);
        for (int n = 0; n < 3; ++n)
            u[n] = FlowField(un[n]);

        const int Nx = u[0].Nx();
        const int Ny = u[0].Ny();
        const int Nz = u[0].Nz();
        const int Nd = u[0].Nd();

        char s = ' ';
        if (u[1].Nx() != Nx || u[2].Nx() != Nx || u[1].Ny() != Ny || u[2].Ny() != Ny || u[1].Nz() != Nz ||
            u[2].Nz() != Nz || u[1].Nd() != Nd || u[2].Nd() != Nd) {
            cerr << "Field grid size mismatch. Nx Ny Nz Nd for ui is " << endl;
            cerr << "u1 : " << u[0].Nx() << s << u[0].Ny() << s << u[0].Nz() << s << u[0].Nd() << endl;
            cerr << "u2 : " << u[1].Nx() << s << u[1].Ny() << s << u[1].Nz() << s << u[1].Nd() << endl;
            cerr << "u3 : " << u[2].Nx() << s << u[2].Ny() << s << u[2].Nz() << s << u[2].Nd() << endl;
            exit(1);
        }

        // Extrapolate indpt param R and box params as function of mu
        cfarray<Real> f(3);

        // const Real Rg = quadraticInterpolate(R, mu, mug);

        for (int i = 0; i < 3; ++i)
            f[i] = u[i].Lx();
        const Real Lx = isconst(f, eps) ? f[0] : quadraticInterpolate(f, mu, mug);

        for (int i = 0; i < 3; ++i)
            f[i] = u[i].Lz();
        const Real Lz = isconst(f, eps) ? f[0] : quadraticInterpolate(f, mu, mug);

        for (int i = 0; i < 3; ++i)
            f[i] = u[i].a();
        const Real a = isconst(f, eps) ? f[0] : quadraticInterpolate(f, mu, mug);

        for (int i = 0; i < 3; ++i)
            f[i] = u[i].b();
        const Real b = isconst(f, eps) ? f[0] : quadraticInterpolate(f, mu, mug);

        for (int i = 0; i < 3; ++i)
            u[i].makePhysical();

        // Extrapolate gridpoint values as function of mu
        FlowField ug(Nx, Ny, Nz, Nd, Lx, Lz, a, b, NULL, Physical, Physical);

        // ug.save("uginit");

        for (int i = 0; i < Nd; ++i)
            for (int ny = 0; ny < Ny; ++ny)
                for (int nx = 0; nx < Nx; ++nx)
                    for (int nz = 0; nz < Nz; ++nz) {
                        for (int j = 0; j < 3; ++j)
                            f[j] = u[j](nx, ny, nz, i);
                        ug(nx, ny, nz, i) = quadraticInterpolate(f, mu, mug);
                    }

        ug.makeSpectral();
        ug.zeroPaddedModes();

        if (fixdiv && ug.Nd() == 3) {
            cout << "Not fixing divergence" << endl;
            //      Vector v;
            //      cout << "divNorm(u)   == " << divNorm(ug) << endl;
            //      cout << "bcNorm(u)    == " << bcNorm(ug) << endl;
            //      cout << "L2Norm(u)    == " << L2Norm(ug) << endl;
            //      cout <<"fixing diveregence..." << endl;
            //      FlowField ug2(ug);
            //      field2vector(ug,v);
            //      vector2field(v,ug);
            //      cout << "divNorm(u')  == " << divNorm(ug) << endl;
            //      cout << "bcNorm(u')   == " << bcNorm(ug) << endl;
            //      cout << "L2Norm(u')   == " << L2Norm(ug) << endl;
            //      cout << "bcNorm(u')   == " << bcNorm(ug) << endl;
            //      cout << "L2Dist(u',u) == " << L2Dist(ug,ug2)<< endl;
        }

        ug.save(outname);
        cout << setprecision(17);
        // cout << "extrapolated R == " << Rg << endl;
    }
    cfMPI_Finalize();
}
