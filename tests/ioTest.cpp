/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <iostream>
#include "channelflow/diffops.h"
#include "channelflow/flowfield.h"

using namespace std;

using namespace chflow;

int main(int argc, char* argv[]) {
    const Real maxerr = 1e-14;
    Real toterr = 0.0;
    Real err;

    CfMPI* cfmpi = NULL;
    CfMPI* cfmpi1 = NULL;
#ifdef HAVE_MPI
    cfMPI_Init(&argc, &argv);
    {
        cfmpi = &CfMPI::getInstance();
        cfmpi1 = &CfMPI::getInstance();
#endif

        bool verbose = true;
        cerr << "ioTest: " << flush;
        if (verbose) {
            cout << "\n====================================================" << endl;
            cout << "ioTest\n\n";
        }

        // Define gridsize
        const int Nx = 32;
        const int Ny = 33;
        const int Nz = 24;

        // Define box size
        const Real Lx = 1.75 * pi;
        const Real a = -1.0;
        const Real b = 1.0;
        const Real Lz = 1.2 * pi;

        // Define size and smoothness of initial disturbance
        const Real decay = 0.5;
        const Real magn = 0.1;
        // const int kxmax = 4;
        // const int kzmax = 4;

        FlowField u(Nx, Ny, Nz, 3, Lx, Lz, a, b, cfmpi);
        FlowField u0(Nx, Ny, Nz, 3, Lx, Lz, a, b, cfmpi1);

        // Test IO on nonpadded field
        u0.setPadded(false);
        u0.perturb(magn, decay);
        u.interpolate(u0);
        u *= magn / L2Norm(u);
        u.save("ufull");
        FlowField v("ufull", cfmpi);
        cout << u.nproc0() << endl;
        cout << v.nproc0() << endl;

        err = L2Dist(u, v);
        cout << "L2Dist(u,v) == " << err << " (nonpadded IO test)" << endl;

        // Test IO on padded field
        u.zeroPaddedModes();
        u.save("upad");
        FlowField w("upad", cfmpi);

        err = L2Dist(u, w);
        toterr += err;
        cout << "L2Dist(u,w) == " << err << " (padded IO test)" << endl;
#ifdef HAVE_MPI
    }
    cfMPI_Finalize();
#endif
    if (toterr < maxerr) {
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
