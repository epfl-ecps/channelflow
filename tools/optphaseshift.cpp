/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <fstream>
#include <iostream>
#include "channelflow/flowfield.h"
#include "channelflow/utilfuncs.h"
using namespace std;
using namespace chflow;

// Integrate 10 timesteps and find phaseshift that minimizes L2(u(10) - u(0))
int main(int argc, char* argv[]) {
    cfMPI_Init(&argc, &argv);
    {
        ArgList args(argc, argv, "Calculate optimal phase shift between first input field and all other fields");
        //   bool s1 = args.getbool("-s1", "--s1", false, "apply shift-and-reflect symmetry");
        //   Real az = args.getreal("-az", "--az", 0., "shift field by az");
        const int nproc0 =
            args.getint("-np0", "--nproc0", 0, "number of MPI-processes for transpose/number of parallel ffts");
        const int nproc1 = args.getint("-np1", "--nproc1", 0, "number of MPI-processes for one fft");
        const Real precis = args.getreal("-prec", "--precision", 1e-7, "set precision");
        const string uname0 = args.getstr(2, "<FlowField>", "First input field");
        const string uname1 = args.getstr(1, "<FlowField>", "Second input field");

        CfMPI* cfmpi = &CfMPI::getInstance(nproc0, nproc1);

        args.save();
        args.check();

        FlowField u0(uname0, cfmpi);
        FlowField u1(uname1, cfmpi);

        Real L2D0 = L2Dist(u0, u1);

        Real ax = optPhaseShiftx(u0, u1, -.5, .5, 5, precis);

        FieldSymmetry sigma(ax, 0);

        u1 *= sigma;
        Real L2D = L2Dist(u0, u1);

        cout << "Optimal shift ax:    " << ax << endl;
        cout << "Error before shift:  " << L2D0 << endl;
        cout << "Error after shift:   " << L2D << endl;
        cout << endl;
        cout << "(The L2 distance of the input fields \nis minimized for sigma * " << uname1 << " - " << uname0
             << " with\nsigma = " << sigma << ")" << endl;

        sigma.save("sigmaopt.asc");
    }
    cfMPI_Finalize();
    return 0;
}
