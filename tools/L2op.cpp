/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <stdlib.h>
#include <sys/stat.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "channelflow/diffops.h"
#include "channelflow/flowfield.h"
#include "channelflow/periodicfunc.h"
#include "channelflow/utilfuncs.h"

using namespace std;

using namespace chflow;

void geomprint(const FlowField& u, ostream& os);

int main(int argc, char* argv[]) {
    cfMPI_Init(&argc, &argv);
    {
        string purpose(
            " \r \n compute the L2 related operations of two fields \r \n possible options are: L2 distance, L2 inner "
            "product");

        ArgList args(argc, argv, purpose);

        // const bool   normal  = args.getflag("-n", "--normalize", "return |u-v|/sqrt(|u||v|)");
        const bool L2dist = args.getflag("-dist", "--distance", "compute the L2 Distance of two fields");
        const bool L2ip =
            args.getflag("-ip", "--innerproduct", "compute the inner product of two fields : L2IP(u0,u1)");
        const bool normalize = args.getflag("-n", "--normalize", "normalize by norms of each field");
        const bool sx = args.getflag("-sx", "--shiftx", "shift by Lx/2");
        const bool sz = args.getflag("-sz", "--shiftz", "shift by Lz/2");
        const bool save_diff = args.getflag("-sd", "--savediff", "save the difference");
        const string u0name = args.getstr(2, "<flowfield>", "input field #1");
        const string u1name = args.getstr(1, "<flowfield>", "input field #2");

        args.check();

        FlowField u0(u0name);
        FlowField u1(u1name);

        u0.makeSpectral();
        u1.makeSpectral();
        if (!u0.congruent(u1)) {
            cout << setprecision(16);
            cout << u0name << " : ";
            geomprint(u0, cout);
            cout << endl;
            cout << u1name << " : ";
            geomprint(u1, cout);
            cout << endl;
            cerr << "The two fields are not congruent!" << endl;
            exit(1);
        }

        if (sx && sz)
            u1 *= FieldSymmetry(1, 1, 1, 0.5, 0.5);
        else if (sx)
            u1 *= FieldSymmetry(1, 1, 1, 0.5, 0.0);
        else if (sz)
            u1 *= FieldSymmetry(1, 1, 1, 0.0, 0.5);

        if (L2dist) {
            Real nrm = normalize ? 1.0 / sqrt(L2Norm(u0) * L2Norm(u1)) : 1;
            cout << nrm * L2Dist(u0, u1) << endl;
            if (save_diff) {
                u0 -= u1;
                u0.save("difference");
            }
        }

        else if (L2ip) {
            if (normalize)
                cout << L2IP(u0, u1) / (L2Norm(u0) * L2Norm(u1)) << endl;
            else
                cout << L2IP(u0, u1) << endl;
        }

        else
            cferror("L2op: please choose a L2 operation, rerun with -h option");
    }
    cfMPI_Finalize();
}

void geomprint(const FlowField& u, ostream& os) {
    cout << u.Nx() << " x " << u.Ny() << " x " << u.Nz() << " x " << u.Nd() << ",  "
         << "[0, " << u.Lx() << "] x "
         << "[" << u.a() << ", " << u.b() << "] x "
         << "[0, " << u.Lz() << "]";
}
