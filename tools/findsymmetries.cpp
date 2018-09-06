/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include "channelflow/diffops.h"
#include "channelflow/flowfield.h"
#include "channelflow/symmetry.h"
#include "channelflow/utilfuncs.h"

using namespace std;

using namespace chflow;

int main(int argc, char* argv[]) {
    cfMPI_Init(&argc, &argv);
    {
        string purpose("find the symmetries satisfied by a given field");

        ArgList args(argc, argv, purpose);
        const bool antisymm = args.getflag("-a", "--antisymmetries", "check antisymmetries as well");
        const bool verbose = args.getflag("-v", "--verbose", "print error of each checked symmetry");
        const int Nx = args.getint("-nx", "--nx", 4, "check x-trans over set {0,nx-1}/nx Lx");
        const int Nz = args.getint("-nz", "--nz", 4, "check z-trans over set {0,nz-1}/nz Lz");
        const Real eps = args.getreal("-e", "--eps", 1e-6, "cut-off for symmetry error");
        const string uname = args.getstr(1, "<flowfield>", "a flowfield");
        args.check();

        int antisymm_inc = antisymm ? 2 : 99;
        FlowField u(uname);
        const Real unorm = L2Norm(u);
        const Real rNx = Real(Nx);
        const Real rNz = Real(Nz);

        vector<FieldSymmetry> symm;

        if (verbose) {
            cout << "% err == L2Dist(u,su)/L2Norm(u)\n";
            cout << "% symm?\terr\ts\n";
            cout.setf(ios::left);
            // cout << scientific << endl;
        }
        for (int nx = 0; nx < Nx; ++nx) {
            for (int nz = 0; nz < Nz; ++nz) {
                for (int sx = 1; sx >= -1; sx -= 2)
                    for (int sy = 1; sy >= -1; sy -= 2)
                        for (int sz = 1; sz >= -1; sz -= 2) {
                            for (int sa = 1; sa >= -1; sa -= antisymm_inc) {
                                //// skip identity
                                // if (nx==0 && nz==0 && sx==1 && sz==1)
                                // break;

                                FieldSymmetry s(sx, sy, sz, nx / rNx, nz / rNz, sa);
                                Real err = L2Dist(u, s(u)) / unorm;
                                if (verbose)
                                    cout << fuzzyless(err, eps) << '\t' << setw(12) << err << '\t' << s << endl;

                                if (err < eps)
                                    symm.push_back(s);
                            }
                        }
            }
        }

        cout << "satisfied " << symm.size() << " symmetries to eps == " << eps << endl;
        ofstream os((removeSuffix(uname, ".ff") + ".symms").c_str());
        os << setprecision(17);
        os << symm.size() << '\n';
        for (uint n = 0; n < symm.size(); ++n) {
            os << symm[n] << '\n';
            cout << symm[n] << '\n';
        }
    }
    cfMPI_Finalize();
}
