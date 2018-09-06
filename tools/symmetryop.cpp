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
        string purpose(
            "apply a symmetry operation to a field:"
            "\n\t(u,v,w)(x,y,z) -> (sx u, sy v, sz w)(sx x + ax*Lx, sy y, sz z + az*Lz)");

        ArgList args(argc, argv, purpose);

        const string symmstr = args.getstr("-symm", "--symmetry", "", "read symmetry from file OR");

        const int sn = args.getint("-sn", "--sn", 0, "apply s1, s2, or s3 symmetry (0 => use sx,sy,sz,ax,az OR");
        const bool sx = args.getflag("-sx", "--x-sign", "change u,x sign");
        const bool sy = args.getflag("-sy", "--y-sign", "change v,y sign");
        const bool sz = args.getflag("-sz", "--z-sign", "change w,z sign");
        const Real ax = args.getreal("-ax", "--axshift", 0.0, "translate x by ax*Lx");
        const Real az = args.getreal("-az", "--azshift", 0.0, "translate z by az*Lz");
        const bool anti = args.getflag("-a", "--anti", "antisymmetry instead of symmetry");
        const bool verbose = args.getflag("-v", "--verbose", "print out some diagnostics");
        const string iname = args.getstr(2, "<flowfield>", "input field");
        const string oname = args.getstr(1, "<flowfield>", "output field");
        args.check();
        args.save("./");

        int asign = anti ? -1 : 1;

        FlowField u(iname);
        u.makeSpectral();

        FieldSymmetry s;

        if (sn == 1)
            s = FieldSymmetry(1, 1, -1, 0.5, 0.0, asign);
        else if (sn == 2)
            s = FieldSymmetry(-1, -1, 1, 0.5, 0.5, asign);
        else if (sn == 3)
            s = FieldSymmetry(-1, -1, -1, 0.0, 0.5, asign);
        else
            s = FieldSymmetry(sx ? -1 : 1, sy ? -1 : 1, sz ? -1 : 1, ax, az, asign);

        if (symmstr != "") {
            cout << "loading symmetry from file" << symmstr << endl;
            s = FieldSymmetry(symmstr);
            cout << " symm == " << s << endl;
        }

        if (verbose) {
            FlowField su = s(u);
            cout << "symmetry s == " << s << endl;
            cout << "L2Dist(u,s(u)) == " << L2Dist(u, su) << endl;
            su.save(oname);
        } else {
            u *= s;
            u.save(oname);
        }
    }
    cfMPI_Finalize();
}
