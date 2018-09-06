/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <fstream>
#include <iomanip>
#include <iostream>
#include "channelflow/dns.h"
#include "channelflow/flowfield.h"
#include "channelflow/symmetry.h"
#include "channelflow/utilfuncs.h"

using namespace std;
using namespace chflow;

int main(int argc, char* argv[]) {
    cfMPI_Init(&argc, &argv);
    {
        string purpose(
            "Randomly perturb a field with another field u with zero divergence and Dirichlet BCs\n"
            "The field is u = sum_{jkl} a_{jkl} T_l(y) exp(2 pi i (jx/Lx + kz/Lz)) where\n"
            "a_{jkl} = (random # in [-1 1]) * smoothness^{|l| + |j| + |k|}, with corrections\n"
            "to assure u has zero divergence and no-slip BCs, and with rescaling\n"
            "u = magnitude/L2Norm(u)");

        ArgList args(argc, argv, purpose);

        const int seed = args.getint("-sd", "--seed", 1, "seed for random number generator");
        const Real smooth = args.getreal("-s", "--smoothness", 0.4, "smoothness of field, 0 < s < 1");
        const Real magn = args.getreal("-m", "--magnitude", 0.20, "magnitude  of field, 0 < m < 1");
        const bool meanfl = args.getflag("-mf", "--meanflow", "perturb the mean");

        const bool s1symm = args.getflag("-s1", "--s1-symmetry", "satisfy s1 symmetry");
        const bool s2symm = args.getflag("-s2", "--s2-symmetry", "satisfy s2 symmetry");
        const bool s3symm = args.getflag("-s3", "--s3-symmetry", "satisfy s3 symmetry");

        const string iname = args.getstr(2, "<flowfield>", "input field filename)");
        const string oname = args.getstr(1, "<flowfield>", "output field filename");

        args.check();
        args.save("./");

        srand48(seed);

        FlowField u0(iname);
        FlowField u(u0);
        u.setToZero();
        u.addPerturbations(u.kxmaxDealiased(), u.kzmaxDealiased(), 1.0, 1 - smooth, meanfl);

        if (s1symm) {
            FieldSymmetry s1(1, 1, -1, 0.5, 0.0);
            u += s1(u);
        }
        if (s2symm) {
            FieldSymmetry s2(-1, -1, 1, 0.5, 0.5);
            u += s2(u);
        }
        if (s3symm) {
            FieldSymmetry s3(-1, -1, -1, 0.0, 0.5);
            u += s3(u);
        }

        u *= magn / L2Norm(u);
        u.setPadded(true);
        u0 += u;
        u0.save(oname);
    }
    cfMPI_Finalize();
}
