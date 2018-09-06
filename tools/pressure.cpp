/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include <sys/stat.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "channelflow/flowfield.h"
#include "channelflow/poissonsolver.h"
#include "channelflow/utilfuncs.h"

using namespace std;
using namespace chflow;

int main(int argc, char* argv[]) {
    cfMPI_Init(&argc, &argv);
    {
        string purpose("compute pressure field of a given velocity field");

        ArgList args(argc, argv, purpose);
        const string nonlstr = args.getstr("-nl", "--nonlinearity", "rot",
                                           "method of calculating "
                                           "nonlinearity, one of [rot conv div skew alt]");
        const string uname = args.getstr(2, "<flowfield>", "input velocity field (deviation from laminar)");
        const string pname =
            args.getstr(1, "<flowfield>", "output pressure field (not including const pressure gradient)");

        string Uname, Wname;
        DNSFlags baseflags = setBaseFlowFlags(args, Uname, Wname);
        baseflags.nonlinearity = s2nonlmethod(nonlstr);
        args.check();

        // define all input for pressure
        FlowField u(uname);
        vector<ChebyCoeff> base_Flow = baseFlow(u.Ny(), u.a(), u.b(), baseflags, Uname, Wname);

        // compute pressure
        PressureSolver poisson(u, base_Flow[0], base_Flow[1], baseflags.nu, baseflags.Vsuck, baseflags.nonlinearity);

        FlowField q = poisson.solve(u);

        q.setPadded(true);
        q.save(pname);
    }
    cfMPI_Finalize();
}
