/**
 * This program adds a base flow, constructed from input parameters,
 * to given velocity and temperature fluctuations.
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 *
 * Original author: Florian Reetz
 */

#include <sys/stat.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "channelflow/flowfield.h"
#include "modules/ilc/ilc.h"

using namespace std;
using namespace channelflow;

int main(int argc, char* argv[]) {
    cfMPI_Init(&argc, &argv);
    {
        string purpose(
            "Compute and add the base flow to velocity and "
            "temerature fields based on the ILCFlags");

        ArgList args(argc, argv, purpose);

        ILCFlags flags(args);
        TimeStep dt(flags);

        const string uname = args.getstr(4, "<flowfield>", "input field of velocity fluctuations");
        const string tname = args.getstr(3, "<flowfield>", "input field of temperature fluctuations");
        const string ubfname = args.getstr(2, "<flowfield>", "output field of total velocity");
        const string tbfname = args.getstr(1, "<flowfield>", "output field of total temperature");

        // put all fields into a vector
        FlowField u(uname);
        FlowField temp(tname);
        vector<FlowField> fields = {u, temp, FlowField(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b())};

        // construct OBE object
        OBE obe(fields, flags);

        //
        ChebyCoeff Ubase(obe.Ubase());
        ChebyCoeff Wbase(obe.Wbase());
        ChebyCoeff Tbase(obe.Tbase());

        // add base flow;
        for (int ny = 0; ny < u.Ny(); ++ny) {
            u.cmplx(0, ny, 0, 0) += Complex(Ubase(ny), 0.0);
            u.cmplx(0, ny, 0, 2) += Complex(Wbase(ny), 0.0);
            temp.cmplx(0, ny, 0, 0) += Complex(Tbase(ny), 0.0);
        }
        u.cmplx(0, 0, 0, 1) -= Complex(flags.Vsuck, 0.);

        u.save(ubfname);
        temp.save(tbfname);
    }
    cfMPI_Finalize();
}
