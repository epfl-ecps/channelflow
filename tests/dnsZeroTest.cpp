/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "channelflow/dns.h"
#include "channelflow/flowfield.h"

// Start a pressure-driven Poisseuille flow with either
// U(y) == 0      and u(x,y,z) == 1-y^2 or
// U(y) == 1-y^20 and u(x,y,z) == 0
// Integrate and verify that nothing changes
//
// Or start U and u from zero and check convergence onto utot = 1-y^2

using namespace std;

using namespace chflow;

int main(int argc, char* argv[]) {
    Real err = 0.0;
    Real maxerr = 1e-8;
    int exitcode = 0;
    int taskid = 0;

    CfMPI* cfmpi = NULL;

#ifdef HAVE_MPI
    cfMPI_Init(&argc, &argv);
    {
        cfmpi = &CfMPI::getInstance();
        taskid = cfmpi->taskid();
#endif

        bool verbose = false;
        bool save = false;

        const int Nx = 8;
        const int Ny = 33;
        const int Nz = 8;

        const int Nd = 3;
        const Real Lx = 2 * pi;
        const Real Lz = 2 * pi;
        const Real a = -1.0;
        const Real b = 1.0;
        // const Real ua= -1.0;
        // const Real ub=  1.0;
        const Real Ly = b - a;
        const Real Reynolds = 400.0;
        // const Real nu = 1.0/Reynolds;
        const Real dPdx = 0.0;
        const Real Ubulk = 0.0;
        const Real dt = 0.1;
        const int n = 10;
        const Real T0 = 0.0;
        const Real T1 = 10.0;

        DNSFlags flags;
        flags.baseflow = LaminarBase;
        flags.timestepping = SBDF3;
        flags.constraint = BulkVelocity;
        flags.dealiasing = DealiasXZ;
        flags.nonlinearity = Rotational;
        flags.verbosity = Silent;
        flags.nu = 1.0 / Reynolds;
        flags.ulowerwall = 0.0;
        flags.uupperwall = 0.0;
        flags.dt = dt;

        // maxerr is determined by a previous DNS computation
        for (int i = 1; i < argc; ++i) {
            string argument(argv[i]);

            if (argument == "--bulkv")
                flags.constraint = BulkVelocity;

            else if (argument == "--gradp")
                flags.constraint = PressureGradient;

            else if (argument == "--cnfe1") {
                flags.timestepping = CNFE1;
                maxerr = 2e-2;
            } else if (argument == "--cnab2") {
                flags.timestepping = CNAB2;
                maxerr = 4e-5;
            } else if (argument == "--cnrk2") {
                flags.timestepping = CNRK2;
                maxerr = 5e-6;
            } else if (argument == "--smrk2") {
                flags.timestepping = SMRK2;
                maxerr = 8e-6;
            } else if (argument == "--sbdf2") {
                flags.timestepping = SBDF2;
                maxerr = 2e-4;
            } else if (argument == "--sbdf3") {
                flags.timestepping = SBDF3;
                maxerr = 2e-6;
            } else if (argument == "--sbdf4") {
                flags.timestepping = SBDF4;
                maxerr = 1e-6;
            }
        }
        char s = ' ';
        if (taskid == 0)
            cerr << "dnsZeroTest     ";
        for (int i = 1; i < argc; ++i) {
            if (taskid == 0)
                cerr << argv[i];
            int pad = 10 - strlen(argv[i]);
            for (int j = 0; j < pad; ++j)  // crude formatting
                if (taskid == 0)
                    cerr << s;
        }
        if (taskid == 0)
            cerr << flush;

        if (taskid == 0 && verbose) {
            cout << "\n====================================================" << endl;
            cout << "dnsZeroTest\n\n";
            for (int i = 1; i < argc; ++i)
                cout << argv[i] << s;
            cout << endl;
            cout << setprecision(14);
            cout << "Nx Ny Nz Nd == " << Nx << s << Ny << s << Nz << s << Nd << endl;
            cout << "Lx Ly Lz == " << Lx << s << Ly << s << Lz << s << endl;
        }

        Vector y = chebypoints(Ny, a, b);
        ChebyTransform trans(Ny);

        // Build DNS
        vector<FlowField> fields = {FlowField(Nx, Ny, Nz, Nd, Lx, Lz, a, b, cfmpi),
                                    FlowField(Nx, Ny, Nz, 1, Lx, Lz, a, b, cfmpi)};
        FlowField ut(Nx, Ny, Nz, Nd, Lx, Lz, a, b, cfmpi);

        DNS dns(fields, flags);

        if (verbose && taskid == 0) {
            cout << "Initial data, prior to time stepping" << endl;
            cout << "L2Norm(un)    == " << L2Norm(fields[0]) << endl;
            cout << "divNorm(un)   == " << divNorm(fields[0]) << endl;
            cout << "L2Norm(qn)    == " << L2Norm(fields[1]) << endl;
            cout << "........................................................" << endl;
            cout << endl;
        }

        ofstream ns;
        ofstream uns;
        ofstream uts;
        ofstream ts;

        if (save && taskid == 0) {
            ns.open("unorms.asc");
            uts.open("ut.asc");
            uns.open("un.asc");
            ts.open("t.asc");

            ns << setprecision(14);
            uts << setprecision(14);
            uts << "% true solution ut(y,t), (y,t) == (columns, rows)" << endl;
            uns << setprecision(14);
            uns << "% numerical solution un(y,t), (y,t) == (columns, rows)" << endl;
            ts << "% time" << endl;
            y.save("y");
        }

        for (Real t = T0; t < T1; t += n * dt) {
            Real l2dist = L2Dist(fields[0], ut);
            Real CFL = dns.CFL(fields[0]);
            Real l2un = L2Norm(fields[0]);
            Real l2ut = L2Norm(ut);

            if (taskid == 0)
                cout << "t == " << t << endl;
            if (taskid == 0)
                cout << "CFL == " << CFL << endl;
            if (taskid == 0)
                cout << "L2Norm(un)    == " << l2un << endl;
            if (taskid == 0)
                cout << "L2Norm(ut)    == " << l2ut << endl;
            if (taskid == 0)
                cout << "L2Dist(un,ut) == " << l2dist << endl;
            if (taskid == 0)
                cout << "dPdx  == " << dns.dPdx() << endl;
            if (taskid == 0)
                cout << "Ubulk == " << dns.Ubulk() << endl;
            err += l2dist;

            ns << L2Dist(fields[0], ut) << ' ' << fabs(dPdx - dns.dPdx()) << ' ' << fabs(Ubulk - dns.Ubulk()) << ' '
               << divNorm(fields[0]) << ' ' << bcNorm(fields[0]) << '\n';

            if (save) {
                ChebyCoeff un00 = Re(fields[0].profile(0, 0, 0));
                ChebyCoeff ut00 = Re(ut.profile(0, 0, 0));

                un00.makePhysical(trans);
                ut00.makePhysical(trans);

                for (int ny = 0; ny < Ny; ++ny)
                    uns << un00(ny) << ' ';
                uns << '\n';
                for (int ny = 0; ny < Ny; ++ny)
                    uts << ut00(ny) << ' ';
                uts << '\n';

                ts << t << '\n';
            }

            if (CFL > 2.0 || l2un > 1) {
                cout << "Problem!" << endl;
                cout << "CFL  == " << CFL << endl;
                cout << "norm == " << l2un << endl;
                cout << "\t** FAIL **" << endl;
                cerr << "\t** FAIL **" << endl;
                return 1;
            }
            if (t + n * dt < T1)
                dns.advance(fields, n);
        }
        if (taskid == 0)
            cout << argv[0] << s << "FinalError == " << err << " < " << maxerr << '\t' << flags << endl;
        if (err < maxerr) {
            if (taskid == 0)
                cerr << "\t\t   pass   " << endl;
            if (taskid == 0)
                cout << "\t\t   pass   " << endl;
            exitcode = 0;
        } else {
            if (taskid == 0)
                cerr << "\t** FAIL **" << endl;
            if (taskid == 0)
                cout << "\t** FAIL **" << endl;
            if (taskid == 0)
                cout << "   err == " << err << endl;
            if (taskid == 0)
                cout << "maxerr == " << maxerr << endl;
            exitcode = 1;
        }
#ifdef HAVE_MPI
    }
    cfMPI_Finalize();
#endif
    return exitcode;
}
