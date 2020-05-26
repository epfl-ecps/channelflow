/**
 * This time integration test ensures compatibility to previous versions.
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 *
 * Original author: Florian Reetz
 */

#include <iostream>
#include "channelflow/flowfield.h"
#include "modules/ilc/ilc.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "channelflow/cfmpi.h"

using namespace std;
using namespace chflow;

int main(int argc, char* argv[]) {
    int failure = 0;
    bool verbose = true;
    cfMPI_Init(&argc, &argv);
    {
        int taskid = mpirank();
        if (taskid == 0) {
            cerr << "time_integrationTest: " << flush;
            if (verbose) {
                cout << "\n====================================================" << endl;
                cout << "time_integrationTest\n\n";
            }
        }

        string purpose(
            "This program loads initial conditions from the harddisk and integrates\nit for 20 time units. The "
            "resulting fields are compared to the final fields \nufinal and tfinal. L2diff decides on success of "
            "test.");
        ArgList args(argc, argv, purpose);
        int nproc0 = args.getint("-np0", "-nproc0", 0, "number of processes for transpose/parallel ffts");
        int nproc1 = args.getint("-np1", "-nproc1", 0, "number of processes for slice fft");
        bool fftwmeasure = args.getflag("-fftwmeasure", "--fftwmeasure", "use fftw_measure instead of fftw_patient");
        Real tol = args.getreal("-t", "--tolerance", 1.0e-14, "max distance allowed for the test to pass");
        string dir = "data";
        args.check();

        if (taskid == 0 && verbose)
            cout << "Creating CfMPI object..." << flush;
        CfMPI* cfmpi = &CfMPI::getInstance(nproc0, nproc1);
        if (taskid == 0 && verbose)
            cout << "done" << endl;

        if (taskid == 0 && verbose)
            cout << "Loading FlowFields..." << endl;
        FlowField u(dir + "/uinit", cfmpi);
        FlowField temp(dir + "/tinit", cfmpi);
        if (taskid == 0 && verbose)
            cout << "done" << endl;

        if (u.taskid() == 0 && verbose) {
            cout << "================================================================\n";
            cout << purpose << endl << endl;
            cout << "Distribution of processes is " << u.nproc0() << "x" << u.nproc1() << endl;
        }
        FlowField utmp(u);
        if (fftwmeasure)
            utmp.optimizeFFTW(FFTW_MEASURE);
        else
            utmp.optimizeFFTW(FFTW_PATIENT);
        fftw_savewisdom();

        // Define integration parameters
        const int n = 40;         // take n steps between printouts
        const Real dt = 1.0 / n;  // integration timestep

        // Define DNS parameters
        ILCFlags flags;
        flags.baseflow =
            ZeroBase;  // should be Laminar base for a better test, but this test was callibrated with ZeroBase

        // Run at default flags. If you change them, recreate the test files.
        flags.dt = dt;
        flags.verbosity = Silent;

        if (u.taskid() == 0 && verbose)
            cout << "Building FlowField q..." << flush;
        vector<FlowField> fields = {
            u, temp, FlowField(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi(), Spectral, Spectral)};
        if (u.taskid() == 0 && verbose)
            cout << "done" << endl;
        if (u.taskid() == 0 && verbose)
            cout << "Building dns..." << flush;
        ILC ilc(fields, flags);
        if (u.taskid() == 0 && verbose)
            cout << "done" << endl;

        Real avtime = 0;
        int i = 0;
        Real T = 20;
        for (Real t = 0; t <= T; t += 1) {
            timeval start, end;
            gettimeofday(&start, 0);
            Real cfl = ilc.CFL(fields[0]);
            if (fields[0].taskid() == 0 && verbose)
                cout << "         t == " << t << endl;
            if (fields[0].taskid() == 0 && verbose)
                cout << "       CFL == " << cfl << endl;
            Real l2n = L2Norm(fields[0]);
            if (fields[0].taskid() == 0 && verbose)
                cout << " L2Norm(u) == " << l2n << endl;

            // Take n steps of length dt
            ilc.advance(fields, n);
            if (verbose) {
                gettimeofday(&end, 0);
                Real sec = (Real)(end.tv_sec - start.tv_sec);
                Real ms = (((Real)end.tv_usec) - ((Real)start.tv_usec));
                Real timeused = sec + ms / 1000000.;
                if (fields[0].taskid() == 0)
                    cout << "duration for this timeunit: " << timeused << endl;
                if (t != 0) {
                    avtime += timeused;
                    i++;
                }
                if (fields[0].taskid() == 0)
                    cout << endl;
            }
        }

        FlowField uf(dir + "/ufinal", cfmpi);
        FlowField tf(dir + "/tfinal", cfmpi);
        Real l2d = L2Dist(uf, fields[0]) + L2Dist(tf, fields[1]);
        if (l2d > tol) {
            if (fields[0].taskid() == 0) {
                if (verbose)
                    cout << endl << "Final L2Dist: " << l2d << endl;
                cerr << "\t** FAIL **" << endl;
                cout << "\t** FAIL **" << endl;
            }
            failure = 1;
        } else {
            if (fields[0].taskid() == 0) {
                if (verbose)
                    cout << endl << "Final L2Dist: " << l2d << endl;
                cerr << "\t   pass   " << endl;
                cout << "\t   pass   " << endl;
            }
        }
        if (fields[0].taskid() == 0 && verbose) {
            cout << "Average time/timeunit: " << avtime / i << "s" << endl;
        }
    }
    cfMPI_Finalize();
    return failure;
}
