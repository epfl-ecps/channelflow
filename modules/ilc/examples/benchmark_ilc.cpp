/**
 * The classic benchmark program, like for pure shear flow
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
    cfMPI_Init(&argc, &argv);
    {
        int taskid = 0;
#ifdef HAVE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
#endif
        if (taskid == 0)
            cout << "Starting channelflow benchmark" << endl;

        string purpose(
            "This program loads the field uinit from the harddisk and integrates\nit for 10 time units. The resulting "
            "field is compared to the field\nufinal. Wall-clock time elapsed for each timeunit as well as an\naverage "
            "are calculated.");
        ArgList args(argc, argv, purpose);
        int nproc0 = args.getint("-np0", "-nproc0", 0, "number of processes for transpose/parallel ffts");
        int nproc1 = args.getint("-np1", "-nproc1", 0, "number of processes for slice fft");
        bool fftwmeasure = args.getflag("-fftwmeasure", "--fftwmeasure", "use fftw_measure instead of fftw_patient");
        bool fftwwisdom = args.getflag("-fftwwisdom", "--fftwwisdom", "try loading fftw wisdom");
        bool saveresult = args.getflag("-s", "--saveresult", "save resulting field to uresult.nc");
        string dir = args.getstr("-d", "--directory", ".", "directory where fields are found and stored");
        args.check();

        if (taskid == 0)
            cout << "Creating CfMPI object..." << flush;
        CfMPI* cfmpi = &CfMPI::getInstance(nproc0, nproc1);
        if (taskid == 0)
            cout << "done" << endl;

        if (taskid == 0)
            cout << "Loading FlowFields..." << endl;
        FlowField u(dir + "/uinit", cfmpi);
        FlowField temp(dir + "/tinit", cfmpi);
        if (taskid == 0)
            cout << "done" << endl;

        if (u.taskid() == 0)
            cout << "Calculating l2norm..." << flush;
        Real ul2n = L2Norm(u);
        Real tl2n = L2Norm(temp);
        if (taskid == 0)
            cout << "done" << endl;

        if (u.taskid() == 0) {
            cout << "================================================================\n";
            cout << purpose << endl << endl;
            cout << "Distribution of processes is " << u.nproc0() << "x" << u.nproc1() << endl;
        }
        if (fftwwisdom && u.taskid() == 0) {
            cout << "Loading fftw wisdom" << endl;
            fftw_loadwisdom();
        }
        if (u.taskid() == 0)
            cout << "Optimizing FFTW..." << flush;
        {
            FlowField utmp(u);
            if (fftwmeasure)
                utmp.optimizeFFTW(FFTW_MEASURE);
            else
                utmp.optimizeFFTW(FFTW_PATIENT);
            fftw_savewisdom();
        }
        if (u.taskid() == 0)
            cout << "done" << endl;
        ul2n = L2Norm(u);
        tl2n = L2Norm(temp);
        if (u.taskid() == 0)
            cout << " L2Norm(u) == " << ul2n << endl;
        if (u.taskid() == 0)
            cout << " L2Norm(T) == " << tl2n << endl;

        // Define integration parameters
        const int n = 40;         // take n steps between printouts
        const Real dt = 1.0 / n;  // integration timestep

        // Define DNS parameters
        ILCFlags flags;

        // Run at default flags. If you change them, recreate the test files.
        flags.dt = dt;
        flags.verbosity = Silent;

        if (u.taskid() == 0)
            cout << "Building FlowField q..." << flush;
        vector<FlowField> fields = {
            u, temp, FlowField(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi(), Spectral, Spectral)};
        if (u.taskid() == 0)
            cout << "done" << endl;
        if (u.taskid() == 0)
            cout << "Building dns..." << flush;
        ILC ilc(fields, flags);
        if (u.taskid() == 0)
            cout << "done" << endl;

        Real avtime = 0;
        int i = 0;
        Real T = 10;
        for (Real t = 0; t <= T; t += 1) {
            timeval start, end;
            gettimeofday(&start, 0);
            Real cfl = ilc.CFL(fields[0]);
            if (fields[0].taskid() == 0)
                cout << "         t == " << t << endl;
            if (fields[0].taskid() == 0)
                cout << "       CFL == " << cfl << endl;
            Real ul2n = L2Norm(fields[0]);
            Real tl2n = L2Norm(fields[1]);
            if (fields[0].taskid() == 0)
                cout << " L2Norm(u) == " << ul2n << endl;
            if (fields[0].taskid() == 0)
                cout << " L2Norm(T) == " << tl2n << endl;

            // Take n steps of length dt
            ilc.advance(fields, n);

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

        if (fields[0].taskid() == 0) {
            cout << "Average time/timeunit: " << avtime / i << "s" << endl;
            ofstream fout("benchmark_results", ios::app);
            fout << "np0 x np1 == " << cfmpi->nproc0() << " x " << cfmpi->nproc1() << endl;
            fout << "fftw_flag == " << (fftwmeasure ? "fftw_measure" : "fftw_patient") << endl;
            fout << "Average time/timeunit: " << avtime / i << "s" << endl << endl;
            fout.close();
        }
        // fftw_mpi_gather_wisdom(MPI_COMM_WORLD);
        fftw_savewisdom();
        if (saveresult)
            fields[0].save(dir + "/uresult.nc");
    }
    cfMPI_Finalize();
    return 0;
}
