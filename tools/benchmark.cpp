/**
 * This file is a part of channelflow version 2.0.
 * License is GNU GPL version 2 or later: https://channelflow.org/license
 */
#include <iostream>
#include "channelflow/dns.h"
#include "channelflow/flowfield.h"
#include "channelflow/utilfuncs.h"
// #include <complex.h>
#include <sys/time.h>
#include <limits>
// typedef ptrdiff_t lint;
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
        bool saveresult = args.getflag("-s", "--saveresult", "save resulting field to uresult.h5");
        bool nocomparison = args.getflag("-nc", "--nocompare", "don't compare integrated field to ufinal.h5");
        string dir = args.getstr("-d", "--directory", ".", "directory where fields are found and stored");
        args.check();

        if (taskid == 0)
            cout << "Creating CfMPI object..." << flush;
        CfMPI* cfmpi = &CfMPI::getInstance(nproc0, nproc1);
        if (taskid == 0)
            cout << "done" << endl;

        if (taskid == 0)
            cout << "Loading FlowField..." << endl;
        FlowField u(dir + "/uinit", cfmpi);
        if (taskid == 0)
            cout << "done" << endl;

        if (u.taskid() == 0)
            cout << "Calculating l2norm..." << flush;
        Real l2n = L2Norm(u);
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
        l2n = L2Norm(u);
        if (u.taskid() == 0)
            cout << " L2Norm(u) == " << l2n << endl;

        // Define integration parameters
        const int n = 40;         // take n steps between printouts
        const Real dt = 1.0 / n;  // integration timestep

        // Define DNS parameters
        DNSFlags flags;
        //         flags.baseflow     = LaminarBase;
        flags.baseflow = SuctionBase;
        flags.timestepping = SBDF3;
        flags.initstepping = SMRK2;
        flags.nonlinearity = Rotational;
        flags.dealiasing = DealiasXZ;
        flags.taucorrection = true;
        flags.constraint = PressureGradient;  // enforce constant pressure gradient
        flags.dPdx = 0;
        flags.dt = dt;
        flags.nu = 1. / 400;
        flags.Vsuck = 1. / 400;

        if (u.taskid() == 0)
            cout << "Building FlowField q..." << flush;
        vector<FlowField> fields = {
            u, FlowField(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi(), Spectral, Spectral)};
        if (u.taskid() == 0)
            cout << "done" << endl;
        if (u.taskid() == 0)
            cout << "Building dns..." << flush;
        DNS dns(fields, flags);
        if (u.taskid() == 0)
            cout << "done" << endl;

        Real avtime = 0;
        int i = 0;
        Real T = 10;
        for (Real t = 0; t <= T; t += 1) {
            timeval start, end;
            gettimeofday(&start, 0);
            Real cfl = dns.CFL(fields[0]);
            if (fields[0].taskid() == 0)
                cout << "         t == " << t << endl;
            if (fields[0].taskid() == 0)
                cout << "       CFL == " << cfl << endl;
            Real l2n = L2Norm(fields[0]);
            if (fields[0].taskid() == 0)
                cout << " L2Norm(u) == " << l2n << endl;

            // Take n steps of length dt
            dns.advance(fields, n);

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

        if (!nocomparison) {
            FlowField v(dir + "/ufinal", cfmpi);
            Real l2d = L2Dist(v, fields[0]);
            if (fields[0].taskid() == 0)
                cout << endl << "Final L2Dist: " << l2d << endl;
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
            fields[0].save(dir + "/ufinal");
    }
    cfMPI_Finalize();
    return 0;
}
