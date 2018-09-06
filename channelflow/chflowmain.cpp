/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <complex.h>
#include <iostream>
#include "channelflow/dns.h"
#include "channelflow/flowfield.h"
#include "channelflow/utilfuncs.h"

#ifdef HAVE_MPI
#include <fftw3-mpi.h>
#include <mpi.h>
#else
#include <fftw3.h>
#endif

#include <sys/time.h>
#include <limits>

using namespace std;
using namespace chflow;

int main(int argc, char* argv[]) {
    cfMPI_Init(&argc, &argv);
    {
        int taskid;
#ifdef HAVE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
#endif

        CfMPI* cfmpi = new CfMPI();

        FlowField u("uinit", cfmpi);
        int numtasks = u.numtasks();

        FlowField v(u.Nx(), u.Ny(), u.Nz(), u.Nd(), u.Lx(), u.Lz(), u.a(), u.b(), NULL);
        FlowField w(u.Nx(), u.Ny(), u.Nz(), u.Nd(), u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());

        u.toprocess0(v);
        w.fromprocess0(v);

        cout << taskid << ": " << L2Norm(u) << endl;
        cout << taskid << ": " << L2Norm(v) << endl;
        cout << taskid << ": " << L2Norm(w) << endl;
        delete cfmpi;
    }
    cfMPI_Finalize();
    return 0;
}
