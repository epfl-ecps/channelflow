/**
 * Store information about MPI process distribution
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "cfmpi.h"
#include "cfbasics/cfbasics.h"

namespace chflow {
int cfMPI_Init(int* argc, char*** argv) {
    int res = 0;

#ifdef HAVE_MPI
    res = MPI_Init(argc, argv);
    int nproc = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // Redirect cout and cerr to "/dev/null" if not on rank 0
    // This is maybe ugly, but efficient
    // Be careful, this opens an ofstream on the heap and the pointer to it is destroyed
    // at return, i.e. a memory leak if this function is called more than once
    int rank = mpirank();
    if (rank != 0) {
        // Open ofstream on heap to preserve it after return
        // TODO "nul" for windows compatibility
        std::ofstream* sink = new std::ofstream("/dev/null");

        // Mute standard output
        std::cout.rdbuf(sink->rdbuf());
        // Optionally mute standard error
        std::cerr.rdbuf(sink->rdbuf());
    }
#endif
    return res;
}

int cfMPI_Finalize() {
#ifdef HAVE_MPI
    return MPI_Finalize();
#else
    return 0;
#endif
}

CfMPI::CfMPI(int nproc0, int nproc1)
    : comm0(0),
      comm1(0),
      comm_world(0),
      nproc0_(nproc0),
      nproc1_(nproc1),
      taskid_(0),
      taskid_world_(0),
      numtasks_(1),
      color0_(0),
      key0_(0),
      color1_(0),
      key1_(0) {
    Init(nproc0, nproc1);
}

void CfMPI::Init(int nproc0, int nproc1) {
#ifdef HAVE_MPI
    static int count = 0;
    count++;
    objcount_ = count;
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        // Until now, CfMPI uses MPI_COMM_WORLD as default
        // This could be replaced by a user-defined communicator
        MPI_Comm_rank(MPI_COMM_WORLD, &taskid_world_);
        MPI_Comm_dup(MPI_COMM_WORLD, &comm_world);

        MPI_Comm_size(comm_world, &numtasks_);
        // old definition was "usempi_=numtasks_ > 1;". To be consistent with the reset of the code, usempi_=HAVE_MPI
        usempi_ = true;
        MPI_Comm_rank(comm_world, &taskid_);

        if (nproc0 == 0 && nproc1 == 0) {
            int nproc = sqrt(numtasks_);
            if (nproc * nproc != numtasks_) {
                nproc0_ = 1;
                nproc1_ = numtasks_;
            } else {
                // cferror ("CfMPI: number of MPI-processes is not a square number and no distribution has been
                // specified");
                nproc0_ = nproc;
                nproc1_ = nproc;
            }
        } else if (nproc0 * nproc1 == 0) {
            cferror("CfMPI: nproc0 or nproc1 is zero");
        } else if (nproc0 * nproc1 > numtasks_) {
            cferror("CfMPI: number of MPI-processes < nproc0*nproc1");
        } else {
            if (nproc0 * nproc1 != numtasks_) {
            }
            nproc0_ = nproc0;
            nproc1_ = nproc1;
        }
        numtasks_ = nproc0_ * nproc1_;

        color0_ = taskid_ % nproc1_;
        key0_ = taskid_ / nproc1_;
        color1_ = taskid_ / nproc1_;  // Yes, color1=key0, I know, but having two variables helps me
        key1_ = taskid_ % nproc1_;

        MPI_Comm_split(comm_world, color0_, key0_, &comm0);
        //    if (taskid_ == 0) {
        //      std::cout << "                     res  " << res << std::endl;
        //      std::cout << std::endl;
        //    }
        MPI_Comm_split(comm_world, color1_, key1_, &comm1);

        fftw_mpi_init();  // This needs to be called once, but additional times aren't harmful
    } else {              // MPI_Initialized()
        nproc0_ = 1;
        nproc1_ = 1;
        numtasks_ = 1;
        usempi_ = false;
    }
#else
    usempi_ = false;
    nproc0_ = 1;
    nproc1_ = 1;
    numtasks_ = 1;
#endif
}

}  // namespace chflow
