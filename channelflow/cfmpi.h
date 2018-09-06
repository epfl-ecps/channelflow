/**
 * Store information about MPI process distribution
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#ifndef CFMPI_H
#define CFMPI_H

#include <vector>
#include "cfbasics/mathdefs.h"

#ifdef HAVE_MPI
#include <fftw3-mpi.h>
#include <mpi.h>
#endif

#ifdef DEBUG
#ifdef HAVE_MPI
#define debug(msg)                            \
    MPI_Barrier(MPI_COMM_WORLD);              \
    cerr << mpirank() << ": " << msg << endl; \
    MPI_Barrier(MPI_COMM_WORLD);
#else
#define debug(msg) cerr << msg << endl;
#endif
#else
#define debug(msg)
#endif

namespace chflow {

// Define a couple of global functions that deal with MPI specific stuff
int cfMPI_Init(int* argc, char*** argv);  // Replaces MPI_Init and mutes output on ranks other than 0
int cfMPI_Finalize();

class CfMPI {
    // Class is implemented as a singleton to avoid enforcing its creation if HAVE_MPI=True.
    // This way, it is possible to create it somewhere, like in a FlowField constructor.
    // The object is created once on first demand of instance and destroyed automatically.
   protected:
    // 		CfMPI();
    CfMPI(int nproc0, int nproc);
    // 		~CfMPI();			  // not implemented
    CfMPI(const CfMPI&);             // not implemented
    CfMPI& operator=(const CfMPI&);  // not implemented
   public:
    // Instantiated on first use. Guaranteed to be destroyed.
    static CfMPI& getInstance(int nproc0 = 0, int nproc = 0) {
        static CfMPI instance(nproc0, nproc);
        return instance;
    }
    // Communicator comm0 is used for the xy-transpose operation
    // Processes in comm0 do independent ffts
    // Communicator comm1 is used for the fftw-mpi of one xz-slice
    MPI_Comm comm0;
    MPI_Comm comm1;
    MPI_Comm comm_world;

    inline int nproc0() const;
    inline int nproc1() const;
    inline int taskid() const;
    inline int taskid_world() const;
    // 		inline int taskid_in_world(int taskid) const;
    inline int numtasks() const;
    inline int color0() const;
    inline int key0() const;
    inline int color1() const;
    inline int key1() const;

    // 		inline bool usempi() const;
    int usempi_;

   private:
    int nproc0_;
    int nproc1_;
    int taskid_;
    int taskid_world_;
    int numtasks_;
    int color0_;
    int key0_;
    int color1_;
    int key1_;

    int objcount_;
    // 		std::vector<int> taskid_in_world_;

    void Init(int nproc0, int nproc1);
};

inline int CfMPI::nproc0() const { return nproc0_; }
inline int CfMPI::nproc1() const { return nproc1_; }
inline int CfMPI::taskid() const { return taskid_; }
inline int CfMPI::taskid_world() const { return taskid_world_; }

inline int CfMPI::numtasks() const { return numtasks_; }
inline int CfMPI::color0() const { return color0_; }
inline int CfMPI::key0() const { return key0_; }
inline int CfMPI::color1() const { return color1_; }
inline int CfMPI::key1() const { return key1_; }

class CfMPI_single : public CfMPI {
    //  Derived class from CfMPI singleton.
    //  Can be used for creating non-distributed data
    //  when running with MPI.
    //  Actually needed as quick fix allowing IO with
    //  legacy formats HDF5 and ff with MPI.
   private:
    using CfMPI::CfMPI;

   public:
    // Instantiated on first use. Guaranteed to be destroyed.
    static CfMPI_single& getInstance() {
        static CfMPI_single instance(1, 1);
        return instance;
    }
};

}  // namespace chflow

#endif  // CFMPI_H
