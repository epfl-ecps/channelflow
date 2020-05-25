/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "channelflow/flowfield.h"
#include "cfbasics/mathdefs.h"
#include "channelflow/cfmpi.h"
#include "channelflow/diffops.h"
#include "channelflow/symmetry.h"
#include "channelflow/utilfuncs.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif

// HAVE_LIBHDF5_CPP is determined by the autoconf system and
// defined in ../config.h when HDF5 is present and left undefined
// when it isn't. Here we define HAVE_HDF5 to 0 or 1 so that it
// can be used in if (HAVE_HDF5) statements later on.

#ifdef HAVE_LIBHDF5_CPP
#include "H5Cpp.h"
#define HAVE_HDF5 1
#else
#define HAVE_HDF5 0
#endif

// Decide if and how (if parallel) NetCDF may be used
#ifdef HAVE_NETCDF_PAR_H
#include <netcdf_par.h>
#define HAVE_NETCDF_PAR 1
#else
#define HAVE_NETCDF_PAR 0
#endif

#ifdef HAVE_NETCDF_H
#include <netcdf.h>
#define HAVE_NETCDF 1
#else
#define HAVE_NETCDF 0
#endif

#include <cstddef>  // for strtok
#include <cstdlib>
#include <cstring>  // for strdupa
#include <fstream>
#include <iomanip>
#include <vector>

using namespace std;
using namespace Eigen;

namespace chflow {

#ifdef HAVE_LIBHDF5_CPP
void hdf5write(int i, const string& name, H5::H5File& h5file);
void hdf5write(Real x, const string& name, H5::H5File& h5file);
void hdf5write(const Vector& v, const string& name, H5::H5File& h5file);
void hdf5write(const FlowField& u, const string& name, H5::H5File& h5file);
bool hdf5query(const string& name, H5::H5File& h5file);  // does attribute exist?
void hdf5read(int& i, const string& name, H5::H5File& h5file);
void hdf5read(Real& x, const string& name, H5::H5File& h5file);
void hdf5read(FlowField& u, const string& name, H5::H5File& h5file);
#endif

void writefloat(std::ofstream& os, float z, bool SwapEndian = true);

FlowField::FlowField() {
#ifdef HAVE_MPI
    cfmpi_ = &CfMPI::getInstance();
#endif
}

FlowField::FlowField(int Nx, int Ny, int Nz, int Nd, Real Lx, Real Lz, Real a, Real b, CfMPI* cfmpi, fieldstate xzstate,
                     fieldstate ystate, uint fftw_flags)
    : xzstate_(xzstate), ystate_(ystate) {
#ifdef HAVE_MPI
    if (cfmpi == nullptr)
        cfmpi = &CfMPI::getInstance();
#endif
    cfmpi_ = cfmpi;
    resize(Nx, Ny, Nz, Nd, Lx, Lz, a, b, cfmpi, fftw_flags);
}

FlowField::FlowField(int Nx, int Ny, int Nz, int Nd, int tensorOrder, Real Lx, Real Lz, Real a, Real b, CfMPI* cfmpi,
                     fieldstate xzstate, fieldstate ystate, uint fftw_flags)
    : Nd_(intpow(Nd, tensorOrder)), xzstate_(xzstate), ystate_(ystate) {
#ifdef HAVE_MPI
    if (cfmpi == nullptr)
        cfmpi = &CfMPI::getInstance();
#endif
    cfmpi_ = cfmpi;
    resize(Nx, Ny, Nz, Nd_, Lx, Lz, a, b, cfmpi, fftw_flags);
}

FlowField::FlowField(const FlowField& f) : padded_(f.padded_), cfmpi_(f.cfmpi_) {
    resize(f.Nx_, f.Ny_, f.Nz_, f.Nd_, f.Lx_, f.Lz_, f.a_, f.b_, f.cfmpi_);

    setState(f.xzstate_, f.ystate_);
    padded_ = f.padded_;
    copy(f.rdata_, f.rdata_ + Nloc_, rdata_);
}

FlowField::FlowField(const string& filebase, CfMPI* cfmpi) {
#ifdef HAVE_MPI
    if (cfmpi == nullptr)
        cfmpi = &CfMPI::getInstance();
#endif
    cfmpi_ = cfmpi;
    // Cases, in order of precedence:
    //    -------conditions---------
    //    filebase exists   HDF5libs   read
    // 1  foo.h5   foo.h5   yes        foo.h5
    // 2  foo      foo.h5   yes        foo.h5
    // 3  foo.ff   foo.ff   either     foo.ff
    // 4  foo      foo.ff   either     foo.ff
    // else fail

    string ncname = appendSuffix(filebase, ".nc");
    string h5name = appendSuffix(filebase, ".h5");
    string ffname = appendSuffix(filebase, ".ff");
    string filename;

    if (HAVE_NETCDF && isReadable(ncname))
        filename = ncname;

    else if (HAVE_HDF5 && isReadable(h5name))  // cases 1 and 2
        filename = h5name;

    else if (isReadable(ffname))  // cases 3 and 4
        filename = ffname;

    else {
        stringstream serr;
        serr << "error in FlowField(string filebase) : can't open filebase==" << filebase << endl;
        if (HAVE_HDF5)
            serr << "neither " << h5name << " nor " << ffname << " is readable" << endl;
        else
            serr << ffname << ".ff is unreadable" << endl;
        cferror(serr.str());
    }

    // At this point, filename should have an .h5 or .ff extension, and that
    // file should be readable. Further, if it has .h5, the HDF5 libs are installed.

    if (hasSuffix(filename, ".nc")) {
        readNetCDF(filename);

    } else if (hasSuffix(filename, ".h5")) {
#ifndef HAVE_LIBHDF5_CPP
        cferror("error in FlowField(string filebase). This line should be unreachable");
#else
        using namespace H5;
        H5File h5file;

        auto Nx = 0;
        auto Ny = 0;
        auto Nz = 0;

        if (taskid() == 0) {
            h5file = H5File(filename.c_str(), H5F_ACC_RDONLY);
            hdf5read(Nx_, "Nxpad", h5file);
            hdf5read(Ny_, "Nypad", h5file);
            hdf5read(Nz_, "Nzpad", h5file);
            hdf5read(Nd_, "Nd", h5file);
            hdf5read(Nx, "Nx", h5file);
            hdf5read(Ny, "Ny", h5file);
            hdf5read(Nz, "Nz", h5file);
            hdf5read(Lx_, "Lx", h5file);
            hdf5read(Lz_, "Lz", h5file);
            hdf5read(a_, "a", h5file);
            hdf5read(b_, "b", h5file);

            if (Ny != Ny_)
                cferror("error in FlowField(string h5filename) : Ny != Nypad in h5 file");
        }

#ifdef HAVE_MPI
        MPI_Bcast(&Nx_, 1, MPI_INT, 0, cfmpi_->comm_world);
        MPI_Bcast(&Ny_, 1, MPI_INT, 0, cfmpi_->comm_world);
        MPI_Bcast(&Nz_, 1, MPI_INT, 0, cfmpi_->comm_world);
        MPI_Bcast(&Nd_, 1, MPI_INT, 0, cfmpi_->comm_world);
        MPI_Bcast(&Nx, 1, MPI_INT, 0, cfmpi_->comm_world);
        MPI_Bcast(&Ny, 1, MPI_INT, 0, cfmpi_->comm_world);
        MPI_Bcast(&Nz, 1, MPI_INT, 0, cfmpi_->comm_world);
        MPI_Bcast(&Lx_, 1, MPI_DOUBLE, 0, cfmpi_->comm_world);
        MPI_Bcast(&Lz_, 1, MPI_DOUBLE, 0, cfmpi_->comm_world);
        MPI_Bcast(&a_, 1, MPI_DOUBLE, 0, cfmpi_->comm_world);
        MPI_Bcast(&b_, 1, MPI_DOUBLE, 0, cfmpi_->comm_world);
#endif

        resize(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, a_, b_, cfmpi_);

        if (Nx == Nx_ && Ny == Ny_ && Nz == Nz_) {  // means that FlowField isn't padded
#ifdef HAVE_MPI
            CfMPI_single* CfMPI_one = &CfMPI_single::getInstance();
            FlowField v(Nx, Ny, Nz, Nd_, Lx_, Lz_, a_, b_, CfMPI_one, Physical, Physical);
            hdf5read(v, "data/u", h5file);
            this->interpolate(v);
#else
            hdf5read(*this, "data/u", h5file);
#endif
            this->setPadded(false);
        } else {  // padded FlowField
            CfMPI_single* CfMPI_one = nullptr;
#ifdef HAVE_MPI
            CfMPI_one = &CfMPI_single::getInstance();
#endif
            FlowField v(Nx, Ny, Nz, Nd_, Lx_, Lz_, a_, b_, CfMPI_one, Physical, Physical);
            hdf5read(v, "data/u", h5file);

            this->interpolate(v);
            this->setPadded(((2 * Nx_) / 3 >= Nx && (2 * Nz_) / 3 >= Nz) ? true : false);
        }
#endif  // HAVE_HDF5_LIB
    } else {
        ifstream is(filename.c_str());
        if (!is) {
            cferror("FlowField::Flowfield(filebase) : can't open file " + filename);
        }

        if (taskid() == 0) {
            int major_version, minor_version, update_version;
            read(is, major_version);
            read(is, minor_version);
            read(is, update_version);
            read(is, Nx_);
            read(is, Ny_);
            read(is, Nz_);
            read(is, Nd_);
            read(is, xzstate_);
            read(is, ystate_);
            read(is, Lx_);
            read(is, Lz_);
            read(is, a_);
            read(is, b_);
            read(is, padded_);
        }
#ifdef HAVE_MPI
        MPI_Bcast(&Nx_, 1, MPI_INT, 0, cfmpi_->comm_world);
        MPI_Bcast(&Ny_, 1, MPI_INT, 0, cfmpi_->comm_world);
        MPI_Bcast(&Nz_, 1, MPI_INT, 0, cfmpi_->comm_world);
        MPI_Bcast(&Nd_, 1, MPI_INT, 0, cfmpi_->comm_world);
        MPI_Bcast(&xzstate_, 1, MPI_INT, 0, cfmpi_->comm_world);
        MPI_Bcast(&ystate_, 1, MPI_INT, 0, cfmpi_->comm_world);
        MPI_Bcast(&Lx_, 1, MPI_DOUBLE, 0, cfmpi_->comm_world);
        MPI_Bcast(&Lz_, 1, MPI_DOUBLE, 0, cfmpi_->comm_world);
        MPI_Bcast(&a_, 1, MPI_DOUBLE, 0, cfmpi_->comm_world);
        MPI_Bcast(&b_, 1, MPI_DOUBLE, 0, cfmpi_->comm_world);
#endif
        resize(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, a_, b_, cfmpi_);
#ifdef HAVE_MPI
        CfMPI_single* CfMPI_one = &CfMPI_single::getInstance();
        FlowField v(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, a_, b_, CfMPI_one);
        v.setState(xzstate_, ystate_);

        if (taskid() == 0) {
            // Read data only for non-aliased modes, assume 0 for aliased.
            if (padded_ == true && xzstate_ == Spectral) {
                int Nxd = 2 * (Nx_ / 6);
                int Nzd = 2 * (Nz_ / 3) + 1;

                // In innermost loop, cfarray index is (nz + Nzpad2_*(nx + Nx_*(ny + Ny_*i))),
                // which is the same as the FlowField::flatten function.
                for (int i = 0; i < Nd_; ++i) {
                    for (int ny = 0; ny < Ny_; ++ny) {
                        for (int nx = 0; nx <= Nxd; ++nx) {
                            for (int nz2 = 0; nz2 <= Nzd / 2; ++nz2) {
                                Real re, im;
                                read(is, re);
                                read(is, im);
                                Complex comp = Complex(re, im);
                                v.cmplx(nx, ny, nz2, i) = comp;
                            }
                        }

                        for (int nx = Nx_ - Nxd; nx < Nx_; ++nx) {
                            for (int nz2 = 0; nz2 <= Nzd / 2; ++nz2) {
                                Real re, im;
                                read(is, re);
                                read(is, im);
                                Complex comp = Complex(re, im);
                                v.cmplx(nx, ny, nz2, i) = comp;
                            }
                        }
                    }
                }

            } else if (padded_ == false && xzstate_ == Spectral) {
                for (int i = 0; i < Nd_; ++i) {
                    for (int ny = 0; ny < Ny_; ++ny) {
                        for (int nx = 0; nx < Nx_; ++nx) {
                            for (int nz2 = 0; nz2 < Nzpad2_; ++nz2) {
                                Real re, im;
                                read(is, re);
                                read(is, im);
                                Complex comp = Complex(re, im);
                                v.cmplx(nx, ny, nz2, i) = comp;
                            }
                        }
                    }
                }

            } else {
                for (int i = 0; i < Nd_; ++i) {
                    for (int ny = 0; ny < Ny_; ++ny) {
                        for (int nx = 0; nx < Nx_; ++nx) {
                            for (int nz = 0; nz < Nzpad_; ++nz) {
                                Real re;
                                read(is, re);
                                v(nx, ny, nz, i) = re;
                            }
                        }
                    }
                }
            }
        }
        this->interpolate(v);

#else
        // Read data only for non-aliased modes, assume 0 for aliased.
        if (padded_ == true && xzstate_ == Spectral) {
            int Nxd = 2 * (Nx_ / 6);
            int Nzd = 2 * (Nz_ / 3) + 1;

            // In innermost loop, cfarray index is (nz + Nzpad2_*(nx + Nx_*(ny + Ny_*i))),
            // which is the same as the FlowField::flatten function.
            for (int i = 0; i < Nd_; ++i) {
                for (int ny = 0; ny < Ny_; ++ny) {
                    for (int nx = 0; nx <= Nxd; ++nx) {
                        for (int nz = 0; nz <= Nzd; ++nz)
                            read(is, rdata_[flatten(nx, ny, nz, i)]);
                        for (int nz = Nzd + 1; nz < Nzpad_; ++nz)
                            rdata_[flatten(nx, ny, nz, i)] = 0.0;
                    }
                    for (int nx = Nxd + 1; nx <= Nxd; ++nx)
                        for (int nz = 0; nz <= Nzpad_; ++nz)
                            rdata_[flatten(nx, ny, nz, i)] = 0.0;

                    for (int nx = Nx_ - Nxd; nx < Nx_; ++nx) {
                        for (int nz = 0; nz <= Nzd; ++nz)
                            read(is, rdata_[flatten(nx, ny, nz, i)]);
                        for (int nz = Nzd + 1; nz < Nzpad_; ++nz)
                            rdata_[flatten(nx, ny, nz, i)] = 0.0;
                    }
                }
            }
        } else {
            int N = Nd_ * Ny_ * Nx_ * Nzpad_;
            for (int i = 0; i < N; ++i)
                read(is, rdata_[i]);
        }

#endif
        makeSpectral();
    }
}

FlowField::FlowField(const string& filebase, int major, int minor, int update) {
    if (mpirank() > 0)
        cferror("Loading flowfield from .ff file is not mpi safe");
    ifstream is;
    string filename = ifstreamOpen(is, filebase, ".ff", ios::in | ios::binary);
    if (!is) {
        cferror("FlowField::FlowField(filebase, major,minor,update) : can't open file " + filename + " or " + filebase);
    }

    if (major > 0 || minor > 9) {
        stringstream serr;
        serr << "FlowField::FlowField(filebase, major,minor,update) :\n"
             << major << '.' << minor << '.' << update << " is higher than 0.9.x" << endl;
        cferror(serr.str());
    }

    if (update >= 16) {
        (*this) = FlowField(filebase);  // not efficient but saves code redundancy
        return;
    } else {
        is.read((char*)&Nx_, sizeof(int));
        is.read((char*)&Ny_, sizeof(int));
        is.read((char*)&Nz_, sizeof(int));
        is.read((char*)&Nd_, sizeof(int));
        is >> xzstate_;
        is >> ystate_;
        is.read((char*)&Lx_, sizeof(Real));
        is.read((char*)&Lz_, sizeof(Real));
        is.read((char*)&a_, sizeof(Real));
        is.read((char*)&b_, sizeof(Real));
        char s;
        is.get(s);
        padded_ = (s == '1') ? true : false;

        resize(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, a_, b_, cfmpi_);

        // Read data only for non-aliased modes, assume 0 for aliased.
        if (padded_ == true && xzstate_ == Spectral) {
            int Nxd = 2 * (Nx_ / 6);
            int Nzd = 2 * (Nz_ / 3) + 1;

            // In innermost loop, cfarray index is (nz + Nzpad2_*(nx + Nx_*(ny+Ny_*i)))
            // which is the same as the FlowField::flatten function.
            for (int i = 0; i < Nd_; ++i) {
                for (int ny = 0; ny < Ny_; ++ny) {
                    for (int nx = 0; nx <= Nxd; ++nx) {
                        for (int nz = 0; nz <= Nzd; ++nz)
                            is.read((char*)(rdata_ + flatten(nx, ny, nz, i)), sizeof(Real));
                        for (int nz = Nzd + 1; nz < Nzpad_; ++nz)
                            rdata_[flatten(nx, ny, nz, i)] = 0.0;
                    }
                    for (int nx = Nxd + 1; nx <= Nxd; ++nx)
                        for (int nz = 0; nz <= Nzpad_; ++nz)
                            rdata_[flatten(nx, ny, nz, i)] = 0.0;

                    for (int nx = Nx_ - Nxd; nx < Nx_; ++nx) {
                        for (int nz = 0; nz <= Nzd; ++nz)
                            is.read((char*)(rdata_ + flatten(nx, ny, nz, i)), sizeof(Real));
                        for (int nz = Nzd + 1; nz < Nzpad_; ++nz)
                            rdata_[flatten(nx, ny, nz, i)] = 0.0;
                    }
                }
            }
        } else {
            int N = Nd_ * Ny_ * Nx_ * Nzpad_;
            for (int i = 0; i < N; ++i)
                is.read((char*)(rdata_ + i), sizeof(Real));
        }
    }
}

Vector FlowField::xgridpts() const {
    Vector xpts(Nx_);
    for (int nx = 0; nx < Nx_; ++nx)
        xpts[nx] = x(nx);
    return xpts;
}
Vector FlowField::ygridpts() const {
    Vector ypts(Ny_);
    Real c = 0.5 * (b_ + a_);
    Real r = 0.5 * (b_ - a_);
    Real piN = pi / (Ny_ - 1);
    for (int ny = 0; ny < Ny_; ++ny)
        ypts[ny] = c + r * cos(piN * ny);
    return ypts;
}
Vector FlowField::zgridpts() const {
    Vector zpts(Nz_);
    for (int nz = 0; nz < Nz_; ++nz)
        zpts[nz] = z(nz);
    return zpts;
}

FlowField& FlowField::operator=(const FlowField& f) {
    cfmpi_ = f.cfmpi_;
    resize(f.Nx_, f.Ny_, f.Nz_, f.Nd_, f.Lx_, f.Lz_, f.a_, f.b_, cfmpi_);
    setState(f.xzstate_, f.ystate_);
    padded_ = f.padded_;
    for (int i = 0; i < Nloc_; ++i)
        rdata_[i] = f.rdata_[i];
    return *this;
}

void FlowField::reconfig(const FlowField& u, uint fftw_flags) {
    resize(u.Nx(), u.Ny(), u.Nz(), u.Nd(), u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi(), fftw_flags);
}

void FlowField::resize(int Nx, int Ny, int Nz, int Nd, Real Lx, Real Lz, Real a, Real b, CfMPI* cfmpi,
                       uint fftw_flags) {
    assert(Nx >= 0);
    assert(Ny >= 0);
    assert(Nz >= 0);
    assert(Nd >= 0);
    assert(Lx >= 0);
    assert(Lz >= 0);
    assert(b >= a);

    if (Nx == Nx_ && Ny == Ny_ && Nz == Nz_ && Nd == Nd_ && Lx == Lx_ && Lz == Lz_ && a == a_ && b == b_ &&
        cfmpi == cfmpi_ && rdata_)
        return;

    Nx_ = Nx;
    Ny_ = Ny;
    Nz_ = Nz;
    Nd_ = Nd;
    Lx_ = Lx;
    Lz_ = Lz;
    a_ = a;
    b_ = b;
    cfmpi_ = cfmpi;

    Nzpad_ = 2 * (Nz_ / 2 + 1);
    Nzpad2_ = Nz_ / 2 + 1;

    Mx_ = Nx_;
    Mz_ = Nz / 2 + 1;  // == Nzpad2_

    int nproc0 = this->nproc0();

    // Calculate data distribution in y
    // The cfarrays are padded in  y to facilitate transpose operations => nylocmax <= nylocmin + Nylocpad
    // Number of y-gridpoints to allocate (padded to the same for each process to facilitate transposing)
    Nylocpad_ = (Ny_ % nproc0 == 0) ? Ny_ / nproc0 : Ny_ / nproc0 + 1;
    nylocmin_ = this->key0() * Nylocpad_;  // Lowest ny on this process
    // nylocmin_ cannot be larger than physical Ny_ (above calculation can yield this in case of large padding)
    nylocmin_ = min((int)nylocmin_, Ny_);
    nylocmax_ = min((int)(nylocmin_ + Nylocpad_), Ny_);
    Nyloc_ = nylocmax_ - nylocmin_;
    Nypad_ = Nylocpad_ * nproc0;

    // Number of gridpoints in spectral state
    Mxloc_ = Mx_ / nproc0;
    if (Mxloc_ * nproc0 != Mx_)  // nproc0 must divide Mx_!
        cferror("Number of processes 0 must divide number of x-gridpoints!");
    mxlocmin_ = Mxloc_ * key0();
    Mxlocmax_ = Mxloc_;

    Mzloc_ = Mx_;
    Mzlocmax_ = Mz_;
    mzlocmin_ = 0;

    Nx_io_ = 2 * (Nx_ / 3);
    Nz_io_ = 2 * (Nz_ / 3);

#ifdef HAVE_MPI
    lint howmany = Nylocpad_ * Nd_;
    // Let fftw determine the data distribution in the xz-plane
    lint rank_0[2] = {Mx_, Mz_};
    Nloc_ = 2 * fftw_mpi_local_size_many_transposed(2, rank_0, howmany, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                                    cfmpi_->comm1, &Nxloc_, &nxlocmin_, &Mzloc_, &mzlocmin_);

    lint rank_2[2] = {Nx_, Nypad_};
    lint alloc_local1 = fftw_mpi_local_size_many_transposed(2, rank_2, 2 * Nd_, Mxloc_, Nylocpad_, cfmpi_->comm0,
                                                            &Mxloc_, &mxlocmin_, &Nylocpad_, &nylocmin_) /
                        2;

    // We transform cfarrays that are part of one greater contiguous cfarray -- verify that fftw doesn't want more
    // memory than the absolute minimum, otherwise it would silently corrupt the flowfield data
    if (taskid() < numtasks()) {
        if (alloc_local1 != Nx_ * howmany)
            cferror(
                "FFTW tries to allocate too large cfarrays for in-place transpose. Another transpose is not "
                "implemented currently!\nThis probably means something is wrong with the MPI communicators.");
    } else {
        Mxloc_ = Mzloc_ = Nxloc_ = Nyloc_ = 0;
    }

    // FFTW uses ptrdiff_t variables for indices
    // Declare int variables to be transmitted by MPI -- should be sufficient for Mx, Mz
    int tmp1 = Mxloc_;
    int tmp2 = 0;
    MPI_Allreduce(&tmp1, &tmp2, 1, MPI_INT, MPI_MAX, cfmpi_->comm_world);
    Mxlocmax_ = tmp2;
    tmp1 = Mzloc_;
    MPI_Allreduce(&tmp1, &tmp2, 1, MPI_INT, MPI_MAX, cfmpi_->comm_world);
    Mzlocmax_ = tmp2;
    tmp1 = Nxloc_;
    MPI_Allreduce(&tmp1, &tmp2, 1, MPI_INT, MPI_MAX, cfmpi_->comm_world);
    Nxlocmax_ = tmp2;
#else
    Nxloc_ = Nx_;
    Nxlocmax_ = Nx_;
    nxlocmin_ = 0;
    Mzloc_ = Mz_;
    mzlocmin_ = 0;
    Nloc_ = Nx_ * Ny_ * Nzpad_ * Nd_;

    scratch_.reset(static_cast<Real*>(fftw_malloc(Ny_ * sizeof(Real))));
#endif

    // INEFFICIENT if geometry doesn't change. Should check.
    data_handle_.reset(fftw_malloc(Nloc_ * sizeof(Real)));
    rdata_ = static_cast<Real*>(data_handle_.get());
    cdata_ = static_cast<Complex*>(data_handle_.get());
    fftw_initialize(fftw_flags);
    fill(rdata_, rdata_ + Nloc_, 0.0);
}

void FlowField::fftw_initialize(uint fftw_flags) {
    static int count = 0;
    static int count2 = 0;
    count++;
#ifndef HAVE_MPI
    fftw_flags = fftw_flags | FFTW_DESTROY_INPUT;
#endif
    if (xz_plan_) {
        count2++;
    }

    // Initialize xz plan
    if (Nx_ != 0 && Nz_ != 0) {
        fftw_complex* fcdata = (fftw_complex*)cdata_;
#ifdef HAVE_MPI
        lint rank_1[2] = {Nx_, Nz_};
        lint howmany = Nylocpad_ * Nd_;
        xz_plan_.reset(fftw_mpi_plan_many_dft_r2c(2, rank_1, howmany, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                                  rdata_, fcdata, cfmpi_->comm1, fftw_flags | FFTW_MPI_TRANSPOSED_OUT));
        xz_iplan_.reset(fftw_mpi_plan_many_dft_c2r(2, rank_1, howmany, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                                   fcdata, rdata_, cfmpi_->comm1, fftw_flags | FFTW_MPI_TRANSPOSED_IN));
        // The transpose plans work on one slice in z at a time. Block distribution changes from x to y, without
        // changing the layout in memory (which is z x y i) howmany = 2*Nd is the number of real values that is stored
        // for the last index i
        t_plan_.reset(fftw_mpi_plan_many_transpose(Nypad_, Mx_, 2 * Nd_, Nylocpad_, Mxlocmax_, rdata_, rdata_,
                                                   cfmpi_->comm0, fftw_flags | FFTW_MPI_TRANSPOSED_IN));
        t_iplan_.reset(fftw_mpi_plan_many_transpose(Mx_, Nypad_, 2 * Nd_, Mxlocmax_, Nylocpad_, rdata_, rdata_,
                                                    cfmpi_->comm0, fftw_flags | FFTW_MPI_TRANSPOSED_OUT));

        // The y plan operates on Nd_*2 real cfarrays at a time and is called Mxloc*Mzloc times
        int ranky[1];
        ranky[0] = Ny_;
        int rankembed[1];
        rankembed[0] = Nypad_;
        int ydist = 1;
        int ystride = Nd_ * 2;
        int yhowmany = Nd_ * 2;
        fftw_r2r_kind kind[1];
        kind[0] = FFTW_REDFT00;
        // Initialize y plan
        if (Ny_ >= 2) {
            y_plan_.reset(fftw_plan_many_r2r(1, ranky, yhowmany, rdata_, rankembed, ystride, ydist, rdata_, rankembed,
                                             ystride, ydist, kind, fftw_flags));
        } else {
            y_plan_ = nullptr;
        }
#else
        const int howmany = Ny_ * Nd_;
        const int rank = 2;

        // These params describe the structure of the real-valued cfarray
        int real_n[rank];
        real_n[0] = Nx_;
        real_n[1] = Nz_;
        int real_embed[rank];
        real_embed[0] = Nx_;
        real_embed[1] = Nzpad_;
        const int real_stride = 1;
        const int real_dist = Nx_ * Nzpad_;

        // These params describe the structure of the complex-valued cfarray
        int cplx_embed[rank];
        cplx_embed[0] = Nx_;
        cplx_embed[1] = Nzpad2_;
        const int cplx_stride = 1;
        const int cplx_dist = Nx_ * Nzpad2_;

        // Real -> Complex transform parameters
        xz_plan_.reset(fftw_plan_many_dft_r2c(rank, real_n, howmany, rdata_, real_embed, real_stride, real_dist, fcdata,
                                              cplx_embed, cplx_stride, cplx_dist, fftw_flags));

        xz_iplan_.reset(fftw_plan_many_dft_c2r(rank, real_n, howmany, fcdata, cplx_embed, cplx_stride, cplx_dist,
                                               rdata_, real_embed, real_stride, real_dist, fftw_flags));

        t_plan_ = nullptr;
        t_iplan_ = nullptr;

        // Initialize y plan
        if (Ny_ >= 2)
            y_plan_.reset(fftw_plan_r2r_1d(Ny_, scratch_.get(), scratch_.get(), FFTW_REDFT00, fftw_flags));
        else
            y_plan_ = nullptr;

#endif
    } else {
        xz_plan_ = nullptr;
        xz_iplan_ = nullptr;
        t_plan_ = nullptr;
        t_iplan_ = nullptr;
    }
}

void FlowField::rescale(Real Lx, Real Lz) {
    assertState(Spectral, Spectral);
    Real e = L2Norm(*this);
    if (Nd_ == 3) {  // change vector component in case of 3D vectors
        Real scaleu = Lx / Lx_;
        Real scalew = Lz / Lz_;
        for (int ny = 0; ny < Ny_; ++ny)
            for (int mx = mxlocmin_; mx < mxlocmin_ + Mxloc_; ++mx)
                for (int mz = mzlocmin_; mz < Mzloc_ + mzlocmin_; ++mz) {
                    cmplx(mx, ny, mz, 0) *= scaleu;
                    cmplx(mx, ny, mz, 2) *= scalew;
                }
    }
    Lx_ = Lx;
    Lz_ = Lz;
    // rescale to have same norm as original
    Real e2 = L2Norm(*this);
    if (e2 > 0)
        (*this) *= e / e2;
    printout("Leaving rescale");
}

void FlowField::interpolate(FlowField f) {
    FlowField& g = *this;

    fieldstate gxstate = g.xzstate();
    fieldstate gystate = g.ystate();

    // Always check these, debugging or not.
    if (f.Lx() != g.Lx() || f.Lz() != g.Lz() || f.a() != g.a() || f.b() != g.b() || f.Nd() != g.Nd()) {
        cferror(
            "FlowField::interpolate(const FlowField& f) error:\nFlowField doesn't match argument f geometrically.\n");
    }

    const int fNy = f.Ny();
    const int gNy = g.Ny();
    const int Nd = f.Nd();
    g.setToZero();

    ComplexChebyCoeff fprof(f.Ny(), a_, b_, Spectral);
    ComplexChebyCoeff gprof(g.Ny(), a_, b_, Spectral);

    const int kxmax = lesser(f.kxmax(), g.kxmax());
    const int kzmax = lesser(f.kzmax(), g.kzmax());
    const int kxmin = Greater(f.kxmin(), g.kxmin());

    f.makeSpectral();
    g.setState(Spectral, Spectral);
    g.setToZero();

    for (int i = 0; i < Nd; ++i) {
        // If going from smaller Chebyshev expansion to larger
        // Just copy the spectral coefficients and loop in fastest order
        if (fNy <= gNy) {
            for (int ny = 0; ny < fNy; ++ny)
                for (int kx = kxmin; kx < kxmax; ++kx) {
                    int fmx = f.mx(kx);
                    int gmx = g.mx(kx);
                    for (int kz = 0; kz <= kzmax; ++kz) {
                        int fmz = f.mz(kz);
                        int gmz = g.mz(kz);
                        transmit_coeff(g, gmx, gmz, f, fmx, fmz, ny, i);
                        //                         g.cmplx(gmx,ny,gmz,i) = f.cmplx(fmx,ny,fmz,i);
                    }
                }
        }
        // Going from larger Chebyshev expansion to smaller
        // Interpolate the spectral coefficients. That requires an inner loop in y.
        else {
            for (int kx = kxmin; kx < kxmax; ++kx) {
                int fmx = f.mx(kx);
                int gmx = g.mx(kx);
                for (int kz = 0; kz <= kzmax; ++kz) {
                    int fmz = f.mz(kz);
                    int gmz = g.mz(kz);

                    int taskid = mpirank();
                    int taskidf = f.task_coeff(fmx, fmz);
                    if (taskid == taskidf) {
                        for (int ny = 0; ny < fNy; ++ny)
                            fprof.set(ny, f.cmplx(fmx, ny, fmz, i));
                        gprof.interpolate(fprof);
                    }
                    for (int ny = 0; ny < gNy; ++ny)
                        transmit_coeff(g, gmx, gmz, ny, i, gprof[ny], taskidf, taskid);
                }
            }
        }

        // Fourier discretizations go from -Mx/2+1 <= kx <= Mx/2
        // So the last mode kx=Mx/2 has an implicit -kx counterpart that is
        // not included in the Fourier representation. For f, this mode is
        // implicitly/automatically included in the Fourier transforms.
        // But if g has more x modes than f, we need to assign
        // g(-kxmax) = conj(f(kxmax)) explicitly.
        if (g.Nx() > f.Nx()) {
            int kx = f.kxmax();
            int fmx = f.mx(kx);
            int gmx = g.mx(-kx);

            if (fNy <= gNy) {
                for (int ny = 0; ny < fNy; ++ny) {
                    for (int kz = f.kzmin(); kz <= f.kzmax(); ++kz) {
                        int fmz = f.mz(kz);
                        int gmz = g.mz(kz);
                        transmit_coeff(g, gmx, gmz, f, fmx, fmz, ny, i);
                    }
                }
            } else {
                for (int kz = f.kzmin(); kz <= f.kzmax(); ++kz) {
                    int fmz = f.mz(kz);
                    int gmz = g.mz(kz);

                    int taskid = mpirank();
                    int taskidf = f.task_coeff(fmx, fmz);
                    if (taskid == taskidf) {
                        for (int ny = 0; ny < fNy; ++ny)
                            fprof.set(ny, f.cmplx(fmx, ny, fmz, i));
                        gprof.interpolate(fprof);
                    }
                    for (int ny = 0; ny < gNy; ++ny)
                        transmit_coeff(g, gmx, gmz, ny, i, gprof[ny], taskidf, taskid);
                }
            }
        }
    }

    g.makeState(gxstate, gystate);
}

void transmit_coeff(FlowField& g, int gmx, int gmz, const FlowField& f, int fmx, int fmz, int ny, int i, int taskid) {
#ifdef HAVE_MPI
    assert(f.taskid_world() == g.taskid_world());  // That would mean that f and g use different communicators ...
    if (taskid == -1)                              // default
        taskid = f.taskid();
    int taskidg = g.task_coeff(gmx, gmz);
    int taskidf = f.task_coeff(fmx, fmz);

    if (taskidg == taskidf && taskid == taskidf)
        g.cmplx(gmx, ny, gmz, i) = f.cmplx(fmx, ny, fmz, i);
    else {
        int tag = (gmz + g.Mz() * (gmx + g.Mx() * (ny + g.Ny() * i)));
        if (taskid == taskidf) {
            Complex fval = f.cmplx(fmx, ny, fmz, i);
            MPI_Ssend(&fval, 1, MPI_DOUBLE_COMPLEX, taskidg, tag, *f.comm_world());
        }
        if (taskid == taskidg) {
            Complex gval = 0;
            MPI_Status status;
            MPI_Recv(&gval, 1, MPI_DOUBLE_COMPLEX, taskidf, tag, *f.comm_world(), &status);
            g.cmplx(gmx, ny, gmz, i) = gval;
        }
    }
#else
    g.cmplx(gmx, ny, gmz, i) = f.cmplx(fmx, ny, fmz, i);
#endif
}

void transmit_coeff(FlowField& g, int gmx, int gmz, int ny, int i, const Complex& val, int taskid_val, int taskid) {
#ifdef HAVE_MPI
    if (taskid == -1)  // default
        taskid = g.taskid();
    int taskidg = g.task_coeff(gmx, gmz);

    if (taskidg == taskid_val && taskid == taskid_val)
        g.cmplx(gmx, ny, gmz, i) = val;
    else {
        int tag = (gmz + g.Mz() * (gmx + g.Mx() * (ny + g.Ny() * i)));
        if (taskid == taskid_val) {
            Complex fval = val;
            MPI_Ssend(&fval, 1, MPI_DOUBLE_COMPLEX, taskidg, tag, *g.comm_world());
        }
        if (taskid == taskidg) {
            Complex gval = 0;
            MPI_Status status;
            MPI_Recv(&gval, 1, MPI_DOUBLE_COMPLEX, taskid_val, tag, *g.comm_world(), &status);
            g.cmplx(gmx, ny, gmz, i) = gval;
        }
    }
#else
    g.cmplx(gmx, ny, gmz, i) = val;
#endif
}

void FlowField::toprocess0(FlowField& v) const {
#ifdef HAVE_MPI

    unsigned long bufsize = 0;
    unsigned long Nloc = Nloc_;
    MPI_Reduce(&Nloc, &bufsize, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, cfmpi_->comm_world);
    int mxlocmin, Mxloc, mzlocmin, Mzloc, Ny, Nd;
    mxlocmin = mxlocmin_;
    Mxloc = Mxloc_;
    mzlocmin = mzlocmin_;
    Mzloc = Mzloc_;
    Ny = Ny_;
    Nd = Nd_;
    if (taskid() == 0) {
        // Argh, cfarray sizes get too large...
        auto recvbuf = vector<double>(bufsize, 0.0);
        for (int commtaskid = 0; commtaskid < numtasks(); commtaskid++) {
            if (commtaskid == 0) {
                // locvariables are already correct

                // just copy flowfield data
                for (int mx = mxlocmin; mx < mxlocmin + Mxloc; mx++) {
                    for (int mz = mzlocmin; mz < mzlocmin + Mzloc; mz++) {
                        for (int my = 0; my < Ny; my++) {
                            for (int i = 0; i < Nd; i++) {
                                v.cmplx(mx, my, mz, i) = cmplx(mx, my, mz, i);
                            }
                        }
                    }
                }
            } else {
                // receive locvariables
                MPI_Status status;
                MPI_Recv(&mxlocmin, 1, MPI_INT, commtaskid, 0, cfmpi_->comm_world, &status);
                MPI_Recv(&Mxloc, 1, MPI_INT, commtaskid, 1, cfmpi_->comm_world, &status);
                MPI_Recv(&mzlocmin, 1, MPI_INT, commtaskid, 2, cfmpi_->comm_world, &status);
                MPI_Recv(&Mzloc, 1, MPI_INT, commtaskid, 3, cfmpi_->comm_world, &status);
                MPI_Recv(&Nloc, 1, MPI_UNSIGNED_LONG, commtaskid, 4, cfmpi_->comm_world, &status);
                // receive data
                MPI_Recv(recvbuf.data(), Nloc, MPI_DOUBLE, commtaskid, 5, cfmpi_->comm_world, &status);
                for (int mx = mxlocmin; mx < mxlocmin + Mxloc; mx++) {
                    for (int mz = mzlocmin; mz < mzlocmin + Mzloc; mz++) {
                        for (int my = 0; my < Ny; my++) {
                            for (int i = 0; i < Nd; i++) {
                                lint index = 2 * (i + Nd * (my + Ny * ((mx - mxlocmin) + Mxloc * (mz - mzlocmin))));
                                v.cmplx(mx, my, mz, i) = Complex(recvbuf[index], recvbuf[index + 1]);
                            }
                        }
                    }
                }
            }
        }

    } else {
        // Prepare sendbuf
        auto sendbuf = vector<double>(Nloc, 0.0);
        for (int mx = mxlocmin; mx < mxlocmin + Mxloc; mx++) {
            for (int mz = mzlocmin; mz < mzlocmin + Mzloc; mz++) {
                for (int my = 0; my < Ny; my++) {
                    for (int i = 0; i < Nd; i++) {
                        lint index = 2 * (i + Nd * (my + Ny * ((mx - mxlocmin) + Mxloc * (mz - mzlocmin))));
                        sendbuf[index] = cmplx(mx, my, mz, i).real();
                        sendbuf[index + 1] = cmplx(mx, my, mz, i).imag();
                    }
                }
            }
        }

        // send locvariables
        MPI_Ssend(&mxlocmin, 1, MPI_INT, 0, 0, cfmpi_->comm_world);
        MPI_Ssend(&Mxloc, 1, MPI_INT, 0, 1, cfmpi_->comm_world);
        MPI_Ssend(&mzlocmin, 1, MPI_INT, 0, 2, cfmpi_->comm_world);
        MPI_Ssend(&Mzloc, 1, MPI_INT, 0, 3, cfmpi_->comm_world);
        MPI_Ssend(&Nloc, 1, MPI_UNSIGNED_LONG, 0, 4, cfmpi_->comm_world);

        // send rdata
        MPI_Ssend(sendbuf.data(), Nloc, MPI_DOUBLE, 0, 5, cfmpi_->comm_world);
    }
#else
    cout << "FlowField::toprocess0 does nothing in serial mode" << endl;
#endif
}

void FlowField::fromprocess0(const FlowField& v) {
#ifdef HAVE_MPI

    unsigned long bufsize = 0;
    unsigned long Nloc = Nloc_;
    MPI_Reduce(&Nloc, &bufsize, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, cfmpi_->comm_world);

    int mxlocmin, Mxloc, mzlocmin, Mzloc, Ny, Nd;
    mxlocmin = mxlocmin_;
    Mxloc = Mxloc_;
    mzlocmin = mzlocmin_;
    Mzloc = Mzloc_;
    Ny = Ny_;
    Nd = Nd_;
    if (taskid() == 0) {
        auto sendbuf = vector<double>(bufsize, 0.0);
        for (int commtaskid = 0; commtaskid < numtasks(); commtaskid++) {
            if (commtaskid == 0) {
                // locvariables are already correct

                // just copy flowfield data
                for (int mx = mxlocmin; mx < mxlocmin + Mxloc; mx++) {
                    for (int mz = mzlocmin; mz < mzlocmin + Mzloc; mz++) {
                        for (int my = 0; my < Ny; my++) {
                            for (int i = 0; i < Nd; i++) {
                                cmplx(mx, my, mz, i) = v.cmplx(mx, my, mz, i);
                            }
                        }
                    }
                }
            } else {
                // receive locvariables
                MPI_Status status;
                MPI_Recv(&mxlocmin, 1, MPI_INT, commtaskid, 0, cfmpi_->comm_world, &status);
                MPI_Recv(&Mxloc, 1, MPI_INT, commtaskid, 1, cfmpi_->comm_world, &status);
                MPI_Recv(&mzlocmin, 1, MPI_INT, commtaskid, 2, cfmpi_->comm_world, &status);
                MPI_Recv(&Mzloc, 1, MPI_INT, commtaskid, 3, cfmpi_->comm_world, &status);
                MPI_Recv(&Nloc, 1, MPI_UNSIGNED_LONG, commtaskid, 4, cfmpi_->comm_world, &status);

                // send data
                for (int mx = mxlocmin; mx < mxlocmin + Mxloc; mx++) {
                    for (int mz = mzlocmin; mz < mzlocmin + Mzloc; mz++) {
                        for (int my = 0; my < Ny; my++) {
                            for (int i = 0; i < Nd; i++) {
                                lint index = 2 * (i + Nd * (my + Ny * ((mx - mxlocmin) + Mxloc * (mz - mzlocmin))));
                                sendbuf[index] = v.cmplx(mx, my, mz, i).real();
                                sendbuf[index + 1] = v.cmplx(mx, my, mz, i).imag();
                            }
                        }
                    }
                }
                // send rdata
                MPI_Ssend(sendbuf.data(), Nloc, MPI_DOUBLE, commtaskid, 5, cfmpi_->comm_world);
            }
        }
    } else {
        // Prepare sendbuf
        auto recvbuf = vector<double>(Nloc, 0.0);

        // send locvariables
        MPI_Ssend(&mxlocmin, 1, MPI_INT, 0, 0, cfmpi_->comm_world);
        MPI_Ssend(&Mxloc, 1, MPI_INT, 0, 1, cfmpi_->comm_world);
        MPI_Ssend(&mzlocmin, 1, MPI_INT, 0, 2, cfmpi_->comm_world);
        MPI_Ssend(&Mzloc, 1, MPI_INT, 0, 3, cfmpi_->comm_world);
        MPI_Ssend(&Nloc, 1, MPI_UNSIGNED_LONG, 0, 4, cfmpi_->comm_world);

        // receive data
        MPI_Status status;
        MPI_Recv(recvbuf.data(), Nloc, MPI_DOUBLE, 0, 5, cfmpi_->comm_world, &status);
        for (int mx = mxlocmin; mx < mxlocmin + Mxloc; mx++) {
            for (int mz = mzlocmin; mz < mzlocmin + Mzloc; mz++) {
                for (int my = 0; my < Ny; my++) {
                    for (int i = 0; i < Nd; i++) {
                        lint index = 2 * (i + Nd * (my + Ny * ((mx - mxlocmin) + Mxloc * (mz - mzlocmin))));
                        cmplx(mx, my, mz, i) = Complex(recvbuf[index], recvbuf[index + 1]);
                    }
                }
            }
        }
    }
#else
    cout << "FlowField::fromprocess0 does nothing in serial mode" << endl;
#endif
}

void FlowField::optimizeFFTW(uint fftw_flags) {
    fftw_flags = fftw_flags | FFTW_DESTROY_INPUT;
    fftw_initialize(fftw_flags);
}

ComplexChebyCoeff FlowField::profile(int mx, int mz, int i) const {
    ComplexChebyCoeff rtn(Ny_, a_, b_, ystate_);
#ifdef HAVE_MPI
    if (xzstate_ == Spectral) {
        if (taskid() == task_coeff(mx, mz)) {
            for (int ny = 0; ny < Ny_; ++ny)
                rtn.set(ny, cmplx(mx, ny, mz, i));
        }
    } else {
        CfMPI_single* CfMPI_one = &CfMPI_single::getInstance();
        FlowField v(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, a_, b_, CfMPI_one);  // FlowField v is only on process 0
        v.interpolate(*this);
        if (taskid() == 0) {
            Real vmxnymzi;
            for (int ny = 0; ny < Ny_; ++ny) {
                vmxnymzi = v(mx, ny, mz, i);
                rtn.re[ny] = vmxnymzi;
            }
        }
    }
#else
    if (xzstate_ == Spectral) {
        for (int ny = 0; ny < Ny_; ++ny)
            rtn.set(ny, cmplx(mx, ny, mz, i));
    } else
        for (int ny = 0; ny < Ny_; ++ny)
            rtn.re[ny] = (*this)(mx, ny, mz, i);
#endif
    return rtn;
}

BasisFunc FlowField::profile(int mx, int mz) const {
    assert(xzstate_ == Spectral);
    BasisFunc rtn(Nd_, Ny_, kx(mx), kz(mz), Lx_, Lz_, a_, b_, ystate_);
    for (int i = 0; i < Nd_; ++i)
        for (int ny = 0; ny < Ny_; ++ny)
            rtn[i].set(ny, cmplx(mx, ny, mz, i));
    return rtn;
}

int sign_i(const FieldSymmetry& sigma, int n, int Nd) {
    switch (Nd) {
        case 0:
        case 1:
            return 1;
        case 3:
            return sigma.s(n);
        case 6: {
            //***** Components of a symmetric tensor *****
            //  ***** 0 ---------- 00 *****
            //  ***** 1 ---------- 11 *****
            //  ***** 2 ---------- 22 *****
            //  ***** 3 ---------- 01 *****
            //  ***** 4 ---------- 02 *****
            //  ***** 5 ---------- 12 *****
            // *****************************
            int i = (n < 3) ? n : ((n < 5) ? 0 : 1);
            int j = (n < 3) ? n : ((n < 4) ? 1 : 2);
            return sigma.s(i) * sigma.s(j);
        }
        case 9:
            // implements inverse of i3j(i,j) -> n function
            return sigma.s(n / 3) * sigma.s(n % 3);
        default:
            cferror("error : symmetry operations defined only for 3d scalar, vector, tensor fields");
            return 0;
    }
}

FlowField& FlowField::project(const FieldSymmetry& sigma) {
    // Identity escape clause
    if (sigma.isIdentity())
        return *this;

    fieldstate xzs = xzstate_;
    fieldstate ys = ystate_;
    makeState(Spectral, Spectral);

    const Real cx = 2 * pi * sigma.ax();
    const Real cz = 2 * pi * sigma.az();

    // (u,v,w)(x,y,z) -> (sa sx u, sa sy v, sa sz w) (sx x + fx*Lx, sy y, sz z + fz*Lz)
    const int s = sigma.s();
    const int sx = sigma.sx();
    const int sy = sigma.sy();
    const int sz = sigma.sz();

    const int Kxmin = padded() ? kxminDealiased() : kxmin();
    const int Kxmax = padded() ? kxmaxDealiased() : kxmax();
    const int Kzmin = padded() ? kzminDealiased() : kzmin();
    const int Kzmax = padded() ? kzmaxDealiased() : kzmax();

    Complex tmp_p;
    Complex tmp_m;

    // If sx,sz are -1,1 or 1,1, we must average kx,kz and -kx,kz modes
    // With MPI, this requires communication
    // We have to make a choice between many (and hence possibly slow) communication operations
    // or few communcation operations but larger storage requirements
    // In this choice, we choose to communciate all data for one kx at a time, which is in between
    // communicating every (kx,ky,kz,i) separately and one large cfarray for all kx

    if (sx + sz == 0) {
        lint bufsize = Nd_ * Ny_ * Mz_ * 2;
#ifdef HAVE_MPI
        auto sendbuf = vector<Real>(bufsize, 0.0);
#endif
        auto conjdata = vector<Real>(bufsize, 0.0);
        for (lint ix = 0; ix < Mx_; ix++) {
            //    for ( lint ix=Mx_-1; ix>=0; ix-- ) {
            // FIXME: improve this ordering so that no process is idly waiting while others communicate
            // But be careful: it is important, that +kx and -kx directly follow each other, otherwise the data might
            // end up at the wrong process
            lint kx_p = ix / 2 - ix * (ix % 2);  // kx = 0,-1,1,-2,2,-3,3...
            if (kx_p >= Kxmin && kx_p <= Kxmax && mx(kx_p) >= mxlocmin_ && mx(kx_p) < mxlocmin_ + Mxloc_) {
                lint mx_p = mx(kx_p);
                lint kx_m = -kx_p;
                lint mx_m = mx(kx_m);

                int commtaskid = task_coeff(mx_m, mzlocmin_);
#ifdef HAVE_MPI
                // Prepare data for transmission
                if (taskid() != commtaskid) {
                    for (lint iz = 0; iz < Mzloc_; iz++) {
                        lint mz = iz + mzlocmin_;
                        for (lint ny = 0; ny < Ny_; ny++) {
                            for (lint i = 0; i < Nd_; i++) {
                                lint index = 2 * (iz + Mzloc_ * (ny + Ny_ * i));
                                sendbuf[index] = cmplx(mx_p, ny, mz, i).real();
                                sendbuf[index + 1] = cmplx(mx_p, ny, mz, i).imag();
                            }
                        }
                    }
                }

                // Transmit -kx data
                // This is necessary because fftw aligns Mz and if the number of processors in Mz is too large, some
                // might remain empty
                if (Mzloc_ > 0) {
                    if (taskid() < commtaskid) {
                        MPI_Status status;
                        MPI_Ssend(sendbuf.data(), bufsize, MPI_DOUBLE, commtaskid, 0, MPI_COMM_WORLD);
                        MPI_Recv(conjdata.data(), bufsize, MPI_DOUBLE, commtaskid, 1, MPI_COMM_WORLD, &status);
                    } else if (taskid() > commtaskid) {
                        MPI_Status status;
                        MPI_Recv(conjdata.data(), bufsize, MPI_DOUBLE, commtaskid, 0, MPI_COMM_WORLD, &status);
                        MPI_Ssend(sendbuf.data(), bufsize, MPI_DOUBLE, commtaskid, 1, MPI_COMM_WORLD);
                    }
                }
#endif

                // Copy -kx data into FlowField
                for (int i = 0; i < Nd_; ++i) {
                    int si = sign_i(sigma, i, Nd_);

                    for (int ky = 0; ky < Ny_; ++ky) {
                        int syl = ((sy == -1) && (ky % 2 == 1)) ? -1 : 1;
                        Real symmsign = Real(s * si * syl);

                        for (int kz = 0; kz <= Kzmax; ++kz) {
                            int mz = this->mz(kz);

                            if (mz >= mzlocmin_ && mz < mzlocmin_ + Mzloc_) {
                                lint iz = mz - mzlocmin_;
                                lint index = 2 * (iz + Mzloc_ * (ky + Ny_ * i));

                                if (taskid() == commtaskid && kx_p >= 0) {  // only kx_p >= 0: don't do things twice
                                    Complex cxyz_p = symmsign * exp(Complex(0.0, cx * sx * kx_p + cz * sz * kz));
                                    Complex cxyz_m = symmsign * exp(Complex(0.0, cx * sx * kx_m + cz * sz * kz));
                                    if (sx == -1) {
                                        // u_{ijkl}     -> (       u_{ijkl} + cxyz_p u_{i,-j,k,l})/2
                                        // u_{i,-j,k,l} -> (cxyz_m u_{ijkl} +        u_{i,-j,k,l})/2
                                        tmp_p = cmplx(mx_p, ky, mz, i);
                                        tmp_m = cmplx(mx_m, ky, mz, i);

                                        cmplx(mx_p, ky, mz, i) = 0.5 * (tmp_p + cxyz_p * tmp_m);
                                        cmplx(mx_m, ky, mz, i) = 0.5 * (cxyz_m * tmp_p + tmp_m);
                                    } else {
                                        // u_{ijkl}     -> cxyz_p u^*_{i,-j,k,l})
                                        // u_{i,-j,k,l} -> cxyz_m u^*_{i,j,k,l}

                                        // u_{ijkl}     -> (       u_{ijkl}   + cxyz_p u^*_{i,-j,k,l})/2
                                        // u_{i,-j,k,l} -> (cxyz_m u^*_{ijkl} +          u_{i,-j,k,l})/2

                                        tmp_p = cmplx(mx_p, ky, mz, i);
                                        tmp_m = cmplx(mx_m, ky, mz, i);

                                        cmplx(mx_p, ky, mz, i) = 0.5 * (tmp_p + cxyz_p * conj(tmp_m));
                                        cmplx(mx_m, ky, mz, i) = 0.5 * (cxyz_m * conj(tmp_p) + tmp_m);
                                    }
                                } else if (taskid() != commtaskid) {
                                    Complex cxyz_p = symmsign * exp(Complex(0.0, cx * sx * kx_p + cz * sz * kz));
                                    tmp_p = cmplx(mx_p, ky, mz, i);
                                    tmp_m = Complex(conjdata[index], conjdata[index + 1]);
                                    if (sx == -1) {
                                        cmplx(mx_p, ky, mz, i) = 0.5 * (tmp_p + cxyz_p * tmp_m);
                                    } else {
                                        // u_{ijkl}     -> cxyz_p u^*_{i,-j,k,l})
                                        cmplx(mx_p, ky, mz, i) = 0.5 * (tmp_p + cxyz_p * conj(tmp_m));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // If sx,sz are 1,1 or -1,-1, can apply symmetry by *=
    else {
        for (int i = 0; i < Nd_; ++i) {
            int si = sign_i(sigma, i, Nd_);
            // int si = sigma.sign(i); // (u,v,w) -> (s0 u, s1 v, s2 w)

            for (int ky = 0; ky < Ny_; ++ky) {
                int syl = (sy == -1 && (ky % 2 == 1)) ? -1 : 1;
                Real symmsign = Real(s * si * syl);

                for (int kx = Kxmin; kx <= Kxmax; ++kx) {
                    int mx = this->mx(kx);
                    if (mx >= mxlocmin_ && mx < mxlocmin_ + Mxloc_) {
                        for (int kz = Kzmin; kz <= Kzmax; ++kz) {
                            int mz = this->mz(kz);
                            if (mz >= mzlocmin_ && mz < mzlocmin_ + Mzloc_) {
                                Complex cxyz = symmsign * exp(Complex(0.0, cx * sx * kx + cz * sz * kz));

                                if (sx == 1)
                                    cmplx(mx, ky, mz, i) *= 0.5 * (1.0 + cxyz);
                                else
                                    cmplx(mx, ky, mz, i) =
                                        0.5 * (cmplx(mx, ky, mz, i) + cxyz * conj(cmplx(mx, ky, mz, i)));
                            }
                        }
                    }
                }
            }
        }
    }
    makeState(xzs, ys);
    return *this;
}

FlowField& FlowField::project(const cfarray<FieldSymmetry>& sigma) {
    for (int n = 0; n < sigma.length(); ++n)
        this->project(sigma[n]);
    return *this;
}

FlowField& FlowField::operator*=(const FieldSymmetry& sigma) {
    // Identity escape clause
    if (sigma.isIdentity())
        return *this;
    stringstream ss;

    const fieldstate xzs = xzstate_;
    const fieldstate ys = ystate_;
    makeState(Spectral, Spectral);

    const Real cx = 2 * pi * sigma.ax();
    const Real cz = 2 * pi * sigma.az();

    // (u,v,w)(x,y,z) -> (sa sx u, sa sy v, sa sz w) (sx x + fx*Lx, sy y, sz z + fz*Lz)
    const int s = sigma.s();
    const int sx = sigma.sx();
    const int sy = sigma.sy();
    const int sz = sigma.sz();

    const int Kxmin = padded() ? kxminDealiased() : kxmin();
    const int Kxmax = padded() ? kxmaxDealiased() : kxmax();
    const int Kzmin = padded() ? kzminDealiased() : kzmin();
    const int Kzmax = padded() ? kzmaxDealiased() : kzmax();

    Complex tmp;
    // If one of sx,sz is -1,1 or 1,-1, we must swap kx,kz and -kx,kz modes
    // With MPI, this requires communication
    // We have to make a choice between many (and hence possibly slow) communication operations
    // or few communcation operations but larger storage requirements
    // In this choice, we choose to communciate all data for one kx at a time, which is in between
    // communicating every (kx,ky,kz,i) separately and one large array for all kx
    lint bufsize = Nd_ * Ny_ * Mz_ * 2;
#ifdef HAVE_MPI
    auto sendbuf = vector<Real>(bufsize, 0.0);
#endif
    auto conjdata = vector<Real>(bufsize, 0.0);
    if (sx + sz == 0) {
        for (lint ix = 0; ix < Mx_; ix++) {
            // FIXME: improve this ordering so that no process is idly waiting while others communicate
            // But be careful: it is important, that +kx and -kx directly follow each other, otherwise the data might
            // end up at the wrong process
            lint kx_p = ix / 2 - ix * (ix % 2);  // kx = 0,-1,1,-2,2,-3,3...
            if (kx_p >= Kxmin && kx_p <= Kxmax && mx(kx_p) >= mxlocmin_ && mx(kx_p) < mxlocmin_ + Mxloc_) {
                lint mx_p = mx(kx_p);
                lint kx_m = -kx_p;
                lint mx_m = mx(kx_m);

                // Prepare data for transmission
                int commtaskid = task_coeff(mx_m, mzlocmin_);
#ifdef HAVE_MPI
                if (taskid() != commtaskid) {
                    for (lint iz = 0; iz < Mzloc_; iz++) {
                        lint mz = iz + mzlocmin_;
                        for (lint ny = 0; ny < Ny_; ny++) {
                            for (lint i = 0; i < Nd_; i++) {
                                lint index = 2 * (iz + Mzloc_ * (ny + Ny_ * i));
                                sendbuf[index] = cmplx(mx_p, ny, mz, i).real();
                                sendbuf[index + 1] = cmplx(mx_p, ny, mz, i).imag();
                            }
                        }
                    }
                }

                // Transmit -kx data
                // This is necessary because fftw aligns Mz and if the number of processors in Mz is too large, some
                // might remain empty
                if (Mzloc_ > 0) {
                    if (taskid() < commtaskid) {
                        MPI_Status status;
                        MPI_Ssend(sendbuf.data(), bufsize, MPI_DOUBLE, commtaskid, 0, MPI_COMM_WORLD);
                        MPI_Recv(conjdata.data(), bufsize, MPI_DOUBLE, commtaskid, 1, MPI_COMM_WORLD, &status);
                    } else if (taskid() > commtaskid) {
                        MPI_Status status;
                        MPI_Recv(conjdata.data(), bufsize, MPI_DOUBLE, commtaskid, 0, MPI_COMM_WORLD, &status);
                        MPI_Ssend(sendbuf.data(), bufsize, MPI_DOUBLE, commtaskid, 1, MPI_COMM_WORLD);
                    }
                }
#endif

                // Copy -kx data into FlowField
                for (int i = 0; i < Nd_; ++i) {
                    int si = sign_i(sigma, i, Nd_);

                    for (int ky = 0; ky < Ny_; ++ky) {
                        int syl = ((sy == -1) && (ky % 2 == 1)) ? -1 : 1;
                        Real symmsign = Real(s * si * syl);

                        for (int kz = 0; kz <= Kzmax; ++kz) {
                            int mz = this->mz(kz);

                            if (mz >= mzlocmin_ && mz < mzlocmin_ + Mzloc_) {
                                lint iz = mz - mzlocmin_;
                                lint index = 2 * (iz + Mzloc_ * (ky + Ny_ * i));

                                if (taskid() == commtaskid && kx_p >= 0) {  // only kx_p >= 0: don't do things twice
                                    Complex cxyz_p = symmsign * exp(Complex(0.0, cx * sx * kx_p + cz * sz * kz));
                                    Complex cxyz_m = symmsign * exp(Complex(0.0, cx * sx * kx_m + cz * sz * kz));
                                    if (sx == -1) {
                                        // u_{ijkl}     -> cxyz_p u_{i,-j,k,l}
                                        // u_{i,-j,k,l} -> cxyz_m u_{i,j,k,l}
                                        tmp = cmplx(mx_p, ky, mz, i);
                                        cmplx(mx_p, ky, mz, i) = cxyz_p * cmplx(mx_m, ky, mz, i);
                                        cmplx(mx_m, ky, mz, i) = cxyz_m * tmp;
                                    } else {
                                        // u_{ijkl}     -> cxyz_p u^*_{i,-j,k,l})
                                        // u_{i,-j,k,l} -> cxyz_m u^*_{i,j,k,l}
                                        tmp = cmplx(mx_p, ky, mz, i);
                                        cmplx(mx_p, ky, mz, i) = cxyz_p * conj(cmplx(mx_m, ky, mz, i));
                                        cmplx(mx_m, ky, mz, i) = cxyz_m * conj(tmp);
                                    }
                                } else if (taskid() != commtaskid) {
                                    Complex cxyz_p = symmsign * exp(Complex(0.0, cx * sx * kx_p + cz * sz * kz));
                                    tmp = Complex(conjdata[index], conjdata[index + 1]);
                                    if (sx == -1) {
                                        // u_{ijkl}     -> cxyz_p u_{i,-j,k,l}
                                        cmplx(mx_p, ky, mz, i) = cxyz_p * tmp;
                                    } else {
                                        // u_{ijkl}     -> cxyz_p u^*_{i,-j,k,l})
                                        cmplx(mx_p, ky, mz, i) = cxyz_p * conj(tmp);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // If sx,sz are 1,1 or -1,-1, can apply symmetry by *=
    else {
        for (int i = 0; i < Nd_; ++i) {
            int si = sign_i(sigma, i, Nd_);

            for (int ky = 0; ky < Ny_; ++ky) {
                int syl = (sy == -1 && (ky % 2 == 1)) ? -1 : 1;
                Real symmsign = Real(s * si * syl);

                for (int kx = Kxmin; kx <= Kxmax; ++kx) {
                    int mx = this->mx(kx);
                    if (mx >= mxlocmin_ && mx < mxlocmin_ + Mxloc_) {
                        for (int kz = Kzmin; kz <= Kzmax; ++kz) {
                            int mz = this->mz(kz);
                            if (mz >= mzlocmin_ && mz < mzlocmin_ + Mzloc_) {
                                Complex cxyz = symmsign * exp(Complex(0.0, cx * sx * kx + cz * sz * kz));

                                if (sx == 1)
                                    cmplx(mx, ky, mz, i) *= cxyz;
                                else
                                    cmplx(mx, ky, mz, i) = cxyz * conj(cmplx(mx, ky, mz, i));
                            }
                        }
                    }
                }
            }
        }
    }
    makeState(xzs, ys);

    return *this;
}

bool FlowField::geomCongruent(const FlowField& v, Real eps) const {
    return ((Nx_ == v.Nx_) && (Ny_ == v.Ny_) && (Nz_ == v.Nz_) && (abs(Lx_ - v.Lx_) / Greater(Lx_, 1.0) < eps) &&
            (abs(Lz_ - v.Lz_) / Greater(Lz_, 1.0) < eps) && (abs(a_ - v.a_) / Greater(abs(a_), 1.0) < eps) &&
            (abs(b_ - v.b_) / Greater(abs(b_), 1.0) < eps));
}

bool FlowField::congruent(const FlowField& v, Real eps) const {
    return ((Nx_ == v.Nx_) && (Ny_ == v.Ny_) && (Nz_ == v.Nz_) && (Nd_ == v.Nd_) &&
            (abs(Lx_ - v.Lx_) / Greater(Lx_, 1.0) < eps) && (abs(Lz_ - v.Lz_) / Greater(Lz_, 1.0) < eps) &&
            (abs(a_ - v.a_) / Greater(abs(a_), 1.0) < eps) && (abs(b_ - v.b_) / Greater(abs(b_), 1.0) < eps) &&
            (xzstate_ == v.xzstate_) && (ystate_ == v.ystate_)) &&
           (nproc0() == v.nproc0());
}

bool FlowField::congruent(const BasisFunc& f) const {
    return ((Ny_ == f.Ny()) && (Lx_ == f.Lx()) && (Lz_ == f.Lz()) && (a_ == f.a()) && (b_ == f.b()) &&
            (ystate_ == f.state()));
}

bool FlowField::congruent(const RealProfileNG& e) const {
    return ((Ny_ == e.Ny()) && (Lx_ == e.Lx()) && (Lz_ == e.Lz()) && (a_ == e.a()) && (b_ == e.b()) &&
            (ystate_ == e.state()));
}

FlowField& FlowField::operator*=(Real x) {
    for (int i = 0; i < Nloc_; ++i)
        rdata_[i] *= x;
    return *this;
}

FlowField operator*(const Real a, const FlowField& w) { return FlowField(w) *= a; }

FlowField operator+(const FlowField& w, const FlowField& v) { return FlowField(w) += v; }

FlowField operator-(const FlowField& w, const FlowField& v) { return FlowField(w) -= v; }

FlowField& FlowField::operator*=(Complex z) {
    assert(xzstate_ == Spectral);
    //     int Ntotal = Nx_ * Ny_ * Nzpad2_ * Nd_;
    for (int i = 0; i < Nloc_; ++i)
        cdata_[i] *= z;
    return *this;
}

FlowField& FlowField::operator+=(const ChebyCoeff& U0) {
    ChebyCoeff U = U0;
    assert(xzstate_ == Spectral);
    U.makeState(ystate_);
    assert(Ny_ == U.N());
    if (task_coeff(0, 0) == taskid())
        for (int ny = 0; ny < Ny_; ++ny)
            cmplx(0, ny, 0, 0) += Complex(U[ny], 0.0);
    return *this;
}

FlowField& FlowField::operator-=(const ChebyCoeff& U) {
    assert(xzstate_ == Spectral);
    assert(ystate_ == U.state());
    assert(Ny_ == U.N());
    if (task_coeff(0, 0) == taskid())
        for (int ny = 0; ny < Ny_; ++ny)
            cmplx(0, ny, 0, 0) -= Complex(U[ny], 0.0);
    return *this;
}

// to add full baseflow

FlowField& FlowField::operator+=(const vector<ChebyCoeff>& UW) {
    ChebyCoeff U = UW[0];
    ChebyCoeff W = UW[1];
    assert(xzstate_ == Spectral);
    U.makeState(ystate_);
    W.makeState(ystate_);
    assert(Ny_ == U.N());
    if (task_coeff(0, 0) == taskid())
        for (int ny = 0; ny < Ny_; ++ny) {
            cmplx(0, ny, 0, 0) += Complex(U[ny], 0.0);
            cmplx(0, ny, 0, 2) += Complex(W[ny], 0.0);
        }
    return *this;
}

FlowField& FlowField::operator-=(const vector<ChebyCoeff>& UW) {
    ChebyCoeff U = UW[0];
    ChebyCoeff W = UW[1];
    assert(xzstate_ == Spectral);
    U.makeState(ystate_);
    W.makeState(ystate_);
    assert(Ny_ == U.N());
    if (task_coeff(0, 0) == taskid())
        for (int ny = 0; ny < Ny_; ++ny) {
            cmplx(0, ny, 0, 0) -= Complex(U[ny], 0.0);
            cmplx(0, ny, 0, 2) -= Complex(W[ny], 0.0);
        }
    return *this;
}

FlowField& FlowField::operator+=(const Real& a) {
    assert(xzstate_ == Spectral);
    assert(ystate_ == Spectral);
    if (task_coeff(0, 0) == taskid())
        cmplx(0, 0, 0, 0) += a;
    return *this;
}

FlowField& FlowField::operator-=(const Real& a) {
    assert(xzstate_ == Spectral);
    assert(ystate_ == Spectral);
    if (task_coeff(0, 0) == taskid())
        cmplx(0, 0, 0, 0) -= a;
    return *this;
}

FlowField& FlowField::operator+=(const ComplexChebyCoeff& U) {
    assert(xzstate_ == Spectral);
    assert(ystate_ == U.state());
    assert(Ny_ == U.N());
    if (task_coeff(0, 0) == taskid())
        for (int ny = 0; ny < Ny_; ++ny)
            cmplx(0, ny, 0, 0) += U[ny];
    return *this;
}

FlowField& FlowField::operator-=(const ComplexChebyCoeff& U) {
    assert(xzstate_ == Spectral);
    assert(ystate_ == U.state());
    assert(Ny_ == U.N());
    if (task_coeff(0, 0) == taskid())
        for (int ny = 0; ny < Ny_; ++ny)
            cmplx(0, ny, 0, 0) -= U[ny];
    return *this;
}

FlowField& FlowField::operator+=(const BasisFunc& f) {
    assert(xzstate_ == Spectral);
    assert(ystate_ == f.state());
    assert(Nd_ == f.Nd());
    int m_x = mx(f.kx());
    int m_z = mz(f.kz());
    for (int i = 0; i < Nd_; ++i)
        for (int ny = 0; ny < Ny_; ++ny)
            cmplx(m_x, ny, m_z, i) += f[i][ny];
    return *this;
}

FlowField& FlowField::operator-=(const BasisFunc& f) {
    assert(xzstate_ == Spectral);
    assert(ystate_ == f.state());
    assert(Nd_ == f.Nd());
    int m_x = mx(f.kx());
    int m_z = mz(f.kz());
    for (int i = 0; i < Nd_; ++i)
        for (int ny = 0; ny < Ny_; ++ny)
            cmplx(m_x, ny, m_z, i) -= f[i][ny];
    return *this;
}

FlowField& FlowField::operator+=(const RealProfile& f) {
    assert(xzstate_ == Spectral);
    assert(ystate_ == f.state());
    assert(Nd_ == f.Nd());
    int fkx = f.kx();
    int fkz = f.kz();

    if (fkx < kxmin() || fkx > kxmax() || fkz < kzmin() || fkz > kzmax())
        return *this;

    int m_x = mx(fkx);
    int m_z = mz(fkz);
    Complex minus_i(0.0, 1.0);
    const BasisFunc& psi = f.psi;

    // f = (psi + psi^*)   case Plus
    // f = (psi - psi^*)/i case Minus
    //
    // where psi.kz >= 0
    //       psi.kx is negative, zero, or positive.

    // Because u is real, u(-kx,-kz) = u^*(kx,kz)
    // or equivalently    u( kx,-kz) = u^*(-kx,kz)

    // Because of this symmetry and the way FFTW works, FlowField stores
    // only u(kx,kz) for kz > 0, u(kx,-kz) is obtained from the symmetry.
    // But for kz=0 and most kx, both u(kx,0) and u(-kx,0) are stored.
    // These are handled by the if statement below
    switch (f.sign()) {
        case Minus:
            for (int i = 0; i < Nd_; ++i)
                for (int ny = 0; ny < Ny_; ++ny)
                    cmplx(m_x, ny, m_z, i) += minus_i * psi[i][ny];
            break;

        case Plus:
            for (int i = 0; i < Nd_; ++i)
                for (int ny = 0; ny < Ny_; ++ny)
                    cmplx(m_x, ny, m_z, i) += psi[i][ny];
            break;
    }

    // When kz=0, both u(kx,0) and u(-kx,0) are stored in the flow field,
    // except for the very largest kx>0 when Nx is even. So to we need to
    // add u(-kx,0) = u^*(kx,0) for kx>1.
    if (fkz == 0 && fkx >= 0 && -fkx >= kxmin()) {
        int nmx = mx(-fkx);
        switch (f.sign()) {
            case Minus:
                for (int i = 0; i < Nd_; ++i)
                    for (int ny = 0; ny < Ny_; ++ny)
                        cmplx(nmx, ny, m_z, i) += conj(minus_i * (psi[i][ny]));
                break;

            case Plus:
                for (int i = 0; i < Nd_; ++i)
                    for (int ny = 0; ny < Ny_; ++ny)
                        cmplx(nmx, ny, m_z, i) += conj(psi[i][ny]);
                break;
        }
    }
    return *this;
}

FlowField& FlowField::operator-=(const RealProfile& f) {
    assert(xzstate_ == Spectral);
    assert(ystate_ == f.state());
    assert(Nd_ == f.Nd());

    int fkx = f.kx();
    int fkz = f.kz();

    if (fkx < kxmin() || fkx > kxmax() || fkz < kzmin() || fkz > kzmax())
        return *this;

    int m_x = mx(fkx);
    int m_z = mz(fkz);
    Complex minus_i(0.0, 1.0);
    const BasisFunc& psi = f.psi;

    switch (f.sign()) {
        case Minus:
            for (int i = 0; i < Nd_; ++i)
                for (int ny = 0; ny < Ny_; ++ny)
                    cmplx(m_x, ny, m_z, i) -= minus_i * psi[i][ny];
            break;

        case Plus:
            for (int i = 0; i < Nd_; ++i)
                for (int ny = 0; ny < Ny_; ++ny)
                    cmplx(m_x, ny, m_z, i) -= psi[i][ny];
            break;
    }

    if (fkz == 0 && fkx >= 0 && -fkx >= kxmin()) {
        int nmx = mx(-fkx);
        switch (f.sign()) {
            case Minus:
                for (int i = 0; i < Nd_; ++i)
                    for (int ny = 0; ny < Ny_; ++ny)
                        cmplx(nmx, ny, m_z, i) -= conj(minus_i * (psi[i][ny]));
                break;

            case Plus:
                for (int i = 0; i < Nd_; ++i)
                    for (int ny = 0; ny < Ny_; ++ny)
                        cmplx(nmx, ny, m_z, i) -= conj(psi[i][ny]);
                break;
        }
    }
    return *this;
}

FlowField& FlowField::operator+=(const RealProfileNG& u) {
    assert(xzstate_ == Spectral);
    assert(congruent(u));

    ComplexChebyCoeff tmp(Ny(), a(), b(), Spectral);
    const int mx_p = mx(abs(u.jx()));
    const int mx_m = mx(-abs(u.jx()));
    const int mz_p = mz(abs(u.jz()));

    for (int i = 0; i < Nd_; ++i) {
        const Complex norm_p = u.normalization_p(i);
        const Complex norm_m = u.normalization_m(i);

        tmp.setToZero();
        tmp.re = u.u_[i];

        tmp *= norm_p;
        for (int ny = 0; ny < Ny(); ++ny)
            cmplx(mx_p, ny, mz_p, i) += tmp[ny];
        if (norm_m != Complex(0, 0)) {
            tmp *= norm_m / norm_p;
            for (int ny = 0; ny < Ny(); ++ny)
                cmplx(mx_m, ny, mz_p, i) += tmp[ny];
        }
    }
    return *this;
}

FlowField& FlowField::operator-=(const RealProfileNG& u) {
    assert(xzstate_ == Spectral);
    assert(congruent(u));

    ComplexChebyCoeff tmp(Ny(), a(), b(), Spectral);
    const int mx_p = mx(abs(u.jx()));
    const int mx_m = mx(-abs(u.jx()));
    const int mz_p = mz(abs(u.jz()));
    for (int i = 0; i < Nd_; ++i) {
        const Complex norm_p = u.normalization_p(i);
        const Complex norm_m = u.normalization_m(i);

        tmp.setToZero();
        tmp.re = u.u_[i];
        tmp *= norm_p;
        for (int ny = 0; ny < Ny(); ++ny)
            cmplx(mx_p, ny, mz_p, i) -= tmp[ny];
        if (norm_m != Complex(0, 0)) {
            tmp *= norm_m / norm_p;
            for (int ny = 0; ny < Ny(); ++ny)
                cmplx(mx_m, ny, mz_p, i) -= tmp[ny];
        }
    }
    return *this;
}

FlowField& FlowField::operator+=(const FlowField& u) {
    assert(congruent(u));
    //     int Ntotal = Nx_ * Ny_ * Nzpad_ * Nd_;
    for (int i = 0; i < Nloc_; ++i)
        rdata_[i] += u.rdata_[i];
    return *this;
}
FlowField& FlowField::operator-=(const FlowField& u) {
    assert(congruent(u));
    for (int i = 0; i < Nloc_; ++i)
        rdata_[i] -= u.rdata_[i];
    return *this;
}

FlowField FlowField::operator[](int i) const {
    FlowField ui(Nx_, Ny_, Nz_, 1, Lx_, Lz_, a_, b_, cfmpi_, xzstate_, ystate_);

    if (xzstate_ == Spectral)
        for (int my = 0; my < My(); ++my)
            for (int mx = mxlocmin_; mx < Mxloc_ + mxlocmin_; ++mx)
                for (int mz = mzlocmin_; mz < Mzloc_ + mzlocmin_; ++mz)
                    ui.cmplx(mx, my, mz, 0) = this->cmplx(mx, my, mz, i);
    else
        for (int ny = nylocmin_; ny < nylocmax_; ++ny)
            for (int nx = nxlocmin_; nx < nxlocmin_ + Nxloc_; ++nx)
                for (int nz = 0; nz < Nz(); ++nz)
                    ui(nx, ny, nz, 0) = (*this)(nx, ny, nz, i);
    return ui;
}

FlowField FlowField::operator[](const cfarray<int>& indices) const {
    const int Nd = indices.length();
    FlowField rtn(Nx_, Ny_, Nz_, Nd, Lx_, Lz_, a_, b_, cfmpi_, xzstate_, ystate_);

    if (xzstate_ == Spectral) {
        for (int i = 0; i < Nd; ++i) {
            const int j = indices[i];
            for (int my = 0; my < My(); ++my)
                for (int mx = mxlocmin_; mx < mxlocmin_ + Mxloc_; ++mx)
                    for (int mz = mzlocmin_; mz < Mzloc_ + mzlocmin_; ++mz)
                        rtn.cmplx(mx, my, mz, i) = this->cmplx(mx, my, mz, j);
        }
    } else {
        for (int i = 0; i < Nd; ++i) {
            const int j = indices[i];
            for (int ny = nylocmin_; ny < nylocmax_; ++ny)
                for (int nx = nxlocmin_; nx < nxlocmin_ + Nxloc_; ++nx)
                    for (int nz = 0; nz < Nz(); ++nz)
                        rtn(nx, ny, nz, i) = (*this)(nx, ny, nz, j);
        }
    }
    return rtn;
}

Real FlowField::eval(Real x, Real y, Real z, int i) const {
    assertState(Spectral, Spectral);

    Real alpha_x = 2 * pi * x / Lx_;
    Real gamma_z = 2 * pi * z / Lz_;

    ChebyCoeff uprof(My(), a_, b_, Spectral);

    // At a given (x,z), collect u(y) as Chebyshev expansion
    // by evaluating complex exponentials of Fourier expansion

    for (int my = 0; my < My(); ++my) {
        // Compute u_my = myth Chebyshev coefficient as tmp.
        // tmp += 2*Re(cmplx(mx,my,mz,i) in the mx,mz loop double-counts
        // the 0,0 mode, so presubtract it
        // Real u_my = -Re(u.cmplx(0,0,0,i));
        Real u_my = 0;

        for (int mx = 0; mx < Mx(); ++mx) {
            int kx = this->kx(mx);
            Real cx = kx * alpha_x;  // cx = 2pi kx x /Lx

            // Unroll kz=0 terms, so as not to double-count the real part
            u_my += Re(this->cmplx(mx, my, 0, i) * exp(Complex(0.0, cx)));

            for (int mz = 1; mz < Mz(); ++mz) {
                // RHS = u_{kx,kz} exp(2pi i (x kx/Lx + z kz/Lz)) + complex conj
                Real cz = mz * gamma_z;
                u_my += 2 * Re(this->cmplx(mx, my, mz, i) * exp(Complex(0.0, (cx + cz))));
            }
        }
        uprof[my] = u_my;
    }
    uprof.makeSpectral();
    return uprof.eval(y);
}

void FlowField::makeSpectral_xz() {
    if (xzstate_ == Spectral)
        return;
    fftw_execute(xz_plan_.get());

    Real scale = 1.0 / (Nx_ * Nz_);
    for (lint i = 0; i < Nloc_; ++i)
        rdata_[i] *= scale;

#ifdef HAVE_MPI
    if (nproc0() > 1) {
        for (lint iz = 0; iz < Mzloc_; iz++) {
            lint offset = 2 * iz * Mxloc_ * Nypad_ * Nd_;
            fftw_mpi_execute_r2r(t_plan_.get(), &rdata_[offset], &rdata_[offset]);
        }
    }
#endif
    xzstate_ = Spectral;
}

void FlowField::makePhysical_xz() {
    if (xzstate_ == Physical)
        return;

#ifdef HAVE_MPI
    if (nproc0() > 1) {
        for (lint iz = 0; iz < Mzloc_; iz++) {
            lint offset = 2 * iz * Mxloc_ * Nypad_ * Nd_;
            fftw_mpi_execute_r2r(t_iplan_.get(), &rdata_[offset], &rdata_[offset]);
        }
    }
#endif

    fftw_execute(xz_iplan_.get());

    xzstate_ = Physical;
}

void FlowField::makeSpectral_y() {
    if (ystate_ == Spectral)
        return;

    if (Ny_ < 2) {
        ystate_ = Spectral;
        return;
    }

#ifdef HAVE_MPI
    Real nrm = 1.0 / (Ny_ - 1);  // needed because FFTW does unnormalized transforms
    for (lint mz = 0; mz < Mzloc_; ++mz)
        for (lint mx = 0; mx < Mxloc_; ++mx) {
            lint offset = Nypad_ * Nd_ * 2 * (mx + Mxloc_ * mz);
            fftw_execute_r2r(y_plan_.get(), &rdata_[offset], &rdata_[offset]);
            for (lint i = 0; i < Nd_; i++) {
                cdata_[complex_flatten(mx + mxlocmin_, 0, mz + mzlocmin_, i)] *= 0.5;
                cdata_[complex_flatten(mx + mxlocmin_, Ny_ - 1, mz + mzlocmin_, i)] *= 0.5;
            }
        }
    for (lint i = 0; i < Nloc_; ++i)
        rdata_[i] *= nrm;

#else
    Real nrm = 1.0 / (Ny_ - 1);  // needed because FFTW does unnormalized transforms
    for (int i = 0; i < Nd_; ++i) {
        unique_ptr<Real, void (*)(void*)> scratch_handle(static_cast<Real*>(fftw_malloc(Ny_ * sizeof(Real))),
                                                         fftw_free);
        auto scratch_t = scratch_handle.get();
        for (int nx = 0; nx < Nx_; ++nx) {
            for (int nz = 0; nz < Nzpad_; ++nz) {
                // Copy data spread through memory into a stride-1 scratch cfarray.
                for (int ny = 0; ny < Ny_; ++ny)
                    scratch_t[ny] = rdata_[flatten(nx, ny, nz, i)];

                // Transform data
                fftw_execute_r2r(y_plan_.get(), scratch_t, scratch_t);

                // Copy back to multi-d cfarrays, normalizing on the way
                // 0th elem is different because of reln btwn cos and Cheb transforms
                rdata_[flatten(nx, 0, nz, i)] = 0.5 * nrm * scratch_t[0];
                for (int ny = 1; ny < Ny_ - 1; ++ny)
                    rdata_[flatten(nx, ny, nz, i)] = nrm * scratch_t[ny];
                rdata_[flatten(nx, Ny_ - 1, nz, i)] = 0.5 * nrm * scratch_t[Ny_ - 1];
            }
        }
    }
#endif
    ystate_ = Spectral;
}

void FlowField::makePhysical_y() {
    if (ystate_ == Physical)
        return;

    if (Ny_ < 2) {
        ystate_ = Physical;
        return;
    }
#ifdef HAVE_MPI
    assert(xzstate_ == Spectral);
    for (lint i = 0; i < Nloc_; ++i)
        rdata_[i] *= 0.5;
    for (lint mz = 0; mz < Mzloc_; ++mz)
        for (lint mx = 0; mx < Mxloc_; ++mx) {
            for (lint i = 0; i < Nd_; i++) {
                cdata_[complex_flatten(mx + mxlocmin_, 0, mz + mzlocmin_, i)] *= 2;
                cdata_[complex_flatten(mx + mxlocmin_, Ny_ - 1, mz + mzlocmin_, i)] *= 2;
            }
            lint offset = Nypad_ * Nd_ * 2 * (mx + Mxloc_ * mz);
            fftw_execute_r2r(y_plan_.get(), &rdata_[offset], &rdata_[offset]);
        }

#else
    for (int i = 0; i < Nd_; ++i) {
        unique_ptr<Real, void (*)(void*)> scratch_handle(static_cast<Real*>(fftw_malloc(Ny_ * sizeof(Real))),
                                                         fftw_free);
        auto scratch_t = scratch_handle.get();
        for (int nx = 0; nx < Nx_; ++nx) {
            for (int nz = 0; nz < Nzpad_; ++nz) {
                // Copy data spread through memory into a stride-1 scratch cfarray.
                // 0th and last elems are different because of reln btwn cos and
                // Cheb transforms.
                scratch_t[0] = rdata_[flatten(nx, 0, nz, i)];
                for (int ny = 1; ny < Ny_ - 1; ++ny)
                    scratch_t[ny] = 0.5 * rdata_[flatten(nx, ny, nz, i)];
                scratch_t[Ny_ - 1] = rdata_[flatten(nx, Ny_ - 1, nz, i)];

                // Transform data
                fftw_execute_r2r(y_plan_.get(), scratch_t, scratch_t);

                // Copy transformed data back into main data cfarray
                for (int ny = 0; ny < Ny_; ++ny)
                    rdata_[flatten(nx, ny, nz, i)] = scratch_t[ny];
            }
        }
    }
#endif
    ystate_ = Physical;
}

void FlowField::makeSpectral() {
    makeSpectral_xz();
    makeSpectral_y();
}
void FlowField::makePhysical() {
    // Reversed order for MPI
    makePhysical_y();
    makePhysical_xz();
}

void FlowField::makeState(fieldstate xzstate, fieldstate ystate) {
    // Right order for MPI
    if (ystate == Physical && xzstate == Physical)
        makePhysical();
    else if (ystate == Spectral && xzstate == Spectral)
        makeSpectral();
    else if (ystate == Physical && xzstate == Spectral) {
        makeSpectral_xz();
        makePhysical_y();
    } else if (ystate == Spectral && xzstate == Physical) {
#ifdef HAVE_MPI
        cferror("The state Physical Spectral is not possible with MPI data-distribution");
#endif
        makeSpectral_y();
        makePhysical_xz();
    }
}

Complex FlowField::Dx(int mx, int n) const {
    Complex rot(0.0, 0.0);
    switch (n % 4) {
        case 0:
            rot = Complex(1.0, 0.0);
            break;
        case 1:
            rot = Complex(0.0, 1.0);
            break;
        case 2:
            rot = Complex(-1.0, 0.0);
            break;
        case 3:
            rot = Complex(0.0, -1.0);
            break;
        default:
            cferror("FlowField::Dx(mx,n) : impossible: n % 4 > 4 !!");
    }
    int kx_ = kx(mx);
    return rot * (std::pow(2 * pi * kx_ / Lx_, n) * zero_last_mode(kx_, kxmax(), n));
}
Complex FlowField::Dz(int mz, int n) const {
    Complex rot(0.0, 0.0);
    switch (n % 4) {
        case 0:
            rot = Complex(1.0, 0.0);
            break;
        case 1:
            rot = Complex(0.0, 1.0);
            break;
        case 2:
            rot = Complex(-1.0, 0.0);
            break;
        case 3:
            rot = Complex(0.0, -1.0);
            break;
        default:
            cferror("FlowField::Dx(mx,n) : impossible: n % 4 > 4 !!");
    }
    int kz_ = kz(mz);
    return rot * (std::pow(2 * pi * kz_ / Lz_, n) * zero_last_mode(kz_, kzmax(), n));
}

Complex FlowField::Dx(int mx) const {
    int kx_ = kx(mx);
    return Complex(0.0, 2 * pi * kx_ / Lx_ * zero_last_mode(kx_, kxmax(), 1));
}
Complex FlowField::Dz(int mz) const {
    int kz_ = kz(mz);
    return Complex(0.0, 2 * pi * kz_ / Lz_ * zero_last_mode(kz_, kzmax(), 1));
}

void FlowField::addPerturbation(int kx, int kz, Real mag, Real decay) {
    assertState(Spectral, Spectral);
    if (mag == 0.0)
        return;

    // Add a div-free perturbation to the base flow.
    ComplexChebyCoeff u(Ny_, a_, b_, Spectral);
    ComplexChebyCoeff v(Ny_, a_, b_, Spectral);
    ComplexChebyCoeff w(Ny_, a_, b_, Spectral);
    randomProfile(u, v, w, kx, kz, Lx_, Lz_, mag, decay);
    Real k = 2 * pi * sqrt(kx * kx / (Lx_ * Lx_) + kz * kz / (Lz_ * Lz_));
    u *= pow(decay, k);
    v *= pow(decay, k);
    w *= pow(decay, k);
    if (mx(kx) >= mxlocmin_ && mx(kx) < mxlocmin_ + Mxloc_ && mz(kz) >= mzlocmin_ && mz(kz) < mzlocmin_ + Mzloc_) {
        int m_x = mx(kx);
        int m_z = mz(kz);
        for (int ny = nylocmin_; ny < nylocmax_; ++ny) {
            cmplx(m_x, ny, m_z, 0) += u[ny];
            cmplx(m_x, ny, m_z, 1) += v[ny];
            cmplx(m_x, ny, m_z, 2) += w[ny];
        }
    }

    if (kz == 0 && kx != 0) {
        int m_x = mx(-kx);
        int m_z = mz(0);  // -kz=0
        if (m_x >= mxlocmin_ && m_x < mxlocmin_ + Mxloc_ && mz(kz) >= mzlocmin_ && mz(kz) < mzlocmin_ + Mzloc_) {
            for (int i = 0; i < Nd_; ++i)
                for (int ny = nylocmin_; ny < nylocmax_; ++ny) {
                    cmplx(m_x, ny, m_z, 0) += conj(u[ny]);
                    cmplx(m_x, ny, m_z, 1) += conj(v[ny]);
                    cmplx(m_x, ny, m_z, 2) += conj(w[ny]);
                }
        }
    }
}

void FlowField::addPerturbation1D(int kx, int kz, Real mag, Real decay) {
    assertState(Spectral, Spectral);
    if (mag == 0.0)
        return;

    // Add a div-free perturbation to the base flow.
    ComplexChebyCoeff u(Ny_, a_, b_, Spectral);
    ComplexChebyCoeff vd(Ny_, a_, b_, Spectral);  // dummy variable
    ComplexChebyCoeff wd(Ny_, a_, b_, Spectral);  // dummy variable
    randomProfile(u, vd, wd, kx, kz, Lx_, Lz_, mag, decay);
    Real k = 2 * pi * sqrt(kx * kx / (Lx_ * Lx_) + kz * kz / (Lz_ * Lz_));
    u *= pow(decay, k);
    if (mx(kx) >= mxlocmin_ && mx(kx) < mxlocmin_ + Mxloc_ && mz(kz) >= mzlocmin_ && mz(kz) < mzlocmin_ + Mzloc_) {
        int m_x = mx(kx);
        int m_z = mz(kz);
        for (int ny = nylocmin_; ny < nylocmax_; ++ny) {
            cmplx(m_x, ny, m_z, 0) += u[ny];
        }
    }

    if (kz == 0 && kx != 0) {
        int m_x = mx(-kx);
        int m_z = mz(0);  // -kz=0
        if (m_x >= mxlocmin_ && m_x < mxlocmin_ + Mxloc_ && mz(kz) >= mzlocmin_ && mz(kz) < mzlocmin_ + Mzloc_) {
            for (int i = 0; i < Nd_; ++i)
                for (int ny = nylocmin_; ny < nylocmax_; ++ny) {
                    cmplx(m_x, ny, m_z, 0) += conj(u[ny]);
                }
        }
    }
}

void FlowField::perturb(Real mag, Real decay, bool meanflow) {
    int Kx = padded() ? kxmaxDealiased() : kxmax();
    int Kz = padded() ? kzmaxDealiased() : kzmax();
    addPerturbations(Kx, Kz, mag, decay, meanflow);
}

void FlowField::addPerturbations(int Kx, int Kz, Real mag, Real decay, bool meanflow) {
    assertState(Spectral, Spectral);
    if (mag == 0.0)
        return;

    int Kxmin = Greater(-Kx, padded() ? kxminDealiased() : kxmin());
    int Kxmax = lesser(Kx, padded() ? kxmaxDealiased() : kxmax() - 1);
    int Kzmax = lesser(Kz, padded() ? kzmaxDealiased() : kzmax() - 1);

    if (Nd_ > 2) {
        // Add a div-free perturbation to the base flow.
        for (int kx = Kxmin; kx <= Kxmax; ++kx)
            for (int kz = 0; kz <= Kzmax; ++kz) {
                // Real norm = pow(10.0, -(abs(2*pi*kx/Lx_) + abs(2*pi*kz/Lz_)));
                // Real norm = pow(decay, 2*(abs(kx) + abs(kz)));
                Real norm = std::pow(decay, abs(2 * pi * kx / Lx_) + abs(2 * pi * kz / Lz_));
                if (meanflow || !(kx == 0 && kz == 0))
                    addPerturbation(kx, kz, mag * norm, decay);
            }
    } else {
        // Add a div-free perturbation to the base flow.
        for (int kx = Kxmin; kx <= Kxmax; ++kx)
            for (int kz = 0; kz <= Kzmax; ++kz) {
                // Real norm = pow(10.0, -(abs(2*pi*kx/Lx_) + abs(2*pi*kz/Lz_)));
                // Real norm = pow(decay, 2*(abs(kx) + abs(kz)));
                Real norm = std::pow(decay, abs(2 * pi * kx / Lx_) + abs(2 * pi * kz / Lz_));
                if (meanflow || !(kx == 0 && kz == 0))
                    addPerturbation1D(kx, kz, mag * norm, decay);
            }
    }
    makePhysical();
    makeSpectral();
}

void FlowField::addPerturbations(Real mag, Real decay, bool meanflow) {
    assertState(Spectral, Spectral);
    if (mag == 0.0)
        return;

    int Kxmin = padded() ? kxminDealiased() : kxmin();
    int Kxmax = padded() ? kxmaxDealiased() : kxmax() - 1;
    int Kzmax = padded() ? kzmaxDealiased() : kzmax() - 1;

    if (Nd_ > 2) {
        // Add a div-free perturbation to the base flow.
        for (int mx = mxlocmin_; mx < mxlocmin_ + Mxloc_; ++mx) {
            int kx_ = kx(mx);
            if (kx_ < Kxmin || kx_ > Kxmax)
                continue;
            for (int mz = mzlocmin_; mz < mzlocmin_ + Mzloc_; ++mz) {
                int kz_ = kz(mz);
                if (abs(kz_) > Kzmax)
                    continue;
                Real norm = std::pow(decay, 2 * (abs(kx_) + abs(kz_)));
                if (meanflow || !(kx_ == 0 && kz_ == 0)) {
                    addPerturbation(kx_, kz_, mag * norm, decay);
                }
            }
        }
    } else {
        // Add a div-free perturbation to the base flow.
        for (int mx = mxlocmin_; mx < mxlocmin_ + Mxloc_; ++mx) {
            int kx_ = kx(mx);
            if (kx_ < Kxmin || kx_ > Kxmax)
                continue;
            for (int mz = mzlocmin_; mz < mzlocmin_ + Mzloc_; ++mz) {
                int kz_ = kz(mz);
                if (abs(kz_) > Kzmax)
                    continue;
                Real norm = std::pow(decay, 2 * (abs(kx_) + abs(kz_)));
                if (meanflow || !(kx_ == 0 && kz_ == 0)) {
                    addPerturbation1D(kx_, kz_, mag * norm, decay);
                }
            }
        }
    }
    makePhysical();
    makeSpectral();
}

bool FlowField::padded() const { return padded_; }

void FlowField::setPadded(bool b) { padded_ = b; }

void FlowField::setToZero() {
    //     int Ntotal = Nx_*Ny_*Nzpad_*Nd_;
    for (int i = 0; i < Nloc_; ++i)
        rdata_[i] = 0.0;
}

void FlowField::zeroPaddedModes() {
    fieldstate xzs = xzstate_;
    fieldstate ys = ystate_;
    makeSpectral_xz();

    // Not as efficient as possible but less prone to error and more robust
    // to changes in aliasing bounds, and the efficiency diff is negligible
    Complex zero(0.0, 0.0);
    for (int mx = mxlocmin_; mx < mxlocmin_ + Mxloc_; ++mx) {
        int kx_ = this->kx(mx);
        for (int mz = mzlocmin_; mz < mzlocmin_ + Mzloc_; ++mz) {
            int kz_ = this->kz(mz);
            if (isAliased(kx_, kz_))
                for (int i = 0; i < Nd_; ++i)
                    for (int ny = 0; ny < Ny_; ++ny)
                        this->cmplx(mx, ny, mz, i) = zero;
        }
    }
    padded_ = true;
    makeState(xzs, ys);
}

void FlowField::print() const {
    cout << Nx_ << " x " << Ny_ << " x " << Nz_ << endl;
    cout << "[0, " << Lx_ << "] x [-1, 1] x [0, " << Lz_ << "]" << endl;
    cout << xzstate_ << " x " << ystate_ << " x " << xzstate_ << endl;
    cout << xzstate_ << " x " << ystate_ << " x " << xzstate_ << endl;
    if (xzstate_ == Spectral) {
        cout << "FlowField::print() real view " << endl;
        for (int i = 0; i < Nd_; ++i) {
            for (int ny = 0; ny < Ny_; ++ny) {
                for (int nx = 0; nx < Nx_; ++nx) {
                    cout << "i=" << i << " ny=" << ny << " nx= " << nx << ' ';
                    int nz;  // MSVC++ FOR-SCOPE BUG
                    for (nz = 0; nz < Nz_; ++nz)
                        cout << rdata_[flatten(nx, ny, nz, i)] << ' ';
                    cout << " pad : ";
                    for (nz = Nz_; nz < Nzpad_; ++nz)
                        cout << rdata_[flatten(nx, ny, nz, i)] << ' ';
                    cout << endl;
                }
            }
        }
    } else {
        cout << "complex view " << endl;
        for (int i = 0; i < Nd_; ++i) {
            for (int ny = 0; ny < Ny_; ++ny) {
                for (int nx = 0; nx < Nx_; ++nx) {
                    cout << "i=" << i << " ny=" << ny << " nx= " << nx << ' ';
                    for (int nz = 0; nz < Nz_ / 2; ++nz)
                        cout << cdata_[complex_flatten(nx, ny, nz, i)] << ' ';
                    cout << endl;
                }
            }
        }
    }
}

// k == direction of normal    (e.g. k=0 means a x-normal slice in yz plane.
// i == component of FlowField (e.g. i=0 means u-component)
// n == nth gridpoint along k direction
void FlowField::saveSlice(int k, int i, int nk, const string& filebase, int xstride, int ystride, int zstride) const {
    assert(k >= 0 && k < 3);
    assert(i >= 0 && i < Nd_);

    if (mpirank() > 0)
        cferror("Function saveSlice is not mpi safe");

    string filename(filebase);
    filename += string(".asc");
    ofstream os(filename.c_str());

    FlowField u(*this);
    fieldstate xzstate = xzstate_;
    fieldstate ystate = ystate_;

    u.makePhysical();

    switch (k) {
        case 0:
            os << "% yz slice\n";
            os << "% (i,j)th elem is field at (x_n, y_i, z_j) with x_n is fixed\n";
            for (int ny = 0; ny < Ny_; ny += ystride) {
                for (int nz = 0; nz < Nz_; nz += zstride)
                    os << u(nk, ny, nz, i) << ' ';
                os << u(nk, ny, 0, i) << '\n';  // repeat nz=0 to complete [0, Lz]
            }
            break;
        case 1:  // xz slice
            os << "% xz slice\n";
            os << "% (i,j)th elem is field at (x_j, y_n, z_i) with y_n fixed\n";
            for (int nz = 0; nz < Nz_; nz += zstride) {
                for (int nx = 0; nx < Nx_; nx += xstride)
                    os << u(nx, nk, nz, i) << ' ';
                os << u(0, nk, nz, i) << '\n';  // repeat nx=0 to complete [0, Lx]
            }
            for (int nx = 0; nx < Nx_; ++nx)  // repeat nz=0 to complete [0, Lz]
                os << u(nx, nk, 0, i) << ' ';
            os << u(0, nk, 0, i) << '\n';  // repeat nx=0 to complete [0, Lx]

            break;
        case 2:
            os << "% xy slice\n";
            os << "% (i,j)th elem is field at (x_j, y_i, z_n)\n";
            for (int ny = 0; ny < Ny_; ny += ystride) {
                for (int nx = 0; nx < Nx_; nx += xstride)
                    os << u(nx, ny, nk, i) << ' ';
                os << u(0, ny, nk, i) << '\n';
            }
            break;
    }
    u.makeState(xzstate, ystate);
}

void FlowField::saveProfile(int mx, int mz, const string& filebase) const {
    ChebyTransform trans(Ny_);
    saveProfile(mx, mz, filebase, trans);
}

void FlowField::saveProfile(int mx, int mz, const string& filebase, const ChebyTransform& trans) const {
    if (mpirank() > 0)
        cferror("Function saveProfile is not mpi safe");

    assert(xzstate_ == Spectral);
    string filename(filebase);
    if (Nd_ == 3)
        filename += string(".bf");  // this convention is unfortunate, need to fix
    else
        filename += string(".asc");

    ofstream os(filename.c_str());
    os << setprecision(REAL_DIGITS);

    if (ystate_ == Physical) {
        for (int ny = 0; ny < Ny_; ++ny) {
            for (int i = 0; i < Nd_; ++i) {
                Complex c = (*this).cmplx(mx, ny, mz, i);
                os << Re(c) << ' ' << Im(c) << ' ';
            }
            os << '\n';
        }
    } else {
        ComplexChebyCoeff* f = new ComplexChebyCoeff[Nd_];
        for (int i = 0; i < Nd_; ++i) {
            f[i] = ComplexChebyCoeff(Ny_, a_, b_, Spectral);
            for (int ny = 0; ny < Ny_; ++ny)
                f[i].set(ny, (*this).cmplx(mx, ny, mz, i));
            f[i].makePhysical(trans);
        }
        for (int ny = 0; ny < Ny_; ++ny) {
            for (int i = 0; i < Nd_; ++i)
                os << f[i].re[ny] << ' ' << f[i].im[ny] << ' ';
            os << '\n';
        }
        delete[] f;
    }
}
void FlowField::saveSpectrum(const string& filebase, int i, int ny, bool kxorder, bool showpadding) const {
    if (mpirank() > 0)
        cferror("Function saveSpectrum is not mpi safe");

    string filename(filebase);
    filename += string(".asc");
    ofstream os(filename.c_str());

    bool sum = (ny == -1) ? true : false;
    assert(xzstate_ == Spectral);

    if (kxorder) {
        int Kxmin = (showpadding || !padded_) ? kxmin() : kxminDealiased();
        int Kxmax = (showpadding || !padded_) ? kxmax() : kxmaxDealiased();
        int Kzmin = (showpadding || !padded_) ? kzmin() : kzminDealiased();
        int Kzmax = (showpadding || !padded_) ? kzmax() : kzmaxDealiased();
        for (int kx = Kxmin; kx <= Kxmax; ++kx) {
            for (int kz = Kzmin; kz <= Kzmax; ++kz) {
                if (sum)
                    os << sqrt(energy(mx(kx), mz(kz))) << ' ';
                else {
                    Complex f = this->cmplx(mx(kx), ny, mz(kz), i);
                    os << Re(f) << ' ' << Im(f) << ' ';
                }
            }
            os << endl;
        }
    } else {
        for (int mx = 0; mx < Mx(); ++mx) {
            for (int mz = 0; mz < Mz(); ++mz) {
                if (sum)
                    os << sqrt(energy(mx, mz)) << ' ';
                else {
                    Complex f = this->cmplx(mx, ny, mz, i);
                    os << Re(f) << ' ' << Im(f) << ' ';
                }
            }
            os << endl;
        }
    }
    os.close();
}

void FlowField::saveSpectrum(const string& filebase, bool kxorder, bool showpadding) const {
    if (mpirank() > 0)
        cferror("Function saveSpectrum is not mpi safe");

    string filename(filebase);
    filename += string(".asc");
    ofstream os(filename.c_str());

    assert(xzstate_ == Spectral && ystate_ == Spectral);

    ComplexChebyCoeff u(Ny_, a_, b_, Spectral);

    if (kxorder) {
        int Kxmin = (showpadding || !padded_) ? kxmin() : kxminDealiased();
        int Kxmax = (showpadding || !padded_) ? kxmax() : kxmaxDealiased();
        int Kzmin = (showpadding || !padded_) ? kzmin() : kzminDealiased();
        int Kzmax = (showpadding || !padded_) ? kzmax() : kzmaxDealiased();
        for (int kx = Kxmin; kx <= Kxmax; ++kx) {
            for (int kz = Kzmin; kz <= Kzmax; ++kz) {
                Real e = 0.0;
                for (int i = 0; i < Nd_; ++i) {
                    for (int ny = 0; ny < Ny_; ++ny)
                        u.set(ny, this->cmplx(mx(kx), ny, mz(kz), i));
                    e += L2Norm2(u);
                }
                os << sqrt(e) << ' ';
            }
            os << endl;
        }
    } else {
        for (int mx = 0; mx < Mx(); ++mx) {
            for (int mz = 0; mz < Mz(); ++mz) {
                Real e = 0.0;
                for (int i = 0; i < Nd_; ++i) {
                    for (int ny = 0; ny < Ny_; ++ny)
                        u.set(ny, this->cmplx(mx, ny, mz, i));
                    e += L2Norm2(u);
                }
                os << sqrt(e) << ' ';
            }
            os << endl;
        }
    }
    os.close();
}

void FlowField::saveDivSpectrum(const string& filebase, bool kxorder, bool showpadding) const {
    string filename(filebase);
    if (mpirank() > 0)
        cferror("Function saveDivSpectrum is not mpi safe");

    filename += string(".asc");
    ofstream os(filename.c_str());

    assert(xzstate_ == Spectral && ystate_ == Spectral);

    // assert(congruent(div));
    assert(Nd_ >= 3);

    ComplexChebyCoeff v(Ny_, a_, b_, Spectral);
    ComplexChebyCoeff vy(Ny_, a_, b_, Spectral);

    // Rely on compiler to pull loop invariants out of loops
    if (kxorder) {
        int Kxmin = kxmin();
        int Kxmax = (showpadding || !padded_) ? kxmax() : kxmaxDealiased();
        int Kzmin = (showpadding || !padded_) ? kzmin() : kzminDealiased();
        int Kzmax = (showpadding || !padded_) ? kzmax() : kzmaxDealiased();
        for (int kx = Kxmin; kx < Kxmax; ++kx) {
            Complex d_dx(0.0, 2 * pi * kx / Lx_ * zero_last_mode(kx, kxmax(), 1));

            for (int kz = Kzmin; kz < Kzmax; ++kz) {
                Complex d_dz(0.0, 2 * pi * kz / Lz_ * zero_last_mode(kz, kzmax(), 1));

                for (int ny = 0; ny < Ny_; ++ny)
                    v.set(ny, cmplx(mx(kx), ny, mz(kz), 1));
                diff(v, vy);

                Real div = 0.0;
                for (int ny = 0; ny < Ny_; ++ny) {
                    Complex ux = d_dx * cmplx(mx(kx), ny, mz(kz), 0);
                    Complex wz = d_dz * cmplx(mx(kx), ny, mz(kz), 2);
                    div += abs2(ux + vy[ny] + wz);
                }
                os << sqrt(div) << ' ';
            }
            os << endl;
        }
    } else {
        for (int mx = 0; mx < Mx(); ++mx) {
            Complex d_dx(0.0, 2 * pi * kx(mx) / Lx_ * zero_last_mode(kx(mx), kxmax(), 1));

            for (int mz = 0; mz < Mz(); ++mz) {
                Complex d_dz(0.0, 2 * pi * kz(mz) / Lz_ * zero_last_mode(kz(mz), kzmax(), 1));

                for (int ny = 0; ny < Ny_; ++ny)
                    v.set(ny, cmplx(mx, ny, mz, 1));
                diff(v, vy);

                Real div = 0.0;
                for (int ny = 0; ny < Ny_; ++ny) {
                    Complex ux = d_dx * cmplx(mx, ny, mz, 0);
                    Complex wz = d_dz * cmplx(mx, ny, mz, 2);
                    div += abs2(ux + vy[ny] + wz);
                }
                os << sqrt(div) << ' ';
            }
            os << endl;
        }
    }
    os.close();
}

void FlowField::asciiSave(const string& filebase) const {
    string filename(filebase);

    if (filename.find(".asc") == string::npos)
        filename += ".asc";
    if (mpirank() > 0)
        cferror("Function asciiSave is not mpi safe");

    ofstream os(filename.c_str());
    os << scientific << setprecision(REAL_DIGITS);

    const char s = ' ';
    const char nl = '\n';
    const int w = REAL_IOWIDTH;
    os << "% Channelflow FlowField data" << nl;
    os << "% xzstate == " << xzstate_ << nl;
    os << "%  ystate == " << ystate_ << nl;
    os << "% Nx Ny Nz Nd == " << Nx_ << s << Ny_ << s << Nz_ << s << Nd_ << " gridpoints\n";
    os << "% Mx My Mz Nd == " << Mx() << s << My() << s << Mz() << s << Nd_ << " spectral modes\n";
    os << "% Lx Lz == " << setw(w) << Lx_ << s << setw(w) << Lz_ << nl;
    os << "% Lx Lz == " << setw(w) << Lx_ << s << setw(w) << Lz_ << nl;
    os << "% a  b  == " << setw(w) << a_ << s << setw(w) << b_ << nl;
    os << "% loop order:\n";
    os << "%   for (int i=0; i<Nd; ++i)\n";
    os << "%     for(long ny=0; ny<Ny; ++ny) // note: Ny == My\n";
    if (xzstate_ == Physical) {
        os << "%       for (int nx=0; nx<Nx; ++nx)\n";
        os << "%         for (int nz=0; nz<Nz; ++nz)\n";
        os << "%           os << f(nx, ny, nz, i) << newline;\n";
    } else {
        os << "%       for (int mx=0; mx<Mx; ++mx)\n";
        os << "%         for (int mz=0; mz<Mz; ++mz)\n";
        os << "%           os << Re(f.cmplx(mx, ny, mz, i) << ' ' << Im(f.cmplx(mx, ny, mz, i) << newline;\n";
    }
    if (xzstate_ == Physical) {
        for (int i = 0; i < Nd_; ++i)
            for (long ny = 0; ny < Ny_; ++ny)
                for (int nx = 0; nx < Nx_; ++nx)
                    for (int nz = 0; nz < Nz_; ++nz)
                        os << setw(w) << (*this)(nx, ny, nz, i) << nl;
    } else {
        for (int i = 0; i < Nd_; ++i)
            for (long ny = 0; ny < Ny_; ++ny)
                for (int mx = 0; mx < Mx(); ++mx)
                    for (int mz = 0; mz < Mz(); ++mz)
                        os << setw(w) << Re(cmplx(mx, ny, mz, i)) << s << setw(w) << Im(cmplx(mx, ny, mz, i)) << nl;
    }
    os.close();
}

void FlowField::save(const string& filebase, vector<string> component_names) const {
    string filename;

    bool ncsuffix = hasSuffix(filebase, ".nc");
    bool h5suffix = hasSuffix(filebase, ".h5");
    bool ffsuffix = hasSuffix(filebase, ".ff");
    bool ascsuffix = hasSuffix(filebase, ".asc");
    bool vtksuffix = hasSuffix(filebase, ".vtk");

    /* suffix is given */
    if (ffsuffix)
        binarySave(filebase);
    else if (h5suffix)
        if (HAVE_HDF5)
            hdf5Save(filebase);
        else {
            cferror(
                "FlowField::save(filename) error : can't save to HDF5 file because HDF5 libraries are not installed. "
                "filename == " +
                filebase);
        }
    else if (ncsuffix)
        if (HAVE_NETCDF)
            writeNetCDF(filebase, component_names);
        else {
            cferror(
                "FlowField::save(filename) error : can't save to NetCDF file because NetCDF libraries are not "
                "installed. filename == " +
                filebase);
        }
    else if (ascsuffix)
        asciiSave(filebase);

    else if (vtksuffix)
        VTKSave(filebase);
    /* suffix is not given */
    else if (HAVE_NETCDF)
        writeNetCDF(filebase, component_names);
    else if (HAVE_HDF5)
        hdf5Save(filebase);
    else
        binarySave(filebase);
}

void FlowField::binarySave(const string& filebase) const {
    string filename = appendSuffix(filebase, ".ff");
    if (mpirank() > 0)
        cferror("Function binarySave is not mpi safe");

    ofstream os(filename.c_str(), ios::out | ios::binary);
    if (!os.good()) {
        cferror("FlowField::binarySave(filebase) : can't open file " + filename);
    }

    int major;
    int minor;
    int update;
    channelflowVersion(major, minor, update);

    write(os, major);
    write(os, minor);
    write(os, update);
    write(os, Nx_);
    write(os, Ny_);
    write(os, Nz_);
    write(os, Nd_);
    write(os, xzstate_);
    write(os, ystate_);
    write(os, Lx_);
    write(os, Lz_);
    write(os, a_);
    write(os, b_);
    write(os, padded_);

    // Write data only for non-aliased modes.
    if (padded_ && xzstate_ == Spectral) {
        int Nxd = 2 * (Nx_ / 6);
        int Nzd = 2 * (Nz_ / 3) + 1;

        // In innermost loop, cfarray index is (nz + Nzpad2_*(nx + Nx_*(ny + Ny_*i))),
        // which is the same as the FlowField::flatten function.
        for (int i = 0; i < Nd_; ++i) {
            for (int ny = 0; ny < Ny_; ++ny) {
                for (int nx = 0; nx <= Nxd; ++nx) {
                    for (int nz2 = 0; nz2 <= Nzd / 2; ++nz2) {
                        Complex comp;
                        comp = cmplx(nx, ny, nz2, i);
                        write(os, comp.real());
                        write(os, comp.imag());
                    }
                }
                for (int nx = Nx_ - Nxd; nx < Nx_; ++nx) {
                    for (int nz2 = 0; nz2 <= Nzd / 2; ++nz2) {
                        Complex comp;
                        comp = cmplx(nx, ny, nz2, i);
                        write(os, comp.real());
                        write(os, comp.imag());
                    }
                }
            }
        }
    } else if (padded_ == false && xzstate_ == Spectral) {
        for (int i = 0; i < Nd_; ++i) {
            for (int ny = 0; ny < Ny_; ++ny) {
                for (int nx = 0; nx < Nx_; ++nx) {
                    for (int nz2 = 0; nz2 < Nzpad2_; ++nz2) {
                        Complex comp;
                        comp = cmplx(nx, ny, nz2, i);
                        write(os, comp.real());
                        write(os, comp.imag());
                    }
                }
            }
        }

    } else {
        for (int i = 0; i < Nd_; ++i) {
            for (int ny = 0; ny < Ny_; ++ny) {
                for (int nx = 0; nx < Nx_; ++nx) {
                    for (int nz = 0; nz < Nzpad_; ++nz)
                        write(os, (*this)(nx, ny, nz, i));
                }
            }
        }
    }
}

#ifndef HAVE_LIBHDF5_CPP

void FlowField::hdf5Save(const string& filebase) const {
    cferror("FlowField::hdf5save requires HDF5 libraries. Please install them and recompile channelflow.");
}

#else
void FlowField::hdf5Save(const string& filebase) const {
    FlowField v;
    // If this FlowField is padded (last 1/3 x,z modes are set to zero)
    // transfer to nopadded grid and save that
    int Nx, Nz;
    if (this->padded()) {
        Nx = (2 * Nx_) / 3;
        Nz = (2 * Nz_) / 3;

    } else {
        Nx = Nx_;
        Nz = Nz_;
        //    v = *this;
    }

    FlowField& u = const_cast<FlowField&>(*this);
    fieldstate initxzstate = u.xzstate();
    fieldstate initystate = u.ystate();
    u.makeSpectral();

    CfMPI_single* CfMPI_one = nullptr;
#ifdef HAVE_MPI
    CfMPI_one = &CfMPI_single::getInstance();
#endif
    v = FlowField(Nx, Ny_, Nz, Nd_, Lx_, Lz_, a_, b_, CfMPI_one);  // FlowField is only on process 0 -- serial io
    v.interpolate(u);

    v.makePhysical();

    if (v.taskid() == 0) {
        H5std_string h5name = appendSuffix(filebase, ".h5");
        H5::H5File h5file(h5name, H5F_ACC_TRUNC);

        // create the groups we will need
        h5file.createGroup("/geom");
        h5file.createGroup("/data");

        hdf5write(v.xgridpts(), "/geom/x", h5file);
        hdf5write(v.ygridpts(), "/geom/y", h5file);
        hdf5write(v.zgridpts(), "/geom/z", h5file);
        hdf5write(Nx, "Nx", h5file);
        hdf5write(Ny_, "Ny", h5file);
        hdf5write(Nz, "Nz", h5file);
        hdf5write(Nd_, "Nd", h5file);
        hdf5write(Nx_, "Nxpad", h5file);
        hdf5write(Ny_, "Nypad", h5file);
        hdf5write(Nz_, "Nzpad", h5file);
        hdf5write(v, "/data/u", h5file);

        hdf5write(Lx_, "Lx", h5file);
        hdf5write(Lz_, "Lz", h5file);
        hdf5write(a_, "a", h5file);
        hdf5write(b_, "b", h5file);
    }
    u.makeState(initxzstate, initystate);
}
#endif  // HAVE_HDF5LIB_CPP

void FlowField::removePaddedModes(Real* rdata_io, lint Nxloc_io, lint nxlocmin_io, lint Mzloc_io,
                                  lint mzlocmin_io) const {
    /* Removal of padded modes in parallel and within the object (in contrast of using interpolation onto a smaller
     * grid, e.g. in hdf5Save)*/

    // bring flow field into intermediate state
    FlowField v(*this);  // make copy to save back transformation from intermediate state
    if (xzstate_ == Physical)
        v.makeSpectral_xz();
    if (ystate_ == Spectral)
        v.makePhysical_y();

        // xy-transform of v=copy(this object) to intermediate state which is xzstate=Spectral and ystate=Physical.
        // The final data layout is z* x y* i (* indicate distributed vars)
#ifdef HAVE_MPI
    if (nproc0() > 1) {
        for (lint iz = 0; iz < Mzloc_; iz++) {
            lint offset = 2 * iz * Mxloc_ * Nypad_ * Nd_;
            fftw_mpi_execute_r2r(v.t_iplan_.get(), &v.rdata_[offset], &v.rdata_[offset]);
        }
    }
#endif
    ///////////////////////////////////////////////////////////////// some parts of block below could be members

    // define data size parameters and allocate memory
    Complex* cdata_io = (Complex*)rdata_io;
    fftw_complex* fcdata_io = (fftw_complex*)cdata_io;

    lint Mx_io = Nx_io_;
    lint Mz_io = Nz_io_ / 2 + 1;
    lint lzero = 0;
    lint Mzloc_trunc = min(Mzloc_, max(Mz_io - mzlocmin_, lzero));  // local z-distribution after truncation

    int tmp1 = Mzloc_io;
    int tmp2 = 0;
    lint Mzlocmax_io = Mz_io;
#ifdef HAVE_MPI
    MPI_Allreduce(&tmp1, &tmp2, 1, MPI_INT, MPI_MAX, cfmpi_->comm_world);
    Mzlocmax_io = tmp2;
#else
    // does nothing, avoids "unused variable" warning in non-MPI compile
    (void)tmp1;
    (void)tmp2;
#endif

    // MPI related
    int np0 = nproc0();
    int np1 = nproc1();
    int np = np0 * np1;
    int rank = taskid();
    int yrank = rank / np1;

    // defined data chunk (z-slice) to be sent via MPI communcation
    lint Mx_chunk = Mx_io + np - (Mx_io % np);  // Must be a multiple of np for Nloc_chunk to be the same on all procs
    lint Mz_chunk = np1;
    lint howmany_io = Nylocpad_ * Nd_;
    lint Nloc_chunk = 2 * Mx_chunk * Mz_chunk * howmany_io;
#ifdef HAVE_MPI
    lint Mxloc_chunk = Mx_chunk;
    lint mxlocmin_chunk = nxlocmin_io;
    lint Mzloc_chunk = Mz_chunk;
    lint mzlocmin_chunk = mzlocmin_io;
    lint rank_chunk[2] = {Mx_chunk, Mz_chunk};
    Nloc_chunk = 2 * fftw_mpi_local_size_many_transposed(2, rank_chunk, howmany_io, FFTW_MPI_DEFAULT_BLOCK,
                                                         FFTW_MPI_DEFAULT_BLOCK, cfmpi_->comm1, &Mxloc_chunk,
                                                         &mxlocmin_chunk, &Mzloc_chunk, &mzlocmin_chunk);
#endif

    vector<unique_ptr<Complex, void (*)(void*)>> cdata_send{};
    vector<unique_ptr<Complex, void (*)(void*)>> cdata_receive{};

    for (int i = 0; i < Mzloc_trunc; i++)
        cdata_send.emplace_back(static_cast<Complex*>(fftw_malloc(Nloc_chunk * sizeof(Real))), fftw_free);
    for (int i = 0; i < Mzloc_io; i++)
        cdata_receive.emplace_back(static_cast<Complex*>(fftw_malloc(Nloc_chunk * sizeof(Real))), fftw_free);

    // design transform plan
    uint fftw_flags_io = FFTW_ESTIMATE;
    fftw_plan_unique_ptr_t xz_iplan_io = {nullptr, fftw_destroy_plan};
#ifdef HAVE_MPI
    lint rank_1_io[2] = {Nx_io_, Nz_io_};
    xz_iplan_io.reset(fftw_mpi_plan_many_dft_c2r(2, rank_1_io, howmany_io, FFTW_MPI_DEFAULT_BLOCK,
                                                 FFTW_MPI_DEFAULT_BLOCK, fcdata_io, rdata_io, cfmpi_->comm1,
                                                 fftw_flags_io | FFTW_MPI_TRANSPOSED_IN));
#else
    const int Nzpad_io = 2 * (Nz_io_ / 2 + 1);
    const int Nzpad2_io = Nz_io_ / 2 + 1;
    const int howmany = Ny_ * Nd_;
    const int rk = 2;
    // These params describe the structure of the real-valued cfarray
    int real_n[rk];
    real_n[0] = Nx_io_;
    real_n[1] = Nz_io_;
    int real_embed[rk];
    real_embed[0] = Nx_io_;
    real_embed[1] = Nzpad_io;
    const int real_stride = 1;
    const int real_dist = Nx_io_ * Nzpad_io;
    // These params describe the structure of the complex-valued cfarray
    int cplx_embed[rk];
    cplx_embed[0] = Nx_io_;
    cplx_embed[1] = Nzpad2_io;
    const int cplx_stride = 1;
    const int cplx_dist = Nx_io_ * Nzpad2_io;
    xz_iplan_io.reset(fftw_plan_many_dft_c2r(rk, real_n, howmany, fcdata_io, cplx_embed, cplx_stride, cplx_dist,
                                             rdata_io, real_embed, real_stride, real_dist, fftw_flags_io));
#endif

    // write data into the slices to be sent
#ifdef HAVE_MPI  // data is accessed differently under MPI
    for (int mz = mzlocmin_; mz < mzlocmin_ + Mzloc_trunc; mz++)
        for (int mx = 0; mx < Mx_io / 2 + 1; mx++)
            for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
                for (int i = 0; i < Nd_; i++)
                    cdata_send[mz - mzlocmin_].get()[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * mx)] =
                        v.cdata_[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * (mx + Mx_ * (mz - mzlocmin_)))];

    for (int mz = mzlocmin_; mz < mzlocmin_ + Mzloc_trunc; mz++)
        for (int mx = Mx_io / 2 + 1; mx < Mx_io; mx++) {
            int mx_cut = mx + (Mx_ - Mx_io);
            for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
                for (int i = 0; i < Nd_; i++)
                    cdata_send[mz - mzlocmin_].get()[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * mx)] =
                        v.cdata_[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * (mx_cut + Mx_ * (mz - mzlocmin_)))];
        }
#else
    for (int mz = mzlocmin_; mz < mzlocmin_ + Mzloc_trunc; mz++)
        for (int mx = 0; mx < Mx_io / 2 + 1; mx++)
            for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
                for (int i = 0; i < Nd_; i++)
                    cdata_send[mz - mzlocmin_].get()[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * mx)] =
                        v.cdata_[complex_flatten(mx, ny, mz, i)];

    for (int mz = mzlocmin_; mz < mzlocmin_ + Mzloc_trunc; mz++)
        for (int mx = Mx_io / 2 + 1; mx < Mx_io; mx++) {
            int mx_cut = mx + (Mx_ - Mx_io);
            for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
                for (int i = 0; i < Nd_; i++)
                    cdata_send[mz - mzlocmin_].get()[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * mx)] =
                        v.cdata_[complex_flatten(mx_cut, ny, mz, i)];
        }
#endif

    // MPI communication
    // receiver loop
    int pos = 0;
    for (int mz = mzlocmin_io; mz < mzlocmin_io + Mzloc_io; mz++) {
        int sender = mz / Mzlocmax_ + yrank * np1;
        int receiver = mz / Mzlocmax_io + yrank * np1;
        if ((rank == receiver) && (sender != receiver)) {
#ifdef HAVE_MPI
            MPI_Status mpistatus;
            MPI_Recv(reinterpret_cast<double*>(cdata_receive[pos].get()), Nloc_chunk, MPI_DOUBLE, sender, mz,
                     *comm_world(), &mpistatus);
#else
            cferror("Trying to use MPI without HAVE_MPI");
#endif
        }
        pos++;
    }
    // sender loop
    for (int mz = mzlocmin_; mz < mzlocmin_ + Mzloc_trunc; mz++) {
        int sender = mz / Mzlocmax_ + yrank * np1;
        int receiver = mz / Mzlocmax_io + yrank * np1;
        int oldpos = mz - mzlocmin_;
        int newpos = mz - mzlocmin_io - Mzloc_io * (receiver - sender);
        if (sender != receiver) {
            if (rank == sender)
#ifdef HAVE_MPI
                // non-blocking comm
                MPI_Ssend(reinterpret_cast<double*>(cdata_send[oldpos].get()), Nloc_chunk, MPI_DOUBLE, receiver, mz,
                          *comm_world());
#else
                cferror("Trying to use MPI without HAVE_MPI");
#endif
        } else {
            std::swap(cdata_receive[newpos], cdata_send[oldpos]);  // shift data from old to new position
        }
    }

    // write received and shifted data into data array
#ifdef HAVE_MPI
    for (int mz = mzlocmin_io; mz < mzlocmin_io + Mzloc_io; mz++)
        for (int mx = 0; mx < Mx_io; mx++)
            for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
                for (int i = 0; i < Nd_; i++)
                    cdata_io[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * (mx + Mx_io * (mz - mzlocmin_io)))] =
                        cdata_receive[mz - mzlocmin_io].get()[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * mx)];
#else
    for (int mz = mzlocmin_io; mz < mzlocmin_io + Mzloc_io; mz++)
        for (int mx = 0; mx < Mx_io; mx++)
            for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
                for (int i = 0; i < Nd_; i++)
                    cdata_io[mz + Nzpad2_io * (mx + Nx_io_ * (ny + Ny_ * i))] =
                        cdata_receive[mz - mzlocmin_io].get()[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * mx)];
#endif

    // final transformation which fills rdata_io with the physical field of the selected Fourier modes
    fftw_execute(xz_iplan_io.get());
}

void FlowField::addPaddedModes(Real* rdata_io, lint Nxloc_io, lint nxlocmin_io, lint Mzloc_io, lint mzlocmin_io) {
    // define data size parameters and allocate memory
    Complex* cdata_io = (Complex*)rdata_io;
    fftw_complex* fcdata_io = (fftw_complex*)cdata_io;

    lint Mx_io = Nx_io_;
    lint Mz_io = Nz_io_ / 2 + 1;
    lint lzero = 0;
    lint Mzloc_trunc = min(Mzloc_, max(Mz_io - mzlocmin_, lzero));  // local z-distribution after truncation

    lint Mzlocmax_io = Mz_io;

#ifdef HAVE_MPI
    int tmp1 = Mzloc_io;
    int tmp2 = 0;
    MPI_Allreduce(&tmp1, &tmp2, 1, MPI_INT, MPI_MAX, cfmpi_->comm_world);
    Mzlocmax_io = tmp2;
#endif

    // MPI related
    int np0 = nproc0();
    int np1 = nproc1();
    int np = np0 * np1;
    int rank = taskid();
    int yrank = rank / np1;

    // defined data chunk (z-slice) to be sent via MPI communcation
    lint Mx_chunk = Mx_io + np - (Mx_io % np);  // Must be a multiple of np for Nloc_chunk to be the same on all procs
    lint Mz_chunk = np1;
    lint howmany_io = Nylocpad_ * Nd_;
    lint Nloc_chunk = 2 * Mx_chunk * Mz_chunk * howmany_io;
#ifdef HAVE_MPI
    lint Mxloc_chunk = Mx_chunk;
    lint mxlocmin_chunk = nxlocmin_io;
    lint Mzloc_chunk = Mz_chunk;
    lint mzlocmin_chunk = mzlocmin_io;
    lint rank_chunk[2] = {Mx_chunk, Mz_chunk};
    Nloc_chunk = 2 * fftw_mpi_local_size_many_transposed(2, rank_chunk, howmany_io, FFTW_MPI_DEFAULT_BLOCK,
                                                         FFTW_MPI_DEFAULT_BLOCK, cfmpi_->comm1, &Mxloc_chunk,
                                                         &mxlocmin_chunk, &Mzloc_chunk, &mzlocmin_chunk);
#endif

    vector<unique_ptr<Complex, void (*)(void*)>> cdata_send{};
    vector<unique_ptr<Complex, void (*)(void*)>> cdata_receive{};
    for (int i = 0; i < Mzloc_io; i++)
        cdata_send.emplace_back(static_cast<Complex*>(fftw_malloc(Nloc_chunk * sizeof(Real))), fftw_free);
    for (int i = 0; i < Mzloc_trunc; i++)
        cdata_receive.emplace_back(static_cast<Complex*>(fftw_malloc(Nloc_chunk * sizeof(Real))), fftw_free);

    // design transform plan
    uint fftw_flags_io = FFTW_ESTIMATE;
    fftw_plan_unique_ptr_t xz_plan_io = {nullptr, fftw_destroy_plan};
#ifdef HAVE_MPI
    lint rank_1_io[2] = {Nx_io_, Nz_io_};
    xz_plan_io.reset(fftw_mpi_plan_many_dft_r2c(2, rank_1_io, howmany_io, FFTW_MPI_DEFAULT_BLOCK,
                                                FFTW_MPI_DEFAULT_BLOCK, rdata_io, fcdata_io, cfmpi_->comm1,
                                                fftw_flags_io | FFTW_MPI_TRANSPOSED_OUT));
#else
    const int Nzpad_io = 2 * (Nz_io_ / 2 + 1);
    const int Nzpad2_io = Nz_io_ / 2 + 1;
    const int howmany = Ny_ * Nd_;
    const int rk = 2;
    // These params describe the structure of the real-valued cfarray
    int real_n[rk];
    real_n[0] = Nx_io_;
    real_n[1] = Nz_io_;
    int real_embed[rk];
    real_embed[0] = Nx_io_;
    real_embed[1] = Nzpad_io;
    const int real_stride = 1;
    const int real_dist = Nx_io_ * Nzpad_io;
    // These params describe the structure of the complex-valued cfarray
    int cplx_embed[rk];
    cplx_embed[0] = Nx_io_;
    cplx_embed[1] = Nzpad2_io;
    const int cplx_stride = 1;
    const int cplx_dist = Nx_io_ * Nzpad2_io;
    xz_plan_io.reset(fftw_plan_many_dft_r2c(rk, real_n, howmany, rdata_io, real_embed, real_stride, real_dist,
                                            fcdata_io, cplx_embed, cplx_stride, cplx_dist, fftw_flags_io));
#endif

    fftw_execute(xz_plan_io.get());  // make xz spectral without xy-transpose which is done after MPI communication

    // write data into the slices to be sent
#ifdef HAVE_MPI
    for (int mz = mzlocmin_io; mz < mzlocmin_io + Mzloc_io; mz++)
        for (int mx = 0; mx < Mx_io; mx++)
            for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
                for (int i = 0; i < Nd_; i++)
                    cdata_send[mz - mzlocmin_io].get()[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * mx)] =
                        cdata_io[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * (mx + Mx_io * (mz - mzlocmin_io)))];
#else
    for (int mz = mzlocmin_io; mz < mzlocmin_io + Mzloc_io; mz++)
        for (int mx = 0; mx < Mx_io; mx++)
            for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
                for (int i = 0; i < Nd_; i++)
                    cdata_send[mz - mzlocmin_io].get()[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * mx)] =
                        cdata_io[mz + Nzpad2_io * (mx + Nx_io_ * (ny + Ny_ * i))];
#endif

    // MPI communication
    // sender loop
    int pos = 0;
    for (int mz = mzlocmin_io; mz < mzlocmin_io + Mzloc_io; mz++) {
        int sender = mz / Mzlocmax_io + yrank * np1;
        int receiver = mz / Mzlocmax_ + yrank * np1;
        if ((rank == sender) && (sender != receiver)) {
#ifdef HAVE_MPI
            // blocking comm
            MPI_Ssend(reinterpret_cast<double*>(cdata_send[pos].get()), Nloc_chunk, MPI_DOUBLE, receiver, mz,
                      *comm_world());
#else
            cferror("Trying to use MPI without HAVE_MPI");
#endif
        }
        pos++;
    }
    // receiver loop
    for (int mz = mzlocmin_; mz < mzlocmin_ + Mzloc_trunc; mz++) {
        int sender = mz / Mzlocmax_io + yrank * np1;
        int receiver = mz / Mzlocmax_ + yrank * np1;
        int oldpos = mz - mzlocmin_io - Mzloc_io * (sender - receiver);
        int newpos = mz - mzlocmin_;
        if (sender != receiver) {
            if (rank == receiver) {
#ifdef HAVE_MPI
                MPI_Status mpistatus;
                MPI_Recv(reinterpret_cast<double*>(cdata_receive[newpos].get()), Nloc_chunk, MPI_DOUBLE, sender, mz,
                         *comm_world(), &mpistatus);
#else
                cferror("Trying to use MPI without HAVE_MPI");
#endif
            }
        } else {
            std::swap(cdata_receive[newpos], cdata_send[oldpos]);  // shift data from old to new position
        }
    }

    // write received and shifted data into data array
#ifdef HAVE_MPI
    for (int mz = mzlocmin_; mz < mzlocmin_ + Mzloc_trunc; mz++)
        for (int mx = 0; mx < Mx_io / 2 + 1; mx++)
            for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
                for (int i = 0; i < Nd_; i++)
                    cdata_[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * (mx + Mx_ * (mz - mzlocmin_)))] =
                        cdata_receive[mz - mzlocmin_].get()[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * mx)];

    for (int mz = mzlocmin_; mz < mzlocmin_ + Mzloc_trunc; mz++)
        for (int mx = Mx_io / 2 + 1; mx < Mx_io; mx++) {
            int mx_cut = mx + (Mx_ - Mx_io);
            for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
                for (int i = 0; i < Nd_; i++)
                    cdata_[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * (mx_cut + Mx_ * (mz - mzlocmin_)))] =
                        cdata_receive[mz - mzlocmin_].get()[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * mx)];
        }
    // pad with zeros
    Complex zero(0.0, 0.0);
    for (int mz = mzlocmin_; mz < mzlocmin_ + Mzloc_trunc; mz++) {
        for (int mx = Mx_io / 2 + 1; mx < Mx_ - Mx_io / 2 + 1; mx++) {
            assert(isAliased(this->kx(mx), this->kz(mz)));
            for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
                for (int i = 0; i < Nd_; i++)
                    cdata_[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * (mx + Mx_ * (mz - mzlocmin_)))] = zero;
        }
    }
    for (int mz = mzlocmin_ + Mzloc_trunc; mz < Mzloc_; mz++) {
        for (int mx = 0; mx < Mx_; mx++) {
            assert(isAliased(this->kx(mx), this->kz(mz)));
            for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
                for (int i = 0; i < Nd_; i++)
                    cdata_[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * (mx + Mx_ * (mz - mzlocmin_)))] = zero;
        }
    }
#else
    for (int mz = mzlocmin_; mz < mzlocmin_ + Mzloc_trunc; mz++)
        for (int mx = 0; mx < Mx_io / 2 + 1; mx++)
            for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
                for (int i = 0; i < Nd_; i++)
                    cdata_[complex_flatten(mx, ny, mz, i)] =
                        cdata_receive[mz - mzlocmin_].get()[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * mx)];

    for (int mz = mzlocmin_; mz < mzlocmin_ + Mzloc_trunc; mz++)
        for (int mx = Mx_io / 2 + 1; mx < Mx_io; mx++) {
            int mx_cut = mx + (Mx_ - Mx_io);
            for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
                for (int i = 0; i < Nd_; i++)
                    cdata_[complex_flatten(mx_cut, ny, mz, i)] =
                        cdata_receive[mz - mzlocmin_].get()[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * mx)];
        }
    // pad with zeros
    Complex zero(0.0, 0.0);
    for (int i = 0; i < Nd_; i++)
        for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
            for (int mx = Mx_io / 2 + 1; mx < Mx_ - Mx_io / 2 + 1; mx++) {
                for (int mz = mzlocmin_; mz < mzlocmin_ + Mzloc_trunc; mz++) {
                    assert(isAliased(this->kx(mx), this->kz(mz)));
                    cdata_[complex_flatten(mx, ny, mz, i)] = zero;
                }
            }
    for (int i = 0; i < Nd_; i++)
        for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
            for (int mx = 0; mx < Mx_; mx++) {
                for (int mz = mzlocmin_ + Mzloc_trunc; mz < Mzloc_; mz++) {
                    assert(isAliased(this->kx(mx), this->kz(mz)));
                    cdata_[complex_flatten(mx, ny, mz, i)] = zero;
                }
            }
#endif

    // xy-transposition
#ifdef HAVE_MPI
    if (np0 > 1) {
        for (lint iz = 0; iz < Mzloc_; iz++) {
            lint offset = 2 * iz * Mxloc_ * Nypad_ * Nd_;
            fftw_mpi_execute_r2r(t_plan_.get(), &rdata_[offset], &rdata_[offset]);
        }
    }
#endif

    Real scale = 1.0 / (Nx_io_ * Nz_io_);
    for (lint i = 0; i < Nloc_; ++i)
        rdata_[i] *= scale;

    xzstate_ = Spectral;
}

#define ERRCODE 2 /* Error message for the NetCDF methods */
#define ERR(e)                                \
    {                                         \
        cerr << "NetCDF: " << nc_strerror(e); \
        exit(ERRCODE);                        \
    }

#if HAVE_NETCDF
void FlowField::writeNetCDF(const string& filebase, vector<string> component_names) const {
    const int format_version = 1;

    /* FlowField object must be in physical state for this method.
     * If object is spectral, this const. method creates a copy,
     * transform it to physical and then calls itself -> 1 RECURSION LEVEL!
     */
    if (not(padded_) && (xzstate_ == Spectral || ystate_ == Spectral)) {
        // the copy for changing the state in the case padded_=true is done in method "removePaddedModes()"
        FlowField v(*this);
        v.makePhysical();
        v.writeNetCDF(filebase);
    } else {
        // define and allocate local memory
        int Nx_io = Nx_;
        int Nz_io = Nz_;
        lint Nxloc_io = Nxloc_;
        lint nxlocmin_io = nxlocmin_;
        int Nzpad_io = Nzpad_;

        unique_ptr<Real, void (*)(void*)> rdata_io_handle = {nullptr, fftw_free};
        auto rdata_io = static_cast<Real*>(nullptr);

        if (not(padded_)) {
            rdata_io = rdata_;
        } else {
            Nx_io = Nx_io_;
            Nz_io = Nz_io_;
            Nxloc_io = Nx_io_;
            nxlocmin_io = nxlocmin_;
            lint Mz_io = Nz_io_ / 2 + 1;  // 2*(Mz_/3);
            // allocate the memory for the data without padded modes
            lint Mzloc_io = Mz_io;
            lint mzlocmin_io = mzlocmin_;
            Nzpad_io = 2 * Mz_io;
            lint howmany_io = Nylocpad_ * Nd_;
            lint Nloc_io = 2 * Nx_io_ * Mz_io * howmany_io;
#ifdef HAVE_MPI
            lint rank_0_io[2] = {Nx_io_, Mz_io};
            Nloc_io = 2 * fftw_mpi_local_size_many_transposed(2, rank_0_io, howmany_io, FFTW_MPI_DEFAULT_BLOCK,
                                                              FFTW_MPI_DEFAULT_BLOCK, cfmpi_->comm1, &Nxloc_io,
                                                              &nxlocmin_io, &Mzloc_io, &mzlocmin_io);
#endif

            rdata_io_handle.reset(static_cast<Real*>(fftw_malloc(Nloc_io * sizeof(Real))));
            rdata_io = rdata_io_handle.get();
            // fill memory in rdata_io with the inverse FFT of only truncated modes
            removePaddedModes(rdata_io, Nxloc_io, nxlocmin_io, Mzloc_io, mzlocmin_io);
        }

        /////////////////////////// NETCDF SETUP //////////////////////////////////////

        /* NetCDF return handles */
        int status;
        const int nd = 3;  // dimension of field (Nd_ is dimension of variable)
        int ncid;
        bool doesIO = false;

        auto dimid = vector<int>(nd, 0);
        auto gridid = vector<int>(nd, 0);
        auto varid = vector<int>(Nd_, 0);

        /* Create the file, with parallel access if possible */
        string nc_name = appendSuffix(filebase, ".nc");

#ifdef HAVE_MPI
#if HAVE_NETCDF_PAR
        MPI_Info info = MPI_INFO_NULL;
        if ((status = nc_create_par(nc_name.c_str(), NC_NETCDF4 | NC_MPIIO, *comm_world(), info, &ncid)))
            ERR(status);
        doesIO = true;
#else
        if (taskid() == 0) {
            if ((status = nc_create(nc_name.c_str(), NC_NETCDF4, &ncid)))
                ERR(status);
            doesIO = true;
        }
#endif
#else
        if ((status = nc_create(nc_name.c_str(), NC_NETCDF4, &ncid)))
            ERR(status);
        doesIO = true;
#endif

        /* Define names for data description */
        size_t dim_size[nd];
        dim_size[0] = Nx_io;
        dim_size[1] = Ny_;
        dim_size[2] = Nz_io;
        vector<string> dim_name;
        dim_name.push_back("X");
        dim_name.push_back("Y");
        dim_name.push_back("Z");
        vector<string> var_name;
        if (int(component_names.size()) == Nd_) {
            for (int i = 0; i < Nd_; i++)
                var_name.push_back(component_names[i]);
        } else if (Nd_ == nd) {
            var_name.push_back("Velocity_X");
            var_name.push_back("Velocity_Y");
            var_name.push_back("Velocity_Z");
        } else {
            for (int i = 0; i < Nd_; i++)
                var_name.push_back("Component_" + i2s(i));
        }

        if (doesIO) {  // header is handled either by master (without parNC) or by all tasks collectively (with parNC)
            /*define global attributes*/
            char cf_conv[] = "CF-1.0";
            char title[] = "FlowField";
            auto fversion = i2s(format_version);
            char reference[] = "Channelflow is free software: www.channelflow.ch.";
            time_t rawtime;  // current time
            struct tm* timeinfo;
            char tbuffer[80];
            time(&rawtime);
            timeinfo = localtime(&rawtime);
            strftime(tbuffer, 80, "%Y-%m-%d %I:%M:%S", timeinfo);
            char hostname[1024];
            gethostname(hostname, 1023);
            /*write global attributes: CF (Climate and Forecast) convention*/
            if ((status = nc_put_att_text(ncid, NC_GLOBAL, "Conventions", strlen(cf_conv), cf_conv)))
                ERR(status);
            if ((status = nc_put_att_text(ncid, NC_GLOBAL, "title", strlen(title), title)))
                ERR(status);
            if ((status = nc_put_att_text(ncid, NC_GLOBAL, "format_version", fversion.size(), fversion.c_str())))
                ERR(status);
            if ((status = nc_put_att_text(ncid, NC_GLOBAL, "channelflow_version", strlen(CHANNELFLOW_VERSION),
                                          CHANNELFLOW_VERSION)))
                ERR(status);
            if ((status =
                     nc_put_att_text(ncid, NC_GLOBAL, "compiler_version", strlen(COMPILER_VERSION), COMPILER_VERSION)))
                ERR(status);
            if ((status = nc_put_att_text(ncid, NC_GLOBAL, "git_revision", strlen(g_GIT_SHA1), g_GIT_SHA1)))
                ERR(status);
            if ((status = nc_put_att_text(ncid, NC_GLOBAL, "time", strlen(tbuffer), tbuffer)))
                ERR(status);
            if ((status = nc_put_att_text(ncid, NC_GLOBAL, "host_name", strlen(hostname), hostname)))
                ERR(status);
            if ((status = nc_put_att_text(ncid, NC_GLOBAL, "references", strlen(reference), reference)))
                ERR(status);
            /*write global attributes: CF (ChannelFlow) requirements*/
            if ((status = nc_put_att_int(ncid, NC_GLOBAL, "Nx", NC_INT, 1, &Nx_)))
                ERR(status);
            if ((status = nc_put_att_int(ncid, NC_GLOBAL, "Ny", NC_INT, 1, &Ny_)))
                ERR(status);
            if ((status = nc_put_att_int(ncid, NC_GLOBAL, "Nz", NC_INT, 1, &Nz_)))
                ERR(status);
            if ((status = nc_put_att_double(ncid, NC_GLOBAL, "Lx", NC_DOUBLE, 1, &Lx_)))
                ERR(status);
            if ((status = nc_put_att_double(ncid, NC_GLOBAL, "Lz", NC_DOUBLE, 1, &Lz_)))
                ERR(status);
            if ((status = nc_put_att_double(ncid, NC_GLOBAL, "a", NC_DOUBLE, 1, &a_)))
                ERR(status);
            if ((status = nc_put_att_double(ncid, NC_GLOBAL, "b", NC_DOUBLE, 1, &b_)))
                ERR(status);

            /* Define the dimensions and the grid variables (chosen here to have the same name) */
            for (int i = 0; i < nd; i++) {
                if ((status = nc_def_dim(ncid, dim_name[i].c_str(), dim_size[i], &dimid[nd - i - 1])))
                    ERR(status);
                int dimid1[1];
                dimid1[0] = dimid[nd - i - 1];
                if ((status = nc_def_var(ncid, dim_name[i].c_str(), NC_DOUBLE, 1, dimid1, &gridid[i])))
                    ERR(status);
            }

            /* Define the data variables. The type of the variable in this case is NC_DOUBLE (NC variable type = 6). */
            for (int i = 0; i < Nd_; i++)
                if ((status = nc_def_var(ncid, var_name[i].c_str(), NC_DOUBLE, nd, dimid.data(), &varid[i])))
                    ERR(status);

            /* End define mode. This tells netCDF we are done defining metadata. */
            if ((status = nc_enddef(ncid)))
                ERR(status);
        }

        /////////////////////////// NETCDF DATA OUTPUT //////////////////////////////////////

        /* define data size for parallel IO */
        int varsize = Nz_io * Nyloc_ * Nxloc_io;
        auto var = vector<double>(varsize, 0.0);
#if HAVE_NETCDF_PAR || !defined HAVE_MPI
        size_t start_var[nd], count_var[nd];
        assert(nd == 3);
        if (varsize > 0) {
            start_var[0] = 0;
            count_var[0] = Nz_io;
            start_var[1] = nylocmin_;
            count_var[1] = Nyloc_;
            start_var[2] = nxlocmin_io;
            count_var[2] = Nxloc_io;
        } else {  // prescribe save indices for jobs with data chunk size 0
            start_var[0] = 0;
            count_var[0] = 0;
            start_var[1] = 0;
            count_var[1] = 0;
            start_var[2] = 0;
            count_var[2] = 0;
        }
#endif

        /* define the grid and write it to file */
        Vector xpts(Nx_io);  // grid points have changed in case padded_=true
        for (int nx = 0; nx < Nx_io; nx++)
            xpts[nx] = nx * Lx_ / Nx_io;
        Vector ypts = ygridpts();
        Vector zpts(Nz_io);
        for (int nz = 0; nz < Nz_io; nz++)
            zpts[nz] = nz * Lz_ / Nz_io;

        size_t start_grid[1], count_grid[1];
#if HAVE_NETCDF_PAR
        int par_access = NC_COLLECTIVE; /* Define the parallel access mode */

        auto xgrid = vector<double>(Nxloc_io, 0.0);
        auto ygrid = vector<double>(Nyloc_, 0.0);
        auto zgrid = vector<double>(Nz_io, 0.0);

        for (int nx = nxlocmin_io; nx < nxlocmin_io + Nxloc_io; nx++)
            xgrid[nx - nxlocmin_io] = xpts(nx);
        for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
            ygrid[ny - nylocmin_] = ypts(ny);
        for (int nz = 0; nz < Nz_io; nz++)
            zgrid[nz] = zpts(nz);
#ifdef HAVE_MPI
        if ((status = nc_var_par_access(ncid, gridid[0], par_access)))
            ERR(status); /* Make parallel access collective */
        if ((status = nc_var_par_access(ncid, gridid[1], par_access)))
            ERR(status);
        if ((status = nc_var_par_access(ncid, gridid[2], par_access)))
            ERR(status);
#endif
        /* Each proc writes its hyperslab of the grid variable */
        start_grid[0] = start_var[2];
        count_grid[0] = count_var[2];
        if ((status = nc_put_vara_double(ncid, gridid[0], start_grid, count_grid, &xgrid[0])))
            ERR(status);
        start_grid[0] = start_var[1];
        count_grid[0] = count_var[1];
        if ((status = nc_put_vara_double(ncid, gridid[1], start_grid, count_grid, &ygrid[0])))
            ERR(status);
        start_grid[0] = start_var[0];
        count_grid[0] = count_var[0];
        if ((status = nc_put_vara_double(ncid, gridid[2], start_grid, count_grid, &zgrid[0])))
            ERR(status);
#else
        if (doesIO) {
            auto xgrid = vector<double>(Nx_io, 0.0);
            auto ygrid = vector<double>(Ny_, 0.0);
            auto zgrid = vector<double>(Nz_io, 0.0);

            for (int nx = 0; nx < Nx_io; nx++)
                xgrid[nx] = xpts(nx);
            for (int ny = 0; ny < Ny_; ny++)
                ygrid[ny] = ypts(ny);
            for (int nz = 0; nz < Nz_io; nz++)
                zgrid[nz] = zpts(nz);
            /* Write the grid variable at once */
            start_grid[0] = 0;
            count_grid[0] = dim_size[0];
            if ((status = nc_put_vara_double(ncid, gridid[0], start_grid, count_grid, &xgrid[0])))
                ERR(status);
            start_grid[0] = 0;
            count_grid[0] = dim_size[1];
            if ((status = nc_put_vara_double(ncid, gridid[1], start_grid, count_grid, &ygrid[0])))
                ERR(status);
            start_grid[0] = 0;
            count_grid[0] = dim_size[2];
            if ((status = nc_put_vara_double(ncid, gridid[2], start_grid, count_grid, &zgrid[0])))
                ERR(status);
        }
#endif

        /* Write the flowfield data to file */
        for (int i = 0; i < Nd_; i++) {
#ifndef HAVE_MPI
            for (int ny = 0; ny < Ny_; ny++)
                for (int nx = 0; nx < Nx_io; nx++)
                    for (int nz = 0; nz < Nz_io; nz++)
                        // explicit formulation of non-mpi flatten method
                        var[nx + Nx_io * (ny + Ny_ * nz)] = rdata_io[nz + Nzpad_io * (nx + Nx_io * (ny + Ny_ * i))];

            /* Write NetCDF variable */
            if ((status = nc_put_vara_double(ncid, varid[i], start_var, count_var, var.data())))
                ERR(status);

#else
            for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
                for (int nz = 0; nz < Nz_io; nz++)
                    for (int nx = nxlocmin_io; nx < nxlocmin_io + Nxloc_io; nx++)
                        // explicit formulation of mpi flatten method
                        var[(nx - nxlocmin_io) + Nxloc_io * ((ny - nylocmin_) + Nyloc_ * nz)] =
                            rdata_io[i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * (nz + Nzpad_io * (nx - nxlocmin_io)))];
#if HAVE_NETCDF_PAR
            /* Define par_access for writing vars */
            if ((status = nc_var_par_access(ncid, varid[i], par_access)))
                ERR(status);
            /* Write NetCDF variable in PARALLEL */
            if ((status = nc_put_vara_double(ncid, varid[i], start_var, count_var, var.data())))
                ERR(status);
#else /* block below defines communication in the case of using mpi without parallel NetCDF */
            ////////////////////////////////////////////////////////////
            int np = nproc0() * nproc1();
            if (taskid() == 0) {
                auto mastervar = vector<double>(Nz_io * Ny_ * Nx_io, 0.0);
                for (int nx = 0; nx < Nxloc_io; nx++)
                    for (int ny = 0; ny < Nyloc_; ny++)
                        for (int nz = 0; nz < Nz_io; nz++)
                            mastervar[nx + Nx_io * (ny + Ny_ * nz)] = var[nx + Nxloc_io * (ny + Nyloc_ * nz)];
                MPI_Status mpistatus;
                // mpi communication: sending location parameters is saver than calculating them on master
                int xstart, xsize, ystart, ysize;
                for (int rk = 1; rk < np; rk++) {
                    MPI_Recv(&xstart, 1, MPI_INT, rk, rk, *comm_world(), &mpistatus);
                    MPI_Recv(&xsize, 1, MPI_INT, rk, rk + np, *comm_world(), &mpistatus);
                    MPI_Recv(&ystart, 1, MPI_INT, rk, rk + np * 2, *comm_world(), &mpistatus);
                    MPI_Recv(&ysize, 1, MPI_INT, rk, rk + np * 3, *comm_world(), &mpistatus);
                    int rec_varsize = Nz_io * ysize * xsize;
                    if (rec_varsize > 0) {
                        // maybe could use var, but local data is not of equal size amongst procs
                        auto rec_var = vector<double>(rec_varsize, 0.0);
                        MPI_Recv(rec_var.data(), rec_varsize, MPI_DOUBLE, rk, rk + 2 * np, *comm_world(), &mpistatus);
                        for (int nx = 0; nx < xsize; nx++)
                            for (int ny = 0; ny < ysize; ny++)
                                for (int nz = 0; nz < Nz_io; nz++)
                                    mastervar[(nx + xstart) + Nx_io * ((ny + ystart) + Ny_ * nz)] =
                                        rec_var[nx + xsize * (ny + ysize * nz)];
                    }
                }
                size_t masterstart[nd], mastercount[nd];
                masterstart[0] = 0;
                masterstart[1] = 0;
                masterstart[2] = 0;
                mastercount[0] = dim_size[2];
                mastercount[1] = dim_size[1];
                mastercount[2] = dim_size[0];
                /* Write NetCDF variable in SERIAL */
                if ((status = nc_put_vara_double(ncid, varid[i], masterstart, mastercount, mastervar.data())))
                    ERR(status);
            } else {
                int tmp1 = nylocmin_;
                int tmp2 = Nyloc_;

                // FIXME: Here we are sending a value of type lint (a.k.a ptrdiff_t a.k.a long int on most
                // archs)
                // FIXME: using a MPI_INT instead of an MPI long. The matching receive should be probably adjusted, but
                // for
                // FIXME: now the hot-fix is to copy the value into an int.
                auto nxlocmin_buffer = static_cast<int>(nxlocmin_io);
                auto nxloc_buffer = static_cast<int>(Nxloc_io);

                MPI_Ssend(&nxlocmin_buffer, 1, MPI_INT, 0, taskid(), *comm_world());
                MPI_Ssend(&nxloc_buffer, 1, MPI_INT, 0, taskid() + np, *comm_world());
                MPI_Ssend(&tmp1, 1, MPI_INT, 0, taskid() + np * 2, *comm_world());
                MPI_Ssend(&tmp2, 1, MPI_INT, 0, taskid() + np * 3, *comm_world());
                if (varsize > 0) {
                    MPI_Ssend(var.data(), varsize, MPI_DOUBLE, 0, taskid() + 2 * np, *comm_world());  // blocking comm
                }
            }
            ////////////////////////////////////////////////////////////
#endif
#endif
        }

        if (doesIO) {
            /* Close the file. This frees up any internal netCDF resources associated with the file, and flushes any
             * buffers. */
            if ((status = nc_close(ncid)))
                ERR(status);
        }
    }
}
#else
void FlowField::writeNetCDF(const string& filebase, vector<string> component_names) const {
    cferror("FlowField::writeNetCDF requires NetCDF libraries. Please install them and recompile channelflow.");
}
#endif

#if HAVE_NETCDF
void FlowField::readNetCDF(const string& filebase) {
    int status;  // return value
    int ncid;    // return handle of file

    /* Open the file, with parallel access if possible. */
    string nc_name = appendSuffix(filebase, ".nc");
#ifdef HAVE_MPI
#if HAVE_NETCDF_PAR
    MPI_Info info = MPI_INFO_NULL;
    if ((status = nc_open_par(nc_name.c_str(), NC_MPIIO, *comm_world(), info, &ncid)))
        ERR(status);
#else
    int np = nproc0() * nproc1();
    for (int rk = 0; rk < np; rk++) {  // enforcing ordered opening of file reduces the risk of dead lock
        if (taskid() == rk) {
            // all proc need to open because all will read in their part (does not work with sending "ncid")
            if ((status = nc_open(nc_name.c_str(), NC_NOWRITE, &ncid)))
                ERR(status);
        }
        MPI_Barrier(*comm_world());
    }
#endif  // HAVE_NETCDF_PAR
#else
    if ((status = nc_open(nc_name.c_str(), NC_NOWRITE, &ncid)))
        ERR(status);
#endif  // HAVE_MPI

    /* General inquiry of how many netCDF variables, dimensions, and global attributes are in the file. */
    auto ndims_in = 0;
    auto nvars_in = 0;
    auto ngatts_in = 0;
    auto unlimdimid_in = 0;

    int Nx_io = 0, Ny_io = 0, Nz_io = 0;
    if (taskid() == 0) {  // header is handled by master node
        if ((status = nc_inq(ncid, &ndims_in, &nvars_in, &ngatts_in, &unlimdimid_in)))
            ERR(status);
        Nd_ = nvars_in - ndims_in;  // the grid along each dimension is also counted as variables

        /* Read necessary global attributes from file */
        char attname[NC_MAX_NAME + 1];
        for (int attid = 0; attid < ngatts_in; attid++) {  // loop over all global attributes
            if ((status = nc_inq_attname(ncid, NC_GLOBAL, attid, attname)))
                ERR(status);
            if (strcmp(attname, "Nx") == 0) {
                if ((status = nc_get_att_int(ncid, NC_GLOBAL, attname, &Nx_)))
                    ERR(status);
            } else if (strcmp(attname, "Ny") == 0) {
                if ((status = nc_get_att_int(ncid, NC_GLOBAL, attname, &Ny_)))
                    ERR(status);
            } else if (strcmp(attname, "Nz") == 0) {
                if ((status = nc_get_att_int(ncid, NC_GLOBAL, attname, &Nz_)))
                    ERR(status);
            } else if (strcmp(attname, "Lx") == 0) {
                if ((status = nc_get_att_double(ncid, NC_GLOBAL, attname, &Lx_)))
                    ERR(status);
            } else if (strcmp(attname, "Lz") == 0) {
                if ((status = nc_get_att_double(ncid, NC_GLOBAL, attname, &Lz_)))
                    ERR(status);
            } else if (strcmp(attname, "a") == 0) {
                if ((status = nc_get_att_double(ncid, NC_GLOBAL, attname, &a_)))
                    ERR(status);
            } else if (strcmp(attname, "b") == 0) {
                if ((status = nc_get_att_double(ncid, NC_GLOBAL, attname, &b_)))
                    ERR(status);
            } else
                continue;
        }

        /* Dimensions inquiry */
        char dimname[NC_MAX_NAME + 1];
        size_t dimlen;
        for (int dimid = 0; dimid < ndims_in; dimid++) {
            if ((status = nc_inq_dim(ncid, dimid, dimname, &dimlen)))
                ERR(status);
            /* Read the coordinate variable data. */
            if (strcmp(dimname, "X") == 0)
                Nx_io = dimlen;
            else if (strcmp(dimname, "Y") == 0)
                Ny_io = dimlen;
            else if (strcmp(dimname, "Z") == 0)
                Nz_io = dimlen;
            else
                cout << "Unknow name of dimension" << endl;
        }
    }

    // distribute the NC inquiry
#ifdef HAVE_MPI
    MPI_Bcast(&ndims_in, 1, MPI_INT, 0, *comm_world());
    MPI_Bcast(&nvars_in, 1, MPI_INT, 0, *comm_world());
    MPI_Bcast(&Nx_, 1, MPI_INT, 0, *comm_world());
    MPI_Bcast(&Ny_, 1, MPI_INT, 0, *comm_world());
    MPI_Bcast(&Nz_, 1, MPI_INT, 0, *comm_world());
    MPI_Bcast(&Nd_, 1, MPI_INT, 0, *comm_world());
    MPI_Bcast(&Nx_io, 1, MPI_INT, 0, *comm_world());
    MPI_Bcast(&Ny_io, 1, MPI_INT, 0, *comm_world());
    MPI_Bcast(&Nz_io, 1, MPI_INT, 0, *comm_world());
    MPI_Bcast(&Lx_, 1, MPI_DOUBLE, 0, *comm_world());
    MPI_Bcast(&Lz_, 1, MPI_DOUBLE, 0, *comm_world());
    MPI_Bcast(&a_, 1, MPI_DOUBLE, 0, *comm_world());
    MPI_Bcast(&b_, 1, MPI_DOUBLE, 0, *comm_world());
#endif

    /* Define flow field members and size of local hyperslab*/
    resize(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, a_, b_, cfmpi_);
    xzstate_ = Physical;
    ystate_ = Physical;
    lint Nxloc_io, nxlocmin_io, Mzloc_io, mzlocmin_io;
    int Nzpad_io = 0;

    unique_ptr<Real, void (*)(void*)> rdata_io_handle = {nullptr, fftw_free};
    auto rdata_io = static_cast<Real*>(nullptr);

    if ((Nx_io == Nx_) && (Ny_io == Ny_) && (Nz_io == Nz_)) {  // IO with padded modes
        padded_ = false;
        Nxloc_io = Nxloc_;
        nxlocmin_io = nxlocmin_;
        Mzloc_io = mzlocmin_io = 0;  // variables not initialized, should not be used when padded_==false
        Nzpad_io = Nzpad_;
        rdata_io = rdata_;
    } else if ((Nx_io == Nx_io_) && (Ny_io == Ny_) && (Nz_io == Nz_io_)) {  // IO without padded modes
        padded_ = true;
        // allocate the memory for the data without padded modes
        lint Mz_io = Nz_io_ / 2 + 1;
        Nxloc_io = Nx_io_;
        nxlocmin_io = nxlocmin_;
        Mzloc_io = Mz_io;
        mzlocmin_io = mzlocmin_;
        Nzpad_io = 2 * Mz_io;
        lint howmany_io = Nylocpad_ * Nd_;
        lint Nloc_io = 2 * Nx_io_ * Mz_io * howmany_io;
#ifdef HAVE_MPI
        lint rank_0_io[2] = {Nx_io_, Mz_io};
        Nloc_io = 2 * fftw_mpi_local_size_many_transposed(2, rank_0_io, howmany_io, FFTW_MPI_DEFAULT_BLOCK,
                                                          FFTW_MPI_DEFAULT_BLOCK, cfmpi_->comm1, &Nxloc_io,
                                                          &nxlocmin_io, &Mzloc_io, &mzlocmin_io);
#endif  // HAVE_MPI
        rdata_io_handle.reset(static_cast<Real*>(fftw_malloc(Nloc_io * sizeof(Real))));
        rdata_io = rdata_io_handle.get();
    } else {
        // Avoids "variable uninitialized" warning. When file is corrupted, results should not be used.
        mzlocmin_io = Mzloc_io = nxlocmin_io = Nxloc_io = 0;

        cout << "Corrupted file! Conflict in file dimension and metadata." << endl;
    }

    /* Read in data variables from file */
    int ivar = 0;
    char varname[NC_MAX_NAME + 1];
    int varsize = Nz_io * Nyloc_ * Nxloc_io;
    for (int varid = 0; varid < nvars_in; varid++) {
        if ((status = nc_inq_varname(ncid, varid, varname)))
            ERR(status);
        if ((strcmp(varname, "X") != 0) && (strcmp(varname, "Y") != 0) && (strcmp(varname, "Z") != 0)) {
            /* Define variable array and corners of the hyperslabs */
            auto var = vector<double>(varsize > 0 ? varsize : 1u, 0.0);

            auto start = vector<size_t>(ndims_in, 0);
            auto count = vector<size_t>(ndims_in, 0);

            assert(ndims_in == 3);
            if (varsize > 0) {
                start[0] = 0;
                count[0] = Nz_io;
                start[1] = nylocmin_;
                count[1] = Nyloc_;
                start[2] = nxlocmin_io;
                count[2] = Nxloc_io;
            } else {
                start[0] = 0;
                count[0] = 1;  // because they do not read in data. This happens for certain combinations
                start[1] = 0;
                count[1] = 1;  // of grid size to job distributions, e.g. np0>Ny. Therefore, all jobs with
                start[2] = 0;
                count[2] = 1;  // varsize==0 will read dummy data, i.e. the first entry.
            }

            /* read data from file */
#ifdef HAVE_MPI
#if HAVE_NETCDF_PAR
            if ((status = nc_var_par_access(ncid, varid, NC_COLLECTIVE)))
                ERR(status); /* Define the parallel access mode */
#endif                       // HAVE_NETCDF_PAR
#endif                       // HAVE_MPI
            if ((status = nc_get_vara_double(ncid, varid, start.data(), count.data(), var.data())))
                ERR(status);

                /* Construct FlowField */
#ifdef HAVE_MPI
            for (int nx = nxlocmin_io; nx < nxlocmin_io + Nxloc_io; nx++)
                for (int ny = nylocmin_; ny < nylocmin_ + Nyloc_; ny++)
                    for (int nz = 0; nz < Nz_io; nz++)
                        rdata_io[ivar + Nd_ * ((ny - nylocmin_) + Nylocpad_ * (nz + Nzpad_io * (nx - nxlocmin_io)))] =
                            var[(nx - nxlocmin_io) + Nxloc_io * ((ny - nylocmin_) + Nyloc_ * nz)];
#else
            for (int nx = 0; nx < Nx_io; nx++)
                for (int ny = 0; ny < Ny_; ny++)
                    for (int nz = 0; nz < Nz_io; nz++) {
                        // explicit formulation of non-mpi flatten method
                        rdata_io[nz + Nzpad_io * (nx + Nx_io * (ny + Ny_ * ivar))] =
                            var[nx + Nx_io * (ny + Nyloc_ * nz)];
                    }
#endif  // HAVE_MPI
            ivar++;
        } else
            continue;
    }

#ifdef HAVE_MPI
    MPI_Barrier(*comm_world());
#endif  // HAVE_MPI
    /* Close the netcdf file. */
    if ((status = nc_close(ncid)))
        ERR(status);

    if (padded_) {
        addPaddedModes(rdata_io, Nxloc_io, nxlocmin_io, Mzloc_io,
                       mzlocmin_io);  // fills rdata_ with rdata_io + padded modes
    }
    makeSpectral();
}
#else
void FlowField::readNetCDF(const string& filebase) {
    cferror("FlowField::readNetCDF requires NetCDF libraries. Please install them and recompile channelflow.");
}
#endif  // HAVE_NETCDF

void writefloat(ofstream& os, float z, bool SwapEndian) {
    if (mpirank() > 0)
        cferror("Function writefloat is not mpi safe");
    int SZ = sizeof(float);
    char* pc;

    pc = (char*)&z;

    if (SwapEndian) {
        for (int i = 0; i < SZ; i++) {
            os.write(pc + 3 - i, 1);
        }
    } else {
        for (int i = 0; i < SZ; i++) {
            os.write(pc + i, 1);
        }
    }
}

void FlowField::VTKSave(const std::string& filebase, bool SwapEndian) const {
    string filename(filebase);

    if (filename.find(".vtk") == string::npos)
        filename += ".vtk";

    if (mpirank() > 0)
        cferror("Function VTKSave is not mpi safe");

    ofstream os(filename.c_str());

    if (xzstate_ == Spectral || ystate_ == Spectral) {
        FlowField v(*this);
        v.makePhysical();
        v.VTKSave(filebase);
    }

    else {
        os << "# vtk DataFile Version 3.0" << endl;
        os << "ChannelFlow data" << endl;
        os << "BINARY" << endl;

        os << "DATASET RECTILINEAR_GRID" << endl;
        os << "DIMENSIONS " << Nx() << " " << Ny() << " " << Nz() << endl;
        os << "X_COORDINATES " << Nx() << " float" << endl;

        float z;
        Vector xg = xgridpts();
        for (int i = 0; i < Nx(); i++) {
            z = (float)xg(i);
            writefloat(os, z, SwapEndian);
        }
        Vector yg = ygridpts();
        os << "Y_COORDINATES " << Ny() << " float" << endl;
        for (int i = 0; i < Ny(); i++) {
            z = (float)yg(i);
            writefloat(os, z, SwapEndian);
        }
        Vector zg = zgridpts();
        os << "Z_COORDINATES " << Nz() << " float" << endl;
        for (int i = 0; i < Nz(); i++) {
            z = (float)zg(i);
            writefloat(os, z, SwapEndian);
        }

        os << "POINT_DATA " << Nx() * Ny() * Nz() << endl;

        os << "VECTORS VelocityField float" << endl;

        for (int k = 0; k < Nz(); k++) {
            for (int j = 0; j < Ny(); j++) {
                for (int i = 0; i < Nx(); i++) {
                    for (int l = 0; l < 3; l++) {
                        z = (float)(*this)(i, j, k, l);
                        writefloat(os, z, SwapEndian);
                    }
                }
            }
        }
    }
}

void FlowField::dump() const {
    for (int i = 0; i < Nx_ * Ny_ * Nzpad_ * Nd_; ++i)
        cout << rdata_[i] << ' ';
    cout << endl;
}

Real FlowField::energy(bool normalize) const {
    assert(xzstate_ == Spectral && ystate_ == Spectral);
    return L2Norm2(*this, normalize);
}

Real FlowField::energy(int mx, int mz, bool normalize) const {
    assert(xzstate_ == Spectral && ystate_ == Spectral);
    ComplexChebyCoeff u(Ny_, a_, b_, Spectral);
    Real e = 0.0;
    for (int i = 0; i < Nd_; ++i) {
        for (int ny = 0; ny < Ny_; ++ny)
            u.set(ny, this->cmplx(mx, ny, mz, i));
        e += L2Norm2(u, normalize);
    }
    if (!normalize)
        e *= Lx_ * Lz_;
    return e;
}

Real FlowField::dudy_a() const {
    assert(ystate_ == Spectral);
    Real result = 0;
    if (taskid() == task_coeff(0, 0)) {
        BasisFunc prof = profile(0, 0);
        ChebyCoeff dudy = diff(Re(prof.u()));
        result = dudy.eval_a();
    }

#ifdef HAVE_MPI
    MPI_Bcast(&result, 1, MPI_DOUBLE, task_coeff(0, 0), cfmpi_->comm_world);
#endif

    return result;
}

Real FlowField::dudy_b() const {
    assert(ystate_ == Spectral);
    Real result = 0;
    if (taskid() == task_coeff(0, 0)) {
        BasisFunc prof = profile(0, 0);
        ChebyCoeff dudy = diff(Re(prof.u()));
        result = dudy.eval_b();
    }

#ifdef HAVE_MPI
    MPI_Bcast(&result, 1, MPI_DOUBLE, task_coeff(0, 0), cfmpi_->comm_world);
#endif

    return result;
    //     BasisFunc prof = profile(0,0);
    //     ChebyCoeff dudy = diff(Re(prof.u()));
    //     return dudy.eval_b();
}

Real FlowField::dwdy_a() const {
    assert(ystate_ == Spectral);
    Real result = 0;
    if (taskid() == task_coeff(0, 0)) {
        BasisFunc prof = profile(0, 0);
        ChebyCoeff dwdy = diff(Re(prof.w()));
        result = dwdy.eval_a();
    }

#ifdef HAVE_MPI
    MPI_Bcast(&result, 1, MPI_DOUBLE, task_coeff(0, 0), cfmpi_->comm_world);
#endif

    return result;
}

Real FlowField::dwdy_b() const {
    assert(ystate_ == Spectral);
    Real result = 0;
    if (taskid() == task_coeff(0, 0)) {
        BasisFunc prof = profile(0, 0);
        ChebyCoeff dwdy = diff(Re(prof.w()));
        result = dwdy.eval_b();
    }

#ifdef HAVE_MPI
    MPI_Bcast(&result, 1, MPI_DOUBLE, task_coeff(0, 0), cfmpi_->comm_world);
#endif

    return result;
}

Real FlowField::CFLfactor() const {
    assert(Nd_ == 3);
    FlowField u(*this);
    fieldstate xzstate = xzstate_;
    fieldstate ystate = ystate_;
    u.makePhysical();
    Vector y = chebypoints(Ny_, a_, b_);
    Real cfl = 0.0;
    Vector dx(3);
    dx[0] = Lx_ / Nx_;
    dx[2] = Lz_ / Nz_;
    for (int i = 0; i < Nd_; ++i)
        for (int ny = nylocmin_; ny < nylocmax_; ++ny) {
            dx[1] = (ny == 0 || ny == Ny_ - 1) ? y[0] - y[1] : (y[ny - 1] - y[ny + 1]) / 2.0;
            for (int nx = nxlocmin_; nx < nxlocmin_ + Nxloc_; ++nx)
                for (int nz = 0; nz < Nz_; ++nz)
                    cfl = Greater(cfl, abs(u(nx, ny, nz, i) / dx[i]));
        }

#ifdef HAVE_MPI
    Real tmp = cfl;
    MPI_Allreduce(&tmp, &cfl, 1, MPI_DOUBLE, MPI_MAX, cfmpi_->comm_world);
#endif

    u.makeState(xzstate, ystate);

    return cfl;
}

Real FlowField::CFLfactor(ChebyCoeff Ubase, ChebyCoeff Wbase) const {
    if (Ubase.N() == 0)
        return CFLfactor();

    assert(Nd_ == 3);
    FlowField u(*this);

    u.makePhysical();
    Ubase.makePhysical();
    Wbase.makePhysical();
    Vector y = chebypoints(Ny_, a_, b_);
    Real cfl = 0.0;
    Vector dx(3);
    dx[0] = Lx_ / Nx_;
    dx[2] = Lz_ / Nz_;
    for (int i = 0; i < 3; ++i)
        for (int ny = nylocmin_; ny < nylocmax_; ++ny) {
            dx[1] = (ny == 0 || ny == Ny_ - 1) ? y[0] - y[1] : (y[ny - 1] - y[ny + 1]) / 2.0;
            Real Ui = 0.0;
            if (i == 0)
                Ui = Ubase(ny);
            else if (i == 2)
                Ui = Wbase(ny);
            for (int nx = nxlocmin_; nx < nxlocmin_ + Nxloc_; ++nx)
                for (int nz = 0; nz < Nz_; ++nz)
                    cfl = Greater(cfl, (u(nx, ny, nz, i) + Ui) / dx[i]);
        }

#ifdef HAVE_MPI
    Real tmp = cfl;
    MPI_Allreduce(&tmp, &cfl, 1, MPI_DOUBLE, MPI_MAX, cfmpi_->comm_world);
#endif
    return cfl;
}

void FlowField::setState(fieldstate xz, fieldstate y) {
    xzstate_ = xz;
    ystate_ = y;
}
void FlowField::assertState(fieldstate xz, fieldstate y) const { assert(xzstate_ == xz && ystate_ == y); }

void swap(FlowField& f, FlowField& g) {
    using std::swap;

    assert(f.congruent(g));

    swap(f.rdata_, g.rdata_);
    swap(f.cdata_, g.cdata_);
    // We must swap the FFTW3 plans, as well, since they're tied to specific
    // memory locations. This burned me when I upgraded FFTW2->FFTW3.
    swap(f.xz_plan_, g.xz_plan_);
    swap(f.xz_iplan_, g.xz_iplan_);
    swap(f.y_plan_, g.y_plan_);
    swap(f.t_plan_, g.t_plan_);
    swap(f.t_iplan_, g.t_iplan_);
}

void orthonormalize(cfarray<FlowField>& e) {
    // Modified Gram-Schmidt orthogonalization
    int N = e.length();
    FlowField em_tmp;

    for (int m = 0; m < N; ++m) {
        FlowField& em = e[m];
        em *= 1.0 / L2Norm(em);
        em *= 1.0 / L2Norm(em);  // reduces floating-point errors

        // orthogonalize
        for (int n = m + 1; n < N; ++n) {
            em_tmp = em;
            em_tmp *= L2InnerProduct(em, e[n]);
            e[n] -= em_tmp;
        }
    }
}

void normalize(cfarray<FlowField>& e) {
    int N = e.length();
    for (int m = 0; m < N; ++m) {
        FlowField& em = e[m];
        em *= 1.0 / L2Norm(em);
        em *= 1.0 / L2Norm(em);
    }
}

void orthogonalize(cfarray<FlowField>& e) {
    // Modified Gram-Schmidt orthogonalization
    int N = e.length();
    FlowField em_tmp;

    for (int m = 0; m < N; ++m) {
        FlowField& em = e[m];
        Real emnorm2 = L2Norm2(em);
        for (int n = m + 1; n < N; ++n) {
            em_tmp = em;
            em_tmp *= L2InnerProduct(em, e[n]) / emnorm2;
            e[n] -= em_tmp;
        }
    }
}

FlowField quadraticInterpolate(cfarray<FlowField>& un, const cfarray<Real>& mun, Real mu, Real eps) {
    // Extrapolate indpt param R and box params as function of mu

    if (un.length() != 3 || mun.length() != 3) {
        stringstream serr;
        serr << "error in quadraticInterpolate(cfarray<FlowField>& un, cfarray<Real>& mun, Real mu, Real eps)\n";
        serr << "un.length() != 3 || mun.length() !=3\n";
        serr << "un.length()  == " << un.length() << '\n';
        serr << "mun.length() == " << mun.length() << '\n';
        serr << "exiting" << endl;
        cferror(serr.str());
    }
    const int Nx = un[0].Nx();
    const int Ny = un[0].Ny();
    const int Nz = un[0].Nz();
    const int Nd = un[0].Nd();

    if (un[1].Nx() != Nx || un[2].Nx() != Nx || un[1].Ny() != Ny || un[2].Ny() != Ny || un[1].Nz() != Nz ||
        un[2].Nz() != Nz || un[1].Nd() != Nd || un[2].Nd() != Nd) {
        cferror(
            "error in quadraticInterpolate(cfarray<FlowField>& un, cfarray<Real>& mun, Real mu, Real "
            "eps)\nIncompatible grids in un. Exiting");
    }

    cfarray<fieldstate> xstate(3);
    cfarray<fieldstate> ystate(3);

    for (int n = 0; n < 3; ++n) {
        xstate[n] = un[n].xzstate();
        ystate[n] = un[n].ystate();
    }
    for (int i = 0; i < 3; ++i)
        un[i].makePhysical();

    cfarray<Real> fn(3);

    for (int i = 0; i < 3; ++i)
        fn[i] = un[i].Lx();
    const Real Lx = isconst(fn) ? fn[0] : quadraticInterpolate(fn, mun, mu);

    for (int i = 0; i < 3; ++i)
        fn[i] = un[i].Lz();
    const Real Lz = isconst(fn) ? fn[0] : quadraticInterpolate(fn, mun, mu);

    for (int i = 0; i < 3; ++i)
        fn[i] = un[i].a();
    const Real a = isconst(fn) ? fn[0] : quadraticInterpolate(fn, mun, mu);

    for (int i = 0; i < 3; ++i)
        fn[i] = un[i].b();
    const Real b = isconst(fn) ? fn[0] : quadraticInterpolate(fn, mun, mu);

    // Extrapolate gridpoint values as function of mu
    FlowField u(Nx, Ny, Nz, Nd, Lx, Lz, a, b, un[0].cfmpi(), Physical, Physical);

    lint nylocmin = u.nylocmin();
    lint nylocmax = u.nylocmax();
    lint nxlocmin = u.nxlocmin();
    lint nxlocmax = u.nxlocmin() + u.Nxloc();
    for (int i = 0; i < Nd; ++i)
        for (int ny = nylocmin; ny < nylocmax; ++ny)
            for (int nx = nxlocmin; nx < nxlocmax; ++nx)
                for (int nz = 0; nz < Nz; ++nz) {
                    for (int n = 0; n < 3; ++n)
                        fn[n] = un[n](nx, ny, nz, i);
                    u(nx, ny, nz, i) = isconst(fn, eps) ? fn[0] : quadraticInterpolate(fn, mun, mu);
                }

    for (int i = 0; i < 3; ++i)
        un[i].makeState(xstate[i], ystate[i]);

    u.makeSpectral();
    if (u[0].padded() && u[1].padded() && u[2].padded())
        u.zeroPaddedModes();
    return u;
}

FlowField polynomialInterpolate(cfarray<FlowField>& un, cfarray<Real>& mun, Real mu) {
    // Extrapolate indpt param R and box params as function of mu

    if (un.length() != mun.length() || un.length() < 1) {
        stringstream serr;
        serr << "error in quadraticInterpolate(cfarray<FlowField>& un, cfarray<Real>& mun, Real mu, Real eps)\n";
        serr << "un.length() != mun.length() or un.length() < 1\n";
        serr << "un.length()  == " << un.length() << '\n';
        serr << "mun.length() == " << mun.length() << '\n';
        serr << "exiting" << endl;
        cferror(serr.str());
    }
    const int N = un.length();
    const int Nx = un[0].Nx();
    const int Ny = un[0].Ny();
    const int Nz = un[0].Nz();
    const int Nd = un[0].Nd();

    if (un[1].Nx() != Nx || un[2].Nx() != Nx || un[1].Ny() != Ny || un[2].Ny() != Ny || un[1].Nz() != Nz ||
        un[2].Nz() != Nz || un[1].Nd() != Nd || un[2].Nd() != Nd) {
        cferror(
            "error in quadraticInterpolate(cfarray<FlowField>& un, cfarray<Real>& mun, Real mu, Real "
            "eps)\nIncompatible grids in un. Exiting");
    }

    cfarray<fieldstate> xstate(N);
    cfarray<fieldstate> ystate(N);

    for (int n = 0; n < N; ++n) {
        xstate[n] = un[n].xzstate();
        ystate[n] = un[n].ystate();
    }
    for (int i = 0; i < N; ++i)
        un[i].makePhysical();

    cfarray<Real> fn(N);

    for (int i = 0; i < N; ++i)
        fn[i] = un[i].Lx();
    const Real Lx = polynomialInterpolate(fn, mun, mu);

    for (int i = 0; i < N; ++i)
        fn[i] = un[i].Lz();
    const Real Lz = polynomialInterpolate(fn, mun, mu);

    for (int i = 0; i < N; ++i)
        fn[i] = un[i].a();
    const Real a = polynomialInterpolate(fn, mun, mu);

    for (int i = 0; i < N; ++i)
        fn[i] = un[i].b();
    const Real b = polynomialInterpolate(fn, mun, mu);

    // Extrapolate gridpoint values as function of mu
    FlowField u(Nx, Ny, Nz, Nd, Lx, Lz, a, b, un[0].cfmpi(), Physical, Physical);
    // u.makeState(Physical,Physical);
    // ug.binarySave("uginit");

    lint nylocmin = u.nylocmin();
    lint nylocmax = u.nylocmax();
    lint nxlocmin = u.nxlocmin();
    lint nxlocmax = u.nxlocmin() + u.Nxloc();
    for (int i = 0; i < Nd; ++i)
        for (lint ny = nylocmin; ny < nylocmax; ++ny)
            for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
                for (int nz = 0; nz < Nz; ++nz) {
                    for (int n = 0; n < 3; ++n)
                        fn[n] = un[n](nx, ny, nz, i);
                    u(nx, ny, nz, i) = polynomialInterpolate(fn, mun, mu);
                }
    for (int i = 0; i < N; ++i)
        un[i].makeState(xstate[i], ystate[i]);

    u.makeSpectral();
    if (u[0].padded())
        u.zeroPaddedModes();
    return u;
}

//=======================================================================
#ifdef HAVE_LIBHDF5_CPP

void hdf5write(int i, const string& name, H5::H5File& h5file) {
    herr_t status = H5Tconvert(H5T_NATIVE_INT, H5T_STD_I32BE, 1, &i, NULL, H5P_DEFAULT);
    if (status < 0) {
        cout << "HDF5 datatype conversion failed! Status: " << status;
    }

    H5::Group rootgr = h5file.openGroup("/");
    H5::DataSpace dspace = H5::DataSpace(H5S_SCALAR);
    H5::Attribute attr = rootgr.createAttribute(name.c_str(), H5::PredType::STD_I32BE, dspace);
    attr.write(H5::PredType::STD_I32BE, &i);
}

void hdf5write(Real x, const string& name, H5::H5File& h5file) {
    herr_t status = H5Tconvert(H5T_NATIVE_DOUBLE, H5T_IEEE_F64BE, 1, &x, NULL, H5P_DEFAULT);
    if (status < 0) {
        cout << "HDF5 datatype conversion failed! Status: " << status;
    }

    H5::Group rootgr = h5file.openGroup("/");
    H5::DataSpace dspace = H5::DataSpace(H5S_SCALAR);
    H5::Attribute attr = rootgr.createAttribute(name.c_str(), H5::PredType::IEEE_F64BE, dspace);
    attr.write(H5::PredType::IEEE_F64BE, &x);
}

void hdf5write(const Vector& v, const string& name, H5::H5File& h5file) {
    const int N = v.length();
    const hsize_t hnx[1] = {(hsize_t)N};

    auto data = vector<Real>(N, 0.0);
    copy(v.pointer(), v.pointer() + N, begin(data));

    // create H5 dataspace, convert from native double to 64-bit IEE
    H5::DataSpace dspace(1, hnx);
    H5::DataSet dset = h5file.createDataSet(name.c_str(), H5::PredType::IEEE_F64BE, dspace);
    herr_t status = H5Tconvert(H5T_NATIVE_DOUBLE, H5T_IEEE_F64BE, N, data.data(), NULL, H5P_DEFAULT);

    if (status < 0) {
        cout << "HDF5 datatype conversion failed! Status: " << status;
    }
    dset.write(data.data(), H5::PredType::IEEE_F64BE);
    dset.close();
}

void hdf5write(const FlowField& u, const string& name, H5::H5File& h5file) {
    const int Nx = u.Nx();
    const int Ny = u.Ny();
    const int Nz = u.Nz();
    const int Nd = u.Nd();
    const int dsize = Nx * Ny * Nz * Nd;

    auto data = vector<double>(dsize, 0.0);  // create cfarray of data to be written

    for (int nx = 0; nx < Nx; ++nx)
        for (int ny = 0; ny < Ny; ++ny)
            for (int nz = 0; nz < Nz; ++nz)
                for (int i = 0; i < Nd; ++i)
                    data[nz + Nz * (ny + Ny * (nx + Nx * i))] = u(nx, ny, nz, i);
    const hsize_t dimsf[4] = {(hsize_t)Nd, (hsize_t)Nz, (hsize_t)Ny, (hsize_t)Nx};  // dataset dimensions

    H5::DataSpace dspace(4, dimsf);  // create dataspace
    H5::DataSet dset = h5file.createDataSet(name.c_str(), H5::PredType::IEEE_F64BE, dspace);
    herr_t status = H5Tconvert(H5T_NATIVE_DOUBLE, H5T_IEEE_F64BE, data.size(), data.data(), NULL, H5P_DEFAULT);
    if (status < 0) {
        cout << "HDF5 datatype conversion failed! Status: " << status;
    }
    dset.write(data.data(), H5::PredType::IEEE_F64BE);  // write data to dataset
    dset.close();
}

bool hdf5query(const string& name, H5::H5File& h5file) {
    // Yucky. HDF5 does not have a "does attribute exist?" function,
    // so we have to loop over all attributes and test.
    H5::Group rootgr = h5file.openGroup("/");
    const uint N = rootgr.getNumAttrs();

    for (uint i = 0; i < N; ++i) {
        H5::Attribute attr = rootgr.openAttribute(i);
        if (attr.getName() == name)
            return true;
    }
    return false;
}

void hdf5read(int& i, const string& name, H5::H5File& h5file) {
    H5::Group rootgr = h5file.openGroup("/");
    H5::Attribute attr = rootgr.openAttribute(name.c_str());
    attr.read(H5::PredType::STD_I32BE, &i);
    H5Tconvert(H5T_STD_I32BE, H5T_NATIVE_INT, 1, &i, NULL, H5P_DEFAULT);
}

void hdf5read(Real& x, const string& name, H5::H5File& h5file) {
    H5::Group rootgr = h5file.openGroup("/");
    H5::Attribute attr = rootgr.openAttribute(name.c_str());
    attr.read(H5::PredType::IEEE_F64BE, &x);
    H5Tconvert(H5T_IEEE_F64BE, H5T_NATIVE_DOUBLE, 1, &x, NULL, H5P_DEFAULT);
}

void hdf5read(FlowField& u, const string& name, H5::H5File& h5file) {
    u.setState(Physical, Physical);
    assert(u.numtasks() == 1);
    if (u.taskid() == 0) {
        const uint Nx = u.Nx();
        const uint Ny = u.Ny();
        const uint Nz = u.Nz();
        const uint Nd = u.Nd();

        H5::DataSet dataset = h5file.openDataSet(name.c_str());
        H5::DataSpace dataspace = dataset.getSpace();
        // int rank = dataspace.getSimpleExtentNdims();
        // hsize_t Nh5[4]; // Nd x Nx x Ny x Nz dimensions of h5 data cfarray
        hsize_t Nh5[4];  // Nd x Nz x Ny x Nx dimensions of h5 data cfarray
        int rank = dataspace.getSimpleExtentDims(Nh5, NULL);

        if (rank != 4)
            cferror("error reading FlowField from hdf5 file : rank of data is not 4");

        Nh5[0] = Nd;
        Nh5[1] = Nz;
        Nh5[2] = Ny;
        Nh5[3] = Nx;

        H5::DataSpace memspace(4, Nh5);

        int N = Nx * Ny * Nz * Nd;
        auto buff = vector<Real>(N, 0.0);
        dataset.read(buff.data(), H5::PredType::IEEE_F64BE, memspace, dataspace);

        H5Tconvert(H5T_IEEE_F64BE, H5T_NATIVE_DOUBLE, buff.size(), buff.data(), NULL, H5P_DEFAULT);

        for (uint nz = 0; nz < Nz; ++nz)
            for (uint ny = 0; ny < Ny; ++ny)
                for (uint nx = 0; nx < Nx; ++nx)
                    for (uint i = 0; i < Nd; ++i)
                        u(nx, ny, nz, i) = buff[nz + Nz * (ny + Ny * (nx + Nx * i))];
        dataset.close();
    }
    u.makeSpectral();
}

void hdf5addstuff(const std::string& filebase, Real nu, ChebyCoeff Ubase, ChebyCoeff Wbase) {
    H5std_string h5name = appendSuffix(filebase, ".h5");
    H5::H5File h5file(h5name, H5F_ACC_TRUNC);
    h5file.createGroup("/extras");

    Ubase.makePhysical();
    Wbase.makePhysical();
    hdf5write(nu, "/extras/nu", h5file);
    hdf5write(Ubase, "/extras/Ubase", h5file);
    hdf5write(Wbase, "/extras/Wbase", h5file);
}

#endif  // HAVE_HDF5LIB_CPP

int field2vector_size(const FlowField& u) {
    int Kx = u.kxmaxDealiased();
    int Kz = u.kzmaxDealiased();
    int Ny = u.Ny();
    // FIXME: Determine array size
    // The original formula was
    //     int N = 4* ( Kx+2*Kx*Kz+Kz ) * ( Ny-3 ) +2* ( Ny-2 );
    // but I've not been able to twist my head enough to adapt it do distributed FlowFields.
    // Since it doesn't take that much time, we now perform the loops twice, once empty to
    // determine cfarray sizes and once with the actual data copying. // Tobias
    int N = 0;
    if (u.taskid() == u.task_coeff(0, 0))
        N += 2 * (Ny - 2);
    for (int kx = 1; kx <= Kx; ++kx)
        if (u.taskid() == u.task_coeff(u.mx(kx), 0))
            N += 2 * (Ny - 2) + 2 * (Ny - 4);
    for (int kz = 1; kz <= Kz; ++kz)
        if (u.taskid() == u.task_coeff(0, u.mz(kz)))
            N += 2 * (Ny - 2) + 2 * (Ny - 4);
    for (int kx = -Kx; kx <= Kx; ++kx) {
        if (kx == 0)
            continue;
        int mx = u.mx(kx);
        for (int kz = 1; kz <= Kz; ++kz) {
            int mz = u.mz(kz);
            if (u.taskid() == u.task_coeff(mx, mz)) {
                N += 2 * (Ny - 2) + 2 * (Ny - 4);
            }
        }
    }
    return N;
}

void field2vector(const FlowField& u, VectorXd& a) {
    // Transform a 3D FlowField into a vector, neglecting all redundant components
    // The implementation minimizes the necessary communication operations (acutally 0 in
    // field2vector and Kz when restoring in vector2field) but results
    // in an unequal number of vector elements per process. The other option - the same
    // number of vector elements per process - would require a lot of communication, which
    // is way less time-efficient in most situations.

    assert(u.xzstate() == Spectral && u.ystate() == Spectral);
    assert(u.Nd() == 3);  // enforcing div(u)=0 assumes velocity field
    int Kx = u.kxmaxDealiased();
    int Kz = u.kzmaxDealiased();
    int Ny = u.Ny();

    int N = field2vector_size(u);

    if (a.size() < N)
        a.resize(N, true);
    setToZero(a);

    int n = 0;
    // These loops follow Gibson, Halcrow, Cvitanovic table 1.
    for (int ny = 2; ny < Ny; ++ny)
        // Ny-2 modes (line 1 of table)
        if (u.taskid() == u.task_coeff(0, 0))
            a(n++) = Re(u.cmplx(0, ny, 0, 0));
    for (int ny = 2; ny < Ny; ++ny)
        // Ny-2 modes (line 2)
        if (u.taskid() == u.task_coeff(0, 0))
            a(n++) = Re(u.cmplx(0, ny, 0, 2));
    // Some coefficients from the FlowField are linearly dependent due
    // to BCs and divergence constraint. Omit these from a(n), and reconstruct
    // them in vector2field(v,u) using the constraint equations. In what follows,
    // which coefficients are omitted is determined by the changing ny loop index
    // e.g. (ny=3; n<Ny-1; ++ny) omits ny=0,1 and Ny-1, and the constraint eqns
    // that allow their reconstruction are mentioned in comments to the right
    // of the loop. 2007-06-07 jfg.
    for (int kx = 1; kx <= Kx; ++kx) {
        int mx = u.mx(kx);
        if (u.taskid() == u.task_coeff(mx, 0)) {
            for (int ny = 2; ny < Ny; ++ny) {        // w BCs => ny=0,1 coefficients
                a(n++) = Re(u.cmplx(mx, ny, 0, 2));  // J(Ny-2) modes (line 3a)
                a(n++) = Im(u.cmplx(mx, ny, 0, 2));  // J(Ny-2) modes (line 3b)
            }
            for (int ny = 3; ny < Ny - 1; ++ny) {    // u BCs => 0,1; v BC => 2; div => Ny-1
                a(n++) = Re(u.cmplx(mx, ny, 0, 0));  // J(Ny-4) modes (line 4a)
                a(n++) = Im(u.cmplx(mx, ny, 0, 0));  // J(Ny-4) modes (line 4b)
            }
        }
    }
    for (int kz = 1; kz <= Kz; ++kz) {
        int mz = u.mz(kz);
        if (u.taskid() == u.task_coeff(0, mz)) {
            for (int ny = 2; ny < Ny; ++ny) {        // u BCs => 0,1; v BC => 2; div => Ny-1
                a(n++) = Re(u.cmplx(0, ny, mz, 0));  // K(Ny-2)  (line 5a)
                a(n++) = Im(u.cmplx(0, ny, mz, 0));  // K(Ny-2)  (line 5b)
            }
            for (int ny = 3; ny < Ny - 1; ++ny) {    // w BCs => 0,1; v BC => 2; div => Ny-1
                a(n++) = Re(u.cmplx(0, ny, mz, 2));  // K(Ny-2)  (line 6a)
                a(n++) = Im(u.cmplx(0, ny, mz, 2));  // K(Ny-2)  (line 6b)
            }
        }
    }
    for (int kx = -Kx; kx <= Kx; ++kx) {
        if (kx == 0)
            continue;

        int mx = u.mx(kx);
        for (int kz = 1; kz <= Kz; ++kz) {
            int mz = u.mz(kz);
            if (u.taskid() == u.task_coeff(mx, mz)) {
                for (int ny = 2; ny < Ny; ++ny) {         // u BCs => 0,1;
                    a(n++) = Re(u.cmplx(mx, ny, mz, 0));  // JK(Ny-2)  (line 7a)
                    a(n++) = Im(u.cmplx(mx, ny, mz, 0));  // JK(Ny-2)  (line 7b)
                }
                for (int ny = 3; ny < Ny - 1; ++ny) {     // w BCs => 0,1; v BC => 2; div => Ny-1
                    a(n++) = Re(u.cmplx(mx, ny, mz, 2));  // JK(Ny-4)  (line 8a)
                    a(n++) = Im(u.cmplx(mx, ny, mz, 2));  // JK(Ny-4)  (line 8b)
                }
            }
        }
    }
}

void vector2field(const VectorXd& a, FlowField& u) {
    assert(u.Nd() == 3);  // enforcing div(u)=0 assumes velocity field
    u.setToZero();

    int Kx = u.kxmaxDealiased();
    int Kz = u.kzmaxDealiased();
    int Ny = u.Ny();  // max polynomial order == number y gridpoints - 1
    Real ya = u.a();
    Real yb = u.b();
    Real Lx = u.Lx();
    Real Lz = u.Lz();

    ComplexChebyCoeff f0(Ny, ya, yb, Spectral);
    ComplexChebyCoeff f1(Ny, ya, yb, Spectral);
    ComplexChebyCoeff f2(Ny, ya, yb, Spectral);

    // These loops follow Gibson, Halcrow, Cvitanovic table 1.
    // The vector contains the linearly independent real numbers in a
    // zero-div no-slip FlowField. The idea is to extract these from the
    // vector and reconstruct the others from the BCs and div constraint
    // See corresponding comments in field2vector(u,v);

    int n = 0;

    // =========================================================
    // (0,0) Fourier mode
    if (u.taskid() == u.task_coeff(0, 0)) {
        for (int ny = 2; ny < Ny; ++ny)
            f0.re[ny] = a(n++);
        fixDiri(f0.re);
        for (int ny = 0; ny < Ny; ++ny)
            u.cmplx(0, ny, 0, 0) = f0[ny];

        for (int ny = 2; ny < Ny; ++ny)
            f2.re[ny] = a(n++);
        fixDiri(f2.re);
        for (int ny = 0; ny < Ny; ++ny)
            u.cmplx(0, ny, 0, 2) = f2[ny];
    }

    // =========================================================
    // (kx,0) Fourier modes, 0<kx
    for (int kx = 1; kx <= Kx; ++kx) {
        int mx = u.mx(kx);

        if (u.taskid() == u.task_coeff(mx, 0)) {
            for (int ny = 2; ny < Ny; ++ny) {
                f2.re[ny] = a(n++);
                f2.im[ny] = a(n++);
            }
            fixDiri(f2);

            for (int ny = 3; ny < Ny - 1; ++ny) {
                f0.re[ny] = a(n++);
                f0.im[ny] = a(n++);
            }
            f0.re[Ny - 1] = 0.0;
            f0.im[Ny - 1] = 0.0;
            fixDiriMean(f0);

            integrate(f0, f1);
            f1.sub(0, 0.5 * (f1.eval_a() + f1.eval_b()));
            f1 *= Complex(0.0, -(2 * pi * kx) / Lx);

            for (int ny = 0; ny < Ny; ++ny)
                u.cmplx(mx, ny, 0, 0) = f0[ny];
            for (int ny = 0; ny < Ny; ++ny)
                u.cmplx(mx, ny, 0, 1) = f1[ny];
            for (int ny = 0; ny < Ny; ++ny)
                u.cmplx(mx, ny, 0, 2) = f2[ny];
        }

        // ------------------------------------------------------
        // Now copy conjugates of u(kx,ny,0,i) to u(-kx,ny,0,i). These are
        // redundant modes stored only for the convenience of FFTW.
        int mxm = u.mx(-kx);
        int send_id = u.task_coeff(mx, 0);
        int rec_id = u.task_coeff(mxm, 0);
        for (int ny = 0; ny < Ny; ++ny) {
            if (u.taskid() == send_id && send_id == rec_id) {  // all is on the same process -> just copy
                u.cmplx(mxm, ny, 0, 0) = conj(f0[ny]);
                u.cmplx(mxm, ny, 0, 1) = conj(f1[ny]);
                u.cmplx(mxm, ny, 0, 2) = conj(f2[ny]);
            }
#ifdef HAVE_MPI     // send_id != rec_id requires multiple processes
            else {  // Transfer the conjugates via MPI
                if (u.taskid() == send_id) {
                    Complex tmp0 = conj(f0[ny]);
                    Complex tmp1 = conj(f1[ny]);
                    Complex tmp2 = conj(f2[ny]);
                    MPI_Send(&tmp0, 1, MPI_DOUBLE_COMPLEX, rec_id, 0, MPI_COMM_WORLD);
                    MPI_Send(&tmp1, 1, MPI_DOUBLE_COMPLEX, rec_id, 1, MPI_COMM_WORLD);
                    MPI_Send(&tmp2, 1, MPI_DOUBLE_COMPLEX, rec_id, 2, MPI_COMM_WORLD);
                }
                if (u.taskid() == rec_id) {
                    Complex tmp0, tmp1, tmp2;
                    MPI_Status status;
                    MPI_Recv(&tmp0, 1, MPI_DOUBLE_COMPLEX, send_id, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(&tmp1, 1, MPI_DOUBLE_COMPLEX, send_id, 1, MPI_COMM_WORLD, &status);
                    MPI_Recv(&tmp2, 1, MPI_DOUBLE_COMPLEX, send_id, 2, MPI_COMM_WORLD, &status);
                    u.cmplx(mxm, ny, 0, 0) = tmp0;
                    u.cmplx(mxm, ny, 0, 1) = tmp1;
                    u.cmplx(mxm, ny, 0, 2) = tmp2;
                }
            }
#endif
        }
    }

    // =========================================================
    // (0,kz) Fourier modes, 0<kz.
    for (int kz = 1; kz <= Kz; ++kz) {
        int mz = u.mz(kz);

        if (u.taskid() == u.task_coeff(0, mz)) {
            for (int ny = 2; ny < Ny; ++ny) {
                f0.re[ny] = a(n++);
                f0.im[ny] = a(n++);
            }
            fixDiri(f0);
            for (int ny = 3; ny < Ny - 1; ++ny) {
                f2.re[ny] = a(n++);
                f2.im[ny] = a(n++);
            }
            f2.re[Ny - 1] = 0.0;
            f2.im[Ny - 1] = 0.0;
            fixDiriMean(f2);

            integrate(f2, f1);
            f1.sub(0, 0.5 * (f1.eval_a() + f1.eval_b()));
            f1 *= Complex(0.0, -(2 * pi * kz) / Lz);

            for (int ny = 0; ny < Ny; ++ny)
                u.cmplx(0, ny, mz, 0) = f0[ny];
            for (int ny = 0; ny < Ny; ++ny)
                u.cmplx(0, ny, mz, 1) = f1[ny];
            for (int ny = 0; ny < Ny; ++ny)
                u.cmplx(0, ny, mz, 2) = f2[ny];
        }
    }

    // =========================================================
    for (int kx = -Kx; kx <= Kx; ++kx) {
        if (kx == 0)
            continue;

        int mx = u.mx(kx);
        for (int kz = 1; kz <= Kz; ++kz) {
            int mz = u.mz(kz);

            if (u.taskid() == u.task_coeff(mx, mz)) {
                for (int ny = 2; ny < Ny; ++ny) {
                    f0.re[ny] = a(n++);
                    f0.im[ny] = a(n++);
                }
                fixDiri(f0);

                // f0 is complete. Load it into u.
                for (int ny = 0; ny < Ny; ++ny)
                    u.cmplx(mx, ny, mz, 0) = f0[ny];

                for (int ny = 3; ny < Ny - 1; ++ny) {
                    f2.re[ny] = a(n++);
                    f2.im[ny] = a(n++);
                }
                f2.re[Ny - 1] = -f0.re[Ny - 1] * (kx * Lz) / (kz * Lx);
                f2.im[Ny - 1] = -f0.im[Ny - 1] * (kx * Lz) / (kz * Lx);

                // Adjust 0,1,2 coeffs of f2 so that f2(+/-1) = 0 and
                // kz/Lz mean(f2) + kx/Lx mean(f0) == 0 (so that  v(1)==v(-1))
                Complex f2a = f2.eval_a();
                Complex f2b = f2.eval_b();
                Complex f0m = f0.mean();
                Complex f2m = f2.mean() + (kx * Lz) / (kz * Lx) * f0m;

                f2.sub(0, 0.125 * (f2a + f2b) + 0.75 * f2m);
                f2.sub(1, 0.5 * (f2b - f2a));
                f2.sub(2, 0.375 * (f2a + f2b) - 0.75 * f2m);

                for (int ny = 0; ny < Ny; ++ny)
                    u.cmplx(mx, ny, mz, 2) = f2[ny];

                f0 *= Complex(0, -2 * pi * kx / Lx);
                f2 *= Complex(0, -2 * pi * kz / Lz);
                f0 += f2;

                integrate(f0, f1);
                f1.sub(0, 0.5 * (f1.eval_a() + f1.eval_b()));

                for (int ny = 0; ny < Ny; ++ny)
                    u.cmplx(mx, ny, mz, 1) = f1[ny];
            }
        }
    }
    u.setPadded(true);
}

void fixdivnoslip(FlowField& u) {
    VectorXd v;
    field2vector(u, v);
    vector2field(v, u);
}

}  // namespace chflow
