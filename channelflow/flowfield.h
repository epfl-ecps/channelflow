/**
 * Class for N-dim Fourier x Chebyshev x Fourier expansions
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef CHANNELFLOW_FLOWFIELD_H
#define CHANNELFLOW_FLOWFIELD_H

#include "cfbasics/cfvector.h"
#include "cfbasics/mathdefs.h"
#include "channelflow/basisfunc.h"
#include "channelflow/cfmpi.h"
#include "channelflow/chebyshev.h"
#include "channelflow/realprofile.h"
#include "channelflow/realprofileng.h"

#include <fftw3.h>

#include <memory>

// A brief overview of FlowField (for more info see channelflow.ps in docs/).
//
// FlowField represents 3d vector fields with Fourier x Chebyshev x Fourier
// spectral expansions. I.e.
// u(x,y,z,i) =
//   sum{kx,kz,ny,i} uhat_{kx,kz,ny,i} T_ny(y) exp(2 pi I [kx x/Lx + kz z/Lz])
//
// FlowField represents such expansions with multidimensional cfarrays of
// spectral coefficients uhat_{mx,mz,my,i} or gridpoint data u(nx,ny,nz,i).
// (see docs for the relation between indices nx, mx, and kx).
//
// FlowFields have independent xz and y spectral transforms, therefore they
// can be in any one of four states: (xzstate, ystate) == (Physical, Physical),
// (Physical, Spectral), (Spectral, Physical), or (Spectral, Spectral).

// Most mathematical operations must be computed on when the FlowField is in
// (Spectral, Spectral) state. Some operations will automatically convert to
// the appropriate state to do the computation, then convert back to original
// state before returning. Not sure whether this is good --it's convenient but
// potentially leads to silent inefficiencies.

// The debugging libraries check that FlowFields are in the proper state,
// when it matters. Test your code for correct FlowField transforms and
// states by compiling and running with debug libs: "make foo.dx; ./foo.dx".

namespace chflow {

class FieldSymmetry;

class FlowField {
   public:
    FlowField();

    FlowField(int Nx, int Ny, int Nz, int Nd, Real Lx, Real Lz, Real a, Real b, CfMPI* cfmpi = NULL,
              fieldstate xzstate = Spectral, fieldstate ystate = Spectral, uint fftw_flags = FFTW_ESTIMATE);

    FlowField(int Nx, int Ny, int Nz, int Nd, int tensorOrder, Real Lx, Real Lz, Real a, Real b, CfMPI* cfmpi = NULL,
              fieldstate xzstate = Spectral, fieldstate ystate = Spectral, uint fftw_flags = FFTW_ESTIMATE);

    FlowField(const FlowField& u);
    FlowField(const std::string& filebase, CfMPI* cfmpi = NULL);  // opens filebase.h5 or filebase.ff
    FlowField(const std::string& filebase, int major, int minor, int update);

    FlowField& operator=(const FlowField& u);  // assign identical copy of U

    // match geom params, set to zero
    void reconfig(const FlowField& u, uint fftw_flags = FFTW_ESTIMATE);
    void resize(int Nx, int Ny, int Nz, int Nd, Real Lx, Real Lz, Real a, Real b, CfMPI* cfmpi,
                uint fftw_flags = FFTW_ESTIMATE);
    void rescale(Real Lx, Real Lz);
    void interpolate(FlowField u);          // interpolate U onto this grid.
    void toprocess0(FlowField& v) const;    // communicate U onto process 0
    void fromprocess0(const FlowField& v);  // fill flowField from v, which is only on process 0

    void optimizeFFTW(uint fftw_flags = FFTW_MEASURE);

    // Real    operator()(nx,ny,nz,i):  0<=nx<Nx, 0<=ny<Ny, 0<=nz<Nz, 0<=i<Nd
    // Complex cmplx(mx,my,mz,i):       0<=mx<Mx, 0<=my<My, 0<=mz<Mz, 0<=i<Nd
    //                           where    Mx==Nx    My==Ny  Mz==Nz_/2+1;

    // Element access methods for vector-valued fields
    inline Real& operator()(int nx, int ny, int nz, int i);
    inline const Real& operator()(int nx, int ny, int nz, int i) const;
    inline Complex& cmplx(int mx, int my, int mz, int i);
    inline const Complex& cmplx(int mx, int my, int mz, int i) const;

    // Element access methods for tensor-valued fields
    inline Real& operator()(int nx, int ny, int nz, int i, int j);
    inline const Real& operator()(int nx, int ny, int nz, int i, int j) const;
    inline Complex& cmplx(int mx, int ny, int mz, int i, int j);
    inline const Complex& cmplx(int mx, int ny, int mz, int i, int j) const;

    FlowField operator[](int i) const;                  // extract ith component
    FlowField operator[](const cfarray<int>& i) const;  // extract {i}th components,

    Real eval(Real x, Real y, Real z, int i) const;

    ComplexChebyCoeff profile(int mx, int mz, int i) const;
    BasisFunc profile(int mx, int mz) const;

    void makeSpectral_xz();
    void makePhysical_xz();
    void makeSpectral_y();
    void makePhysical_y();
    void makeSpectral();
    void makePhysical();
    void makeState(fieldstate xzstate, fieldstate ystate);

    void setToZero();
    void perturb(Real magnitude, Real spectralDecay, bool meanflow = true);
    void addPerturbation(int kx, int kz, Real mag, Real spectralDecay);
    void addPerturbation1D(int kx, int kz, Real mag, Real spectralDecay);
    void addPerturbations(int kxmax, int kzmax, Real mag, Real spectralDecay, bool meanflow = true);
    void addPerturbations(Real magnitude, Real spectralDecay, bool meanflow = true);

    FlowField& operator*=(const FieldSymmetry& s);
    FlowField& project(const FieldSymmetry& s);
    FlowField& project(const cfarray<FieldSymmetry>& s);

    inline int numXmodes() const;  // should eliminate
    inline int numYmodes() const;
    inline int numZmodes() const;
    inline int numXgridpts() const;  // should change to Nxmodes
    inline int numYgridpts() const;
    inline int numZgridpts() const;
    inline int vectorDim() const;  // should eliminate

    inline int Nx() const;  // same as numXgridpts()
    inline int Ny() const;  // same as numYgridpts()
    inline int Nz() const;  // same as numZgridpts()
    inline int Nd() const;  // same as vectorDim()
    inline int Mx() const;  // same as numXmodes()
    inline int My() const;  // same as numYmodes()
    inline int Mz() const;  // same as numZmodes()

    inline lint Nloc() const;
    inline lint Nxloc() const;
    inline lint Nxlocmax() const;
    inline lint nxlocmin() const;
    inline lint Mxloc() const;
    inline lint mxlocmin() const;
    inline lint Nyloc() const;
    inline lint Nylocpad() const;
    inline lint Nypad() const;
    inline lint nylocmin() const;
    inline lint nylocmax() const;
    inline lint Mzloc() const;
    inline lint mzlocmin() const;

    inline int mx(int kx) const;  // where, in the cfarray, is kx?
    inline int mz(int kz) const;  // should be mx,mz rather than nx,nz
    inline int kx(int mx) const;  // the wavenumber of the nxth cfarray elem
    inline int kz(int mz) const;

    inline int kxmax() const;  // the largest value kx takes on
    inline int kzmax() const;
    inline int kxmin() const;  // the smallest value kx takes on
    inline int kzmin() const;
    inline int kxminDealiased() const;  // |kx| > kxmaxDealiased is aliased mode
    inline int kxmaxDealiased() const;  // |kx| > kxmaxDealiased is aliased mode
    inline int kzminDealiased() const;  // |kz| > kzmaxDealiased is aliased mode
    inline int kzmaxDealiased() const;
    inline bool isAliased(int kx, int kz) const;

    inline Real Lx() const;
    inline Real Ly() const;
    inline Real Lz() const;
    inline Real a() const;
    inline Real b() const;
    inline Real x(int nx) const;  // the x coord of the nxth gridpoint
    inline Real y(int ny) const;
    inline Real z(int nz) const;

    inline int nproc0() const;
    inline int nproc1() const;
    inline int taskid() const;
    inline int key0() const;
    inline int color0() const;
    inline int taskid_world() const;
    inline int numtasks() const;
    inline int task_coeff(int mx, int mz) const;
    inline int task_coeffp(int nx, int ny) const;
    inline int task_coeff(int mx, int my, int mz, int i) const;
    inline MPI_Comm* comm_world() const;

    Vector xgridpts() const;
    Vector ygridpts() const;
    Vector zgridpts() const;

    Complex Dx(int mx) const;         // spectral diff operator
    Complex Dz(int mz) const;         // spectral diff operator
    Complex Dx(int mx, int n) const;  // spectral diff operator
    Complex Dz(int mz, int n) const;  // spectral diff operator

    FlowField& operator*=(Real x);
    FlowField& operator*=(Complex x);
    FlowField& operator+=(const ChebyCoeff& U);                //  i.e. u(0,*,0,0) += U
    FlowField& operator-=(const ChebyCoeff& U);                //       u(0,*,0,0) -= U
    FlowField& operator+=(const std::vector<ChebyCoeff>& UW);  // i.e. u(0,*,0,0) += U and u(0,*,0.2) += W
    FlowField& operator-=(const std::vector<ChebyCoeff>& UW);  // i.e. u(0,*,0,0) -= U and u(0,*,0.2) -= W
    FlowField& operator+=(const Real& a);                      //  i.e. u(0,*,0,0) += a
    FlowField& operator-=(const Real& a);                      //       u(0,*,0,0) -= a
    FlowField& operator+=(const ComplexChebyCoeff& U);         // u.cmplx(0,*,0,0) += U
    FlowField& operator-=(const ComplexChebyCoeff& U);         // u.cmplx(0,*,0,0) += U
    FlowField& operator+=(const BasisFunc& U);
    FlowField& operator-=(const BasisFunc& U);
    FlowField& operator+=(const RealProfile& U);
    FlowField& operator-=(const RealProfile& U);
    FlowField& operator+=(const FlowField& u);
    FlowField& operator-=(const FlowField& u);
    FlowField& operator+=(const RealProfileNG& U);
    FlowField& operator-=(const RealProfileNG& U);

    inline void add(const Real a, const FlowField& u);
    inline void add(const Real a, const FlowField& u, const Real b, const FlowField& v);

    bool geomCongruent(const FlowField& f, Real eps = 1e-13) const;
    bool congruent(const FlowField& f, Real eps = 1e-13) const;
    bool congruent(const BasisFunc& phi) const;
    bool congruent(const RealProfileNG& e) const;
    friend void swap(FlowField& f, FlowField& g);  // exchange data of two congruent fields.

    // save methods add extension .asc or .ff ("flow field")
    void asciiSave(const std::string& filebase) const;
    void binarySave(const std::string& filebase) const;
    void hdf5Save(const std::string& filebase) const;
    void writeNetCDF(const std::string& filebase,
                     std::vector<std::string> component_names = std::vector<std::string>()) const;
    void VTKSave(const std::string& filebase, bool SwapEndian = true) const;

    // read methods
    void readNetCDF(const std::string& filebase);

    // save to .h5 or .ff based on file extension, or if none, presence of HDF5 libs
    void save(const std::string& filebase, std::vector<std::string> component_names = std::vector<std::string>()) const;

    // save k-normal slice of ith component at nkth gridpoint (along k-direction)
    void saveSlice(int k, int i, int nk, const std::string& filebase, int xstride = 1, int ystride = 1,
                   int zstride = 1) const;
    void saveProfile(int mx, int mz, const std::string& filebase) const;
    void saveProfile(int mx, int mz, const std::string& filebase, const ChebyTransform& t) const;

    // save L2Norm(u(kx,kz)) to file, kxorder => order in kx,kz, drop last mode
    void saveSpectrum(const std::string& filebase, int i, int ny = -1, bool kxorder = true,
                      bool showpadding = false) const;
    void saveSpectrum(const std::string& filebase, bool kxorder = true, bool showpadding = false) const;
    void saveDivSpectrum(const std::string& filebase, bool kxorder = true, bool showpadding = false) const;

    void print() const;
    void dump() const;

    Real energy(bool normalize = true) const;
    Real energy(int mx, int mz, bool normalize = true) const;
    Real dudy_a() const;
    Real dudy_b() const;
    Real dwdy_a() const;
    Real dwdy_b() const;
    Real CFLfactor() const;
    Real CFLfactor(ChebyCoeff Ubase, ChebyCoeff Wbase) const;

    void setState(fieldstate xz, fieldstate y);
    void assertState(fieldstate xz, fieldstate y) const;

    inline fieldstate xzstate() const;
    inline fieldstate ystate() const;

    void zeroPaddedModes();  // set padded modes to zero
    void setPadded(bool b);  // turn on padded flag
    bool padded() const;     // true implies that upper 1/3 modes are zero

    // returns pointer to rdata_ array for IO which does not contain padding
    void removePaddedModes(Real* rdata_io, lint Nxloc_io, lint nxlocmin_io, lint Mzloc_io, lint mzlocmin_io) const;
    void addPaddedModes(Real* rdata_io, lint Nxloc_io, lint nxlocmin_io, lint Mzloc_io, lint mzlocmin_io);

    inline CfMPI* cfmpi() const;

   private:
    int Nx_ = 0;      // number of X gridpoints and modes
    int Ny_ = 0;      // number of Y gridpoints and modes
    int Nz_ = 0;      // number of Z gridpoints
    int Nzpad_ = 0;   // 2*(Nz/2+1)
    int Nzpad2_ = 0;  // number of Z modes == Nz/2+1.
    int Nd_ = 0;

    lint Nloc_;      // total number of gridpoints on this process
    lint Nxloc_;     // number of x gridpoints on this proc in physical state
    lint nxlocmin_;  // first gridpoint local
    lint Nxlocmax_;  // max number over all processes (not really used)
    lint Mx_;        // same as Nx
    lint Mxloc_;     // number of x gridpoints on this proc in spectral state
    lint Mxlocmax_;  // max number of gridpoints on any process in spectral state
    lint mxlocmin_;  // first locally stored gridpoint
    lint Nyloc_;     // number of y gridpoints on this proc in physical state
    lint Nylocpad_;  // number of y gridpoints on this proc in physical state incl padding
    lint Nypad_;     // Ny incl padding
    lint nylocmin_;  // first locally stored gridpoint
    lint nylocmax_;  // nylocmin_ + Nyloc, can be used in loops as nylocmin <= ny < nylocmax
    lint Mz_;        // number of z gridpoints in spectral state
    lint Mzloc_;     // number of z gridpoints on this proc in spectral state
    lint Mzlocmax_;  // max number over all processes (not really used)
    lint mzlocmin_;  // first locally stored gridpoint

    Real Lx_ = 0;
    Real Lz_ = 0;
    Real a_ = 0;
    Real b_ = 0;

    // size parameters for dealiased I/O
    int Nx_io_;
    int Nz_io_;

    bool padded_ = false;  // flag, simplifies IO and norm calcs on dealiased fields

    // Manages the lifetime of FFTW data buffer
    std::unique_ptr<void, void (*)(void*)> data_handle_ = {nullptr, fftw_free};
    Real* rdata_ = nullptr;     // stored with indices in order d, Ny, Nx, Nz
    Complex* cdata_ = nullptr;  // Complex alias for rdata_ (cdata_ = (Complex*) rdata_).

#ifndef HAVE_MPI
    std::unique_ptr<Real, void (*)(void*)> scratch_ = {nullptr, fftw_free};
#endif

    fieldstate xzstate_ = Spectral;
    fieldstate ystate_ = Spectral;

    using fftw_plan_unique_ptr_t = std::unique_ptr<std::remove_pointer<fftw_plan>::type, void (*)(fftw_plan)>;

    fftw_plan_unique_ptr_t xz_plan_ = {nullptr, fftw_destroy_plan};
    fftw_plan_unique_ptr_t xz_iplan_ = {nullptr, fftw_destroy_plan};
    fftw_plan_unique_ptr_t y_plan_ = {nullptr, fftw_destroy_plan};

    // Plans for transposing in MPI
    fftw_plan_unique_ptr_t t_plan_ = {nullptr, fftw_destroy_plan};
    fftw_plan_unique_ptr_t t_iplan_ = {nullptr, fftw_destroy_plan};

    CfMPI* cfmpi_ = nullptr;

    inline int flatten(int nx, int ny, int nz, int i) const;
    inline int complex_flatten(int mx, int my, int mz, int i) const;
    inline int flatten(int nx, int ny, int nz, int i, int j) const;
    inline int complex_flatten(int mx, int my, int mz, int i, int j) const;
    void fftw_initialize(uint fftw_flags = FFTW_ESTIMATE);
};

FlowField operator*(const Real a, const FlowField& w);
FlowField operator+(const FlowField& v, const FlowField& w);
FlowField operator-(const FlowField& v, const FlowField& w);

void normalize(cfarray<FlowField>& e);
void orthogonalize(cfarray<FlowField>& e);
void orthonormalize(cfarray<FlowField>& e);

// Quadratic interpolation/expolate of FlowField as function of parameter mu.
// Input cfarrays are length 3: un[0],un[1],un[2] at values mun[0], mun[1], mun[2].
// At any gridpoint, if difference is less than eps, use un[0] instead of interpolating
FlowField quadraticInterpolate(cfarray<FlowField>& un, const cfarray<Real>& mun, Real mu, Real eps = 1e-13);
FlowField polynomialInterpolate(cfarray<FlowField>& un, cfarray<Real>& mun, Real mu);

void transmit_coeff(FlowField& g, int gmx, int gmz, const FlowField& f, int fmx, int fmz, int ny, int i,
                    int taskid = -1);

// To set g(gmx,ny,gmz,i) on one processor with a complex value from another processor
void transmit_coeff(FlowField& g, int gmx, int gmz, int ny, int i, const Complex& fval, int taskidf, int taskid = -1);

void hdf5addstuff(const std::string& filebase, Real nu, ChebyCoeff& Ubase, ChebyCoeff& Wbase);

// Vector-valued access methods
inline int FlowField::flatten(int nx, int ny, int nz, int i) const {
#ifdef HAVE_MPI
    assert(nx >= nxlocmin_ && nx < nxlocmin_ + Nxloc_);
    assert(ny >= nylocmin_ && ny < nylocmax_);
    assert(nz >= 0 && nz < Nzpad_);
    assert(i >= 0 && i < Nd_);
    // cfarray format is x* z y* i, * indicate distributed vars
    return i + Nd_ * ((ny - nylocmin_) + Nylocpad_ * (nz + Nzpad_ * (nx - nxlocmin_)));
#else
    assert(nx >= 0 && nx < Nx_);
    assert(ny >= 0 && ny < Ny_);
    assert(nz >= 0 && nz < Nzpad_);
    assert(i >= 0 && i < Nd_);
    return nz + Nzpad_ * (nx + Nx_ * (ny + Ny_ * i));
#endif
}

inline int FlowField::complex_flatten(int mx, int my, int mz, int i) const {
#ifdef HAVE_MPI
    assert(mx >= mxlocmin_ && mx < Mxloc_ + mxlocmin_);
    assert(mz >= mzlocmin_ && mz < Mzloc_ + mzlocmin_);
    assert(i >= 0 && i < Nd_);
    assert(my >= 0 && my < Ny_);
    // cfarray format is z* x* y i, * indicate distributed vars
    return i + Nd_ * (my + Nypad_ * ((mx - mxlocmin_) + Mxloc_ * (mz - mzlocmin_)));
#else
    assert(mx >= 0 && mx < Nx_);
    assert(my >= 0 && my < Ny_);
    assert(mz >= 0 && mz < Nzpad2_);
    assert(i >= 0 && i < Nd_);
    return (mz + Nzpad2_ * (mx + Nx_ * (my + Ny_ * i)));
#endif
}

inline const Real& FlowField::operator()(int nx, int ny, int nz, int i) const {
    assert(xzstate_ == Physical);
    return rdata_[flatten(nx, ny, nz, i)];
}
inline Real& FlowField::operator()(int nx, int ny, int nz, int i) {
    assert(xzstate_ == Physical);
    return rdata_[flatten(nx, ny, nz, i)];
}
inline Complex& FlowField::cmplx(int mx, int my, int mz, int i) {
    assert(xzstate_ == Spectral);
    return cdata_[complex_flatten(mx, my, mz, i)];
}
inline const Complex& FlowField::cmplx(int mx, int my, int mz, int i) const {
    assert(xzstate_ == Spectral);
    return cdata_[complex_flatten(mx, my, mz, i)];
}

// Tensor-valued access methods
inline int FlowField::flatten(int nx, int ny, int nz, int i, int j) const {
    assert(nx >= 0 && nx < Nx_);
    assert(ny >= 0 && ny < Ny_);
    assert(nz >= 0 && nz < Nzpad_);
    assert(i >= 0 && i < Nd_);
    assert(j >= 0 && j < Nd_);
    return nz + Nzpad_ * (nx + Nx_ * (ny + Ny_ * (i + Nd_ * j)));
}

inline int FlowField::complex_flatten(int mx, int my, int mz, int i, int j) const {
    assert(mx >= 0 && mx < Nx_);
    assert(my >= 0 && my < Ny_);
    assert(mz >= 0 && mz < Nzpad2_);
    assert(i >= 0 && i < Nd_);
    return (mz + Nzpad2_ * (mx + Nx_ * (my + Ny_ * (i + Nd_ * j))));
}

inline const Real& FlowField::operator()(int nx, int ny, int nz, int i, int j) const {
    assert(xzstate_ == Physical);
    return rdata_[flatten(nx, ny, nz, i, j)];
}
inline Real& FlowField::operator()(int nx, int ny, int nz, int i, int j) {
    assert(xzstate_ == Physical);
    return rdata_[flatten(nx, ny, nz, i, j)];
}
inline Complex& FlowField::cmplx(int mx, int my, int mz, int i, int j) {
    assert(xzstate_ == Spectral);
    return cdata_[complex_flatten(mx, my, mz, i, j)];
}
inline const Complex& FlowField::cmplx(int mx, int my, int mz, int i, int j) const {
    assert(xzstate_ == Spectral);
    return cdata_[complex_flatten(mx, my, mz, i, j)];
}

inline Real FlowField::Lx() const { return Lx_; }
inline Real FlowField::Ly() const { return b_ - a_; }
inline Real FlowField::Lz() const { return Lz_; }
inline Real FlowField::a() const { return a_; }
inline Real FlowField::b() const { return b_; }

inline int FlowField::kx(int mx) const {
    assert(mx >= 0 && mx < Nx_);
    return (mx <= Nx_ / 2) ? mx : mx - Nx_;
}

inline int FlowField::kz(int mz) const {
    assert(mz >= 0 && mz < Nzpad2_);
    return mz;
}

inline int FlowField::mx(int kx) const {
    assert(kx >= Nx_ / 2 + 1 - Nx_ && kx <= Nx_ / 2);
    return (kx >= 0) ? kx : kx + Nx_;
}

inline int FlowField::mz(int kz) const {
    assert(kz >= 0 && kz < Nzpad2_);
    return kz;
}

inline Real FlowField::x(int nx) const { return nx * Lx_ / Nx_; }
inline Real FlowField::y(int ny) const { return 0.5 * ((b_ + a_) + (b_ - a_) * cos(pi * ny / (Ny_ - 1))); }
inline Real FlowField::z(int nz) const { return nz * Lz_ / Nz_; }

inline int FlowField::numXmodes() const { return Nx_; }
inline int FlowField::numYmodes() const { return Ny_; }
inline int FlowField::numZmodes() const { return Nzpad2_; }  // Nzpad2 = Nz/2+1

inline int FlowField::numXgridpts() const { return Nx_; }
inline int FlowField::numYgridpts() const { return Ny_; }
inline int FlowField::numZgridpts() const { return Nz_; }

inline int FlowField::Nx() const { return Nx_; }
inline int FlowField::Ny() const { return Ny_; }
inline int FlowField::Nz() const { return Nz_; }

inline int FlowField::Mx() const { return Nx_; }
inline int FlowField::My() const { return Ny_; }
inline int FlowField::Mz() const { return Nzpad2_; }

inline lint FlowField::Nloc() const { return Nloc_; }
inline lint FlowField::Nxloc() const { return Nxloc_; }
inline lint FlowField::nxlocmin() const { return nxlocmin_; }
inline lint FlowField::Nxlocmax() const { return Nxlocmax_; }
inline lint FlowField::Mxloc() const { return Mxloc_; }
inline lint FlowField::mxlocmin() const { return mxlocmin_; }
inline lint FlowField::Nyloc() const { return Nyloc_; }
inline lint FlowField::Nylocpad() const { return Nylocpad_; }
inline lint FlowField::Nypad() const { return Nypad_; }
inline lint FlowField::nylocmin() const { return nylocmin_; }
inline lint FlowField::nylocmax() const { return nylocmax_; }
inline lint FlowField::Mzloc() const { return Mzloc_; }
inline lint FlowField::mzlocmin() const { return mzlocmin_; }

inline int FlowField::nproc0() const {
    if (cfmpi_ == NULL)
        return 1;
    return cfmpi_->nproc0();
}
inline int FlowField::nproc1() const {
    if (cfmpi_ == NULL)
        return 1;
    return cfmpi_->nproc1();
}
inline int FlowField::taskid() const {
    if (cfmpi_ == NULL)
        return 0;
    return cfmpi_->taskid();
}
inline int FlowField::key0() const {
    if (cfmpi_ == NULL)
        return 0;
    return cfmpi_->key0();
}
inline int FlowField::color0() const {
    if (cfmpi_ == NULL)
        return 0;
    return cfmpi_->color0();
}
inline int FlowField::taskid_world() const {
    if (cfmpi_ == NULL)
        return 0;
    return cfmpi_->taskid_world();
}
inline int FlowField::numtasks() const {
    if (cfmpi_ == NULL)
        return 1;
    return cfmpi_->numtasks();
}

inline int FlowField::task_coeff(int mx, int mz) const {
    assert(xzstate_ == Spectral);

    int res = mx / Mxlocmax_ * nproc1() + mz / Mzlocmax_;
    return res;
}
inline int FlowField::task_coeff(int mx, int my, int mz, int i) const { return task_coeff(mx, mz); }
inline int FlowField::task_coeffp(int nx, int ny) const {
    assert(xzstate_ == Physical);
    int res = ny / Nylocpad_ * nproc1() + nx / Nxlocmax_;
    return res;
}
#ifdef HAVE_MPI
inline MPI_Comm* FlowField::comm_world() const {
    if (cfmpi_ == NULL)
        return NULL;
    return &cfmpi_->comm_world;
}
#endif

inline CfMPI* FlowField::cfmpi() const { return cfmpi_; }

inline int FlowField::kxmax() const { return Nx_ / 2; }
inline int FlowField::kzmax() const { return Nz_ / 2; }
inline int FlowField::kxmin() const { return Nx_ / 2 + 1 - Nx_; }
inline int FlowField::kzmin() const { return 0; }
inline int FlowField::kxmaxDealiased() const { return Nx_ / 3 - 1; }  // CHQZ06 p139
inline int FlowField::kzmaxDealiased() const { return Nz_ / 3 - 1; }  // CHQZ06 p139
inline int FlowField::kxminDealiased() const { return -(Nx_ / 3 - 1); }
inline int FlowField::kzminDealiased() const { return 0; }
inline bool FlowField::isAliased(int kx, int kz) const {
    return (abs(kx) > kxmaxDealiased() || abs(kz) > kzmaxDealiased()) ? true : false;
}
inline int FlowField::vectorDim() const { return Nd_; }
inline int FlowField::Nd() const { return Nd_; }

inline fieldstate FlowField::xzstate() const { return xzstate_; }
inline fieldstate FlowField::ystate() const { return ystate_; }

// helper func for zeroing highest-order mode under odd differentiation
// See Trefethen Spectral Methods in Matlab pg 19.
inline int zero_last_mode(int k, int kmax, int n) { return ((k == kmax) && (n % 2 == 1)) ? 0 : 1; }

// functions for fast addition of FlowFields
inline void FlowField::add(const Real a, const FlowField& u) {
    assert(congruent(u));
    const auto* __restrict__ urdata_pnt = u.rdata_;
    auto* __restrict__ rdata_pnt = rdata_;

    for (int ii = 0; ii < Nloc_; ++ii) {
        rdata_pnt[ii] += a * urdata_pnt[ii];
    }
}

inline void FlowField::add(const Real a, const FlowField& u, const Real b, const FlowField& v) {
    assert(congruent(u));
    const auto* __restrict__ urdata_pnt = u.rdata_;
    const auto* __restrict__ vrdata_pnt = v.rdata_;
    auto* __restrict__ rdata_pnt = rdata_;

    for (int ii = 0; ii < Nloc_; ++ii) {
        rdata_pnt[ii] += a * urdata_pnt[ii] + b * vrdata_pnt[ii];
    }
}

// The field2vector and vector2field functions assume zero divergece and no-slip BCs.
int field2vector_size(const FlowField& u);
void field2vector(const FlowField& u, Eigen::VectorXd& v);
void vector2field(const Eigen::VectorXd& v, FlowField& u);
void fixdivnoslip(FlowField& u);

}  // namespace chflow
#endif
