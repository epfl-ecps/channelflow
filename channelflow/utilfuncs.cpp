/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "channelflow/utilfuncs.h"
#include <sys/types.h>
#include <unistd.h>
#include <ctime>
#include "cfbasics/brent.h"
#include "channelflow/periodicfunc.h"

using namespace std;
using namespace Eigen;

namespace chflow {

/**
 * Write date, hostname, pid and cwd to file.
 * @param filename filename
 * @param mode ios::openmode
 */
void WriteProcessInfo(int argc, char* argv[], string filename, ios::openmode mode) {
    if (mpirank() == 0) {
        ofstream fout(filename.c_str(), mode);

        // Save command-line
        fout << "Command:  ";
        for (int n = 0; n < argc; ++n)
            fout << argv[n] << ' ';
        fout << endl;

        // Save current path
        fout << "PWD:      ";
        char currentPath[1024];
        if (getcwd(currentPath, 1023) == NULL)
            cferror("Error in getcwd())");
        fout << currentPath << endl;

        // Save host and pid
        fout << "Host:     ";
        char hostname[1024];
        gethostname(hostname, 1023);
        fout << hostname << ", PID: " << getpid() << endl;

        // Save current time
        time_t rawtime;
        struct tm* timeinfo;
        char buffer[80];
        time(&rawtime);
        timeinfo = localtime(&rawtime);
        strftime(buffer, 80, "%Y-%m-%d %I:%M:%S", timeinfo);
        fout << "Time:     " << buffer << endl;

        // Save code revision and compiler version
        fout << "Version:  " << CHANNELFLOW_VERSION << endl;
        fout << "Git rev:  " << g_GIT_SHA1 << endl;
        fout << "Compiler: " << COMPILER_VERSION << endl;

        // Add two newlines to separate entries if there are things appended to this file
        fout << endl << endl;
    }
}

FieldSeries::FieldSeries() : t_(0), f_(0), emptiness_(0) {}

FieldSeries::FieldSeries(int N) : t_(N), f_(N), emptiness_(N) {}

// Still working on first draft of these functions

void FieldSeries::push(const FlowField& f, Real t) {
    for (int n = f_.N() - 1; n > 0; --n) {
        if (f_[n].congruent(f_[n - 1]))
            swap(f_[n], f_[n - 1]);
        else
            f_[n] = f_[n - 1];
        t_[n] = t_[n - 1];
    }
    if (f_.N() > 0) {
        f_[0] = f;
        t_[0] = t;
    }
    --emptiness_;
}

bool FieldSeries::full() const { return (emptiness_ == 0) ? true : false; }

void FieldSeries::interpolate(FlowField& f, Real t) const {
    if (!(this->full()))
        cferror(
            "FieldSeries::interpolate(Real t, FlowField& f) : FieldSeries is not completely initialized. Take more "
            "time steps before interpolating");

    for (int n = 0; n < f_.N(); ++n) {
        assert(f_[0].congruent(f_[n]));
        ;
    }

    if (f_[0].xzstate() == Spectral) {
        //         lint Mx = f_[0].Mx();
        //         lint Mz = f_[0].Mz();
        lint mxlocmin = f_[0].mxlocmin();
        lint mxlocmax = mxlocmin + f_[0].Mxloc();
        lint mzlocmin = f_[0].mzlocmin();
        lint mzlocmax = mzlocmin + f_[0].Mzloc();

        lint Ny = f_[0].Ny();
        lint Nd = f_[0].Nd();
        int N = f_.N();
        cfarray<Real> fn(N);
        for (lint i = 0; i < Nd; ++i)
            for (lint ny = 0; ny < Ny; ++ny)
                for (lint mx = mxlocmin; mx < mxlocmax; ++mx)
                    for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                        for (int n = 0; n < N; ++n)
                            fn[n] = Re(f_[n].cmplx(mx, ny, mz, i));
                        Real a = polyInterp(fn, t_, t);

                        for (int n = 0; n < N; ++n)
                            fn[n] = Im(f_[n].cmplx(mx, ny, mz, i));
                        Real b = polyInterp(fn, t_, t);

                        f.cmplx(mx, ny, mz, i) = Complex(a, b);
                    }
    } else {
        //         int Nx = f_[0].Nx();
        //         int Ny = f_[0].Ny();
        lint nxlocmin = f_[0].nxlocmin();
        lint nxlocmax = f_[0].Nxloc() + nxlocmin;
        lint nylocmin = f_[0].nylocmin();
        lint nylocmax = f_[0].nylocmax();
        lint Nz = f_[0].Nz();
        lint Nd = f_[0].Nd();
        int N = f_.N();
        cfarray<Real> fn(N);
        for (lint i = 0; i < Nd; ++i)
            for (lint ny = nylocmin; ny < nylocmax; ++ny)
                for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
                    for (lint nz = 0; nz < Nz; ++nz) {
                        for (int n = 0; n < N; ++n)
                            fn[n] = f_[n](nx, ny, nz, i);
                        f(nx, ny, nz, i) = polyInterp(fn, t_, t);
                    }
    }
}

void save(const std::string& filebase, Real T, const FieldSymmetry& tau) {
    if (mpirank() == 0) {
        string filename = appendSuffix(filebase, ".asc");
        ofstream os(filename.c_str());
        const int p = 16;     // digits of precision
        const int w = p + 7;  // width of field
        os << setprecision(p) << scientific;
        os << setw(w) << T << " %T\n";
        os << setw(w) << tau.s() << " %s\n";
        os << setw(w) << tau.sx() << " %sx\n";
        os << setw(w) << tau.sy() << " %sy\n";
        os << setw(w) << tau.sz() << " %sz\n";
        os << setw(w) << tau.ax() << " %ax\n";
        os << setw(w) << tau.az() << " %az\n";
    }
}

void load(const std::string& filebase, Real& T, FieldSymmetry& tau) {
    auto su = 0;
    auto sx = 0;
    auto sy = 0;
    auto sz = 0;

    auto ax = Real(0.0);
    auto az = Real(0.0);

    bool error = false;
    string filename = appendSuffix(filebase, ".asc");

    if (mpirank() == 0) {
        ifstream is(filename.c_str());
        string junk;
        is >> T >> junk;
        if (junk != "%T")
            error = true;
        is >> su >> junk;
        if (junk != "%s" || abs(su) != 1)
            error = true;
        is >> sx >> junk;
        if (junk != "%sx" || abs(sx) != 1)
            error = true;
        is >> sy >> junk;
        if (junk != "%sy" || abs(sy) != 1)
            error = true;
        is >> sz >> junk;
        if (junk != "%sz" || abs(sz) != 1)
            error = true;
        is >> ax >> junk;
        if (junk != "%ax" || abs(ax) > 0.5)
            error = true;
        is >> az >> junk;
        if (junk != "%az" || abs(az) > 0.5)
            error = true;
    }
#ifdef HAVE_MPI
    MPI_Bcast(&ax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&az, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sy, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&su, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    if (error) {
        cferror("Error in loading T,su,sx,sy,sz,ax,az from file " + filename + "\nExiting.");
    }
    tau = FieldSymmetry(sx, sy, sz, ax, az, su);
}

void plotfield(const FlowField& u_, const string& outdir, const string& label, int xstride, int ystride, int zstride,
               int nx, int ny, int nz) {
    string preface = outdir;
    if (label.length() > 0)
        preface += label + "_";

    mkdir(outdir);

    FlowField& u = (FlowField&)u_;
    fieldstate xs = u.xzstate();
    fieldstate ys = u.ystate();

    if (ny < 0 || ny >= u.Ny())
        ny = (u.Ny() - 1) / 2;

    // Vector x = periodicpoints(Nx, u.Lx());
    // Vector y = chebypoints(Ny, u.a(), u.b());
    // Vector z = periodicpoints(Nz, u.Lz());
    // x.save(outdir + "x");
    // y.save(outdir + "y");
    // z.save(outdir + "z");

    u.makePhysical();

    int Ndim = u.Nd();
    switch (Ndim) {
        case 3:
            u.saveSlice(0, 2, nx, preface + "w_yz");
            u.saveSlice(1, 2, ny, preface + "w_xz");
            u.saveSlice(2, 2, nz, preface + "w_xy");
            // fall through to case 2
        case 2:
            u.saveSlice(0, 0, nx, preface + "u_yz");
            u.saveSlice(0, 1, nx, preface + "v_yz");
            u.saveSlice(1, 0, ny, preface + "u_xz");
            u.saveSlice(1, 1, ny, preface + "v_xz");
            u.saveSlice(2, 0, nz, preface + "u_xy");
            u.saveSlice(2, 1, nz, preface + "v_xy");
            break;
        case 1:
            u.saveSlice(0, 0, nx, preface + "yz");
            u.saveSlice(1, 0, ny, preface + "xz");
            u.saveSlice(2, 0, nz, preface + "xy");
            break;
        case 0:
            break;
        default:
            string preface2 = preface + "_";
            for (int i = 0; i < Ndim; ++i) {
                u.saveSlice(0, i, 0, preface2 + i2s(i) + "_yz");
                u.saveSlice(1, i, ny, preface2 + i2s(i) + "_xz");
                u.saveSlice(2, i, 0, preface2 + i2s(i) + "_xy");
            }
    }
    u.makeSpectral();

    // FlowField vort = curl(u);
    // vort.makePhysical();
    // vort.saveSlice(0, 0, 0, preface + "vort_yz");

    // FlowField drag = ydiff(u);
    // drag.makePhysical();
    // drag.saveSlice(1, 0, Ny-1, preface + "drag_xz");

    // ChebyCoeff U = Re(u.profile(0,0,0));
    // U.makePhysical();
    // U.save("U");

    u.makeState(xs, ys);
}

// Produce plots of various slices, etc.
void plotxavg(const FlowField& u_, const std::string& outdir, const std::string& label) {
    FlowField& u = (FlowField&)u_;
    fieldstate xs = u.xzstate();
    fieldstate ys = u.ystate();
    u.makeSpectral();

    FlowField uxavg(4, u.Ny(), u.Nz(), u.Nd(), u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi(), Spectral, Spectral);
    for (int i = 0; i < u.Nd(); ++i)
        for (int my = 0; my < u.My(); ++my)
            for (int mz = 0; mz < u.Mz(); ++mz)
                uxavg.cmplx(0, my, mz, i) = u.cmplx(0, my, mz, i);

    mkdir(outdir);
    string preface = outdir;
    if (label.length() > 0)
        preface += label + "_";

    uxavg.makePhysical();
    uxavg.saveSlice(0, 0, 0, preface + "u_yz_xavg");
    uxavg.saveSlice(0, 1, 0, preface + "v_yz_xavg");
    uxavg.saveSlice(0, 2, 0, preface + "w_yz_xavg");

    uxavg.makeSpectral();
    FlowField uxyavg(4, 1, u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi(), Spectral, Spectral);
    for (int mz = 0; mz < u.Mz(); ++mz)
        uxyavg.cmplx(0, 0, mz, 0) = uxavg.cmplx(0, 0, mz, 0);

    ofstream os((preface + "u_xyavg.asc").c_str());
    uxyavg.makePhysical();
    int Nz = u.Nz();
    for (int nz = 0; nz <= Nz; ++nz)
        os << uxyavg(0, 0, nz % Nz, 0) << '\n';

    u.makeState(xs, ys);
}

void plotspectra(const FlowField& u_, const string& outdir, const string& label, bool showpadding) {
    FlowField& u = (FlowField&)u_;
    fieldstate xs = u.xzstate();
    fieldstate ys = u.ystate();

    string preface = outdir;
    if (label.length() > 0)
        preface += label + "_";

    mkdir(outdir);
    u.makeSpectral();
    u.saveSpectrum(preface + "xzspec", true, showpadding);

    u.makePhysical_y();
    ofstream ucs((preface + "yspec.asc").c_str());

    ucs << "% sum of abs vals of Cheby coeffs of components u_{kx,kz}(y) " << endl;

    ChebyTransform trans(u.Ny());

    const int kxmax = lesser(u.kxmax(), 8);
    const int kzmax = lesser(u.kzmax(), 8);
    const int dk = 1;

    for (int kx = 0; kx <= kxmax; kx += dk) {
        // for (int kz=0; kz<=kzmax; kz += dk) {
        int kz = lesser(kx, kzmax);
        BasisFunc prof = u.profile(u.mx(kx), u.mz(kz));
        prof.makeSpectral(trans);
        ucs << "% kx,kz == " << kx << ", " << kz << endl;
        for (int ny = 0; ny < prof.Ny(); ++ny) {
            Real sum = 0.0;
            for (int i = 0; i < prof.Nd(); ++i)
                sum += abs(prof[i][ny]);
            ucs << sum << ' ';
        }
        ucs << '\n';
    }

    u.makeState(xs, ys);
}

/*****************
TimeStepParams::TimeStepParams(Real dt_, Real dtmin_, Real dtmax_,
           Real CFLmin_, Real CFLmax_, bool variable_)
:
dt(dt_),
dtmin(dtmin_),
dtmax(dtmax_),
CFLmin(CFLmin_),
CFLmax(CFLmax_),
variable(variable_)
{}


void params2timestep(const TimeStepParams& p, Real T, int& N, TimeStep& dt) {

// Set CFL adjustment time dT to an integer division of T near 1
N = Greater(iround(T), 1);
const Real dT = T/N;
if (p.variable)
  dt = TimeStep(p.dt, p.dtmin, p.dtmax, dT, p.CFLmin, p.CFLmax);
else
  dt = TimeStep(p.dt, p.dt, p.dt, dT, p.CFLmin, p.CFLmax);
}
******************/

Real PuFraction(const FlowField& u, const FieldSymmetry& s, int sign) {
    FlowField Pu = s(u);
    if (sign == 1)
        Pu += u;
    else
        Pu -= u;
    return 0.25 * L2Norm2(Pu) / L2Norm2(u);  // 0.25 is square of 1/2 from Pu = 1/2 (u+s(u))
}

FieldSymmetry xfixphasehack(FlowField& u, Real axguess, int i, parity p, string mode) {
    if (mode == "cm") {  // use center of mass

        u.makePhysical();

        // Calculate center of mass (over periodic bcs)
        // See http://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions

        Real cxm = 0, cym = 0;  // Averaged circle coordinates of cm
        for (int nx = u.nxlocmin(); nx < u.nxlocmin() + u.Nxloc(); nx++) {
            Real Lx = u.Lx();
            Real theta = u.x(nx) / Lx * 2 * pi;  // map x coordinate to angle
            Real cx = cos(theta);                // x-circle-coord for angle theta
            Real cy = sin(theta);                // y-circle-coord for angle theta

            // Calculate weight by averaging y,z and dimension
            Real mi = 0;
            for (int nz = 0; nz < u.Nz(); nz++) {
                for (int ny = u.nylocmin(); ny < u.nylocmax(); ny++) {
                    for (int i = 0; i < u.Nd(); i++) {
                        mi += pow(u(nx, ny, nz, i), 2);
                    }
                }
            }

            cxm += cx * mi;
            cym += cy * mi;
        }

#ifdef HAVE_MPI
        Real tmp = cxm;
        MPI_Allreduce(&tmp, &cxm, 1, MPI_DOUBLE, MPI_SUM, u.cfmpi()->comm_world);
        tmp = cym;
        MPI_Allreduce(&tmp, &cym, 1, MPI_DOUBLE, MPI_SUM, u.cfmpi()->comm_world);
#endif

        Real r = sqrt(cxm * cxm + cym * cym);
        cxm /= r;
        cym /= r;

        Real cmx = atan2(cym, cxm) * u.Lx() / 2 / pi;
        if (cmx < 0)
            cmx += u.Lx();

        u.makeSpectral();
        FieldSymmetry phaseshift(1, 1, 1, cmx / u.Lx(), 0.0);
        return phaseshift;
    }

    const int Ny2 = (u.Ny() - 1) / 2;

    // move solution from centered about box center to center about origin
    FieldSymmetry halfshift(1, 1, 1, 0.5, 0.5);
    u *= halfshift;
    u.makePhysical();

    std::cout << "arglar  " << u.taskid() << std::endl;
    PeriodicFunc f(u.Nx(), u.Lx(), Physical);
    for (int nx = 0; nx < u.Nx(); ++nx) {
        Real tmp = 0;
#ifdef HAVE_MPI
        if (u.task_coeffp(nx, Ny2) == u.taskid()) {
            tmp = u(nx, Ny2, 0, i);
        }
        MPI_Bcast(&tmp, 1, MPI_DOUBLE, u.task_coeffp(nx, Ny2), *u.comm_world());
#else
        tmp = u(nx, Ny2, 0, i);
#endif

        f(nx) = tmp;
    }

    u.makeSpectral();
    u *= halfshift;

    f.makeSpectral();
    if (p == Odd)
        f.cmplx(0) = Complex(0.0, 0.0);  // set mean to zero
    else
        f = diff(f);  // even parity means find zero of df/dz

    Real xcenter = newtonSearch(f, axguess * u.Lx());
    FieldSymmetry phaseshift(1, 1, 1, xcenter / u.Lx(), 0.0);

    return phaseshift;
}

// z-shift until f'(z) == Lz/2 for f(z) = <u>_{xy}(z)
FieldSymmetry zfixphasehack(FlowField& u, Real azguess, int i, parity p, string mode) {
    // move solution from centered about z=Lz/2 to center about z=0
#ifdef HAVE_MPI
    const int Mx = 0;
#endif
    FieldSymmetry halfshift(1, 1, 1, 0.0, 0.5);
    u.makeSpectral();
    u *= halfshift;

    PeriodicFunc f(u.Nz(), u.Lz(), Spectral);
    for (int mz = 0; mz < u.Mz(); ++mz) {
        Complex tmp = 0;
#ifdef HAVE_MPI
        if (u.task_coeff(Mx, mz) == u.taskid()) {
            tmp = u.cmplx(0, 0, mz, i);
        }
        MPI_Bcast(&tmp, 1, MPI_DOUBLE_COMPLEX, u.task_coeff(Mx, mz), *u.comm_world());
#else
        tmp = u.cmplx(0, 0, mz, i);
#endif

        f.cmplx(mz) = tmp;  // should do zshift here
    }
    u *= halfshift;  // shift u back to original phase
    if (p == Odd)
        f.cmplx(0) = Complex(0.0, 0.0);  // set mean to zero
    else
        f = diff(f);  // even parity means find zero of df/dz

    Real zcenter = newtonSearch(f, azguess * u.Lz());
    FieldSymmetry phaseshift(1, 1, 1, 0.0, zcenter / u.Lz());

    return phaseshift;
}

void fixuUbasehack(FlowField& u, ChebyCoeff U) {
    // Construct Udiff = 1 - (y/h)^2
    ChebyCoeff u00 = Re(u.profile(0, 0, 0));
    ChebyCoeff u00y = diff(u00);
    Real ubulk = u00.mean();
    Real uya = u00y.eval_a();
    Real uyb = u00y.eval_b();

    ChebyCoeff Uy = diff(U);
    Real Ubulk = U.mean();
    Real Uya = Uy.eval_a();
    Real Uyb = Uy.eval_b();

    ChebyCoeff utot = u00;
    utot += U;
    ChebyCoeff utoty = diff(utot);
    Real utotbulk = utot.mean();
    Real utotya = utoty.eval_a();
    Real utotyb = utoty.eval_b();

    cout << "fixUbasehack input: " << endl;
    cout << "    ubulk == " << ubulk << endl;
    cout << "    Ubulk == " << Ubulk << endl;
    cout << " utotbulk == " << utotbulk << endl;
    cout << "  uya-uyb == " << uya - uyb << endl;
    cout << "  Uya-Uya == " << Uyb - Uya << endl;
    cout << "utya-utyb == " << utotyb - utotya << endl;

    // Construct Udiff = c*(1 - (y/h)^2) where c is set so that Uffyb-Udiffya == -(uyb-uya)
    ChebyCoeff Udiff(u.Ny(), u.a(), u.b(), Spectral);
    Udiff[0] = 0.5;
    Udiff[2] = -0.5;
    ChebyCoeff Udiffy = diff(Udiff);
    Udiff *= -(uyb - uya) / (Udiffy.eval_b() - Udiffy.eval_a());

    // Move Udiff from u to U
    u += Udiff;
    U -= Udiff;

    u00 = Re(u.profile(0, 0, 0));
    u00y = diff(u00);
    ubulk = u00.mean();
    uya = u00y.eval_a();
    uyb = u00y.eval_b();

    Uy = diff(U);
    Ubulk = U.mean();
    Uya = Uy.eval_a();
    Uyb = Uy.eval_b();

    utot = u00;
    utot += U;
    utoty = diff(utot);
    utotbulk = utot.mean();
    utotya = utoty.eval_a();
    utotyb = utoty.eval_b();

    cout << "fixUbasehack output: " << endl;
    cout << "    ubulk == " << ubulk << endl;
    cout << "    Ubulk == " << Ubulk << endl;
    cout << " utotbulk == " << utotbulk << endl;
    cout << "  uya-uyb == " << uya - uyb << endl;
    cout << "  Uya-Uya == " << Uyb - Uya << endl;
    cout << "utya-utyb == " << utotyb - utotya << endl;
}

// Return a translation that maximizes the s-symmetry of u.
// (i.e. return tau // that minimizes L2Norm(s(tau(u)) - tau(u))).
FieldSymmetry optimizePhase(const FlowField& u, const FieldSymmetry& s, int Nsteps, Real residual, Real damp,
                            bool verbose, Real x0, Real z0) {
    FieldSymmetry tau;
    if (s.sx() == 1 && s.sz() == 1)
        ;
    else if ((s.sx() == -1) ^ (s.sz() == -1)) {
        // Newton search for x-translation that maximizes shift-rotate symmetry
        if (verbose && u.taskid() == 0)
            cout << "\noptimizePhase: Newton search on translation" << endl;

        int i = (s.sx() == -1) ? 0 : 2;  // are we optimizing on x or z?
        Real L = (s.sx() == -1) ? u.Lx() : u.Lz();
        cfarray<Real> x(3);
        x[0] = x0;
        x[2] = z0;

        FlowField tu, stu, tux, stux, tuxx, stuxx;

        for (int n = 0; n < Nsteps; ++n) {
            tau = FieldSymmetry(x[0] / u.Lx(), x[2] / u.Lz());

            tu = tau(u);
            stu = s(tu);
            tux = diff(tu, i, 1);
            stux = diff(stu, i, 1);
            tuxx = diff(tux, i, 1);
            stuxx = diff(stux, i, 1);

            // g(x) is df/dx where f == (L2Norm(stu - tu)) and t = tau(x/Lx,0)
            Real g = L2IP(stux, tu) - L2IP(stu, tux);

            // Note: I think this could simplify via (stuxx, tu) = (stu,tuxx)
            Real dgdx = 2 * L2IP(stux, tux) - L2IP(stuxx, tu) - L2IP(stu, tuxx);

            if (verbose) {
                Real f = L2Dist(stu, tu);
                cout << endl;
                cout << "Newton step n  == " << n << endl;
                cout << "tau.ax         == " << tau.ax() << endl;
                cout << "tau.az         == " << tau.az() << endl;
                cout << "zeroable g     == " << g << endl;
                cout << "L2Norm(stu-tu) == " << f << endl;
            }
            if (abs(g) < residual && verbose) {
                cout << "\noptimizePhase: Newton search converged." << endl;
                break;
            }

            Real dx = -g / dgdx;

            if (abs(damp * dx) / L > 0.5) {
                cout << "\nerror in optimizePhase(FlowField, FieldSymmetry) :" << endl;
                cout << "Newton search went beyond bounds. " << endl;
                cout << "     g == " << g << endl;
                cout << " dg/dx == " << dgdx << endl;
                cout << "    dx == " << damp * dx << endl;
                cout << "  L/2  == " << 0.5 * L << endl;
                cout << "Jostle by L/10 and try again." << endl;
                x[i] += L / 10;
            } else
                x[i] += damp * dx;
        }
    } else {
        cout << "\nerror in optimizePhase(FlowField, FieldSymmetry) :" << endl;
        cout << "2d symmetrization is not yet implemented. Returning identity." << endl;
    }

    if (verbose) {
        FlowField tu = tau(u);
        FlowField stu = s(tu);
        cout << "\noptimal translation == " << tau << endl;
        cout << "Post-translation symmetry error == " << L2Dist(stu, tu) << endl;
    }
    return tau;
}

Real optPhaseShiftx(const FlowField& u0, const FlowField& u1, Real amin, Real amax, int nSampling, Real tolerance) {
    if (amax <= amin)
        cferror("optPhaseShiftx(): a0 must be < a1");
    if (nSampling < 3)
        nSampling = 3;
    Real astep = (amax - amin) / (nSampling - 1);

    std::function<Real(Real)> f = [&](Real ax) {
        FieldSymmetry sigma = FieldSymmetry(1, 1, 1, ax, 0);
        FlowField v = sigma * u1;
        return L2Dist(v, u0);
    };

    VectorXd fa(nSampling);  // f(ax_i)
    int imin = 0;
    Real fmin = 0;
    for (int i = 0; i < nSampling; ++i) {
        fa(i) = f(amin + i * astep);
        if (i == 0 || fa(i) < fmin) {
            fmin = fa(i);
            imin = i;
        }
    }

    if (amax - amin < 1 - 1e-12) {
        // We have not sampled the whole domain, so check if our minimum is not at the boundaries
        if (imin == 0 || imin == nSampling - 1)
            cferror("optPhaseShiftx(): no minimum is found and interval is not periodic. Increase nSampling?");
    }

    Real a1 = amin + imin * astep;
    Real a0 = a1 - astep;
    Real a2 = a1 + astep;

    Real f1 = fa(imin);
    Real f0 = (imin > 0) ? fa(imin - 1) : fa(nSampling - 2);
    Real f2 = (imin < nSampling - 1) ? fa(imin + 1) : fa(1);

    if (fabs(f1 - f2) < tolerance || fabs(f1 - f0) < tolerance)
        return 0;

    Brent brent(f, a1, f1, a0, f0, a2, f2);
    return brent.minimize(100, tolerance, 0);
}

void fixDiri(ChebyCoeff& f) {
    Real fa = f.eval_a();
    Real fb = f.eval_b();
    Real mean = 0.5 * (fb + fa);
    Real slop = 0.5 * (fb - fa);
    f[0] -= mean;
    f[1] -= slop;
}

void fixDiriMean(ChebyCoeff& f) {
    Real fa = f.eval_a();
    Real fb = f.eval_b();
    Real fm = f.mean();
    f[0] -= 0.125 * (fa + fb) + 0.75 * fm;
    f[1] -= 0.5 * (fb - fa);
    f[2] -= 0.375 * (fa + fb) - 0.75 * fm;
}

void fixDiriNeum(ChebyCoeff& f) {
    Real ya = f.a();
    Real yb = f.b();
    f.setBounds(-1, 1);
    Real a = f.eval_a();
    Real b = f.eval_b();
    Real c = f.slope_a();
    Real d = f.slope_b();

    // The numercial coeffs are inverse of the matrix (values found with Maple)
    // T0(-1)  T1(-1)  T2(-1)  T3(-1)     s0      a
    // T0(1)   T1(1)   T2(1)   T3(1)      s1      b
    // T0'(-1) T1'(-1) T2'(-1) T3'(-1)    s2  ==  c
    // T0'(1)  T1'(1)  T2'(1)  T3'(1)     s3      d

    Real s0 = 0.5 * (a + b) + 0.125 * (c - d);
    Real s1 = 0.5625 * (b - a) - 0.0625 * (c + d);
    Real s2 = 0.125 * (d - c);
    Real s3 = 0.0625 * (a - b + c + d);

    f[0] -= s0;
    f[1] -= s1;
    f[2] -= s2;
    f[3] -= s3;
    f.setBounds(ya, yb);
}

void fixDiri(ComplexChebyCoeff& f) {
    fixDiri(f.re);
    fixDiri(f.im);
}
void fixDiriMean(ComplexChebyCoeff& f) {
    fixDiriMean(f.re);
    fixDiriMean(f.im);
}

void fixDiriNeum(ComplexChebyCoeff& f) {
    fixDiriNeum(f.re);
    fixDiriNeum(f.im);
}

Real tFromFilename(const string filename) {
    // The expected filename format is ..../ {letters} {time} {letters}

    string f = filename.substr(filename.rfind("/") + 1);
    int len = f.size();

    // find the position of figures or dot in the string
    const char* s = f.c_str();

    int i0 = 0;
    auto isnumber = [](char c) { return ((int)c >= 48 && (int)c <= 57) || (int)c == 46; };
    while (i0 < len && !isnumber(s[i0]))
        i0++;
    int i1 = i0;
    while (i1 < len && isnumber(s[i1]))
        i1++;
    while (i1 > i0 && (int)s[i1 - 1] == 46)
        i1--;

    if (i1 == i0)  // The filename doesn't contain any numbers
        return 0;
    else {
        // convert to Real
        f = f.substr(i0, i1 - i0);
        stringstream ss;
        ss << f;
        Real res;
        ss >> res;
        return res;
    }
}

bool comparetimes(const string& s0, const string& s1) {
    int t0 = tFromFilename(s0);
    int t1 = tFromFilename(s1);
    return t0 < t1;
}

// Extract the version numbers from the PACKAGE_STRING provided by autoconf
void channelflowVersion(int& major, int& minor, int& update) {
    major = 2;
    minor = 0;  // This function does not work with cmake....
    update = 9;
    //   const char delimiters[] = ".";
    //    char* cp = strdup(CF_PACKAGE_VERSION);  // Make writable copy of version str
    //    assert(cp != NULL);
    //    major  = atoi(strtok(cp, delimiters));
    //    minor  = atoi(strtok(NULL, delimiters));
    //    update = atoi(strtok(NULL, delimiters));
    //    free(cp);
}

DNSFlags setBaseFlowFlags(ArgList& args, string& Uname, string& Wname) {
    // base flow options
    args.section("Base flow options");
    const string bf =
        args.getstr("-bf", "--baseflow", "laminar", "set base flow to one of [zero|laminar|linear|parabolic|suction]");
    Uname = args.getstr("-ub", "--Ubase", "",
                        "input baseflow file of arbitrary U-baseflow (takes precedence over -bf option)");
    Wname = args.getstr("-wb", "--Wbase", "",
                        "input baseflow file of arbitrary W-baseflow (takes precedence over -bf option)");
    const Real Reynolds = args.getreal("-R", "--Reynolds", 400, "pseudo-Reynolds number == 1/nu");
    const Real nuarg =
        args.getreal("-nu", "--nu", 0, "kinematic viscosity (takes precedence over Reynolds, if nonzero)");
    const string meanstr_ =
        args.getstr("-mc", "--meanconstraint", "gradp", "fix one of two flow constraints [gradp|bulkv]");
    const Real dPds_ =
        args.getreal("-dPds", "--dPds", 0.0, "magnitude of imposed pressure gradient along streamwise s");
    const Real Ubulk_ = args.getreal("-Ubulk", "--Ubulk", 0.0, "magnitude of imposed bulk velocity");
    const Real Uwall_ =
        args.getreal("-Uwall", "--Uwall", 1.0, "magnitude of imposed wall velocity, +/-Uwall at y = +/-h");
    const Real theta_ = args.getreal("-theta", "--theta", 0.0, "angle of base flow relative to x-axis");
    const Real Vsuck_ = args.getreal("-Vs", "--Vsuck", 0.0, "wall-normal suction velocity");

    DNSFlags flags;
    flags.baseflow = s2baseflow(bf);
    flags.nu = (nuarg != 0) ? nuarg : 1.0 / Reynolds;
    flags.constraint = s2constraint(meanstr_);
    flags.theta = theta_;
    flags.Uwall = Uwall_;
    flags.ulowerwall = -Uwall_ * cos(theta_);
    flags.uupperwall = Uwall_ * cos(theta_);
    flags.wlowerwall = -Uwall_ * sin(theta_);
    flags.wupperwall = Uwall_ * sin(theta_);
    flags.Vsuck = Vsuck_;
    flags.dPdx = dPds_ * cos(theta_);
    flags.dPdz = dPds_ * sin(theta_);
    flags.Ubulk = Ubulk_ * cos(theta_);
    flags.Wbulk = Ubulk_ * sin(theta_);

    return flags;
}

vector<ChebyCoeff> baseFlow(int Ny, Real a, Real b, DNSFlags& flags, string Uname, string Wname) {
    ChebyCoeff Ubase = ChebyCoeff(Ny, a, b, Spectral);
    ChebyCoeff Wbase = ChebyCoeff(Ny, a, b, Spectral);

    if (Uname != "" || Wname != "")
        flags.baseflow = ArbitraryBase;

    if (flags.baseflow == ZeroBase) {
        cout << "Baseflow: zero" << endl;
    } else if (flags.baseflow == LinearBase) {
        cout << "Baseflow: linear" << endl;
        Ubase[1] = 1;
    } else if (flags.baseflow == ParabolicBase) {
        cout << "Baseflow: parabolic" << endl;
        Ubase[0] = 0.5;
        Ubase[2] = -0.5;
    } else if (flags.baseflow == SuctionBase) {
        cout << "Baseflow: suction" << endl;
        Ubase = laminarProfile(flags.nu, PressureGradient, 0, flags.Ubulk, flags.Vsuck, a, b, -0.5, 0.5, Ny);
    } else if (flags.baseflow == LaminarBase) {
        cout << "Baseflow: laminar" << endl;
        Ubase = laminarProfile(flags.nu, flags.constraint, flags.dPdx, flags.Ubulk, flags.Vsuck, a, b, flags.ulowerwall,
                               flags.uupperwall, Ny);

        Wbase = laminarProfile(flags.nu, flags.constraint, flags.dPdz, flags.Wbulk, flags.Vsuck, a, b, flags.wlowerwall,
                               flags.wupperwall, Ny);
    } else if (flags.baseflow == ArbitraryBase) {
        cout << "Baseflow: reading from file" << endl;
        if (Uname != "") {
            Ubase = ChebyCoeff(Uname);
        }
        if (Wname != "") {
            Wbase = ChebyCoeff(Wname);
        }
    } else {
        cferror("Unknown base flow !!!");
    }

    vector<ChebyCoeff> baseflow = {Ubase, Wbase};
    return baseflow;
}

}  // namespace chflow
