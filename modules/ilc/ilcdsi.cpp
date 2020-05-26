/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 *
 * Original author: Florian Reetz
 */

#include "modules/ilc/ilcdsi.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "cfbasics/mathdefs.h"
#include "channelflow/diffops.h"
// #include "viscoelastic/veutils.h"
#include "modules/ilc/ilc.h"

using namespace std;

namespace chflow {

/*utility functions*/

std::vector<Real> ilcstats(const FlowField& u, const FlowField& temp, const ILCFlags flags) {
    double l2n = L2Norm(u);
    if (std::isnan(l2n)) {
        cferror("L2Norm(u) is nan");
    }

    FlowField u_tot = totalVelocity(u, flags);
    FlowField temp_tot = totalTemperature(temp, flags);

    std::vector<Real> stats;
    stats.push_back(L2Norm(u));
    stats.push_back(heatcontent(temp_tot, flags));
    stats.push_back(L2Norm(u_tot));
    stats.push_back(L2Norm(temp_tot));
    stats.push_back(L2Norm3d(u));
    stats.push_back(Ecf(u));
    stats.push_back(getUbulk(u));
    stats.push_back(getWbulk(u));
    stats.push_back(wallshear(u_tot));
    stats.push_back(buoyPowerInput(u_tot, temp_tot, flags));
    stats.push_back(dissipation(u_tot, flags));
    stats.push_back(heatinflux(temp_tot, flags));
    stats.push_back(L2Norm(temp));
    stats.push_back(Nusselt_plane(u_tot, temp_tot, flags));
    return stats;
}

string ilcfieldstatsheader(const ILCFlags flags) {
    stringstream header;
    header << setw(14) << "L2(u')" << setw(10) << "<T>(y=" << flags.ystats << ")"  // change position with L2(T')
           << setw(14) << "L2(u)" << setw(14) << "L2(T)" << setw(14) << "e3d" << setw(14) << "ecf" << setw(14)
           << "ubulk" << setw(14) << "wbulk" << setw(14) << "wallshear" << setw(14) << "buoyPowIn" << setw(14)
           << "totalDiss" << setw(14) << "heatinflux" << setw(14) << "L2(T')" << setw(10) << "Nu(y=" << flags.ystats
           << ")";
    return header.str();
}

string ilcfieldstatsheader_t(const string tname, const ILCFlags flags) {
    stringstream header;
    header << setw(6) << "#(" << tname << ")" << ilcfieldstatsheader(flags);
    return header.str();
}

string ilcfieldstats(const FlowField& u, const FlowField& temp, const ILCFlags flags) {
    std::vector<Real> stats = ilcstats(u, temp, flags);
    // Return string
    stringstream s;
    for (uint i = 0; i < stats.size(); i++) {
        s << setw(14) << stats[i];
    }
    return s.str();
}

string ilcfieldstats_t(const FlowField& u, const FlowField& temp, const Real t, const ILCFlags flags) {
    std::vector<Real> stats = ilcstats(u, temp, flags);
    // Return string
    stringstream s;
    s << setw(8) << t;
    for (uint i = 0; i < stats.size(); i++) {
        s << setw(14) << stats[i];
    }
    return s.str();
}

FlowField totalVelocity(const FlowField& velo, const ILCFlags flags) {
    // copy
    FlowField u(velo);
    FlowField tmp(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());

    // get base flow
    ILC ilc({u, tmp, tmp}, flags);
    ChebyCoeff Ubase = ilc.Ubase();
    ChebyCoeff Wbase = ilc.Wbase();

    // add base flow (should be identical to code in function temperatureNL in OBE
    for (int ny = 0; ny < u.Ny(); ++ny) {
        if (u.taskid() == u.task_coeff(0, 0)) {
            u.cmplx(0, ny, 0, 0) += Complex(Ubase(ny), 0.0);
            u.cmplx(0, ny, 0, 2) += Complex(Wbase(ny), 0.0);
        }
    }
    if (u.taskid() == u.task_coeff(0, 0)) {
        u.cmplx(0, 0, 0, 1) -= Complex(flags.Vsuck, 0.);
    }

    return u;
}

FlowField totalTemperature(const FlowField& temp, const ILCFlags flags) {
    // copy
    FlowField T(temp);
    FlowField tmp(T.Nx(), T.Ny(), T.Nz(), 3, T.Lx(), T.Lz(), T.a(), T.b(), T.cfmpi());

    // get base flow
    ILC ilc({tmp, T, T}, flags);
    ChebyCoeff Tbase = ilc.Tbase();

    // add base flow (should be identical to code in function temperatureNL in OBE
    for (int ny = 0; ny < T.Ny(); ++ny) {
        if (T.taskid() == T.task_coeff(0, 0))
            T.cmplx(0, ny, 0, 0) += Complex(Tbase(ny), 0.0);
    }

    return T;
}

Real buoyPowerInput(const FlowField& utot, const FlowField& ttot, const ILCFlags flags, bool relative) {
    // calculate the bouyancy force along the velocity field to get
    // the power input to the kinetic energy equation

    // get parameters
    Real nu = flags.nu;
    Real sing = sin(flags.gammax);
    Real cosg = cos(flags.gammax);
    Real grav = flags.grav;
    Real laminarInput = grav * sing * sing / (720 * nu);  // normalized by Volume

    // prepare loop over field
    FlowField u(utot);
    FlowField T(ttot);
    FlowField xinput(T.Nx(), T.Ny(), T.Nz(), T.Nd(), T.Lx(), T.Lz(), T.a(), T.b(), T.cfmpi(), Physical, Physical);
    FlowField yinput(T.Nx(), T.Ny(), T.Nz(), T.Nd(), T.Lx(), T.Lz(), T.a(), T.b(), T.cfmpi(), Physical, Physical);
    lint Nz = u.Nz();
    lint nxlocmin = u.nxlocmin();
    lint nxlocmax = u.nxlocmin() + u.Nxloc();
    lint nylocmin = u.nylocmin();
    lint nylocmax = u.nylocmax();

    // sum up buoyancy term
    u.makePhysical();
    T.makePhysical();
#ifdef HAVE_MPI
    for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
        for (lint nz = 0; nz < Nz; ++nz)
            for (lint ny = nylocmin; ny < nylocmax; ++ny) {
                xinput(nx, ny, nz, 0) = u(nx, ny, nz, 0) * T(nx, ny, nz, 0);
                yinput(nx, ny, nz, 0) = u(nx, ny, nz, 1) * T(nx, ny, nz, 0);
            }
#else
    for (lint ny = nylocmin; ny < nylocmax; ++ny)
        for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
            for (lint nz = 0; nz < Nz; ++nz) {
                xinput(nx, ny, nz, 0) = u(nx, ny, nz, 0) * T(nx, ny, nz, 0);
                yinput(nx, ny, nz, 0) = u(nx, ny, nz, 1) * T(nx, ny, nz, 0);
            }
#endif
    xinput.makeSpectral();
    yinput.makeSpectral();

    // calculate the input mean with cheby profile (code is taken from OBE::initConstraint)
    ChebyCoeff xprof(T.My(), T.a(), T.b(), Spectral);
    ChebyCoeff yprof(T.My(), T.a(), T.b(), Spectral);
    Real xtmp = 0;
    Real ytmp = 0;
    for (int ny = 0; ny < T.My(); ++ny) {
        if (utot.taskid() == 0) {
            xtmp = sing * Re(xinput.cmplx(0, ny, 0, 0));
            ytmp = cosg * Re(yinput.cmplx(0, ny, 0, 0));
        }
#ifdef HAVE_MPI
        MPI_Bcast(&xtmp, 1, MPI_DOUBLE, utot.task_coeff(0, 0), *utot.comm_world());
        MPI_Bcast(&ytmp, 1, MPI_DOUBLE, utot.task_coeff(0, 0), *utot.comm_world());
#endif
        xprof[ny] = xtmp;
        yprof[ny] = ytmp;
    }

    Real buoyancyInput = xprof.mean() + yprof.mean();
    // difference between full input and laminar input
    if (relative && abs(laminarInput) > 1e-12) {
        buoyancyInput *= 1.0 / laminarInput;
        buoyancyInput -= 1;
    }

    return buoyancyInput;
}

Real dissipation(const FlowField& utot, const ILCFlags flags, bool normalize, bool relative) {
    Real diss = flags.nu * dissipation(utot, normalize);

    // analytic laminar dissipation (for standard base flow)
    Real sing = sin(flags.gammax);
    Real laminarDiss = flags.grav * sing * sing / (720 * flags.nu);  // normalized by Volume
    // difference between full input and laminar input
    if (relative && abs(laminarDiss) > 1e-12) {
        diss *= 1.0 / laminarDiss;
        diss -= 1;
    }

    return diss;
}

Real heatinflux(const FlowField& temp, const ILCFlags flags, bool normalize, bool relative) {
    // with reference to wallshear, but only lower wall
    assert(temp.ystate() == Spectral);
    Real I = 0;
    if (temp.taskid() == temp.task_coeff(0, 0)) {
        ChebyCoeff tprof = Re(temp.profile(0, 0, 0));
        ChebyCoeff dTdy = diff(tprof);
        I = flags.kappa * abs(dTdy.eval_a());
    }
#ifdef HAVE_MPI
    MPI_Bcast(&I, 1, MPI_DOUBLE, temp.task_coeff(0, 0), temp.cfmpi()->comm_world);
#endif
    if (!normalize)
        I *= 2 * temp.Lx() * temp.Lz();
    if (relative) {
        Real kappa = flags.kappa;
        Real deltaT = flags.tlowerwall - flags.tupperwall;
        Real H = temp.b() - temp.a();
        I *= 1.0 / kappa / deltaT * H;
        I -= 1.0;
    }
    return I;
}

Real heatcontent(const FlowField& ttot, const ILCFlags flags) {
    assert(ttot.ystate() == Spectral);
    int N = 100;
    Real dy = (flags.ystats - ttot.a()) / (N - 1);
    Real avt = 0;  // average temperature
    if (ttot.taskid() == ttot.task_coeff(0, 0)) {
        ChebyCoeff tprof = Re(ttot.profile(0, 0, 0));
        for (int i = 0; i < N; ++i) {
            Real y = ttot.a() + i * dy;
            avt += tprof.eval(y);
        }
        avt *= 1.0 / N;
    }
#ifdef HAVE_MPI
    MPI_Bcast(&avt, 1, MPI_DOUBLE, ttot.task_coeff(0, 0), ttot.cfmpi()->comm_world);
#endif
    return avt;
}

Real Nusselt_plane(const FlowField& utot, const FlowField& ttot, const ILCFlags flags, bool relative) {
    assert(utot.ystate() == Spectral && ttot.ystate() == Spectral);

    // calculate product for advective heat transport
    FlowField u(utot);
    FlowField T(ttot);
    FlowField vt(T.Nx(), T.Ny(), T.Nz(), T.Nd(), T.Lx(), T.Lz(), T.a(), T.b(), T.cfmpi(), Physical, Physical);
    lint Nz = u.Nz();
    lint nxlocmin = u.nxlocmin();
    lint nxlocmax = u.nxlocmin() + u.Nxloc();
    lint nylocmin = u.nylocmin();
    lint nylocmax = u.nylocmax();

    // loop to form product
    u.makePhysical();
    T.makePhysical();
#ifdef HAVE_MPI
    for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
        for (lint nz = 0; nz < Nz; ++nz)
            for (lint ny = nylocmin; ny < nylocmax; ++ny) {
                vt(nx, ny, nz, 0) = u(nx, ny, nz, 1) * T(nx, ny, nz, 0);
            }
#else
    for (lint ny = nylocmin; ny < nylocmax; ++ny)
        for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
            for (lint nz = 0; nz < Nz; ++nz) {
                vt(nx, ny, nz, 0) = u(nx, ny, nz, 1) * T(nx, ny, nz, 0);
            }
#endif
    vt.makeSpectral();

    // calculate Nusselt number
    Real Nu = 0;
    Real y = flags.ystats;
    Real kappa = flags.kappa;
    if (ttot.taskid() == ttot.task_coeff(0, 0)) {
        ChebyCoeff tprof = Re(ttot.profile(0, 0, 0));
        ChebyCoeff dTdy = diff(tprof);
        ChebyCoeff vtprof = Re(vt.profile(0, 0, 0));
        Nu = vtprof.eval(y) -
             kappa * dTdy.eval(y);  // Formula 10 in Chilla&Schumacher 2012 (together with normalization below)
    }

#ifdef HAVE_MPI
    MPI_Bcast(&Nu, 1, MPI_DOUBLE, ttot.task_coeff(0, 0), ttot.cfmpi()->comm_world);
#endif

    if (relative) {
        Real deltaT = flags.tlowerwall - flags.tupperwall;
        Real H = utot.b() - utot.a();
        Nu *= 1.0 / kappa / deltaT * H;
        Nu -= 1.0;
    }
    return Nu;
}

/* Begin of ilcDSI class*/

ilcDSI::ilcDSI() {}

ilcDSI::ilcDSI(ILCFlags& ilcflags, FieldSymmetry sigma, PoincareCondition* h, TimeStep dt, bool Tsearch, bool xrelative,
               bool zrelative, bool Tnormalize, Real Unormalize, const FlowField& u, const FlowField& temp, ostream* os)
    : cfDSI(ilcflags, sigma, h, dt, Tsearch, xrelative, zrelative, Tnormalize, Unormalize, u, os),
      ilcflags_(ilcflags) {}

Eigen::VectorXd ilcDSI::eval(const Eigen::VectorXd& x) {
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField temp(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);

    Real T;
    extractVectorILC(x, u, temp, sigma_, T);

    FlowField Gu(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField Gtemp(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);
    G(u, temp, T, h_, sigma_, Gu, Gtemp, ilcflags_, dt_, Tnormalize_, Unormalize_, fcount_, CFL_, *os_);
    Eigen::VectorXd Gx(Eigen::VectorXd::Zero(x.rows()));
    //   Galpha *= 1./vednsflags_.b_para;
    field2vector(Gu, Gtemp, Gx);  // This does not change the size of Gx and automatically leaves the last entries zero

    return Gx;
}

Eigen::VectorXd ilcDSI::eval(const Eigen::VectorXd& x0, const Eigen::VectorXd& x1, bool symopt) {
    FlowField u0(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField u1(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField temp0(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField temp1(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);
    Real T0, T1;
    FieldSymmetry sigma0, sigma1;
    extractVectorILC(x0, u0, temp0, sigma0, T0);
    extractVectorILC(x1, u1, temp1, sigma1, T1);

    FlowField Gu(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField Gtemp(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);

    f(u0, temp0, T0, h_, Gu, Gtemp, ilcflags_, dt_, fcount_, CFL_, *os_);
    if (symopt) {
        Gu *= sigma0;
        if (sigma0.sy() == -1) {
            // wall-normal mirroring in velocity requires sign change in temperature
            FieldSymmetry inv(-1);
            sigma0 *= inv;
        }
        Gtemp *= sigma0;
    }
    Gu -= u1;
    Gtemp -= temp1;

    // normalize
    if (Tnormalize_) {
        Gu *= 1.0 / T0;
        Gtemp *= 1.0 / T0;
    }
    if (Unormalize_ != 0.0) {
        Real funorm = L2Norm3d(Gu);
        Gu *= 1. / sqrt(abs(funorm * (Unormalize_ - funorm)));
        // u should stay off zero, so normalize with u for now - temp should also stay away from zero
        Gtemp *= 1. / sqrt(abs(funorm * (Unormalize_ - funorm)));
    }

    Eigen::VectorXd Gx(Eigen::VectorXd::Zero(x0.rows()));
    field2vector(Gu, Gtemp, Gx);  // This does not change the size of Gx and automatically leaves the last entries zero

    return Gx;
}

void ilcDSI::save(const Eigen::VectorXd& x, const string filebase, const string outdir, const bool fieldsonly) {
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField temp(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);
    FieldSymmetry sigma;
    Real T;
    extractVectorILC(x, u, temp, sigma, T);

    u.save(outdir + "u" + filebase);
    temp.save(outdir + "t" + filebase);

    if (!fieldsonly) {
        string fs = ilcfieldstats(u, temp, ilcflags_);
        if (u.taskid() == 0) {
            if (xrelative_ || zrelative_ || !sigma.isIdentity())
                sigma.save(outdir + "sigma" + filebase);
            if (Tsearch_)
                chflow::save(T, outdir + "T" + filebase);
            // sigma.save (outdir+"sigmaconverge.asc", ios::app);
            ofstream fout((outdir + "fieldconverge.asc").c_str(), ios::app);
            long pos = fout.tellp();
            if (pos == 0)
                fout << ilcfieldstatsheader() << endl;
            fout << fs << endl;
            fout.close();
            ilcflags_.save(outdir);
        }
    }
}

string ilcDSI::stats(const Eigen::VectorXd& x) {
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField temp(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);
    FieldSymmetry sigma;
    Real T;
    extractVectorILC(x, u, temp, sigma, T);
    return ilcfieldstats_t(u, temp, mu_, ilcflags_);
}

pair<string, string> ilcDSI::stats_minmax(const Eigen::VectorXd& x) {
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField temp(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField Gu(u);
    FlowField Gtemp(temp);
    FieldSymmetry sigma;
    Real T;
    extractVectorILC(x, u, temp, sigma, T);

    std::vector<Real> stats = ilcstats(u, temp, ilcflags_);
    std::vector<Real> minstats(stats);
    std::vector<Real> maxstats(stats);

    // quick hack to avoid new interface or creating simple f() for ILC
    TimeStep dt = TimeStep(ilcflags_.dt, 0, 1, 1, 0, 0, false);
    int fcount = 0;
    PoincareCondition* h = 0;
    Real CFL = 0.0;
    std::ostream muted_os(0);
    Real timep = T / 100.0;

    *os_ << "Using flag -orbOut: Calculate minmax-statistics of periodic orbit." << endl;
    for (int t = 0; t < 100; t++) {
        f(u, temp, timep, h, Gu, Gtemp, ilcflags_, dt, fcount, CFL, muted_os);
        stats = ilcstats(Gu, Gtemp, ilcflags_);
        for (uint i = 0; i < stats.size(); i++) {
            minstats[i] = (minstats[i] < stats[i]) ? minstats[i] : stats[i];
            maxstats[i] = (maxstats[i] > stats[i]) ? maxstats[i] : stats[i];
        }
        u = Gu;
        temp = Gtemp;
    }
    // Return string
    stringstream smin;
    stringstream smax;
    smin << setw(8) << mu_;
    smax << setw(8) << mu_;
    for (uint i = 0; i < stats.size(); i++) {
        smin << setw(14) << minstats[i];
        smax << setw(14) << maxstats[i];
    }

    pair<string, string> minmax;
    minmax = make_pair(smin.str(), smax.str());
    return minmax;
}

string ilcDSI::statsHeader() { return ilcfieldstatsheader_t(ilc_cPar2s(ilc_cPar_), ilcflags_); }

/// after finding new solution fix phases
void ilcDSI::phaseShift(Eigen::VectorXd& x) {
    if (xphasehack_ || zphasehack_) {
        FlowField unew(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
        FlowField tnew(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);
        FieldSymmetry sigma;
        Real T;
        extractVectorILC(x, unew, tnew, sigma, T);
        // vector2field (x,unew);
        const int phasehackcoord = 0;  // Those values were fixed in continuesoln anyway
        const parity phasehackparity = Odd;
        const Real phasehackguess = 0.0;

        if (zphasehack_) {
            FieldSymmetry tau = zfixphasehack(unew, phasehackguess, phasehackcoord, phasehackparity);
            cout << "fixing z phase of potential solution with phase shift tau == " << tau << endl;
            unew *= tau;
            tnew *= tau;
        }
        if (xphasehack_) {
            FieldSymmetry tau = xfixphasehack(unew, phasehackguess, phasehackcoord, phasehackparity);
            cout << "fixing x phase of potential solution with phase shift tau == " << tau << endl;
            unew *= tau;
            tnew *= tau;
        }
        if (uUbasehack_) {
            cout << "fixing u+Ubase decomposition so that <du/dy> = 0 at walls (i.e. Ubase balances mean pressure "
                    "gradient))"
                 << endl;
            Real ubulk = Re(unew.profile(0, 0, 0)).mean();
            if (abs(ubulk) < 1e-15)
                ubulk = 0.0;

            ChebyCoeff Ubase = laminarProfile(ilcflags_.nu, ilcflags_.constraint, ilcflags_.dPdx,
                                              ilcflags_.Ubulk - ubulk, ilcflags_.Vsuck, unew.a(), unew.b(),
                                              ilcflags_.ulowerwall, ilcflags_.uupperwall, unew.Ny());

            fixuUbasehack(unew, Ubase);
        }
        makeVectorILC(unew, tnew, sigma, T, x);
    }
}

void ilcDSI::phaseShift(Eigen::MatrixXd& y) {
    if (xphasehack_ || zphasehack_) {
        FlowField unew(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
        FlowField tnew(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);
        Eigen::VectorXd yvec;
        FieldSymmetry sigma;
        Real T;

        const int phasehackcoord = 0;  // Those values were fixed in continuesoln anyway
        const parity phasehackparity = Odd;
        const Real phasehackguess = 0.0;

        FieldSymmetry taux(0.0, 0.0);
        FieldSymmetry tauz(0.0, 0.0);

        extractVectorILC(y.col(0), unew, tnew, sigma, T);

        if (xphasehack_) {
            taux = xfixphasehack(unew, phasehackguess, phasehackcoord, phasehackparity);
            cout << "fixing x phase of potential solution with phase shift tau == " << taux << endl;
        }
        if (zphasehack_) {
            tauz = zfixphasehack(unew, phasehackguess, phasehackcoord, phasehackparity);
            cout << "fixing z phase of potential solution with phase shift tau == " << tauz << endl;
        }

        for (int i = 0; i < y.cols(); i++) {
            extractVectorILC(y.col(i), unew, tnew, sigma, T);
            unew *= taux;
            tnew *= taux;
            unew *= tauz;
            tnew *= tauz;
            makeVectorILC(unew, tnew, sigma, T, yvec);
            y.col(i) = yvec;
        }
    }
}

Real ilcDSI::extractT(const Eigen::VectorXd& x) {  // inefficient hack
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField temp(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);
    FieldSymmetry sigma;
    Real T;
    extractVectorILC(x, u, temp, sigma, T);
    return T;
}

Real ilcDSI::extractXshift(const Eigen::VectorXd& x) {  // inefficient hack
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField temp(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);
    FieldSymmetry sigma;
    Real T;
    extractVectorILC(x, u, temp, sigma, T);
    return sigma.ax();
}

Real ilcDSI::extractZshift(const Eigen::VectorXd& x) {  // inefficient hack
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField temp(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);
    FieldSymmetry sigma;
    Real T;
    extractVectorILC(x, u, temp, sigma, T);
    return sigma.az();
}

void ilcDSI::makeVectorILC(const FlowField& u, const FlowField& temp, const FieldSymmetry& sigma, const Real T,
                           Eigen::VectorXd& x) {
    if (u.Nd() != 3)
        cferror("ilcDSI::makeVector(): u.Nd() = " + i2s(u.Nd()) + " != 3");
    if (temp.Nd() != 1)
        cferror("ilcDSI::makeVector(): temp.Nd() = " + i2s(temp.Nd()) + " != 1");
    int taskid = u.taskid();

    int uunk = field2vector_size(u, temp);                   // # of variables for u and alpha unknonwn
    const int Tunk = (Tsearch_ && taskid == 0) ? uunk : -1;  // index for T unknown
    const int xunk = (xrelative_ && taskid == 0) ? uunk + Tsearch_ : -1;
    const int zunk = (zrelative_ && taskid == 0) ? uunk + Tsearch_ + xrelative_ : -1;
    int Nunk = (taskid == 0) ? uunk + Tsearch_ + xrelative_ + zrelative_ : uunk;
    if (x.rows() < Nunk)
        x.resize(Nunk);
    field2vector(u, temp, x);
    if (taskid == 0) {
        if (Tsearch_)
            x(Tunk) = T;
        if (xrelative_)
            x(xunk) = sigma.ax();
        if (zrelative_)
            x(zunk) = sigma.az();
    }
}

void ilcDSI::extractVectorILC(const Eigen::VectorXd& x, FlowField& u, FlowField& temp, FieldSymmetry& sigma, Real& T) {
    int uunk = field2vector_size(u, temp);  // number of components in x that corresond to u and alpha
    vector2field(x, u, temp);
    const int Tunk = uunk + Tsearch_ - 1;
    const int xunk = uunk + Tsearch_ + xrelative_ - 1;
    const int zunk = uunk + Tsearch_ + xrelative_ + zrelative_ - 1;
    Real ax = 0;
    Real az = 0;
    if (u.taskid() == 0) {
        T = Tsearch_ ? x(Tunk) : Tinit_;
        ax = xrelative_ ? x(xunk) : axinit_;
        az = zrelative_ ? x(zunk) : azinit_;
    }
#ifdef HAVE_MPI
    MPI_Bcast(&T, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&az, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    sigma = FieldSymmetry(sigma_.sx(), sigma_.sy(), sigma_.sz(), ax, az, sigma_.s());
}

Eigen::VectorXd ilcDSI::xdiff(const Eigen::VectorXd& a) {
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField temp(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);
    vector2field(a, u, temp);
    Eigen::VectorXd dadx(a.size());
    dadx.setZero();
    u = chflow::xdiff(u);
    temp = chflow::xdiff(temp);
    field2vector(u, temp, dadx);
    dadx *= 1. / L2Norm(dadx);
    return dadx;
}

Eigen::VectorXd ilcDSI::zdiff(const Eigen::VectorXd& a) {
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField temp(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);
    vector2field(a, u, temp);
    Eigen::VectorXd dadz(a.size());
    dadz.setZero();
    u = chflow::zdiff(u);
    temp = chflow::zdiff(temp);
    field2vector(u, temp, dadz);
    dadz *= 1. / L2Norm(dadz);
    return dadz;
}

Eigen::VectorXd ilcDSI::tdiff(const Eigen::VectorXd& a, Real epsDt) {
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField temp(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);
    FieldSymmetry sigma;
    Real T;
    // quick hack to avoid new interface or creating simple f() for ILC
    TimeStep dt = TimeStep(epsDt, 0, 1, 1, 0, 0, false);
    int fcount = 0;
    PoincareCondition* h = 0;
    Real CFL = 0.0;
    FlowField edudtf(u);
    FlowField edtempdtf(temp);
    std::ostream muted_os(0);
    //   vector2field (a, u, temp);
    extractVectorILC(a, u, temp, sigma, T);
    // use existing f() instead of simple
    f(u, temp, epsDt, h, edudtf, edtempdtf, ilcflags_, dt, fcount, CFL, muted_os);
    //   f (temp, 1,epsDt, edtempdtf, dnsflags_, *os_);
    edudtf -= u;
    edtempdtf -= temp;
    Eigen::VectorXd dadt(a.size());
    field2vector(edudtf, edtempdtf, dadt);
    dadt *= 1. / L2Norm(dadt);
    return dadt;
}

void ilcDSI::updateMu(Real mu) {
    DSI::updateMu(mu);
    if (ilc_cPar_ == ilc_continuationParameter::none) {
        cfDSI::updateMu(mu);
    } else if (ilc_cPar_ == ilc_continuationParameter::Ra) {
        Real Prandtl = ilcflags_.nu / ilcflags_.kappa;
        ilcflags_.nu = sqrt(Prandtl / mu);
        ilcflags_.kappa = 1.0 / sqrt(Prandtl * mu);
    } else if (ilc_cPar_ == ilc_continuationParameter::Pr) {
        Real Rayleigh = 1 / (ilcflags_.nu * ilcflags_.kappa);
        ilcflags_.nu = sqrt(mu / Rayleigh);
        ilcflags_.kappa = 1.0 / sqrt(mu * Rayleigh);
    } else if (ilc_cPar_ == ilc_continuationParameter::gx) {
        ilcflags_.gammax = mu;
    } else if (ilc_cPar_ == ilc_continuationParameter::gz) {
        ilcflags_.gammaz = mu;
    } else if (ilc_cPar_ == ilc_continuationParameter::gxEps) {
        Real gx = ilcflags_.gammax;
        Real Ra_old = 1 / (ilcflags_.nu * ilcflags_.kappa);
        Real Prandtl = ilcflags_.nu / ilcflags_.kappa;
        Real Rc = 1707.76;
        Real Rc2 = 8053.1;
        Real gc2 = 77.7567;
        // get old threshold
        Real Rac_old = 0;
        if (gx / pi * 180.0 < gc2)
            Rac_old = Rc / cos(gx);
        else
            Rac_old = 1.0 / 41.0 * pow(gx / pi * 180.0 - gc2, 3) + 5.0 / 14.0 * pow(gx / pi * 180.0 - gc2, 2) +
                      29 * (gx / pi * 180.0 - gc2) + Rc2;
        // get new threshold
        Real Rac_new = 0;
        if (mu / pi * 180.0 < gc2)
            Rac_new = Rc / cos(mu);
        else
            Rac_new = 1.0 / 41.0 * pow(mu / pi * 180.0 - gc2, 3) + 5.0 / 14.0 * pow(mu / pi * 180.0 - gc2, 2) +
                      29 * (mu / pi * 180.0 - gc2) + Rc2;
        Real Ra_new = Ra_old / Rac_old * Rac_new;
        cout << "Normalized Rayleigh number is epsilon = " << Ra_new / Rac_new - 1 << endl;
        ilcflags_.nu = sqrt(Prandtl / Ra_new);
        ilcflags_.kappa = 1.0 / sqrt(Prandtl * Ra_new);
        ilcflags_.gammax = mu;
    } else if (ilc_cPar_ == ilc_continuationParameter::Grav) {
        ilcflags_.grav = mu;
    } else if (ilc_cPar_ == ilc_continuationParameter::Tref) {
        ilcflags_.t_ref = mu;
    } else if (ilc_cPar_ == ilc_continuationParameter::P) {
        ilcflags_.dPdx = mu;
    } else if (ilc_cPar_ == ilc_continuationParameter::Uw) {
        ilcflags_.Uwall = mu;
        ilcflags_.ulowerwall = -mu * cos(ilcflags_.theta);
        ilcflags_.uupperwall = mu * cos(ilcflags_.theta);
        ilcflags_.wlowerwall = -mu * sin(ilcflags_.theta);
        ilcflags_.wupperwall = mu * sin(ilcflags_.theta);
        ;
    } else if (ilc_cPar_ == ilc_continuationParameter::UwGrav) {
        ilcflags_.Uwall = mu;
        ilcflags_.ulowerwall = -mu * cos(ilcflags_.theta);
        ilcflags_.uupperwall = mu * cos(ilcflags_.theta);
        ilcflags_.wlowerwall = -mu * sin(ilcflags_.theta);
        ilcflags_.wupperwall = mu * sin(ilcflags_.theta);
        ;
        ilcflags_.grav = 1 - mu;
    } else if (ilc_cPar_ == ilc_continuationParameter::Rot) {
        ilcflags_.rotation = mu;
    } else if (ilc_cPar_ == ilc_continuationParameter::Theta) {
        ilcflags_.theta = mu;
        ilcflags_.ulowerwall = -ilcflags_.Uwall * cos(mu);
        ilcflags_.uupperwall = ilcflags_.Uwall * cos(mu);
        ilcflags_.wlowerwall = -ilcflags_.Uwall * sin(mu);
        ilcflags_.wupperwall = ilcflags_.Uwall * sin(mu);
        ;
    } else if (ilc_cPar_ == ilc_continuationParameter::ThArc) {
        Real xleg = Lz_ / tan(ilcflags_.theta);  // hypothetical Lx
        Lz_ *= sin(mu) / sin(ilcflags_.theta);   // rescale Lz for new angle at const diagonal (of xleg x Lz_)
        Lx_ *= Lz_ / tan(mu) / xleg;
        ilcflags_.theta = mu;
        ilcflags_.ulowerwall = -ilcflags_.Uwall * cos(mu);
        ilcflags_.uupperwall = ilcflags_.Uwall * cos(mu);
        ilcflags_.wlowerwall = -ilcflags_.Uwall * sin(mu);
        ilcflags_.wupperwall = ilcflags_.Uwall * sin(mu);
        ;
    } else if (ilc_cPar_ == ilc_continuationParameter::ThLx) {
        Lx_ *= tan(ilcflags_.theta) / tan(mu);
        ilcflags_.theta = mu;
        ilcflags_.ulowerwall = -ilcflags_.Uwall * cos(mu);
        ilcflags_.uupperwall = ilcflags_.Uwall * cos(mu);
        ilcflags_.wlowerwall = -ilcflags_.Uwall * sin(mu);
        ilcflags_.wupperwall = ilcflags_.Uwall * sin(mu);
        ;
    } else if (ilc_cPar_ == ilc_continuationParameter::ThLz) {
        Lz_ *= tan(mu) / tan(ilcflags_.theta);
        ilcflags_.theta = mu;
        ilcflags_.ulowerwall = -ilcflags_.Uwall * cos(mu);
        ilcflags_.uupperwall = ilcflags_.Uwall * cos(mu);
        ilcflags_.wlowerwall = -ilcflags_.Uwall * sin(mu);
        ilcflags_.wupperwall = ilcflags_.Uwall * sin(mu);
        ;
    } else if (ilc_cPar_ == ilc_continuationParameter::Lx) {
        Lx_ = mu;
    } else if (ilc_cPar_ == ilc_continuationParameter::Lz) {
        Lz_ = mu;
    } else if (ilc_cPar_ == ilc_continuationParameter::Aspect) {
        Real aspect = Lx_ / Lz_;
        Real update = mu / aspect;

        // do only half the adjustment in x, the other in z (i.e. equivalent to Lx_new = 0.5Lx +0.5Lx * update)
        Lx_ *= 1 + (update - 1) / 2.0;
        Lz_ = Lx_ / mu;
    } else if (ilc_cPar_ == ilc_continuationParameter::Diag) {
        Real aspect = Lx_ / Lz_;
        Real theta = atan(aspect);
        Real diagonal = sqrt(Lx_ * Lx_ + Lz_ * Lz_);
        Real update = mu - diagonal;
        //     Lx_ += sqrt ( (update*update * aspect*aspect) / (1 + aspect*aspect));
        Lx_ += update * sin(theta);
        Lz_ = Lx_ / aspect;
    } else if (ilc_cPar_ == ilc_continuationParameter::Vs) {
        ilcflags_.Vsuck = mu;
    } else if (ilc_cPar_ == ilc_continuationParameter::VsNu) {
        ilcflags_.nu = mu;
        ilcflags_.Vsuck = mu;
    } else if (ilc_cPar_ == ilc_continuationParameter::VsH) {
        Real nu = ilcflags_.nu;
        Real Vs = nu * (1 - exp(-mu));
        Real H = nu * mu / Vs;
        ya_ = 0;
        yb_ = H;
        ilcflags_.Vsuck = Vs;
    } else if (ilc_cPar_ == ilc_continuationParameter::H) {
        ya_ = 0;
        yb_ = mu;
    } else {
        throw invalid_argument("ilcDSI::updateMu(): continuation parameter is unknown");
    }
}

void ilcDSI::chooseMuILC(string muName) {
    ilc_continuationParameter ilc_cPar = s2ilc_cPar(muName);

    if (ilc_cPar == ilc_continuationParameter::none)
        ilcDSI::chooseMu(muName);
    else
        chooseMuILC(ilc_cPar);
}

void ilcDSI::chooseMuILC(ilc_continuationParameter mu) {
    ilc_cPar_ = mu;
    Real Rayleigh;
    Real Prandtl;
    switch (mu) {
        case ilc_continuationParameter::Ra:
            Rayleigh = 1 / (ilcflags_.nu * ilcflags_.kappa);
            updateMu(Rayleigh);
            break;
        case ilc_continuationParameter::Pr:
            Prandtl = ilcflags_.nu / ilcflags_.kappa;
            updateMu(Prandtl);
            break;
        case ilc_continuationParameter::gx:
            updateMu(ilcflags_.gammax);
            break;
        case ilc_continuationParameter::gz:
            updateMu(ilcflags_.gammaz);
            break;
        case ilc_continuationParameter::gxEps:
            updateMu(ilcflags_.gammax);
            break;
        case ilc_continuationParameter::Grav:
            updateMu(ilcflags_.grav);
            break;
        case ilc_continuationParameter::Tref:
            updateMu(ilcflags_.t_ref);
            break;
        case ilc_continuationParameter::P:
            updateMu(ilcflags_.dPdx);
            break;
        case ilc_continuationParameter::Uw:
            updateMu(ilcflags_.uupperwall / cos(ilcflags_.theta));
            break;
        case ilc_continuationParameter::UwGrav:
            updateMu(ilcflags_.uupperwall / cos(ilcflags_.theta));
            break;
        case ilc_continuationParameter::Rot:
            updateMu(ilcflags_.rotation);
            break;
        case ilc_continuationParameter::Theta:
            updateMu(ilcflags_.theta);
            break;
        case ilc_continuationParameter::ThArc:
            updateMu(ilcflags_.theta);
            break;
        case ilc_continuationParameter::ThLx:
            updateMu(ilcflags_.theta);
            break;
        case ilc_continuationParameter::ThLz:
            updateMu(ilcflags_.theta);
            break;
        case ilc_continuationParameter::Lx:
            updateMu(Lx_);
            break;
        case ilc_continuationParameter::Lz:
            updateMu(Lz_);
            break;
        case ilc_continuationParameter::Aspect:
            updateMu(Lx_ / Lz_);
            break;
        case ilc_continuationParameter::Diag:
            updateMu(sqrt(Lx_ * Lx_ + Lz_ * Lz_));
            break;
        case ilc_continuationParameter::Vs:
            updateMu(ilcflags_.Vsuck);
            break;
        case ilc_continuationParameter::VsNu:
            updateMu(ilcflags_.Vsuck);
            break;
        case ilc_continuationParameter::VsH:
            updateMu(-log(1 - ilcflags_.Vsuck / ilcflags_.nu));
            break;
        case ilc_continuationParameter::H:
            updateMu(yb_ - ya_);
            break;
        case ilc_continuationParameter::none:
            throw invalid_argument(
                "ilcDSI::chooseMu(): continuation parameter is none, we should not reach this point");
        default:
            throw invalid_argument("ilcDSI::chooseMu(): continuation parameter is unknown");
    }
}

ilc_continuationParameter ilcDSI::s2ilc_cPar(string muname) {
    std::transform(muname.begin(), muname.end(), muname.begin(), ::tolower);  // why is the string made lower case?
    if (muname == "ra")
        return ilc_continuationParameter::Ra;
    else if (muname == "pr")
        return ilc_continuationParameter::Pr;
    else if (muname == "gx")
        return ilc_continuationParameter::gx;
    else if (muname == "gz")
        return ilc_continuationParameter::gz;
    else if (muname == "gxeps")
        return ilc_continuationParameter::gxEps;
    else if (muname == "grav")
        return ilc_continuationParameter::Grav;
    else if (muname == "tref")
        return ilc_continuationParameter::Tref;
    else if (muname == "p")
        return ilc_continuationParameter::P;
    else if (muname == "uw")
        return ilc_continuationParameter::Uw;
    else if (muname == "uwgrav")
        return ilc_continuationParameter::UwGrav;
    else if (muname == "rot")
        return ilc_continuationParameter::Rot;
    else if (muname == "theta")
        return ilc_continuationParameter::Theta;
    else if (muname == "tharc")
        return ilc_continuationParameter::ThArc;
    else if (muname == "thlx")
        return ilc_continuationParameter::ThLx;
    else if (muname == "thlz")
        return ilc_continuationParameter::ThLz;
    else if (muname == "lx")
        return ilc_continuationParameter::Lx;
    else if (muname == "lz")
        return ilc_continuationParameter::Lz;
    else if (muname == "aspect")
        return ilc_continuationParameter::Aspect;
    else if (muname == "diag")
        return ilc_continuationParameter::Diag;
    else if (muname == "vs")
        return ilc_continuationParameter::Vs;
    else if (muname == "vsnu")
        return ilc_continuationParameter::VsNu;
    else if (muname == "vsh")
        return ilc_continuationParameter::VsH;
    else if (muname == "h")
        return ilc_continuationParameter::H;
    else
        // cout << "ilcDSI::s2ilc_cPar(): ilc_continuation parameter '"+muname+"' is unknown, defaults to 'none'" <<
        // endl;
        return ilc_continuationParameter::none;
}

string ilcDSI::printMu() { return ilc_cPar2s(ilc_cPar_); }

string ilcDSI::ilc_cPar2s(ilc_continuationParameter ilc_cPar) {
    if (ilc_cPar == ilc_continuationParameter::none)
        return cfDSI::cPar2s(cPar_);
    else if (ilc_cPar == ilc_continuationParameter::Ra)
        return "Ra";
    else if (ilc_cPar == ilc_continuationParameter::Pr)
        return "Pr";
    else if (ilc_cPar == ilc_continuationParameter::gx)
        return "gx";
    else if (ilc_cPar == ilc_continuationParameter::gz)
        return "gz";
    else if (ilc_cPar == ilc_continuationParameter::gxEps)
        return "gxeps";
    else if (ilc_cPar == ilc_continuationParameter::Grav)
        return "Grav";
    else if (ilc_cPar == ilc_continuationParameter::Tref)
        return "Tref";
    else if (ilc_cPar == ilc_continuationParameter::P)
        return "P";
    else if (ilc_cPar == ilc_continuationParameter::Uw)
        return "Uw";
    else if (ilc_cPar == ilc_continuationParameter::UwGrav)
        return "Uw(Grav)";
    else if (ilc_cPar == ilc_continuationParameter::Rot)
        return "Rot";
    else if (ilc_cPar == ilc_continuationParameter::Theta)
        return "Theta";
    else if (ilc_cPar == ilc_continuationParameter::ThArc)
        return "Theta(DiagArc)";
    else if (ilc_cPar == ilc_continuationParameter::ThLx)
        return "Theta(Lx)";
    else if (ilc_cPar == ilc_continuationParameter::ThLz)
        return "Theta(Lz)";
    else if (ilc_cPar == ilc_continuationParameter::Lx)
        return "Lx";
    else if (ilc_cPar == ilc_continuationParameter::Lz)
        return "Lz";
    else if (ilc_cPar == ilc_continuationParameter::Aspect)
        return "Aspect";
    else if (ilc_cPar == ilc_continuationParameter::Diag)
        return "Diag";
    else if (ilc_cPar == ilc_continuationParameter::Vs)
        return "Vs";
    else if (ilc_cPar == ilc_continuationParameter::VsNu)
        return "VsNu";
    else if (ilc_cPar == ilc_continuationParameter::VsH)
        return "VsH";
    else if (ilc_cPar == ilc_continuationParameter::H)
        return "H";
    else
        throw invalid_argument("ilcDSI::ilc_cPar2s(): continuation parameter is not convertible to string");
}

void ilcDSI::saveParameters(string searchdir) {
    // cfDSI::saveParameters (searchdir);
    ilcflags_.save(searchdir);
}

void ilcDSI::saveEigenvec(const Eigen::VectorXd& ev, const string label, const string outdir) {
    FlowField efu(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField eft(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);
    vector2field(ev, efu, eft);
    efu *= 1.0 / L2Norm(efu);
    eft *= 1.0 / L2Norm(eft);
    efu.save(outdir + "efu" + label);
    eft.save(outdir + "eft" + label);
}

void ilcDSI::saveEigenvec(const Eigen::VectorXd& evA, const Eigen::VectorXd& evB, const string label1,
                          const string label2, const string outdir) {
    FlowField efAu(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField efBu(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField efAt(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField efBt(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);
    vector2field(evA, efAu, efAt);
    vector2field(evB, efBu, efBt);
    Real cu = 1.0 / sqrt(L2Norm2(efAu) + L2Norm2(efBu));
    Real ct = 1.0 / sqrt(L2Norm2(efAt) + L2Norm2(efBt));
    efAu *= cu;
    efBu *= cu;
    efAt *= ct;
    efBt *= ct;
    efAu.save(outdir + "efu" + label1);
    efBu.save(outdir + "efu" + label2);
    efAt.save(outdir + "eft" + label1);
    efBt.save(outdir + "eft" + label2);
}

/* OUTSIDE CLASS */

// G(x) = G(u,sigma) = (sigma f^T(u) - u) for orbits
void G(const FlowField& u, const FlowField& temp, Real& T, PoincareCondition* h, const FieldSymmetry& sigma,
       FlowField& Gu, FlowField& Gtemp, const ILCFlags& ilcflags, const TimeStep& dt, bool Tnormalize, Real Unormalize,
       int& fcount, Real& CFL, ostream& os) {
    f(u, temp, T, h, Gu, Gtemp, ilcflags, dt, fcount, CFL, os);
    Real funorm = L2Norm3d(Gu);
    Gu *= sigma;
    Gu -= u;
    if (sigma.sy() == -1) {
        // wall-normal mirroring in velocity requires sign change in temperature
        FieldSymmetry tsigma(sigma);
        FieldSymmetry inv(-1);
        tsigma *= inv;
        Gtemp *= tsigma;
    } else {
        Gtemp *= sigma;
    }
    Gtemp -= temp;
    if (Tnormalize) {
        Gu *= 1.0 / T;
        Gtemp *= 1.0 / T;
    }
    if (Unormalize != 0.0) {
        Gu *= 1. / sqrt(abs(funorm * (Unormalize - funorm)));
        // u should stay off zero, so normalize with u for now - temp should also stay away from zero
        Gtemp *= 1. / sqrt(abs(funorm * (Unormalize - funorm)));
    }
}

void f(const FlowField& u, const FlowField& temp, Real& T, PoincareCondition* h, FlowField& f_u, FlowField& f_temp,
       const ILCFlags& ilcflags_, const TimeStep& dt_, int& fcount, Real& CFL, ostream& os) {
    if (!isfinite(L2Norm(u))) {
        os << "error in f: u is not finite. exiting." << endl;
        exit(1);
    }
    ILCFlags flags(ilcflags_);
    flags.logstream = &os;
    TimeStep dt(dt_);
    vector<FlowField> fields = {u, temp, FlowField(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi())};

    //   f_u = u;
    //   f_temp = temp;
    // No Poincare section, just integration to time T
    if (h == 0) {
        if (T < 0) {
            os << "f: negative integration time T == " << T << endl
               << "returning f(u,T) == (1+abs(T))*u" << endl
               << "returning f(temp,T) == (1+abs(T))*temp" << endl;
            fields[0] *= 1 + abs(T);
            fields[1] *= 1 + abs(T);
            return;
        }
        // Special case #1: no time steps
        if (T == 0) {
            os << "f: T==0, no integration, returning u and temp" << endl;
            return;
        }
        dt.adjust_for_T(T, false);
        flags.dt = dt;
        // Adjust dt for CFL if necessary
        ILC ilc(fields, flags);
        ilc.advance(fields, 1);
        if (dt.variable()) {
            dt.adjust(ilc.CFL(fields[0]), false);
            ilc.reset_dt(dt);
        }
        //  t == current time in integration
        //  T == total integration time
        // dT == CFL-check interval
        // dt == DNS time-step
        //  N == T/dT,  n == dT/dt;
        //  T == N dT, dT == n dt
        //  t == s dT (s is loop index)

        os << "f^T: " << flush;
        for (int s = 1; s <= dt.N(); ++s) {
            Real t = s * dt.dT();
            CFL = ilc.CFL(fields[0]);
            if (s % 10 == 0)
                os << iround(t) << flush;
            else if (s % 2 == 0) {
                if (CFL < dt.CFLmin())
                    os << '<' << flush;
                else if (CFL > dt.CFLmax())
                    os << '>' << flush;
                else
                    os << '.' << flush;
            }
            ilc.advance(fields, dt.n());
            if (dt.variable() && dt.adjust(CFL, false))
                ilc.reset_dt(dt);
        }

    }
    // Poincare section computation: return Poincare crossing nearest to t=T, with Tmin < t < Tmax.
    else {
        cout << "Poincare sectioning not yet implemented (markd as experimental)." << endl;
        exit(1);
        /*    // Adjust dt for CFL if necessary
            DNSPoincare dns (f_u, h, flags);
            if (dt.variable()) {
              dns.advance (f_u, p, 1);
              dt.adjust (dns.CFL());
              dns.reset_dt (dt,u);
              f_u = u;
            }
            // Collect all Poincare crossings between Tfudgemin and Tfudgemax
            // If we don't find one in that range, go as far as Tlastchance
            Real dTfudge = 1.05;
            Real Tfudgemin = lesser (0.90*T, T - dTfudge*dt.dT());
            Real Tfudgemax = Greater (1.02*T, T + dTfudge*dt.dT());
            Real Tlastchance = 10*T;

            vector<FlowField> ucross;
            vector<Real> tcross;
            vector<int>  scross;
            int s=0;
            int crosssign = 0; // look for crossings in either direction

            os << "f^t: " << flush;

            for (Real t=0; t<=Tlastchance; t += dt.dT(), ++s) {

              CFL = dns.CFL();

              if (s % 10 == 0)   os << iround (t);
              else if (s % 2 == 0) {
                if (CFL > dt.CFLmax())  os << '>';
                else if (CFL < dt.CFLmin())  os << '<';
                else  os << '.';
                os << flush;
              }

              // Collect any Poincare crossings
              bool crossed = dns.advanceToSection (f_u, p, dt.n(), crosssign, Tfudgemin);
              if (crossed && t >= Tfudgemin) {
                ucross.push_back (dns.ucrossing());
                tcross.push_back (dns.tcrossing());
                scross.push_back (dns.scrossing());
              }

              // If we've found at least one crossing within the fudge range, stop.
              // Otherwise continue trying until Tlastchance
              if (ucross.size() > 0 && t >= Tfudgemax)
                break;
            }

            if (ucross.size() <1) {

              os << "\nError in f(u, T, f_u, flags, dt, fcount, CFL, os) :\n";
              os << "the integration did not reach the Poincare section.\n";
              os << "Returning laminar solution and a b.s. value for the crossing time.\n";
              os << "I hope you can do something useful with them." << endl;
              f_u.setToZero();
              T = dns.time();
              ++fcount;
              return;
            }
            os << "  " << flush;

            // Now select the crossing that is closest to the estimated crossing time
            FlowField ubest = ucross[0];
            Real  Tbest = tcross[0];
            int   sbest = scross[0];
            int   nbest = 0;

            for (uint n=1; n<ucross.size(); ++n) {
              if (abs (tcross[n]-T) < abs (Tbest-T)) {
                ubest = ucross[n];
                Tbest = tcross[n];
                sbest = scross[n];
                nbest = n;
              }
            }
            os << nbest << (sbest==1 ? '+' : '-') << " at t== " << Tbest << flush;

            T = Tbest;
            f_u = ubest;

            // Now check if there are any crossings of opposite sign close by.
            // This signals near-tangency to Poincare section, which'll mess up
            // the search. Just print warning and let user intervene manually.
            Real l2distubestucross = 0;
            for (uint n=0; n<ucross.size(); ++n) {
              l2distubestucross = L2Dist (ubest, ucross[n]);
              if ( (u.taskid() == 0) && (scross[n] != sbest)) {
                os << "\nWARNING : There is a nearby Poincare crossing of opposite sign," << endl;
                os << "signalling near-tangency to section. You should probably switch " << endl;
                os << "to another Poincare crossing." << endl;
                os << "(ubest, unear) signs == " << sbest << ", " << scross[n] << endl;
                os << "(ubest, unear) times == " << Tbest << ", " << tcross[n] << endl;
                os << "(ubest, unear) dist  == " << l2distubestucross << endl;
              }
            }*/
    }

    if (!isfinite(L2Norm(f_u))) {
        os << "error in f: f(u,t) is not finite. exiting." << endl;
        exit(1);
    }
    if (!isfinite(L2Norm(f_temp))) {
        os << "error in f: f(temp,t) is not finite. exiting." << endl;
        exit(1);
    }

    ++fcount;
    f_u = fields[0];
    f_temp = fields[1];
    return;
}

}  // namespace chflow
