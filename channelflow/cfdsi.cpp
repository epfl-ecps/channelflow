/**
 * Channelflow Dynamical System Interface
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "cfdsi.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "cfbasics/mathdefs.h"
#include "channelflow/diffops.h"
#include "channelflow/poissonsolver.h"

using namespace std;
using namespace Eigen;

namespace chflow {

cfDSI::cfDSI() {}

cfDSI::cfDSI(DNSFlags& dnsflags, FieldSymmetry sigma, PoincareCondition* h, TimeStep dt, bool Tsearch, bool xrelative,
             bool zrelative, bool Tnormalize, Real Unormalize, const FlowField& u, ostream* os)
    : DSI(os),
      dnsflags_(dnsflags),
      cfmpi_(u.cfmpi()),
      sigma_(sigma),
      h_(h),
      dt_(dt),
      Tsearch_(Tsearch),
      xrelative_(xrelative),
      zrelative_(zrelative),
      Tinit_(dnsflags.T),
      axinit_(sigma.ax()),
      azinit_(sigma.az()),
      Tnormalize_(Tnormalize),
      Unormalize_(Unormalize),
      fcount_(0),
      Nx_(u.Nx()),
      Ny_(u.Ny()),
      Nz_(u.Nz()),
      Nd_(u.Nd()),
      Lx_(u.Lx()),
      Lz_(u.Lz()),
      ya_(u.a()),
      yb_(u.b()),
      CFL_(0),
      uunk_(u.taskid() == 0 ? xrelative_ + zrelative_ + Tsearch_ : 0) {}

VectorXd cfDSI::eval(const VectorXd& x) {
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    Real T;
    extractVector(x, u, sigma_, T);
    FlowField Gu(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    G(u, T, h_, sigma_, Gu, dnsflags_, dt_, Tnormalize_, Unormalize_, fcount_, CFL_, *os_);
    VectorXd Gx(VectorXd::Zero(x.rows()));
    field2vector(Gu, Gx);  // This does not change the size of Gx and automatically leaves the last entries zero
    return Gx;
}

VectorXd cfDSI::eval(const VectorXd& x0, const VectorXd& x1, bool symopt) {
    FlowField u0(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField u1(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    Real T0, T1;
    FieldSymmetry sigma0, sigma1;
    extractVector(x0, u0, sigma0, T0);
    extractVector(x1, u1, sigma1, T1);

    FlowField Gu(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);

    f(u0, T0, h_, Gu, dnsflags_, dt_, fcount_, CFL_, *os_);
    Real funorm = L2Norm3d(Gu);
    if (symopt)
        Gu *= sigma0;
    Gu -= u1;

    if (Tnormalize_)
        Gu *= 1.0 / T0;
    if (Unormalize_ != 0.0)
        Gu *= 1. / sqrt(abs(funorm * (Unormalize_ - funorm)));

    VectorXd Gx(VectorXd::Zero(x0.rows()));

    field2vector(Gu, Gx);  // This does not change the size of Gx and automatically leaves the last entries zero

    return Gx;
}

void cfDSI::save(const VectorXd& x, const string filebase, const string outdir, const bool fieldsonly) {
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FieldSymmetry sigma;
    Real T;
    extractVector(x, u, sigma, T);
    u.save(outdir + "u" + filebase);
    dnsflags_.T = T;

    if (!fieldsonly) {
        string fs = fieldstats(u);
        if (u.taskid() == 0) {
            if (xrelative_ || zrelative_ || !sigma.isIdentity())
                sigma.save(outdir + "sigma" + filebase);
            if (Tsearch_)
                chflow::save(T, outdir + "T" + filebase);
            ofstream fout((outdir + "fieldconverge.asc").c_str(), ios::app);
            long pos = fout.tellp();
            if (pos == 0)
                fout << fieldstatsheader() << endl;
            fout << fs << endl;
            fout.close();
            dnsflags_.save(outdir);
            // save_sp(T,outdir);
        }
    }
}

void cfDSI::saveEigenvec(const VectorXd& ev, const string label, const string outdir) {
    FlowField ef(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    vector2field(ev, ef);
    ef *= 1.0 / L2Norm(ef);
    ef.save(outdir + "ef" + label);
}

void cfDSI::saveEigenvec(const VectorXd& evA, const VectorXd& evB, const string label1, const string label2,
                         const string outdir) {
    FlowField efA(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField efB(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    vector2field(evA, efA);
    vector2field(evB, efB);
    Real c = 1.0 / sqrt(L2Norm2(efA) + L2Norm2(efB));
    efA *= c;
    efB *= c;
    efA.save(outdir + "ef" + label1);
    efB.save(outdir + "ef" + label2);
}

string cfDSI::stats(const VectorXd& x) {
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    vector2field(x, u);
    return fieldstats_t(u, mu_);
}

pair<string, string> cfDSI::stats_minmax(const VectorXd& x) {
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField Gu(u);
    FieldSymmetry sigma;
    Real T;
    extractVector(x, u, sigma, T);

    vector<Real> stats = fieldstats_vector(u);
    vector<Real> minstats(stats);
    vector<Real> maxstats(stats);

    // quick hack to avoid new interface or creating simple f()
    TimeStep dt = TimeStep(dnsflags_.dt, 0, 1, 1, 0, 0, false);
    int fcount = 0;
    PoincareCondition* h = 0;
    Real CFL = 0.0;
    ostream muted_os(0);
    Real timep = T / 100.0;

    *os_ << "Using flag -orbOut: Calculate minmax-statistics of periodic orbit." << endl;
    for (int t = 0; t < 100; t++) {
        f(u, timep, h, Gu, dnsflags_, dt, fcount, CFL, muted_os);
        stats = fieldstats_vector(u);
        for (uint i = 0; i < stats.size(); i++) {
            minstats[i] = (minstats[i] < stats[i]) ? minstats[i] : stats[i];
            maxstats[i] = (maxstats[i] > stats[i]) ? maxstats[i] : stats[i];
        }
        u = Gu;
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

string cfDSI::statsHeader() { return fieldstatsheader_t(cPar2s(cPar_)); }

void cfDSI::updateMu(Real mu) {
    DSI::updateMu(mu);

    if (cPar_ == continuationParameter::Re) {
        dnsflags_.nu = 1. / mu;
    } else if (cPar_ == continuationParameter::T) {
        Tinit_ = mu;
    } else if (cPar_ == continuationParameter::P) {
        dnsflags_.dPdx = mu;
    } else if (cPar_ == continuationParameter::Ub) {
        dnsflags_.Ubulk = mu;
    } else if (cPar_ == continuationParameter::Uw) {
        dnsflags_.ulowerwall = -mu * cos(dnsflags_.theta);
        dnsflags_.uupperwall = mu * cos(dnsflags_.theta);
        dnsflags_.wlowerwall = -mu * sin(dnsflags_.theta);
        dnsflags_.wupperwall = mu * sin(dnsflags_.theta);
        ;
    } else if (cPar_ == continuationParameter::ReP) {
        Real ratio = 1 / (mu * dnsflags_.nu);  // (Re_old/Re_new), Re_new = mu
        dnsflags_.nu = 1. / mu;
        dnsflags_.dPdx *= ratio;
    } else if (cPar_ == continuationParameter::Theta) {
        dnsflags_.theta = mu;
        dnsflags_.ulowerwall = -dnsflags_.Uwall * cos(mu);
        dnsflags_.uupperwall = dnsflags_.Uwall * cos(mu);
        dnsflags_.wlowerwall = -dnsflags_.Uwall * sin(mu);
        dnsflags_.wupperwall = dnsflags_.Uwall * sin(mu);
        ;
    } else if (cPar_ == continuationParameter::ThArc) {
        Real xleg = Lz_ / tan(dnsflags_.theta);  // hypothetical Lx
        Lz_ *= sin(mu) / sin(dnsflags_.theta);   // rescale Lz for new angle at const diagonal (of xleg x Lz_)
        Lx_ *= Lz_ / tan(mu) / xleg;
        dnsflags_.theta = mu;
        dnsflags_.ulowerwall = -dnsflags_.Uwall * cos(mu);
        dnsflags_.uupperwall = dnsflags_.Uwall * cos(mu);
        dnsflags_.wlowerwall = -dnsflags_.Uwall * sin(mu);
        dnsflags_.wupperwall = dnsflags_.Uwall * sin(mu);
        ;
    } else if (cPar_ == continuationParameter::ThLx) {
        Lx_ *= tan(dnsflags_.theta) / tan(mu);
        dnsflags_.theta = mu;
        dnsflags_.ulowerwall = -dnsflags_.Uwall * cos(mu);
        dnsflags_.uupperwall = dnsflags_.Uwall * cos(mu);
        dnsflags_.wlowerwall = -dnsflags_.Uwall * sin(mu);
        dnsflags_.wupperwall = dnsflags_.Uwall * sin(mu);
        ;
    } else if (cPar_ == continuationParameter::ThLz) {
        Lz_ *= tan(mu) / tan(dnsflags_.theta);
        dnsflags_.theta = mu;
        dnsflags_.ulowerwall = -dnsflags_.Uwall * cos(mu);
        dnsflags_.uupperwall = dnsflags_.Uwall * cos(mu);
        dnsflags_.wlowerwall = -dnsflags_.Uwall * sin(mu);
        dnsflags_.wupperwall = dnsflags_.Uwall * sin(mu);
        ;
    } else if (cPar_ == continuationParameter::Lx) {
        Lx_ = mu;
    } else if (cPar_ == continuationParameter::Lz) {
        Lz_ = mu;
    } else if (cPar_ == continuationParameter::Aspect) {
        Real aspect = Lx_ / Lz_;
        Real update = mu / aspect;
        // do only half the adjustment in x, the other in z (i.e. equivalent to Lx_new = 0.5Lx +0.5Lx * update)
        Lx_ *= 1 + (update - 1) / 2.0;
        Lz_ = Lx_ / mu;
    } else if (cPar_ == continuationParameter::Diag) {
        Real aspect = Lx_ / Lz_;
        Real theta = atan(aspect);
        Real diagonal = sqrt(Lx_ * Lx_ + Lz_ * Lz_);
        Real update = mu - diagonal;
        //     Lx_ += sqrt ( (update*update * aspect*aspect) / (1 + aspect*aspect));
        Lx_ += update * sin(theta);
        Lz_ = Lx_ / aspect;
    } else if (cPar_ == continuationParameter::Lt) {
        cferror("continuationParameter::Lt not implemented");
        //     Real Lxdist = Lxtarg_ - Lx_, Lzdist = Lztarg_ - Lz_;
        //     Real dist = sqrt (Lxdist*Lxdist + Lzdist*Lzdist);
        //     Real diff = dist - mu;
        //     Lx_ += diff/dist * Lxdist;
        //     Lz_ += diff/dist * Lzdist;
    } else if (cPar_ == continuationParameter::Vs) {
        dnsflags_.Vsuck = mu;
    } else if (cPar_ == continuationParameter::ReVs) {
        dnsflags_.nu = 1. / mu;
        dnsflags_.Vsuck = 1. / mu;
    } else if (cPar_ == continuationParameter::H) {
        ya_ = 0;
        yb_ = mu;
    } else if (cPar_ == continuationParameter::HVs) {
        Real nu = dnsflags_.nu;
        Real Vs = nu * (1 - exp(-mu));
        Real H = nu * mu / Vs;
        ya_ = 0;
        yb_ = H;
        dnsflags_.Vsuck = Vs;
    } else if (cPar_ == continuationParameter::Rot) {
        dnsflags_.rotation = mu;
    } else {
        throw invalid_argument("cfDSI::updateMu(): continuation parameter is unknown");
    }
}

void cfDSI::chooseMu(string muName) { chooseMu(s2cPar(muName)); }

void cfDSI::chooseMu(continuationParameter mu) {
    cPar_ = mu;
    switch (mu) {
        case continuationParameter::Re:
            updateMu(1. / dnsflags_.nu);
            break;
        case continuationParameter::T:
            updateMu(Tinit_);
            break;
        case continuationParameter::P:
            updateMu(dnsflags_.dPdx);
            break;
        case continuationParameter::Ub:
            updateMu(dnsflags_.Ubulk);
            break;
        case continuationParameter::Uw:
            updateMu(dnsflags_.uupperwall / cos(dnsflags_.theta));
            break;
        case continuationParameter::ReP:
            updateMu(1. / dnsflags_.nu);
            break;
        case continuationParameter::Theta:
            updateMu(dnsflags_.theta);
            break;
        case continuationParameter::ThArc:
            updateMu(dnsflags_.theta);
            break;
        case continuationParameter::ThLx:
            updateMu(dnsflags_.theta);
            break;
        case continuationParameter::ThLz:
            updateMu(dnsflags_.theta);
            break;
        case continuationParameter::Lx:
            updateMu(Lx_);
            break;
        case continuationParameter::Lz:
            updateMu(Lz_);
            break;
        case continuationParameter::Aspect:
            updateMu(Lx_ / Lz_);
            break;
        case continuationParameter::Diag:
            updateMu(sqrt(Lx_ * Lx_ + Lz_ * Lz_));
            break;
        case continuationParameter::Lt:
            cferror("continuationParameter::Lt not implemented");
            // updateMu (0);
            break;
        case continuationParameter::Vs:
            updateMu(dnsflags_.Vsuck);
            break;
        case continuationParameter::ReVs:
            updateMu(1. / dnsflags_.nu);
            break;
        case continuationParameter::H:
            updateMu(yb_ - ya_);
            break;
        case continuationParameter::HVs:
            updateMu(-log(1 - dnsflags_.Vsuck / dnsflags_.nu));
            break;
        case continuationParameter::Rot:
            updateMu(dnsflags_.rotation);
            break;
        default:
            throw invalid_argument("cfDSI::chooseMu(): continuation parameter is unknown");
    }
}

continuationParameter cfDSI::s2cPar(string muname) {
    std::transform(muname.begin(), muname.end(), muname.begin(), ::tolower);
    if (muname == "re")
        return continuationParameter::Re;
    if (muname == "T")
        return continuationParameter::T;
    else if (muname == "p")
        return continuationParameter::P;
    else if (muname == "ub")
        return continuationParameter::Ub;
    else if (muname == "uw")
        return continuationParameter::Uw;
    else if (muname == "rep")
        return continuationParameter::ReP;
    else if (muname == "theta")
        return continuationParameter::Theta;
    else if (muname == "tharc")
        return continuationParameter::ThArc;
    else if (muname == "thlx")
        return continuationParameter::ThLx;
    else if (muname == "thlz")
        return continuationParameter::ThLz;
    else if (muname == "lx")
        return continuationParameter::Lx;
    else if (muname == "lz")
        return continuationParameter::Lz;
    else if (muname == "aspect")
        return continuationParameter::Aspect;
    else if (muname == "diag")
        return continuationParameter::Diag;
    else if (muname == "lt")
        return continuationParameter::Lt;
    else if (muname == "vs")
        return continuationParameter::Vs;
    else if (muname == "revs")
        return continuationParameter::ReVs;
    else if (muname == "h")
        return continuationParameter::H;
    else if (muname == "hvs")
        return continuationParameter::HVs;
    else if (muname == "rot")
        return continuationParameter::Rot;
    else
        throw invalid_argument("cfDSI::s2cPar(): continuation parameter '" + muname + "' is unknown");
}

string cfDSI::printMu() { return cPar2s(cPar_); }

string cfDSI::cPar2s(continuationParameter cPar) {
    if (cPar == continuationParameter::Re)
        return "Re";
    else if (cPar == continuationParameter::P)
        return "P";
    else if (cPar == continuationParameter::Ub)
        return "Ub";
    else if (cPar == continuationParameter::Uw)
        return "uw";
    else if (cPar == continuationParameter::ReP)
        return "ReP";
    else if (cPar == continuationParameter::Theta)
        return "Theta";
    else if (cPar == continuationParameter::ThArc)
        return "Theta(DiagArc)";
    else if (cPar == continuationParameter::ThLx)
        return "Theta(Lx)";
    else if (cPar == continuationParameter::ThLz)
        return "Theta(Lz)";
    else if (cPar == continuationParameter::Lx)
        return "Lx";
    else if (cPar == continuationParameter::Lz)
        return "Lz";
    else if (cPar == continuationParameter::Aspect)
        return "Aspect";
    else if (cPar == continuationParameter::Diag)
        return "Diag";
    else if (cPar == continuationParameter::Lt)
        return "Lt";
    else if (cPar == continuationParameter::Vs)
        return "Vs";
    else if (cPar == continuationParameter::ReVs)
        return "ReVs";
    else if (cPar == continuationParameter::H)
        return "H";
    else if (cPar == continuationParameter::HVs)
        return "HVs";
    else if (cPar == continuationParameter::Rot)
        return "Rot";
    else
        throw invalid_argument("cfDSI::cPar2s(): continuation parameter is not convertible to string");
}

void cfDSI::saveParameters(string searchdir) { dnsflags_.save(searchdir); }

/// after finding new solution fix phases
void cfDSI::phaseShift(VectorXd& x) {
    FlowField unew(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FieldSymmetry sigma;
    Real T;
    extractVector(x, unew, sigma, T);
    // vector2field (x,unew);
    const int phasehackcoord = 0;  // Those values were fixed in continuesoln anyway
    const parity phasehackparity = Odd;
    const Real phasehackguess = 0.0;

    if (zphasehack_) {
        FieldSymmetry tau = zfixphasehack(unew, phasehackguess, phasehackcoord, phasehackparity);
        cout << "fixing z phase of potential solution with phase shift tau == " << tau << endl;
        unew *= tau;
    }
    if (xphasehack_) {
        FieldSymmetry tau = xfixphasehack(unew, phasehackguess, phasehackcoord, phasehackparity);
        cout << "fixing x phase of potential solution with phase shift tau == " << tau << endl;
        unew *= tau;
    }
    if (uUbasehack_) {
        cout
            << "fixing u+Ubase decomposition so that <du/dy> = 0 at walls (i.e. Ubase balances mean pressure gradient))"
            << endl;
        Real ubulk = Re(unew.profile(0, 0, 0)).mean();
        if (abs(ubulk) < 1e-15)
            ubulk = 0.0;

        ChebyCoeff Ubase =
            laminarProfile(dnsflags_.nu, dnsflags_.constraint, dnsflags_.dPdx, dnsflags_.Ubulk - ubulk, dnsflags_.Vsuck,
                           unew.a(), unew.b(), dnsflags_.ulowerwall, dnsflags_.uupperwall, unew.Ny());

        fixuUbasehack(unew, Ubase);
    }
    makeVector(unew, sigma, T, x);
}

void cfDSI::phaseShift(MatrixXd& y) {
    FlowField unew(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    VectorXd yvec;
    FieldSymmetry sigma;
    Real T;

    const int phasehackcoord = 0;  // Those values were fixed in continuesoln anyway
    const parity phasehackparity = Odd;
    const Real phasehackguess = 0.0;

    FieldSymmetry taux(0.0, 0.0);
    FieldSymmetry tauz(0.0, 0.0);

    extractVector(y.col(0), unew, sigma, T);

    if (xphasehack_) {
        taux = xfixphasehack(unew, phasehackguess, phasehackcoord, phasehackparity);
        cout << "fixing x phase of potential solution with phase shift tau == " << taux << endl;
    }
    if (zphasehack_) {
        tauz = zfixphasehack(unew, phasehackguess, phasehackcoord, phasehackparity);
        cout << "fixing z phase of potential solution with phase shift tau == " << tauz << endl;
    }

    for (int i = 0; i < y.cols(); i++) {
        extractVector(y.col(i), unew, sigma, T);
        unew *= taux;
        unew *= tauz;
        makeVector(unew, sigma, T, yvec);
        y.col(i) = yvec;
    }
}

Real cfDSI::DSIL2Norm(const VectorXd& x) {
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    vector2field(x, u);
    return L2Norm(u);
}

void cfDSI::makeVector(const FlowField& u, const FieldSymmetry& sigma, const Real T, VectorXd& x) {
    int taskid = u.taskid();
    int uunk = field2vector_size(u);                         // # of variables for u unknonwn
    const int Tunk = (Tsearch_ && taskid == 0) ? uunk : -1;  // index for T unknown
    const int xunk = (xrelative_ && taskid == 0) ? uunk + Tsearch_ : -1;
    const int zunk = (zrelative_ && taskid == 0) ? uunk + Tsearch_ + xrelative_ : -1;
    int Nunk = (taskid == 0) ? uunk + Tsearch_ + xrelative_ + zrelative_ : uunk;

    //   VectorXd x(Nunk);
    if (x.rows() < Nunk)
        x.resize(Nunk);
    field2vector(u, x);

    if (taskid == 0) {
        if (Tsearch_)
            x(Tunk) = T;
        if (xrelative_)
            x(xunk) = sigma.ax();
        if (zrelative_)
            x(zunk) = sigma.az();
    }
}

void cfDSI::extractVector(const VectorXd& x, FlowField& u, FieldSymmetry& sigma, Real& T) {
    int uunk = field2vector_size(u);  // Important for arclength method

    vector2field(x, u);
    const int Tunk = uunk + Tsearch_ - 1;
    const int xunk = uunk + Tsearch_ + xrelative_ - 1;
    const int zunk = uunk + Tsearch_ + xrelative_ + zrelative_ - 1;
    Real ax, az;

#ifdef HAVE_MPI
    if (u.taskid() == 0) {
#else
    {
#endif
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

Real cfDSI::extractT(const VectorXd& x) {
    Real Tvec;
    FieldSymmetry sigma;
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    extractVector(x, u, sigma, Tvec);
    return Tvec;
}

Real cfDSI::extractXshift(const VectorXd& x) {
    Real Tvec;
    FieldSymmetry sigma;
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    extractVector(x, u, sigma, Tvec);
    return sigma.ax();
}

Real cfDSI::extractZshift(const VectorXd& x) {
    Real Tvec;
    FieldSymmetry sigma;
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    extractVector(x, u, sigma, Tvec);
    return sigma.az();
}

VectorXd cfDSI::xdiff(const VectorXd& a) {
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    vector2field(a, u);
    VectorXd dadx(a.size());
    dadx.setZero();
    u = chflow::xdiff(u);
    field2vector(u, dadx);
    dadx *= 1. / L2Norm(dadx);
    return dadx;
}

VectorXd cfDSI::zdiff(const VectorXd& a) {
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    vector2field(a, u);
    VectorXd dadx(a.size());
    dadx.setZero();
    u = chflow::zdiff(u);
    field2vector(u, dadx);
    dadx *= 1. / L2Norm(dadx);
    return dadx;
}

VectorXd cfDSI::tdiff(const VectorXd& a, Real epsDt) {
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    FlowField edudtf(u);
    vector2field(a, u);
    f(u, epsDt, 0, edudtf, dnsflags_, dt_, fcount_, CFL_, *os_);
    edudtf -= u;
    VectorXd dadt(a.size());
    field2vector(edudtf, dadt);
    dadt *= 1. / L2Norm(dadt);
    return dadt;
}

Real cfDSI::observable(VectorXd& x) {
    printout("computing mean dissipation", false);
    FlowField uarg(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    Real T;
    extractVector(x, uarg, sigma_, T);

    // just the current dissipation, was :
    // obs = meandissipation (u, T, dnsflags_, dt_, solntype);
    FlowField u(uarg);

    ChebyCoeff Ubase(u.Ny(), u.a(), u.b());
    if (dnsflags_.baseflow == LinearBase)
        Ubase[1] = 1;
    else if (dnsflags_.baseflow == ParabolicBase) {
        Ubase[0] = 0.5;
        Ubase[2] = -0.5;
    } else if (dnsflags_.baseflow == LaminarBase) {
        Real ubulk = Re(uarg.profile(0, 0, 0)).mean();
        if (abs(ubulk) < 1e-15)
            ubulk = 0.0;
        Ubase =
            laminarProfile(dnsflags_.nu, dnsflags_.constraint, dnsflags_.dPdx, dnsflags_.Ubulk - ubulk, dnsflags_.Vsuck,
                           uarg.a(), uarg.b(), dnsflags_.ulowerwall, dnsflags_.uupperwall, uarg.Ny());
    }
    printout(" of EQB/TW...", false);
    u += Ubase;
    printout("done");
    return dissipation(u);
}

Real cfDSI::tph_observable(VectorXd& x) {
    FlowField u(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    Real T;
    extractVector(x, u, sigma_, T);
    return Ecf(u);
}

// return f^{N dt}(u) = time-(N dt) DNS integration of u
void f(const FlowField& u, int N, Real dt, FlowField& f_u, const DNSFlags& flags_, ostream& os) {
    os << "f(u, N, dt, f_u, flags, dt) : " << flush;
    DNSFlags flags(flags_);
    flags.logstream = &os;
    vector<FlowField> fields(2);
    fields[0] = u;
    DNS dns(fields, flags);
    dns.advance(fields, N);
    f_u = fields[0];
    return;
}

// G(x) = G(u,sigma) = (sigma f^T(u) - u) for orbits
void G(const FlowField& u, Real& T, PoincareCondition* h, const FieldSymmetry& sigma, FlowField& Gu,
       const DNSFlags& flags, const TimeStep& dt, bool Tnormalize, Real Unormalize, int& fcount, Real& CFL,
       ostream& os) {
    f(u, T, h, Gu, flags, dt, fcount, CFL, os);
    Real funorm = L2Norm3d(Gu);
    Gu *= sigma;
    Gu -= u;
    if (Tnormalize)
        Gu *= 1.0 / T;
    if (Unormalize != 0.0) {
        Gu *= 1. / sqrt(abs(funorm * (Unormalize - funorm)));
    }
}

void f(const FlowField& u, Real& T, PoincareCondition* h, FlowField& f_u, const DNSFlags& flags_, const TimeStep& dt_,
       int& fcount, Real& CFL, ostream& os) {
    if (!isfinite(L2Norm(u))) {
        os << "error in f: u is not finite. exiting." << endl;
        exit(1);
    }

    DNSFlags flags(flags_);
    flags.logstream = &os;
    TimeStep dt(dt_);
    vector<FlowField> fields = {u, FlowField(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi())};
    f_u = u;
    // No Poincare section, just integration to time T
    if (h == 0) {
        if (T < 0) {
            os << "f: negative integration time T == " << T << endl << "returning f(u,T) == (1+abs(T))*u" << endl;
            f_u *= 1 + abs(T);
            return;
        }
        // Special case #1: no time steps
        if (T == 0) {
            os << "f: T==0, no integration, returning u " << endl;
            return;
        }
        dt.adjust_for_T(T, false);
        flags.dt = dt;
        // Adjust dt for CFL if necessary
        DNS dns(fields, flags);
        if (dt.variable()) {
            dns.advance(fields, 1);
            dt.adjust(dns.CFL(fields[0]), false);
            dns.reset_dt(dt);
            fields[0] = u;
        }

        //  t == current time in integration
        //  T == total integration time
        // dT == CFL-check interval
        // dt == DNS time-stepn
        //  N == T/dT,  n == dT/dt;
        //  T == N dT, dT == n dt
        //  t == s dT (s is loop index)

        os << "f^T: " << flush;
        for (int s = 1; s <= dt.N(); ++s) {
            Real t = s * dt.dT();
            CFL = dns.CFL(fields[0]);

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

            dns.advance(fields, dt.n());

            if (dt.variable() && dt.adjust(CFL, false))
                dns.reset_dt(dt);
        }
        double l2normfu = L2Norm(fields[0]);
        if (!isfinite(l2normfu)) {
            os << "L2Norm(f^t(u)) == " << l2normfu << endl;
            os << "error in f: f^t(u) is not finite. exiting." << endl;
            exit(1);
        }
    }
    // Poincare section computation: return Poincare crossing nearest to t=T, with Tmin < t < Tmax.
    else {
        // Adjust dt for CFL if necessary
        DNSPoincare dns(fields[0], h, flags);
        if (dt.variable()) {
            dns.advance(fields, 1);
            dt.adjust(dns.CFL(fields[0]));
            dns.reset_dt(dt);
        }
        // Collect all Poincare crossings between Tfudgemin and Tfudgemax
        // If we don't find one in that range, go as far as Tlastchance
        Real dTfudge = 1.05;
        Real Tfudgemin = lesser(0.90 * T, T - dTfudge * dt.dT());
        Real Tfudgemax = Greater(1.02 * T, T + dTfudge * dt.dT());
        Real Tlastchance = 10 * T;

        vector<FlowField> ucross;
        vector<Real> tcross;
        vector<int> scross;
        int s = 0;
        int crosssign = 0;  // look for crossings in either direction

        os << "f^t: " << flush;

        for (Real t = 0; t <= Tlastchance; t += dt.dT(), ++s) {
            CFL = dns.CFL(fields[0]);

            if (s % 10 == 0)
                os << iround(t);
            else if (s % 2 == 0) {
                if (CFL > dt.CFLmax())
                    os << '>';
                else if (CFL < dt.CFLmin())
                    os << '<';
                else
                    os << '.';
                os << flush;
            }

            // Collect any Poincare crossings
            bool crossed = dns.advanceToSection(fields[0], fields[1], dt.n(), crosssign, Tfudgemin);
            if (crossed && t >= Tfudgemin) {
                ucross.push_back(dns.ucrossing());
                tcross.push_back(dns.tcrossing());
                scross.push_back(dns.scrossing());
            }

            // If we've found at least one crossing within the fudge range, stop.
            // Otherwise continue trying until Tlastchance
            if (ucross.size() > 0 && t >= Tfudgemax)
                break;
        }

        if (ucross.size() < 1) {
            os << "\nError in f(u, T, f_u, flags, dt, fcount, CFL, os) :\n";
            os << "the integration did not reach the Poincare section.\n";
            os << "Returning laminar solution and a b.s. value for the crossing time.\n";
            os << "I hope you can do something useful with them." << endl;
            fields[0].setToZero();
            T = dns.time();
            ++fcount;
            return;
        }
        os << "  " << flush;

        // Now select the crossing that is closest to the estimated crossing time
        FlowField ubest = ucross[0];
        Real Tbest = tcross[0];
        int sbest = scross[0];
        int nbest = 0;

        for (uint n = 1; n < ucross.size(); ++n) {
            if (abs(tcross[n] - T) < abs(Tbest - T)) {
                ubest = ucross[n];
                Tbest = tcross[n];
                sbest = scross[n];
                nbest = n;
            }
        }
        os << nbest << (sbest == 1 ? '+' : '-') << " at t== " << Tbest << flush;

        T = Tbest;
        fields[0] = ubest;

        // Now check if there are any crossings of opposite sign close by.
        // This signals near-tangency to Poincare section, which'll mess up
        // the search. Just print warning and let user intervene manually.
        Real l2distubestucross = 0;
        for (uint n = 0; n < ucross.size(); ++n) {
            l2distubestucross = L2Dist(ubest, ucross[n]);
            if ((u.taskid() == 0) && (scross[n] != sbest)) {
                os << "\nWARNING : There is a nearby Poincare crossing of opposite sign," << endl;
                os << "signalling near-tangency to section. You should probably switch " << endl;
                os << "to another Poincare crossing." << endl;
                os << "(ubest, unear) signs == " << sbest << ", " << scross[n] << endl;
                os << "(ubest, unear) times == " << Tbest << ", " << tcross[n] << endl;
                os << "(ubest, unear) dist  == " << l2distubestucross << endl;
            }
        }
    }
    if (!isfinite(L2Norm(fields[0]))) {
        os << "error in f: f(u,t) is not finite. exiting." << endl;
        exit(1);
    }
    ++fcount;
    f_u = fields[0];
    return;
}

// Is this function maybe obsolete? (FR)
Real GMRESHookstep_vector(FlowField& u, Real& T, FieldSymmetry& sigma, PoincareCondition* hpoincare,
                          const NewtonSearchFlags& searchflags, DNSFlags& dnsflags, TimeStep& dt, Real& CFL,
                          Real Unormalize) {
    Real residual = 0;
    ostream* os = searchflags.logstream;  // a short name for ease of use
    project(dnsflags.symmetries, u, "initial guess u", *os);
    fixdivnoslip(u);

    bool Tsearch = (searchflags.solntype == PeriodicOrbit && hpoincare == 0) ? true : false;
    const bool Tnormalize = (searchflags.solntype == PeriodicOrbit) ? false : true;
    cfDSI Gx(dnsflags, sigma, hpoincare, dt, Tsearch, searchflags.xrelative, searchflags.zrelative, Tnormalize,
             Unormalize, u, os);

    VectorXd x;
    Gx.makeVector(u, sigma, T, x);
    int Nunk = x.rows();
    int Nunk_total = Nunk;
#ifdef HAVE_MPI
    MPI_Allreduce(&Nunk, &Nunk_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    *os << Nunk_total << " unknowns" << endl;

    //********************************* Do search and deal with results *********************************************//
    VectorXd SearchResult = hookstepSearch(Gx, x, searchflags, residual);

    Gx.extractVector(SearchResult, u, sigma, T);

    u.save(searchflags.outdir + "xbest");

    if (u.taskid() == 0) {
        sigma.save(searchflags.outdir + "sigmabest");
        save(T, searchflags.outdir + "Tbest");
    }
    return residual;
}

vector<Real> fieldstats_vector(const FlowField& u) {
    vector<Real> stats;
    string temp = fieldstats(u);
    stringstream ss(temp);
    int stats_size = 0;
    for (int i = 0; ss.tellg() != -1; i++) {
        stats_size = i + 1;
        stats.resize(stats_size);
        ss >> stats[i];
    }
    return stats;
}

}  // namespace chflow
