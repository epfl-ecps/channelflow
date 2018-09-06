/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include "channelflow/laurettedsi.h"
#include <memory>

using namespace std;
using namespace Eigen;

namespace chflow {

LauretteDSI::LauretteDSI(FlowField& u, DNSFlags& flags, Real dt, bool xrel, bool zrel, FieldSymmetry sigma)
    : cfDSI(flags, sigma, NULL, TimeStep(), false, xrel, zrel, false, false, u, &cout),
      fieldst_(0),
      fieldsdt_(0),
      U_(u),
      //   p(fields[1]),
      dt_(dt) {
    FlowField p(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi(), Spectral, Spectral);
    fieldst_ = {u, p};
    fieldsdt_ = {u, p};

    dnsflags_.dt = dt_;

    ChebyCoeff Ubase(laminarProfile(dnsflags_.nu, dnsflags_.constraint, dnsflags_.dPdx, dnsflags_.Ubulk,
                                    dnsflags_.Vsuck, fieldsdt_[0].a(), fieldsdt_[0].b(), dnsflags_.ulowerwall,
                                    dnsflags_.uupperwall, fieldsdt_[0].Ny()));
    ChebyCoeff Wbase(laminarProfile(dnsflags_.nu, dnsflags_.constraint, dnsflags_.dPdz, dnsflags_.Wbulk,
                                    dnsflags_.Vsuck, fieldsdt_[0].a(), fieldsdt_[0].b(), dnsflags_.wlowerwall,
                                    dnsflags_.wupperwall, fieldsdt_[0].Ny()));

    //   vector<FlowField> fields={udt_,p};
    vector<ChebyCoeff> base = {Ubase, Wbase};
    nse = shared_ptr<NSE>(new NSE(fieldsdt_, base, dnsflags_));
    alg = unique_ptr<EulerDNS>(new EulerDNS(fieldsdt_, nse, dnsflags_));
}

LauretteDSI::~LauretteDSI() {
    if (nse.unique())
        nse.reset();
    else if (nse) {
        cerr << "Destruction of DNS does not destroy the instance of MAIN_NSE which it created.\n"
             << "Identify object which still holds copy of shared_ptr<NSE>." << endl;
    } else {
    }
}

Real LauretteDSI::shift2speed(Real ax) { return -ax / Tinit_ * fieldst_[0].Lx(); }

VectorXd LauretteDSI::eval(const VectorXd& x) {
    int uunk = field2vector_size(fieldsdt_[0]);

    vector2field(x, fieldst_[0]);

    Real Cx = 0;
    if (xrelative_) {
        Real ax = (mpirank() == 0) ? x(uunk) : 0;
#ifdef HAVE_MPI
        MPI_Bcast(&ax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
        Cx = shift2speed(ax);
    }
    fieldsdt_[0] = fieldst_[0];
    alg->advance(fieldsdt_, 1, fieldsdt_[0], false, Cx, 0, false);
    fieldsdt_[0] -= fieldst_[0];

    VectorXd result(VectorXd::Zero(x.size()));
    field2vector(fieldsdt_[0], result);
    return result;
}

VectorXd LauretteDSI::Jacobian(const VectorXd& x, const VectorXd& dx, const VectorXd& Gx, const Real& epsDx,
                               bool centdiff, int& fcount) {
    int uunk = field2vector_size(fieldsdt_[0]);
    Real Cx = 0, cx = 0;
    int Nx = x.size();
    if (xrelative_) {
        int xunk = uunk;
        Real dax = (mpirank() == 0) ? dx(xunk) : 0;
        Real ax = (mpirank() == 0) ? x(xunk) : 0;
#ifdef HAVE_MPI
        MPI_Bcast(&dax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
        Cx = shift2speed(ax);
        cx = -dax / Tinit_ * fieldst_[0].Lx();
    }

    vector2field(x, U_);
    vector2field(dx, fieldst_[0]);
    fieldsdt_[0] = fieldst_[0];
    alg->advance(fieldsdt_, 1, U_, true, Cx, cx, false);
    fieldsdt_[0] -= fieldst_[0];

    VectorXd xt(VectorXd::Zero(Nx));
    field2vector(fieldsdt_[0], xt);

    fcount++;
    return xt;
}

void LauretteDSI::updateMu(Real mu) {
    cfDSI::updateMu(mu);  // this updates dnsflags
    ChebyCoeff Ubase(laminarProfile(dnsflags_.nu, dnsflags_.constraint, dnsflags_.dPdx, dnsflags_.Ubulk,
                                    dnsflags_.Vsuck, fieldsdt_[0].a(), fieldsdt_[0].b(), dnsflags_.ulowerwall,
                                    dnsflags_.uupperwall, fieldsdt_[0].Ny()));
    ChebyCoeff Wbase(laminarProfile(dnsflags_.nu, dnsflags_.constraint, dnsflags_.dPdz, dnsflags_.Wbulk,
                                    dnsflags_.Vsuck, fieldsdt_[0].a(), fieldsdt_[0].b(), dnsflags_.wlowerwall,
                                    dnsflags_.wupperwall, fieldsdt_[0].Ny()));

    // Adapt FlowFields to new size if the size changed
    fieldsdt_[0] = FlowField(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    fieldst_[0] = FlowField(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    U_ = FlowField(Nx_, Ny_, Nz_, Nd_, Lx_, Lz_, ya_, yb_, cfmpi_);
    fieldsdt_[1] = FlowField(Nx_, Ny_, Nz_, 1, Lx_, Lz_, ya_, yb_, cfmpi_);

    //   vector<FlowField> fields={udt_,p};
    vector<ChebyCoeff> base = {Ubase, Wbase};
    nse.reset(new NSE(fieldsdt_, base, dnsflags_));
    alg.reset(new EulerDNS(fieldsdt_, nse, dnsflags_));  // delete old EulerDNS and construct a new one
}

void rotatereal(FlowField& u, int cxmx, int cxmy, int cxmz, int cxi) {
    Real phi = atan(u.cmplx(cxmx, cxmy, cxmz, cxi).imag() / u.cmplx(cxmx, cxmy, cxmz, cxi).real()) / cxmx;

    Complex factor = 0;

    int Mz = u.Mz();
    int Mx = u.Mx();
    int My = u.My();

    for (int mx = 0; mx < Mx; ++mx) {
        factor = exp(-I * phi * Real(u.kx(mx)));
        for (int mz = 0; mz < Mz; ++mz) {
            for (int my = 0; my < My; ++my) {
                for (int i = 0; i < 3; ++i) {
                    u.cmplx(mx, my, mz, i) *= factor;
                }
            }
        }
    }
}

EulerDNS::EulerDNS() : MultistepDNS() {}

EulerDNS::EulerDNS(const EulerDNS& dns)
    : MultistepDNS(dns)
// copy all internal variables here
{
    // Copy tausolvers is not needed here anymore.
    // Done in DNSAlgorithm constructor by copying nse object.
}

EulerDNS& EulerDNS::operator=(const EulerDNS& dns) {
    cerr << "EulerDNS::operator=(const EulerDNS& dns) unimplemented\n";
    exit(1);
}

EulerDNS::EulerDNS(const vector<FlowField>& fields, const shared_ptr<NSE>& nse, const DNSFlags& flags)

    : MultistepDNS(fields, nse, flags) {
    TimeStepMethod algorithm = flags.timestepping;
    switch (algorithm) {
        case SBDF1:  // old version used redundant FEBE
            //       coefficients are already set in parent class
            break;
        default:
            cerr << "EulerDNS::EulerDNS(un,Ubase,nu,dt,flags,t0)\n"
                 << "error: flags.timestepping == " << algorithm << "is not the Euler algorithm" << endl;
            exit(1);
    }
    // Configure tausolvers is not needed here anymore. -> MultistepDNS constructor
}

EulerDNS::~EulerDNS() {
    //   cerr << "~EulerDNS" << endl;
}

DNSAlgorithm* EulerDNS::clone(const shared_ptr<NSE>& nse) const {
    DNSAlgorithm* clone = new EulerDNS(*this);
    clone->reset_nse(nse);
    return clone;
}

// Basically a copy of SBDF1, but with linearization about linearU if linearize is true
void EulerDNS::advance(vector<FlowField>& fieldsn, int Nsteps, FlowField& linearU, bool linearize, Real Cx, Real cx,
                       bool quad) {
    const int J = order_ - 1;
    vector<FlowField> rhs(nse_->createRHS(fieldsn));  // Number of fields and number of RHS's can be different
    int len = rhs.size();
    fields_[0] = fieldsn;
    // Start of time stepping loop
    for (int step = 0; step < Nsteps; ++step) {
        //>>> new part for nonlinear term
        FlowField dudx(fields_[0][0]);

        if (linearize) {
            // use dudx as tmp_
            navierstokesNL_linearU(fields_[0][0], linearU, nse_->Ubase(), nse_->Wbase(), nonlf_[0][0], dudx, flags_);
            xdiff(linearU, dudx);
            dudx *= cx;
            nonlf_[0][0] += dudx;
        } else {
            nse_->nonlinear(fields_[0], nonlf_[0]);
        }

        xdiff(fields_[0][0], dudx);
        dudx *= Cx;
        nonlf_[0][0] += dudx;

        // Add up multistepping terms of linear and nonlinear terms
        for (int l = 0; l < len; ++l) {
            rhs[l].setToZero();  // RHS must be zero before sum over multistep loop
            for (int j = 0; j < order_; ++j) {
                const Real a = -alpha_[j] / flags_.dt;
                const Real b = -beta_[j];
                rhs[l].add(a, fields_[j][l]);
                rhs[l].add(b, nonlf_[j][l]);
            }
        }

        // Solve the implicit problem
        nse_->solve(fields_[J], rhs);
        // The solution is currently stored in fields_[J]. Shift entire fields and nonlf vectors
        // to move it into fields_[0]. Ie shift fields_[J] <- fields_[J-1] <- ... <- fields_[0] <- fields_[J]
        for (int j = order_ - 1; j > 0; --j) {
            for (int l = 0; l < numfields_; ++l) {
                swap(nonlf_[j][l], nonlf_[j - 1][l]);
                swap(fields_[j][l], fields_[j - 1][l]);
            }
        }
        t_ += flags_.dt;

        // printStack();
        //*flags_.logstream << "} Multistep::advance(...) step == " << step << " }" <<endl;
        if (nse_->taskid() == 0) {
            if (flags_.verbosity == PrintTime || flags_.verbosity == PrintAll)
                *flags_.logstream << t_ << ' ' << flush;
            else if (flags_.verbosity == PrintTicks)
                *flags_.logstream << '.' << flush;
        }
    }  // End of time stepping loop

    fieldsn = fields_[0];  // update velocity

    //   cfl_ = u_[0].CFLfactor (Ubase_, Wbase_);				//calculated on demand through DNS class
    //   cfl_ *= flags_.dealias_xz() ? 2.0*pi/3.0*flags_.dt : pi*flags_.dt;

    if (nse_->taskid() == 0)
        if (flags_.verbosity == PrintTime || flags_.verbosity == PrintAll || flags_.verbosity == PrintTicks)
            *flags_.logstream << endl;

    return;
}

void navierstokesNL_linearU(const FlowField& u_, const FlowField& U_, const ChebyCoeff Ubase, const ChebyCoeff Wbase,
                            FlowField& f, FlowField& tmp, DNSFlags& flags, bool lauretteHack) {
    FlowField& u = (FlowField&)u_;
    FlowField& U = (FlowField&)U_;
    assert(u_.xzstate() == Spectral && u_.ystate() == Spectral);
    assert(Ubase.state() == Spectral);

    U += Ubase;
    U += Wbase;
    if (u.taskid() == u.task_coeff(0, 0))
        u.cmplx(0, 0, 0, 1) -= Complex(flags.Vsuck, 0.);

    f.makeSpectral();

    assert(u.Nd() == 3);

    if (!u.geomCongruent(f) || f.Nd() != 3)
        f.resize(u.Nx(), u.Ny(), u.Nz(), 3, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());

    // create a function pointer to the form of the nonlinear term for ease of use
    void (*NSNL)(const FlowField&, FlowField&, FlowField&, fieldstate);

    switch (flags.nonlinearity) {
        case Rotational:
            NSNL = &rotationalNL;
            break;
        case SkewSymmetric:
            NSNL = &skewsymmetricNL;
            break;
        case Divergence:
            NSNL = &divergenceNL;
            break;
        case Convection:
            NSNL = &convectionNL;
            break;
        default:
            cout << "nonlinearity " << flags.nonlinearity << " is currently not implemented in navierstokesNL_linearU"
                 << endl;
            cout << "Falling back to SkewSymmetric" << endl;
            NSNL = &skewsymmetricNL;
            break;
    }

    // Want to calculate (u.V) U + (U.V)u  // V == nabla
    // this is equal to ((u+U).V)(u+U) - (U.V)U - (u.V)u

    FlowField utot(U);
    utot += u;
    NSNL(utot, f, tmp, Spectral);
    FlowField tmpf(f);
    NSNL(u, tmpf, tmp, Spectral);
    f -= tmpf;
    NSNL(U, tmpf, tmp, Spectral);
    f -= tmpf;

    U -= Ubase;
    U -= Wbase;
    if (u.taskid() == u.task_coeff(0, 0))
        u.cmplx(0, 0, 0, 1) += Complex(flags.Vsuck, 0.);
}

VectorXd LauretteDSI::Q(const VectorXd& x) {
    vector2field(x, fieldst_[0]);

    alg->advance(fieldst_, 1, fieldst_[0], false, 0, 0, true);
    //   fieldsdt_[0] -= fieldst_[0];

    VectorXd result(VectorXd::Zero(x.size()));
    field2vector(fieldst_[0], result);
    return result;
}

}  // namespace chflow
