/**
 * time-integration classes for spectral Navier-Stokes simulation
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "channelflow/dns.h"

using namespace std;

namespace chflow {

DNS::DNS() : main_nse_(0), init_nse_(0), main_algorithm_(0), init_algorithm_(0) {}

DNS::DNS(const DNS& dns)
    : main_nse_(dns.main_nse_ ? new NSE(*dns.main_nse_) : 0),
      init_nse_(dns.init_nse_ ? new NSE(*dns.init_nse_) : 0),
      main_algorithm_(dns.main_algorithm_ ? dns.main_algorithm_->clone(main_nse_) : 0),
      init_algorithm_(dns.init_algorithm_ ? dns.init_algorithm_->clone(init_nse_) : 0) {}

DNS::DNS(const vector<FlowField>& fields, const DNSFlags& flags)
    : main_nse_(0), init_nse_(0), main_algorithm_(0), init_algorithm_(0) {
    main_nse_ = newNSE(fields, flags);
    main_algorithm_ = newAlgorithm(fields, main_nse_, flags);
    if (!main_algorithm_->full() && flags.initstepping != flags.timestepping) {
        DNSFlags initflags = flags;
        initflags.timestepping = flags.initstepping;
        initflags.dt = flags.dt;
        init_nse_ = newNSE(fields, initflags);
        init_algorithm_ = newAlgorithm(fields, init_nse_, initflags);
        // Safety check

        if (init_algorithm_->Ninitsteps() != 0)
            cerr << "DNS::DNS(fields, flags) :\n"
                 << flags.initstepping << " can't initialize " << flags.timestepping
                 << " since it needs initialization itself.\n";
    }
}

DNS::DNS(const vector<FlowField>& fields, const vector<ChebyCoeff>& base, const DNSFlags& flags_)
    : main_nse_(0), init_nse_(0), main_algorithm_(0), init_algorithm_(0) {
    DNSFlags flags = flags_;
    flags.baseflow = ArbitraryBase;
    main_nse_ = newNSE(fields, base, flags);
    main_algorithm_ = newAlgorithm(fields, main_nse_, flags);

    //   u.setnu (flags.nu);

    if (!main_algorithm_->full() && flags.initstepping != flags.timestepping) {
        DNSFlags initflags = flags;
        initflags.timestepping = flags.initstepping;
        init_nse_ = newNSE(fields, base, initflags);
        init_algorithm_ = newAlgorithm(fields, init_nse_, initflags);

        // Safety check
        if (init_algorithm_->Ninitsteps() != 0)
            cerr << "DNS::DNS(fields, base, flags) :\n"
                 << flags.initstepping << " can't initialize " << flags.timestepping
                 << " since it needs initialization itself.\n";
    }
}

DNS::~DNS() {
    if (main_algorithm_)
        delete main_algorithm_;
    if (init_algorithm_)
        delete init_algorithm_;
    if (main_nse_.unique())
        main_nse_.reset();
    else if (main_nse_) {
        cerr << "Destruction of DNS does not destroy the instance of MAIN_NSE which it created.\n"
             << "Identify object which still holds copy of shared_ptr<NSE>." << endl;
    } else {
    }
    if (init_nse_.unique())
        init_nse_.reset();
    else if (init_nse_) {
        cerr << "Destruction of DNS does not destroy the instance of INIT_NSE which it created.\n"
             << "Identify object which still holds copy of shared_ptr<NSE>." << endl;
    } else {
    }
}

shared_ptr<NSE> DNS::newNSE(const vector<FlowField>& fields, const DNSFlags& flags) {
    shared_ptr<NSE> nse(new NSE(fields, flags));
    return nse;
}

shared_ptr<NSE> DNS::newNSE(const vector<FlowField>& fields, const vector<ChebyCoeff>& base, const DNSFlags& flags) {
    shared_ptr<NSE> nse(new NSE(fields, base, flags));
    return nse;
}

DNSAlgorithm* DNS::newAlgorithm(const vector<FlowField>& fields, const shared_ptr<NSE>& nse, const DNSFlags& flags) {
    DNSAlgorithm* alg = 0;
    switch (flags.timestepping) {
        case CNFE1:
        case SBDF1:
        case SBDF2:
        case SBDF3:
        case SBDF4:
            alg = new MultistepDNS(fields, nse, flags);
            break;
        case CNRK2:
            alg = new RungeKuttaDNS(fields, nse, flags);
            break;
        case SMRK2:
        case CNAB2:
            alg = new CNABstyleDNS(fields, nse, flags);
            break;
        default:
            cerr << "DNS::newAlgorithm : algorithm " << flags.timestepping << " is unimplemented" << endl;
    }
    return alg;
}

DNS& DNS::operator=(const DNS& dns) {
    if (main_algorithm_)
        delete main_algorithm_;
    if (init_algorithm_)
        delete init_algorithm_;
    if (main_nse_)
        main_nse_.reset();
    if (init_nse_)
        init_nse_.reset();

    main_nse_ = shared_ptr<NSE>(new NSE(*dns.main_nse_));
    init_nse_ = shared_ptr<NSE>(new NSE(*dns.init_nse_));
    main_algorithm_ = dns.main_algorithm_ ? dns.main_algorithm_->clone(dns.main_nse_) : 0;
    init_algorithm_ = dns.init_algorithm_ ? dns.init_algorithm_->clone(dns.init_nse_) : 0;
    return *this;
}

void DNS::advance(vector<FlowField>& fields, int Nsteps) {
    assert(main_algorithm_);
    // Error check
    if (!main_algorithm_->full() && !init_algorithm_) {
        cerr << "DNS::advance(u,q,Nsteps) : the main algorithm is uninitialized,\n"
             << "and the initialization algorithm is not set. This should not be\n"
             << "possible. Please submit a bug report (see documentation)." << endl;
        exit(1);
    }
    // check if 2nd, 3rd, ... field has the same size as 1st field (usually velocity)
    for (uint j = 1; j < fields.size(); ++j) {
        if (!fields[j].geomCongruent(fields[0]))  // are velocity and pressure of the same size?
            fields[j].resize(fields[0].Nx(), fields[0].Ny(), fields[0].Nz(), fields[j].Nd(), fields[0].Lx(),
                             fields[0].Lz(), fields[0].a(), fields[0].b(), fields[0].cfmpi());
    }

    int n = 0;
    if ((int)(main_algorithm_->time()) % (int)(flags().symmetryprojectioninterval) == 0) {
        main_algorithm_->project();  // projects the flowfield and nonlin. term held by the DNSAlgo object onto symm
        for (uint j = 0; j < fields.size(); ++j)
            fields[j].project(main_algorithm_->symmetries(j));
    }
    while (!main_algorithm_->full() && n < Nsteps) {  // initial stepping
        main_algorithm_->push(fields);
        init_algorithm_->advance(fields, 1);
        ++n;
    }
    main_algorithm_->advance(fields, Nsteps - n);
}

// void DNS::reset() {
//}

void DNS::project() {
    if (init_algorithm_)
        init_algorithm_->project();
    if (main_algorithm_)
        main_algorithm_->project();
}

void DNS::operator*=(const vector<FieldSymmetry>& sigma) {
    if (init_algorithm_)
        *init_algorithm_ *= sigma;
    if (main_algorithm_)
        *main_algorithm_ *= sigma;
}

/***************************
void DNS::reset() {
if (main_algorithm_)
  main_algorithm_->reset();
if (init_algorithm_)
  init_algorithm_->reset();
}
**************************/

void DNS::reset_dt(Real dt) {
    assert(main_algorithm_);
    main_algorithm_->reset_dt(dt);
    if (init_algorithm_)
        init_algorithm_->reset_dt(dt);
}

// The mindless hassle of wrapper classes in C++ follows
void DNS::reset_time(Real t) {
    if (init_algorithm_)
        init_algorithm_->reset_time(t);
    if (main_algorithm_)
        main_algorithm_->reset_time(t);
}

/****************************************
void DNS::reset_dPdx(Real dPdx) {
if (init_algorithm_)
  init_algorithm_->reset_dPdx(dPdx);
if (main_algorithm_)
  main_algorithm_->reset_dPdx(dPdx);
}
void DNS::reset_Ubulk(Real Ubulk) {
if (init_algorithm_)
  init_algorithm_->reset_Ubulk(Ubulk);
if (main_algorithm_)
  main_algorithm_->reset_Ubulk(Ubulk);
}
**************************************/

void DNS::reset_gradp(Real dPdx, Real dPdz) {
    if (init_nse_)
        init_nse_->reset_gradp(dPdx, dPdz);
    if (main_nse_)
        main_nse_->reset_gradp(dPdx, dPdz);
}
void DNS::reset_bulkv(Real Ubulk, Real Wbulk) {
    if (init_nse_)
        init_nse_->reset_bulkv(Ubulk, Wbulk);
    if (main_nse_)
        main_nse_->reset_bulkv(Ubulk, Wbulk);
}

// void DNS::reset_uj(const FlowField& uj, int j) {
// assert(main_algorithm_);
// main_algorithm_->reset_uj(uj, j);
//}
bool DNS::push(const vector<FlowField>& fields) {
    if (main_algorithm_)
        return main_algorithm_->push(fields);
    else
        return false;
}
bool DNS::full() const {
    if (main_algorithm_)
        return main_algorithm_->full();
    else
        return false;
}
int DNS::order() const {
    if (main_algorithm_)
        return main_algorithm_->order();
    else if (init_algorithm_)
        return init_algorithm_->order();
    else
        return 0;
}

int DNS::Ninitsteps() const {
    if (main_algorithm_)
        return main_algorithm_->Ninitsteps();
    else
        return 0;
}

Real DNS::nu() const {
    if (main_nse_)
        return main_nse_->nu();
    else if (init_nse_)
        return init_nse_->nu();
    else
        return 0.0;
}
Real DNS::dt() const {
    if (main_algorithm_)
        return main_algorithm_->dt();
    else if (init_algorithm_)
        return init_algorithm_->dt();
    else
        return 0.0;
}
Real DNS::CFL(FlowField& u) const {
    if (main_algorithm_)
        return main_algorithm_->CFL(u);
    else if (init_algorithm_)
        return init_algorithm_->CFL(u);
    else
        return 0.0;
}
Real DNS::time() const {
    if (main_algorithm_)
        return main_algorithm_->time();
    else if (init_algorithm_)
        return init_algorithm_->time();
    else
        return 0.0;
}
Real DNS::dPdx() const {
    if (main_nse_)
        return main_nse_->dPdx();
    else if (init_nse_)
        return init_nse_->dPdx();
    else
        return 0.0;
}
Real DNS::Ubulk() const {
    if (main_nse_)
        return main_nse_->Ubulk();
    else if (init_nse_)
        return init_nse_->Ubulk();
    else
        return 0.0;
}
Real DNS::dPdxRef() const {
    if (main_nse_)
        return main_nse_->dPdxRef();
    else if (init_nse_)
        return init_nse_->dPdxRef();
    else
        return 0.0;
}
Real DNS::UbulkRef() const {  // the bulk velocity enforced during integ.
    if (main_nse_)
        return main_nse_->UbulkRef();
    else if (init_nse_)
        return init_nse_->UbulkRef();
    else
        return 0.0;
}
const ChebyCoeff& DNS::Ubase() const {
    if (main_nse_)
        return main_nse_->Ubase();
    else if (init_nse_)
        return init_nse_->Ubase();
    else {
        cerr << "Error in DNS::Ubase(): Ubase is currently undefined" << endl;
        exit(1);
        return init_nse_->Ubase();  // to make compiler happy
    }
}
const ChebyCoeff& DNS::Wbase() const {
    if (main_nse_)
        return main_nse_->Wbase();
    else if (init_nse_)
        return init_nse_->Wbase();
    else {
        cerr << "Error in DNS::Wbase(): Wbase is currently undefined" << endl;
        exit(1);
        return init_nse_->Wbase();  // to make compiler happy
    }
}
const DNSFlags& DNS::flags() const {
    if (main_algorithm_)
        return main_algorithm_->flags();
    else if (init_algorithm_)
        return init_algorithm_->flags();
    else {
        cerr << "Error in DNS::flags(): flags are currently undefined" << endl;
        exit(1);
        return init_algorithm_->flags();  // to make compiler happy
    }
}
TimeStepMethod DNS::timestepping() const {
    if (main_algorithm_)
        return main_algorithm_->timestepping();
    else if (init_algorithm_)
        return init_algorithm_->timestepping();
    else
        return CNFE1;
}

void DNS::uq2p(FlowField u, FlowField q, FlowField& p) const {
    if (flags().nonlinearity != Rotational) {
        p = q;
        return;
    }

    assert(u.Nd() == 3);
    assert(q.Nd() == 1);
    assert(main_algorithm_);
    ChebyCoeff U(main_nse_->Ubase());
    ChebyCoeff W(main_nse_->Wbase());
    fieldstate qxzstate = q.xzstate();
    fieldstate qystate = q.ystate();

    u.makePhysical();
    q.makePhysical();
    U.makePhysical();
    W.makePhysical();

    int Nx = u.Nx();
    int Ny = u.Ny();
    int Nz = u.Nz();

    // Set p = q - 1/2 (u+U) dot (u+U)
    p = q;
    for (int ny = 0; ny < Ny; ++ny) {
        Real Uny = U(ny);
        Real Wny = W(ny);
        for (int nx = 0; nx < Nx; ++nx)
            for (int nz = 0; nz < Nz; ++nz)
                p(nx, ny, nz, 0) -=
                    0.5 * (square(u(nx, ny, nz, 0) + Uny) + square(u(nx, ny, nz, 1)) + square(u(nx, ny, nz, 2) + Wny));
    }
    p.makeState(qxzstate, qystate);
}

void DNS::up2q(FlowField u, FlowField p, FlowField& q) const {
    if (flags().nonlinearity != Rotational) {
        q = p;
        return;
    }

    assert(main_algorithm_);
    assert(u.Nd() == 3);
    assert(p.Nd() == 1);
    ChebyCoeff U(main_nse_->Ubase());
    ChebyCoeff W(main_nse_->Wbase());
    fieldstate pxzstate = p.xzstate();
    fieldstate pystate = p.ystate();

    u.makePhysical();
    p.makePhysical();
    W.makePhysical();

    int Nx = u.Nx();
    int Ny = u.Ny();
    int Nz = u.Nz();

    // Set q = p + 1/2 (u+U) dot (u+U) to q
    q.makePhysical();
    for (int ny = 0; ny < Ny; ++ny) {
        Real Uny = U(ny);
        Real Wny = U(ny);
        for (int nx = 0; nx < Nx; ++nx)
            for (int nz = 0; nz < Nz; ++nz)
                q(nx, ny, nz, 0) +=
                    0.5 * (square(u(nx, ny, nz, 0) + Uny) + square(u(nx, ny, nz, 1)) + square(u(nx, ny, nz, 2) + Wny));
    }
    q.makeState(pxzstate, pystate);
}

void DNS::printStack() const {
    assert(main_algorithm_);
    main_algorithm_->printStack();
}

// *******************************************************************************************
// BEGIN EXPERIMENTAL CODE

PoincareCondition::~PoincareCondition() {}
PoincareCondition::PoincareCondition() {}

PlaneIntersection::~PlaneIntersection() {}
PlaneIntersection::PlaneIntersection() {}
PlaneIntersection::PlaneIntersection(const FlowField& ustar, const FlowField& estar)
    : estar_(estar), cstar_(L2IP(ustar, estar)) {}

Real PlaneIntersection::operator()(const FlowField& u) { return L2IP(u, estar_) - cstar_; }

DragDissipation::~DragDissipation() {}
DragDissipation::DragDissipation() {}

Real DragDissipation::operator()(const FlowField& u) { return wallshear(u) - dissipation(u); }

DNSPoincare::DNSPoincare()
    : DNS(),
      e_(),
      sigma_(),
      h_(0),
      ucrossing_(),
      pcrossing_(),
      tcrossing_(0),
      scrossing_(0),
      hcrossing_(0.0),
      hcurrent_(0.0),
      t0_(0) {}

DNSPoincare::DNSPoincare(FlowField& u, PoincareCondition* h, const DNSFlags& flags)
    : DNS({u}, flags),
      e_(),
      sigma_(),
      h_(h),
      ucrossing_(),
      pcrossing_(),
      tcrossing_(0),
      scrossing_(0),
      hcrossing_(0.0),
      hcurrent_((*h)(u)),
      t0_(flags.t0) {}

DNSPoincare::DNSPoincare(FlowField& u, const cfarray<FlowField>& e, const cfarray<FieldSymmetry>& sigma,
                         PoincareCondition* h, const DNSFlags& flags)
    : DNS({u}, flags),
      e_(e),
      sigma_(sigma),
      h_(h),
      ucrossing_(),
      pcrossing_(),
      tcrossing_(0),
      scrossing_(0),
      hcrossing_(0.0),
      hcurrent_((*h)(u)),
      t0_(flags.t0) {
    // check that sigma[n] e[n] = -e[n]
    //*flags_.logstream << "Checking symmetries and basis sets for fundamental domain " << endl;
    // FlowField tmp;
    //*flags_.logstream << "n \t L2Norm(e[n] + s[n] e[n])/L2Norm(e[n])" << endl;
    // for (int n=0; n<e.length(); ++n) {
    // tmp = e_[n];
    // tmp += sigma_[n](e_[n]);
    //*flags_.logstream << n << '\t' << L2Norm(tmp)/L2Norm(e_[n]) << endl;
    //}
}

const FlowField& DNSPoincare::ucrossing() const { return ucrossing_; }
const FlowField& DNSPoincare::pcrossing() const { return pcrossing_; }
Real DNSPoincare::tcrossing() const { return tcrossing_; }
int DNSPoincare::scrossing() const { return scrossing_; }
Real DNSPoincare::hcrossing() const { return hcrossing_; }
Real DNSPoincare::hcurrent() const { return hcurrent_; }
// Real DNSPoincare::f(const FlowField& u) const {
//  return L2IP(u, estar_) - cstar_;
//}

bool DNSPoincare::advanceToSection(FlowField& u, FlowField& p, int nSteps, int crosssign, Real Tmin, Real epsilon) {
    ostream* os = flags().logstream;
    FlowField uprev(u);
    FlowField pprev(p);

    // Take nSteps of length dt, advancing t -> t + nSteps dt
    vector<FlowField> fields = {u, p};
    advance(fields, nSteps);

    // Check for u(t) cross of fundamental domain boundary, map back in
    // Map uprev, pprev back, too, so that
    FieldSymmetry s, identity;  // both identity at this point
    for (int n = 0; n < e_.length(); ++n) {
        if (L2IP(u, e_[n]) < 0) {
            // *os << "Crossed " << n << "th boundary of fundamental domain" << endl;
            s *= sigma_[n];
        }
    }
    if (s != identity) {
        // *os << "Mapping fields back into fund. domain w s = " << s << endl;
        u *= s;
        p *= s;
        uprev *= s;
        pprev *= s;
        FieldSymmetry psym;
        vector<FieldSymmetry> svec = {s, psym};
        (*this) *= svec;  // maps all FlowField members of DNS
    }

    Real tcoarse = DNS::time();
    if (tcoarse - t0_ > Tmin) {
        // Real cstar = L2IP(ustar_, estar_);
        Real hprev = (*h_)(uprev);
        Real hcurr = (*h_)(u);
        // hcrossing_ = hcurr;
        hcurrent_ = hcurr;
        //*os << tcoarse << '\t' << c << endl;

        bool dhdt_pos_cross = (hprev < 0 && 0 <= hcurr) ? true : false;
        bool dhdt_neg_cross = (hprev > 0 && 0 >= hcurr) ? true : false;

        // If we cross the Poincare section in required direction, back up and
        // reintegrate, checking condition every integration time step dt.
        if ((crosssign > 0 && dhdt_pos_cross) || (crosssign < 0 && dhdt_neg_cross) ||
            (crosssign == 0 && (dhdt_pos_cross || dhdt_neg_cross))) {
            //*os << "u(t) crossed Poincare section..." << endl;
            *os << (dhdt_pos_cross ? '+' : '-') << flush;
            scrossing_ = dhdt_pos_cross ? +1 : -1;

            // Allocate cfarrays to store time, velocity, and Poincare conditions
            // at three consecutive time steps for quadratic interpolation. E.g.
            // v[n],q[n],f[n] are u,p,f at three successive fine-scale timesteps
            cfarray<Real> s(3);       // s[n] = tcoarse - dT + n dt    time-like variable
            cfarray<Real> h(3);       // h[n] = h(s[n])                space-like variable
            cfarray<FlowField> v(3);  // v[n] = u(s[n])                velocity field
            cfarray<FlowField> q(3);  // q[n] = p(s[n])                pressure field

            Real dt = DNS::dt();
            Real dT = dt * nSteps;

            s[0] = tcoarse - dT;
            s[1] = 0.0;
            s[2] = 0.0;
            // s[3] = 0.0;
            h[0] = hprev;
            h[1] = 0.0;
            h[2] = 0.0;
            // h[3] = 0.0;
            v[0] = uprev;
            q[0] = pprev;

            //*os << "h:";
            // for (int i=0; i<3; ++i)
            //*os << h[i] << '\t';
            //*os << endl;
            //*os << "s:";
            // for (int i=0; i<3; ++i)
            //*os << s[i] << '\t';
            //*os << endl;

            DNSFlags fineflags = DNS::flags();
            fineflags.verbosity = Silent;
            //*os << "constructing DNS for fine-time integration..." << endl;
            fineflags.t0 = tcoarse - dT;

            DNS dns({v[0]}, fineflags);
            //*os << "finished contructing DNS..." << endl;

            int count = 1;  // need four data points for cubic interpolation

            // Now take a number of small-scale (dt) time-steps until we hit the section
            // Hitting the section will be defined by 0.0 lying between d[0] and d[2]
            for (Real tfine = tcoarse - dT; tfine <= tcoarse + dt; tfine += dt) {
                //*os << "time  shifts..." << endl;
                // Time-shift velocity and Poincare condition cfarrays
                // v[2] <- v[1] <- v[0], same w d in prep for advancing v[0],q[0] under DNS
                for (int n = 2; n > 0; --n) {
                    s[n] = s[n - 1];
                    v[n] = v[n - 1];
                    q[n] = q[n - 1];
                    h[n] = h[n - 1];
                }

                //*os << "time step..." << endl;
                vector<FlowField> fields = {v[0], q[0]};
                dns.advance(fields, 1);  // take one step of length dt
                h[0] = (*h_)(v[0]);
                s[0] = tfine + dt;
                *os << ':' << flush;

                //*os << "crossing check..." << endl;
                // Check for Poincare section crossing in midpoint of h[0],h[1],h[2],h[3]
                if (++count >= 3 && ((h[2] < 0 && 0 <= h[0]) || (h[2] > 0 && 0 >= h[0]))) {
                    // Newton search for zero of h(s) == h(v(s)) == 0 where v(s) is
                    // quadratic interpolant of v. Interpolating s as a function of h
                    // at h==0 gives a good initial guess for s

                    // Newton iteration variables
                    Real sN = polynomialInterpolate(s, h, 0);
                    Real eps = 1e-9;  // used for approximation dh/ds = (h(s + eps s) - h(s))/(eps s)
                    Real hsN, hsN_ds;
                    FlowField vN;
                    // os << "Newtown iteration on interpolated Poincare crossing" << endl;

                    int Newtsteps = 6;
                    for (int n = 0; n < Newtsteps; ++n) {
                        vN = polynomialInterpolate(v, s, sN);
                        vN.makeSpectral();
                        hsN = (*h_)(vN);
                        //*os << n << flush;

                        if (abs(hsN) < epsilon / 2 || n == Newtsteps - 1) {
                            if (abs(hsN) < epsilon / 2)
                                *os << "|" << flush;  // signal an accurate computation of a Poincare crossing
                            else
                                *os << "~|" << flush;  // signal an inaccurate computation of a Poincare crossing

                            tcrossing_ = sN;
                            ucrossing_ = vN;
                            pcrossing_ = polynomialInterpolate(q, s, sN);
                            pcrossing_.makeSpectral();
                            break;
                        } else {
                            vN = polynomialInterpolate(v, s, sN + eps * sN);
                            vN.makeSpectral();
                            hsN_ds = (*h_)(vN);
                            Real dhds = (hsN_ds - hsN) / (eps * sN);
                            Real ds = -hsN / dhds;
                            sN += ds;

                            //*os << "Not good enough. Taking Newton step. " << endl;
                            //*os << "dhds == " << dhds << endl;
                            //*os << "ds   == " << ds << endl;
                            //*os << "s+ds == " << sN << endl;
                        }
                    }

                    // output time of crossing
                    hcrossing_ = hsN;
                    // Real cross = (*h_)(ucrossing_);
                    //*os << "Estimated poincare crossing: " << endl;
                    //*os << "  h(u) == " << cross << endl;
                    //*os << "  time == " << tcrossing_ << endl;

                    return true;
                }
            }
            *os << "Strange.... the large-scale steps crossed the Poincare section,\n";
            *os << "but when we went back and looked with finer-scale steps, there\n";
            *os << "was no crossing. Exiting." << endl;
            exit(1);
        }
    }
    return false;  // didn't cross Poincare section
}
// END EXPERIMENTAL CODE
// *******************************************************************************************

}  // namespace chflow
