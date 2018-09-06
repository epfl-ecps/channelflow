/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "channelflow/dnsalgo.h"

using namespace std;

namespace chflow {

DNSAlgorithm::~DNSAlgorithm() {}

DNSAlgorithm::DNSAlgorithm()
    : flags_(),
      order_(0),
      numfields_(0),
      Ninitsteps_(0),
      t_(0),
      //   cfl_ (0),
      lambda_t_(),
      nse_(0) {}

DNSAlgorithm::DNSAlgorithm(const DNSAlgorithm& d)
    : flags_(d.flags_),
      order_(d.order_),
      numfields_(d.numfields_),
      Ninitsteps_(d.Ninitsteps_),
      t_(d.t_),
      //   cfl_ (d.cfl_),
      lambda_t_(d.lambda_t_),
      nse_(d.nse_),
      symmetries_(d.symmetries_) {}

DNSAlgorithm::DNSAlgorithm(const vector<FlowField>& fields, const shared_ptr<NSE>& nse, const DNSFlags& flags)
    : flags_(flags),
      order_(0),
      numfields_(fields.size()),
      Ninitsteps_(0),
      t_(flags.t0),
      //   cfl_ (0),
      lambda_t_(0),
      nse_(nse),
      symmetries_(0) {
    symmetries_ = nse_->createSymmVec();
}

void DNSAlgorithm::project() {}

void DNSAlgorithm::operator*=(const vector<FieldSymmetry>& symm) {}

bool DNSAlgorithm::push(const vector<FlowField>& fields) { return true; }

bool DNSAlgorithm::full() const { return true; }

void DNSAlgorithm::reset_time(Real t) { t_ = t; }
void DNSAlgorithm::reset_nse(shared_ptr<NSE> nse) {
    nse_.reset(new NSE);
    nse_ = nse;
}

Real DNSAlgorithm::dt() const { return flags_.dt; }
Real DNSAlgorithm::CFL(FlowField& u) const {
    Real cfl = u.CFLfactor(nse_->Ubase(), nse_->Wbase());
    cfl *= flags_.dealias_xz() ? 2.0 * pi / 3.0 * flags_.dt : pi * flags_.dt;
    return cfl;
}
Real DNSAlgorithm::time() const { return t_; }
cfarray<FieldSymmetry> DNSAlgorithm::symmetries(int ifield) const { return symmetries_[ifield]; }
int DNSAlgorithm::order() const { return order_; }
int DNSAlgorithm::Ninitsteps() const { return Ninitsteps_; }
const DNSFlags& DNSAlgorithm::flags() const { return flags_; }

// const FlowField&   DNSAlgorithm::ubase() const {return ubase_;}
TimeStepMethod DNSAlgorithm::timestepping() const { return flags_.timestepping; }

void DNSAlgorithm::printStack() const {
    // os << "DNSAlgorithm::printStack()" << endl;
}

// ====================================================================
// Multistep algorithms

MultistepDNS::MultistepDNS() : DNSAlgorithm() {}

MultistepDNS::MultistepDNS(const MultistepDNS& dns)
    : DNSAlgorithm(dns),
      //  lambda_t_ (dns.lambda_t_),
      eta_(dns.eta_),
      alpha_(dns.alpha_),
      beta_(dns.beta_),
      fields_(dns.fields_),
      nonlf_(dns.nonlf_),
      //   nse_ (new NSE( * dns.nse_)),
      countdown_(dns.countdown_) {}

MultistepDNS& MultistepDNS::operator=(const MultistepDNS& dns) {
    cerr << "MultistepDNS::operator=(const MultistepDNS& dns) unimplemented\n";
    exit(1);
}

MultistepDNS::MultistepDNS(const vector<FlowField>& fields, const shared_ptr<NSE>& nse, const DNSFlags& flags)
    : DNSAlgorithm(fields, nse, flags) {
    TimeStepMethod algorithm = flags.timestepping;
    switch (algorithm) {
        case CNFE1:
        case SBDF1:
            order_ = 1;
            eta_ = 1.0;
            alpha_.resize(order_);
            beta_.resize(order_);
            alpha_[0] = -1.0;
            beta_[0] = 1.0;
            break;
        case SBDF2:
            order_ = 2;
            alpha_.resize(order_);
            beta_.resize(order_);
            eta_ = 1.5;
            alpha_[0] = -2.0;
            alpha_[1] = 0.5;
            beta_[0] = 2.0;
            beta_[1] = -1.0;
            break;
        case SBDF3:
            order_ = 3;
            alpha_.resize(order_);
            beta_.resize(order_);
            eta_ = 11.0 / 6.0;
            alpha_[0] = -3.0;
            alpha_[1] = 1.5;
            alpha_[2] = -1.0 / 3.0;
            beta_[0] = 3.0;
            beta_[1] = -3.0;
            beta_[2] = 1.0;
            break;
        case SBDF4:
            order_ = 4;
            alpha_.resize(order_);
            beta_.resize(order_);
            eta_ = 25.0 / 12.0;
            alpha_[0] = -4.0;
            alpha_[1] = 3.0;
            alpha_[2] = -4.0 / 3.0;
            alpha_[3] = 0.25;
            beta_[0] = 4.0;
            beta_[1] = -6.0;
            beta_[2] = 4.0;
            beta_[3] = -1.0;
            break;
        default:
            cerr << "MultistepDNS::MultistepDNS(un,Ubase,nu,dt,flags,t0)\n"
                 << "error: flags.timestepping == " << algorithm << "is a non-multistepping algorithm" << endl;
            exit(1);
    }
    // make nse object allocate the TauSolvers depending on time stepping constant
    lambda_t_ = {eta_ / flags_.dt};
    nse_->reset_lambda(lambda_t_);  // allocates memory in nse on first call
    //   nse_=std::move(makeNSE (fields, base, lambda_t_, flags));

    // construct field members for linear and nonlinear part
    vector<FlowField> tmpar(fields);
    for (int i = 0; i < numfields_; ++i)
        tmpar[i].setToZero();

    fields_.resize(order_);
    nonlf_.resize(order_);
    for (int j = 0; j < order_; ++j) {
        fields_[j] = tmpar;
        nonlf_[j] = tmpar;
    }
    // if (order_ > 0)  // should always be true
    // u_[0] = u;
    //   cfl_ = fields[0].CFLfactor (nse_->Ubase(),nse_->Wbase());
    //   cfl_ *= flags_.dealias_xz() ? 2.0*pi/3.0*flags.dt : pi*flags.dt;

    Ninitsteps_ = order_ - 1;
    countdown_ = Ninitsteps_;
}

MultistepDNS::~MultistepDNS() {
    // unique pointer destroys itself when out of scope
}

DNSAlgorithm* MultistepDNS::clone(const shared_ptr<NSE>& nse) const {
    DNSAlgorithm* cloned = new MultistepDNS(*this);
    cloned->reset_nse(nse);
    return cloned;
}

void MultistepDNS::reset_dt(Real dt) {
    //   cfl_ *= dt/flags_.dt;
    flags_.dt = dt;
    lambda_t_.resize(1);
    lambda_t_[0] = eta_ / flags_.dt;
    // The construction of new TauSolver objects is done in NSE::reset_lambda():
    nse_->reset_lambda(lambda_t_);
    // Start from beginning on initialization
    countdown_ = Ninitsteps_;
}

void MultistepDNS::advance(vector<FlowField>& fieldsn, int Nsteps) {
    // This calculation follows Peyret section 4.5.1(b) pg 131.
    const int J = order_ - 1;
    vector<FlowField> rhs(nse_->createRHS(fieldsn));  // Number of fields and number of RHS's can be different
    int len = rhs.size();
    fields_[0] = fieldsn;
    // Start of time stepping loop

    for (int step = 0; step < Nsteps; ++step) {
        // Calculate nonlinearity, includes dealiasing if applicable
        if (order_ > 0) {
            nse_->nonlinear(fields_[0], nonlf_[0]);
        }

        // Add up multistepping terms of linear and nonlinear terms
        for (int l = 0; l < len; ++l) {
            rhs[l].setToZero();  // RHS must be zero before sum over multistep loop
            for (int j = 0; j < order_; ++j) {
                const Real a = -alpha_[j] / flags_.dt;
                const Real b = -beta_[j];
                rhs[l].add(a, fields_[j][l], b, nonlf_[j][l]);
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

    if (nse_->taskid() == 0)
        if (flags_.verbosity == PrintTime || flags_.verbosity == PrintAll || flags_.verbosity == PrintTicks)
            *flags_.logstream << endl;
    return;
}

void MultistepDNS::project() {
    for (int n = 0; n < order_; ++n)
        for (int m = 0; m < numfields_; ++m)
            fields_[n][m].project(symmetries_[m]);
    for (int n = 0; n < order_; ++n)
        for (int m = 0; m < numfields_; ++m)
            nonlf_[n][m].project(symmetries_[m]);
}

void MultistepDNS::operator*=(const vector<FieldSymmetry>& sigma) {
    assert(static_cast<int>(sigma.size()) == numfields_);
    for (int n = 0; n < order_; ++n)
        for (int m = 0; m < numfields_; ++m)
            fields_[n][m] *= sigma[m];
    for (int n = 0; n < order_; ++n)
        for (int m = 0; m < numfields_; ++m)
            nonlf_[n][m] *= sigma[m];
}

void MultistepDNS::printStack() const {
    *flags_.logstream << "Multistep::printStack() {" << endl;
    *flags_.logstream << "        t == " << t_ << endl;
    *flags_.logstream << "countdown == " << countdown_ << endl;
    *flags_.logstream << "     full == " << full() << endl;

    for (int j = order_ - 1; j >= 0; --j)
        printf("j=%2d t=%5.2f L2(uj)=%13.10f L2(fj)=%13.10f\n", j, t_ - j * flags_.dt, L2Norm(fields_[j][0]),
               L2Norm(nonlf_[j][0]));
    *flags_.logstream << endl;
    *flags_.logstream << "}" << endl;
}

bool MultistepDNS::push(const vector<FlowField>& fields) {
    //*flags_.logstream << "MultistepDNS::push(const FlowField& un) { " << endl;
    // printStack();
    // Let K = order-1. Arrays are then u_[0:K], f_[0:K]
    // Shift u_[K] <- u_[K-1] <- ... <- u_[0] <- un
    // Shift u_[K] <- u_[K-1] <- ... <- u_[0] <- un
    for (int j = order_ - 1; j > 0; --j) {
        for (int l = 0; l < numfields_; ++l) {
            swap(nonlf_[j][l], nonlf_[j - 1][l]);
            swap(fields_[j][l], fields_[j - 1][l]);
        }
    }

    // push in current field. TODO: why does field not enter at leading position?
    if (order_ > 1) {
        fields_[1] = fields;
        // navierstokesNL(u_[0], ubase_, Ubase_, f_[0], tmp_, tmp2_, flags_.nonlinearity);
        nse_->nonlinear(fields_[1], nonlf_[1]);

        //     cfl_ = fields_[1][0].CFLfactor (nse_->Ubase(), nse_->Wbase());
        //     cfl_ *= flags_.dealias_xz() ? 2.0*pi/3.0*flags_.dt : pi*flags_.dt;
    }

    t_ += flags_.dt;
    --countdown_;
    // printStack();
    //*flags_.logstream << "}" << endl;
    return full();
}

bool MultistepDNS::full() const { return (countdown_ == 0) ? true : false; }

// ==============================================================
// Runge-Kutta algorithms

RungeKuttaDNS::RungeKuttaDNS()
    : DNSAlgorithm(),
      //   lambda_t_ ({0}),
      //   nse_ (),
      Qj1_(0),
      Qj_(0) {
    //   nse_.resize(1);
    //   unique_ptr<NSE> nse(new NSE());
    //   nse_[0]=std::move(nse);
}

RungeKuttaDNS::RungeKuttaDNS(const RungeKuttaDNS& dns)
    : DNSAlgorithm(dns),
      //   lambda_t_(dns.lambda_t_),
      //   nse_ (),
      Nsubsteps_(dns.Nsubsteps_),
      Qj1_(dns.Qj1_),
      Qj_(dns.Qj_),
      A_(dns.A_),
      B_(dns.B_),
      C_(dns.C_) {
    //   //make copy of nse objects
    //   nse_.resize(Nsubsteps_);
    //   for(int j=0; j<Nsubsteps_; ++j){
    //     unique_ptr<NSE> nse(new NSE( * dns.nse_[j]));
    //     nse_[j] = std::move(nse);
    //   }
}

// This algorithm is described in "Spectral Methods for Incompressible Viscous
// Flow", Roger Peyret, Springer-Verlag Appl Math Sci series vol 148, 2002.
// section 4.5.2.c.2 "Three-stage scheme (RK3/CN)". I use C for his B'
RungeKuttaDNS::RungeKuttaDNS(const vector<FlowField>& fields, const shared_ptr<NSE>& nse, const DNSFlags& flags)
    : DNSAlgorithm(fields, nse, flags), Qj1_(fields), Qj_(fields) {
    TimeStepMethod algorithm = flags_.timestepping;
    switch (algorithm) {
        case CNRK2:
            order_ = 2;
            Nsubsteps_ = 3;
            Ninitsteps_ = 0;
            A_.resize(Nsubsteps_);
            B_.resize(Nsubsteps_);
            C_.resize(Nsubsteps_);
            A_[0] = 0.0;
            A_[1] = -5.0 / 9.0;
            A_[2] = -153.0 / 128.0;  // Peyret A
            B_[0] = 1.0 / 3.0;
            B_[1] = 15.0 / 16.0;
            B_[2] = 8.0 / 15.0;  // Peyret B
            C_[0] = 1.0 / 6.0;
            C_[1] = 5.0 / 24.0;
            C_[2] = 1.0 / 8.0;  // Peyret B'
            break;
        default:
            cerr << "RungeKuttaDNS::RungeKuttaDNS(un,Ubase,nu,dt,flags,t0)\n"
                 << "error: flags.timestepping == " << algorithm << " is a non-runge-kutta algorithm" << endl;
            exit(1);
    }

    // initialize fields for nonlinear terms with zero
    for (int i = 0; i < numfields_; ++i) {
        Qj1_[i].setToZero();
        Qj_[i].setToZero();
    }

    // define time stepping const. for impl. solver and create new nse objects
    lambda_t_.resize(Nsubsteps_);
    //   nse_.resize(Nsubsteps_);
    for (int j = 0; j < Nsubsteps_; ++j)  //{
        lambda_t_[j] = 1.0 / (C_[j] * flags_.dt);
    //     nse_[j]=std::move(makeNSE (fields, base, lambda_t_[j], flags));
    //   }
    nse_->reset_lambda(lambda_t_);
}

RungeKuttaDNS::~RungeKuttaDNS() {
    // unique pointer delete themselve
}

DNSAlgorithm* RungeKuttaDNS::clone(const shared_ptr<NSE>& nse) const {
    DNSAlgorithm* clone = new RungeKuttaDNS(*this);
    clone->reset_nse(nse);
    return clone;
}

void RungeKuttaDNS::reset_dt(Real dt) {
    //   cfl_ *= dt/flags_.dt;
    flags_.dt = dt;

    // This loop reconfigures the TauSolver objects held by nse object
    for (int j = 0; j < Nsubsteps_; ++j)  //{
        lambda_t_[j] = 1.0 / (C_[j] * flags_.dt);
    //     nse_[j]->reset_lambda(lambda_t_[j]);
    //   }
    nse_->reset_lambda(lambda_t_);
}

void RungeKuttaDNS::advance(vector<FlowField>& fields, int Nsteps) {
    vector<FlowField> rhs(nse_->createRHS(fields));
    vector<FlowField> lt(rhs);
    int len = rhs.size();

    // Start with time stepping loop
    for (int n = 0; n < Nsteps; ++n) {
        for (int j = 0; j < Nsubsteps_; ++j) {
            // Store uj in un during substeps, reflect in notation, might even increase performance
            vector<FlowField>& uj(fields);

            // Goal is to compute
            // RHS = lt + lambda_t* uj  + Bj/Cj Qj

            // Efficient implementation of
            // Q_{j+1} = A_j Q_j + N(u_j)}  where N = -u grad u
            // Q_{j+1} = A_j Q_j - f(u_j)}  where f =  u grad u
            // Q_j = Q_{j+1}
            Qj_[0] *= A_[j];
            nse_->nonlinear(uj, Qj1_);
            Qj_[0] -= Qj1_[0];  // subtract because navierstokesNL(u) = u grad u = -N(u)

            // get linear term
            nse_->linear(uj, lt);

            // Add up multistepping terms of linear and nonlinear terms
            const Real B_C = B_[j] / C_[j];
            for (int l = 0; l < len; ++l) {
                rhs[l] = lt[l];
                rhs[l].add(lambda_t_[j], uj[l], B_C, Qj_[l]);
            }

            // solve the implicit problem and store solution in fields
            nse_->solve(fields, rhs, j);

        }  // End of loop over substeps j

        t_ += flags_.dt;
        if (fields[0].taskid() == 0) {
            if (flags_.verbosity == PrintTime || flags_.verbosity == PrintAll)
                *flags_.logstream << t_ << ' ' << flush;
            else if (flags_.verbosity == PrintTicks)
                *flags_.logstream << '.' << flush;
        }
    }  // End of loop over main time steps n

    if (fields[0].taskid() == 0)
        if (flags_.verbosity == PrintTime || flags_.verbosity == PrintAll)
            *flags_.logstream << endl;

    return;
}

// void RungeKuttaDNS::project() {}
// void RungeKuttaDNS::operator*= {}

// ==============================================================
// CNAB-style algorithms

CNABstyleDNS::CNABstyleDNS()
    : DNSAlgorithm(),
      //   lambda_t_ ({0}),
      //   nse_ (),
      full_(false),
      fj1_(),
      fj_() {
    //   nse_.resize(1);
    //   unique_ptr<NSE> nse(new NSE());
    //   nse_[0]=std::move(nse);
}

CNABstyleDNS::CNABstyleDNS(const CNABstyleDNS& dns)
    : DNSAlgorithm(dns),
      //   lambda_t_(dns.lambda_t_),
      //   nse_ (),
      Nsubsteps_(dns.Nsubsteps_),
      full_(dns.full_),
      fj1_(dns.fj1_),
      fj_(dns.fj_),
      alpha_(dns.alpha_),
      beta_(dns.beta_),
      gamma_(dns.gamma_),
      zeta_(dns.zeta_) {
    // make copy of nse objects
    //   nse_.resize(Nsubsteps_);
    //   for(int j=0; j<Nsubsteps_; ++j){
    //     unique_ptr<NSE> nse(new NSE( * dns.nse_[j]));
    //     nse_[j] = std::move(nse);
    //   }
}

CNABstyleDNS::CNABstyleDNS(const vector<FlowField>& fields, const shared_ptr<NSE>& nse, const DNSFlags& flags)
    : DNSAlgorithm(fields, nse, flags), full_(false), fj1_(fields), fj_(fields) {
    TimeStepMethod algorithm = flags_.timestepping;
    switch (algorithm) {
            // CNAB2 == the classic Crank-Nicolson/Adams-bashforth algorithm
        case CNAB2:
            order_ = 2;
            Nsubsteps_ = 1;
            Ninitsteps_ = 1;
            full_ = false;
            alpha_.resize(Nsubsteps_);
            beta_.resize(Nsubsteps_);
            gamma_.resize(Nsubsteps_);
            zeta_.resize(Nsubsteps_);
            alpha_[0] = 0.5;
            beta_[0] = 0.5;
            gamma_[0] = 1.5;
            zeta_[0] = -0.5;
            break;

            // SMRK2 == Spalart-Moser-Rogers Runge-Kutta method
            // Constants taken from P.R. Spalart, R.D. Moser, M.M. Rogers,
            // Spectral methods for the Navier-Stokes equations with one infinite and
            // two periodic directions, J. Comp. Phys. 96, 297–324 (1990).
        case SMRK2:
            order_ = 2;
            Nsubsteps_ = 3;
            Ninitsteps_ = 0;
            full_ = true;
            alpha_.resize(Nsubsteps_);
            beta_.resize(Nsubsteps_);
            gamma_.resize(Nsubsteps_);
            zeta_.resize(Nsubsteps_);
            alpha_[0] = 29.0 / 96.0;
            alpha_[1] = -3.0 / 40.0;
            alpha_[2] = 1.0 / 6.0;
            beta_[0] = 37.0 / 160.0;
            beta_[1] = 5.0 / 24.0;
            beta_[2] = 1.0 / 6.0;
            gamma_[0] = 8.0 / 15.0;
            gamma_[1] = 5.0 / 12.0;
            gamma_[2] = 3.0 / 4.0;
            zeta_[0] = 0.0;
            zeta_[1] = -17.0 / 60.0;
            zeta_[2] = -5.0 / 12.0;
            break;
        default:
            cerr << "CNABstyleDNS::CNABstyleDNS(un,Ubase,nu,dt,flags,t0)\n"
                 << "error: flags.timestepping == " << algorithm << " is not a CNAB-style algorithm." << endl;
            exit(1);
    }

    // initialize fields for nonlinear terms with zero
    for (int i = 0; i < numfields_; ++i) {
        fj1_[i].setToZero();
        fj_[i].setToZero();
    }

    // define time stepping const. for impl. solver and create new nse objects
    lambda_t_.resize(Nsubsteps_);
    //   nse_.resize(Nsubsteps_);
    for (int j = 0; j < Nsubsteps_; ++j)  //{
        lambda_t_[j] = 1.0 / (beta_[j] * flags_.dt);
    //     nse_[j]=std::move(makeNSE (fields, base, lambda_t_[j], flags));
    //   }
    nse_->reset_lambda(lambda_t_);
}

CNABstyleDNS::~CNABstyleDNS() {
    // unique pointer delete themselve
}

DNSAlgorithm* CNABstyleDNS::clone(const shared_ptr<NSE>& nse) const {
    DNSAlgorithm* clone = new CNABstyleDNS(*this);
    clone->reset_nse(nse);
    return clone;
}

void CNABstyleDNS::reset_dt(Real dt) {
    //   cfl_ *= dt/flags_.dt;
    flags_.dt = dt;

    // This loop reconfigures the TauSolver objects held by nse object
    for (int j = 0; j < Nsubsteps_; ++j)  //{
        lambda_t_[j] = 1.0 / (beta_[j] * flags_.dt);
    //     nse_[j]->reset_lambda(lambda_t_[j]);
    //   }
    nse_->reset_lambda(lambda_t_);

    // For CNAB2, need to take one forward Euler step to initialize u and f cfarrays.
    switch (flags_.timestepping) {
        case CNAB2:
            full_ = false;
            break;
        default:;
    }
}

bool CNABstyleDNS::push(const vector<FlowField>& fields) {
    for (int l = 0; l < numfields_; ++l)
        swap(fj_[l], fj1_[l]);

    nse_->nonlinear(fields, fj_);

    t_ += flags_.dt;
    full_ = true;
    return full_;
}

void CNABstyleDNS::printStack() const {
    // os << "CNABstyleDNS::printStack() {" << endl;
    // printf("L2(fj) =%13.10f\n",    L2Norm(fj_));
    // printf("L2(fj1)=%13.10f\n",    L2Norm(fj1_));
    // os << "}" << endl;
}

bool CNABstyleDNS::full() const { return full_; }

void CNABstyleDNS::project() {
    for (int l = 0; l < numfields_; ++l)
        fj_[l].project(symmetries_[l]);
    // project(symm, fj1_); // no need; gets overwritten w fj_ first thing in advance
}

void CNABstyleDNS::operator*=(const vector<FieldSymmetry>& sigma) {
    assert(static_cast<int>(sigma.size()) == numfields_);
    for (int l = 0; l < numfields_; ++l)
        fj_[l] *= sigma[l];
    // project(symm, fj1_); // no need; gets overwritten w fj_ first thing in advance
}

void CNABstyleDNS::advance(vector<FlowField>& fields, int Nsteps) {
    vector<FlowField> rhs(nse_->createRHS(fields));
    vector<FlowField> lt(rhs);
    int len = rhs.size();

    // Start with time stepping loop
    for (int n = 0; n < Nsteps; ++n) {
        for (int j = 0; j < Nsubsteps_; ++j) {
            // put most recent NLT into last
            for (int l = 0; l < numfields_; ++l)
                swap(fj_[l], fj1_[l]);

            // navierstokesNL(un, ubase_, Ubase_, fj_, tmp_, tmp2_, flags_.nonlinearity);
            nse_->nonlinear(fields, fj_);

            // Goal is to compute
            // RHS = lambda_t uj + a/b lt - g/b fj - z/b fj1

            // Set convenience variables
            Real a_b = alpha_[j] / beta_[j];
            Real g_b = gamma_[j] / beta_[j];
            Real z_b = zeta_[j] / beta_[j];

            // get linear term
            nse_->linear(fields, lt);

            // Add up linear and nonlinear terms to RHS
            for (int l = 0; l < len; ++l) {
                rhs[l] = fields[l];
                rhs[l] *= lambda_t_[j];
                rhs[l].add(a_b, lt[l]);
                rhs[l].add(-g_b, fj_[l], -z_b, fj1_[l]);
            }

            // solve the implicit problem and store solution in fields
            nse_->solve(fields, rhs, j);

        }  // End of loop over substeps j

        t_ += flags_.dt;
        if (fields[0].taskid() == 0) {
            if (flags_.verbosity == PrintTime || flags_.verbosity == PrintAll)
                *flags_.logstream << t_ << ' ' << flush;
            else if (flags_.verbosity == PrintTicks)
                *flags_.logstream << '.' << flush;
        }
    }  // End of loop over main time steps n

    //   cfl_ = fields[0].CFLfactor (Ubase_, Wbase_);         //calculated on demand through DNS class
    //   cfl_ *= flags_.dealias_xz() ? 2.0*pi/3.0*flags_.dt : pi*flags_.dt;

    if (fields[0].taskid() == 0)
        if (flags_.verbosity == PrintTime || flags_.verbosity == PrintAll)
            *flags_.logstream << endl;

    return;
}

}  // namespace chflow
