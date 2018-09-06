/**
 * Time-stepping schemes for spectral Navier-Stokes simulation
 * TimeStep manages variable time-stepping, adjusting dt to keep CFL in range
 * DNSAlgorithm is the parent class of all time-stepper classes
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef CHANNELFLOW_DNSALGO_H
#define CHANNELFLOW_DNSALGO_H

#include <memory>
#include "cfbasics/cfvector.h"
#include "cfbasics/mathdefs.h"
#include "channelflow/chebyshev.h"
#include "channelflow/dnsflags.h"
#include "channelflow/flowfield.h"
#include "channelflow/nse.h"
#include "channelflow/symmetry.h"

namespace chflow {

// DNSAlgorithm is a base class for classes representing time-stepping
// algorithms for the Navier-Stokes equations, using a Fourier x Chebyshev
// x Fourier FlowField for spatial discretization and finite-differencing
// and tau method for temporal discretization.
// NEW: Handling of the spatial discretization, geometric aspects of the
// flowfield and all about physics and boundary conditions (e.g. BaseFlow)
// have been transferred to the NSE class (FS, 20/08/2016)
class DNSAlgorithm {
   public:
    DNSAlgorithm();
    DNSAlgorithm(const DNSAlgorithm& dns);
    DNSAlgorithm(const std::vector<FlowField>& fields, const std::shared_ptr<NSE>& nse, const DNSFlags& flags);

    virtual ~DNSAlgorithm();
    // DNSAlgorithm& operator=(const DNSAlgorithm& dns);

    // PURE VIRT, definition of time stepping scheme
    virtual void advance(std::vector<FlowField>& fields, int nSteps = 1) = 0;
    virtual void project();                                           // project onto symm subspace (a member of flags)
    virtual void operator*=(const std::vector<FieldSymmetry>& symm);  // apply symmetry operator
    // virtual void reset() = 0;					// flush state, prepare for new integration
    virtual void reset_dt(Real dt) = 0;                       // PURE VIRT, somewhat expensive
    virtual bool push(const std::vector<FlowField>& fields);  // push u onto u[j] stack, t += dt
    virtual bool full() const;                                // have enough init data?

    void reset_time(Real t);
    void reset_nse(std::shared_ptr<NSE> nse);

    int order() const;       // err should scale as dt^order
    int Ninitsteps() const;  // number of steps needed to initialize

    Real dt() const;
    Real CFL(FlowField& u) const;
    Real time() const;

    cfarray<FieldSymmetry> symmetries(int ifield) const;

    const DNSFlags& flags() const;
    TimeStepMethod timestepping() const;

    virtual DNSAlgorithm* clone(const std::shared_ptr<NSE>& nse) const = 0;  // PURE VIRT, new copy of *this

    virtual void printStack() const;

   protected:
    // Temporal integration parameters
    DNSFlags flags_;  // User-defined integration parameters
    int order_;
    int numfields_;
    int Ninitsteps_;  // number of initialization steps required

    Real t_;  // time in convective units

    std::vector<Real> lambda_t_;  // time stepping factors for implicit solver
    std::shared_ptr<NSE> nse_;    // copy of pointer to navier-stokes equations of channelflow
    // defined and space-discretized in separate NSE class, ptr is given at construction
    std::vector<cfarray<FieldSymmetry>> symmetries_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////

// Multistep algorithms, all using Backwards Differentiation: SBDFk
// Based on Peyret section. The order is set by flags.timestepping.
class MultistepDNS : public DNSAlgorithm {
   public:
    MultistepDNS();
    MultistepDNS(const MultistepDNS& dns);
    MultistepDNS(const std::vector<FlowField>& fields, const std::shared_ptr<NSE>& nse, const DNSFlags& flags);

    ~MultistepDNS();

    MultistepDNS& operator=(const MultistepDNS& dns);

    virtual void advance(std::vector<FlowField>& fields, int nSteps = 1);
    virtual void project();
    virtual void operator*=(const std::vector<FieldSymmetry>& symm);
    virtual void reset_dt(Real dt);
    virtual bool push(const std::vector<FlowField>& fields);  // for initialization
    virtual bool full() const;                                // have enough init data?
    // virtual void reset();       // flush state, prepare for new integration

    virtual DNSAlgorithm* clone(const std::shared_ptr<NSE>& nse) const;  // new copy of *this

    virtual void printStack() const;

   protected:
    Real eta_;                                // new field coeff for implicit term, equals a0 in Peyret
    cfarray<Real> alpha_;                     // coefficients of field history
    cfarray<Real> beta_;                      // coefficients of nonlin. field history
    cfarray<std::vector<FlowField>> fields_;  // u[j] == u at t-j*dt for multistep algorithms
    cfarray<std::vector<FlowField>> nonlf_;   // f[j] == f at t-j*dt for multistep algorithms (nonlinear term)

    int countdown_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////
// CNRK2 and hopefully another. Based on algorithm in Peyret pg 149
class RungeKuttaDNS : public DNSAlgorithm {
   public:
    RungeKuttaDNS();
    RungeKuttaDNS(const RungeKuttaDNS& dns);
    RungeKuttaDNS(const std::vector<FlowField>& fields, const std::shared_ptr<NSE>& nse, const DNSFlags& flags);

    ~RungeKuttaDNS();

    virtual void advance(std::vector<FlowField>& fields, int nSteps = 1);
    virtual void reset_dt(Real dt);

    virtual DNSAlgorithm* clone(const std::shared_ptr<NSE>& nse) const;  // new copy of *this
   protected:
    int Nsubsteps_;
    std::vector<FlowField> Qj1_;  // Q_{j-1} (Q at previous substep)
    std::vector<FlowField> Qj_;   // Q_j     (Q at current  substep)
    cfarray<Real> A_;             // Q_{j+1} = A_j Q_j + N(u_j)
    cfarray<Real> B_;             // u_{j+1} = u_j + dt B_j Q_j + dt C_j (L u_j + L u_{j+1})
    cfarray<Real> C_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////
// A generalization of CNAB2 with substeps. Implements CNAB2 and SMRK2
class CNABstyleDNS : public DNSAlgorithm {
   public:
    CNABstyleDNS();
    CNABstyleDNS(const CNABstyleDNS& dns);
    CNABstyleDNS(const std::vector<FlowField>& fields, const std::shared_ptr<NSE>& nse, const DNSFlags& flags);

    ~CNABstyleDNS();

    //     CNABstyleDNS & operator= (const CNABstyleDNS & dns);

    virtual void advance(std::vector<FlowField>& fields, int nSteps = 1);
    virtual void project();
    virtual void operator*=(const std::vector<FieldSymmetry>& symm);  // apply symmetry operator
    virtual void reset_dt(Real dt);
    virtual bool push(const std::vector<FlowField>& fields);  // push u onto u[j] stack, t += dt for initial steps
    virtual bool full() const;                                // have enough init data?
    virtual void printStack() const;

    virtual DNSAlgorithm* clone(const std::shared_ptr<NSE>& nse) const;  // new copy of *this

   protected:
    int Nsubsteps_;  // time integration takes Nsubsteps per dt step
    bool full_;
    std::vector<FlowField> fj1_;  // f_{j-1} (nonlinear term f at previous substep)
    std::vector<FlowField> fj_;   // f_j     (nonlinear term f at current  substep)
    cfarray<Real> alpha_;         // u_{j+1} = u_j + dt L (alpha_j u_j + beta_j u_{j+1})
    cfarray<Real> beta_;          //           + dt gamma_j N(u_j) + dt zeta N(u_{j-1})
    cfarray<Real> gamma_;
    cfarray<Real> zeta_;
};

}  // namespace chflow
#endif
