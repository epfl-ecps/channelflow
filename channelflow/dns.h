/**
 * Time-integration class for spectral Navier-Stokes simulator
 * DNS is a class for integrating Navier-Stokes equation.
 * DNSFlags is used to specify the integration parameters of DNS.
 * TimeStep manages variable time-stepping, adjusting dt to keep CFL in range
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef CHANNELFLOW_DNS_H
#define CHANNELFLOW_DNS_H

#include <memory>
#include "cfbasics/mathdefs.h"
#include "channelflow/dnsalgo.h"
#include "channelflow/dnsflags.h"
#include "channelflow/flowfield.h"
#include "channelflow/nse.h"

namespace chflow {

// DNS is a wrapper class for DNSAlgorithms. It's the main class for
// integrating the Navier-Stokes equations in top-level programs.
// Specify the integration algorithm and other parameters in the DNSFlags.
// If you like, you can construct and use specific DNS algorithms like
// MultiStepDNS in top-level programs --any class derived from DNSAlgorithm.
// Look in example codes for examples of initialization and use.

class DNS {
   public:
    DNS();
    DNS(const DNS& dns);
    DNS(const std::vector<FlowField>& fields, const DNSFlags& flags);
    DNS(const std::vector<FlowField>& fields, const std::vector<ChebyCoeff>& base, const DNSFlags& flags);

    virtual ~DNS();

    DNS& operator=(const DNS& dns);

    void loaddnsflags(int taskid, DNSFlags& flags, TimeStep& dt,  // UNIMPLEMENTED
                      const std::string indir);
    void advance(std::vector<FlowField>& fields, int nSteps = 1);

    void project();                                           // Project onto symmetric subspace
    void operator*=(const std::vector<FieldSymmetry>& symm);  // Apply symmetry to internal fields

    // Convert potentially fake pressure q and true pressure p, back and forth.
    // TODO: check if still need. If yes, implement for general vector of fields
    void up2q(FlowField u, FlowField p, FlowField& q) const;
    void uq2p(FlowField u, FlowField q, FlowField& p) const;

    // void reset();                  // flush state, prepare for new integration
    virtual void reset_dt(Real dt);
    virtual void reset_time(Real t);
    virtual void reset_gradp(Real dPdx, Real dPdz);    // change dPdx and enforce const dPdx
    virtual void reset_bulkv(Real Ubulk, Real Wbulk);  // change Ubulk and enforce const Ubulk

    bool push(const std::vector<FlowField>& fields);  // push into u[j] stack from another DNS,
    virtual bool full() const;                        // is u[j] full, can we commence timestepping?

    virtual int order() const;       // err should scale as dt^order
    virtual int Ninitsteps() const;  // number of steps needed to initialize

    Real nu() const;
    virtual Real dt() const;
    virtual Real CFL(FlowField& u) const;
    virtual Real time() const;
    virtual Real dPdx() const;  // the mean pressure gradient at the current time
    Real dPdz() const;
    virtual Real Ubulk() const;  // the actual bulk velocity at the current time
    Real Wbulk() const;
    virtual Real dPdxRef() const;  // the mean press grad enforced during integration
    Real dPdzRef() const;
    virtual Real UbulkRef() const;  // the bulk velocity enforced during integ.
    Real WbulkRef() const;          // the bulk velocity enforced during integ.

    virtual const ChebyCoeff& Ubase() const;
    virtual const ChebyCoeff& Wbase() const;
    const DNSFlags& flags() const;
    virtual TimeStepMethod timestepping() const;

    virtual void printStack() const;

   protected:
    std::shared_ptr<NSE> main_nse_;
    std::shared_ptr<NSE> init_nse_;
    DNSAlgorithm* main_algorithm_;
    DNSAlgorithm* init_algorithm_;

    std::shared_ptr<NSE> newNSE(const std::vector<FlowField>& fields, const DNSFlags& flags);
    std::shared_ptr<NSE> newNSE(const std::vector<FlowField>& fields, const std::vector<ChebyCoeff>& base,
                                const DNSFlags& flags);

    DNSAlgorithm* newAlgorithm(const std::vector<FlowField>& fields, const std::shared_ptr<NSE>& nse,
                               const DNSFlags& flags);
};

/****************************************************************************
 // Return the nu that produces a given Reynolds = Uh/nu for given flow
 // parameters, where U is defined as the maximum difference between the
 // mean flow of the laminar solution for these flow parameters and the
 // speed of either wall, i.e. greater(abs(Umean-ua), abs(Umean-ub)).
 // This function provides a conversion mechanism for Reynolds number
 // (which is a convenient input quantity) to viscosity (which is better
 // for specifying a flow precisely), which works for both plane Couette
 // flow and Poiseuille flow and any flow in between.

Real viscosity(Real Reynolds, MeanConstraint constraint, Real dPdx,
       Real Ubulk, Real a, Real b, Real ua, Real ub);

Real viscosity(Real Reynolds, const DNSFlags& flags, Real a, Real b);

// Return Reynolds = Uh/nu where U is max diff between mean(U(y)) and walls.
// For classic plane couette, this works out to U = 1/2 relative wall speed
// For classic poiseuille,    this works out to U = mean(U(y))
Real reynolds(Real nu, MeanConstraint constraint, Real dPdx, Real Ubulk,
      Real a, Real b, Real ua, Real ub);
Real reynolds(const DNSFlags& flags, Real a, Real b);
//Real reynolds(Real nu, const ChebyCoeff& u);
******************************************************************************/

// **********************************************************************************
// BEGIN EXPERIMENTAL CODE: DNS that integrates to a Poincare section and maps back to a
// fundamental domain via symmetries, whenever certain boundaries are crossed.

// DNSPoincare is a class for integrating u to a Poincare section and mapping u back
// into a fundamental domain of a discrete symmetry group. The Poincare intersections
// are well-tested. The fundamental domain stuff is not. For the Poincare section,
// the stopping condition is a geometric condition rather than a stopping time.
// As of now there are two forms for the Poincare condtion, I-D=0 (DragDissipation)
// or (u(t) - ustar, estar) = 0 (PlaneIntersection). The fundamental domain is defined as
// (u(t), e[n]) >= 0. e[n] is antisymmetric under symmetry sigma[n], so that when
// (u(t), e[n]) <  0, we can get back to the fundamental domain by applying sigma[n],
// since (sigma[n] u(t), e[n]) = (u(t), sigma[n] e[n]) = (u(t), -e[n]) > 0.

// advanceToSection should be used this way
// 1. Start with some initial condition (u,q,t), t arbitrary
// 2. Make repeated calls to advanceToSection(u,q,nsteps,crosssign,eps). Each call will
//    advance (u,q,t) by dT = nSteps*dt and map u(t) back into the fundamental domain
//    should it leave. The sign argument determines which kinds of crossings will
//    return: sign<0 => only dh/dt < 0,
//            sign=0 => either direction
//            sign>0 => only dh/dt > 0,
// 3. Check the return value of advanceToSection. It will be
//      FALSE if u(t) does not cross the section during the the advancement and
//      TRUE  if u(t) does cross the section
// 4. When the return value is TRUE, you can then access the values of (u,q,t)
//    at the crossing through ucrossing(), pcrossing(), and tcrossing().
// 5. The signcrossing() member function returns the sign of dh/dt at h==0
// 6. Continue on with more calls to advanceToSection to find the next intersection,
//    if you like.
// The integration to the section is done in multiple steps and multiple calls
// to advanceToSection so that the field can be saved, projected, etc.
// over the course of integration by the caller.

class PoincareCondition {
   public:
    virtual ~PoincareCondition();
    PoincareCondition();
    virtual Real operator()(const FlowField& u) = 0;
};

// Section defined by (u - ustar, estar) == (u, estar) - (ustar, estar)
class PlaneIntersection : public PoincareCondition {
   public:
    ~PlaneIntersection();
    PlaneIntersection();
    PlaneIntersection(const FlowField& ustar, const FlowField& estar);
    Real operator()(const FlowField& u);

   private:
    FlowField estar_;  // A normal that defines orientation of section
    Real cstar_;       // L2IP(ustar, estar), defines location of section
};

// Section defined by I-D == drag(u) - dissipation(u) == 0
class DragDissipation : public PoincareCondition {
   public:
    ~DragDissipation();
    DragDissipation();
    Real operator()(const FlowField& u);
};

class DNSPoincare : public DNS {
   public:
    DNSPoincare();

    DNSPoincare(FlowField& u, PoincareCondition* h, const DNSFlags& flags);

    DNSPoincare(FlowField& u, const cfarray<FlowField>& e, const cfarray<FieldSymmetry>& sigma, PoincareCondition* h,
                const DNSFlags& flags);

    bool advanceToSection(FlowField& u, FlowField& q, int nSteps, int crosssign = 0, Real Tmin = 0,
                          Real epsilon = 1e-13);

    const FlowField& ucrossing() const;
    const FlowField& pcrossing() const;
    Real hcrossing() const;  // value of poincare condition at crossing
    Real tcrossing() const;  // time of poincare crossing
    int scrossing() const;   // -1 or 1 for dh/dt<0 or dh/dt>0

    Real hcurrent() const;  // return h(u) at current timestep
   private:
    cfarray<FlowField> e_;          // Defines fundamental domain. See comments above.
    cfarray<FieldSymmetry> sigma_;  // Maps u(t) back into fundamental domain. See above.
    PoincareCondition* h_;          // The poincare condition h(u) == 0.

    FlowField ucrossing_;  // velocity field at crossing
    FlowField pcrossing_;  // pressure field at crossing
    Real tcrossing_;       // time at crossing
    int scrossing_;        // sign of dh/dt at crossing
    Real hcrossing_;       // value of (*h)(ucrossing_)
    Real hcurrent_;        // value of (*h)(u)
    Real t0_;              // starting time, used to check t-t0 >= Tmin
};
// END EXPERIMENTAL CODE

}  // namespace chflow
#endif
