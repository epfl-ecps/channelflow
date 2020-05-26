/**
 * System class of Oberbeck-Boussinesq equations for standard channel flows
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 *
 * Original author: Florian Reetz
 */

#ifndef OBE_H
#define OBE_H

#include "channelflow/diffops.h"
#include "channelflow/flowfield.h"
#include "channelflow/nse.h"
#include "channelflow/tausolver.h"
#include "modules/ilc/ilcflags.h"

namespace chflow {

// nonlinear term of NSE plus the linear coupling term to the temperature equation
void momentumNL(const FlowField& u, const FlowField& T, ChebyCoeff Ubase, ChebyCoeff Wbase, FlowField& f,
                FlowField& tmp, ILCFlags flags);

// nonlinear term of heat equation plus the linear coupling term to the momentum equation
void temperatureNL(const FlowField& u, const FlowField& T, ChebyCoeff Ubase, ChebyCoeff Wbase, ChebyCoeff Tbase,
                   FlowField& f, FlowField& tmp, ILCFlags flags);

class OBE : public NSE {
   public:
    //     OBE ();
    //     OBE (const NSE& nse);
    OBE(const std::vector<FlowField>& fields, const ILCFlags& flags);
    OBE(const std::vector<FlowField>& fields, const std::vector<ChebyCoeff>& base, const ILCFlags& flags);
    virtual ~OBE();

    void nonlinear(const std::vector<FlowField>& infields, std::vector<FlowField>& outfields) override;
    void linear(const std::vector<FlowField>& infields, std::vector<FlowField>& outfields) override;

    // calls a tausolver for each Fourier mode
    void solve(std::vector<FlowField>& outfields, const std::vector<FlowField>& infields, const int i = 0) override;

    // redefines the tausolver objects with new time-stepping constant (allocates memory for tausolver at first use)
    void reset_lambda(const std::vector<Real> lambda_t) override;

    // vector of RHS is smaller than of fields because of missing pressure equation
    std::vector<FlowField> createRHS(const std::vector<FlowField>& fields) const override;

    // returns vector of symmetries confining the vector of fields to a subspace
    std::vector<cfarray<FieldSymmetry>> createSymmVec() const override;

    const ChebyCoeff& Tbase() const;
    const ChebyCoeff& Ubase() const override;
    const ChebyCoeff& Wbase() const override;

   protected:
    HelmholtzSolver*** heatsolver_;  // 3d cfarray of tausolvers, indexed by [i][mx][mz] for substep, Fourier Mode x,z

    ILCFlags flags_;  // User-defined integration parameters
    Real gsingx_;
    Real gcosgx_;
    Real Tref_;

    // additional base solution profiles
    ChebyCoeff Tbase_;    // temperature base profile (physical)
    ChebyCoeff Tbaseyy_;  // 2. deriv. of temperature base profile
    ChebyCoeff Pbasey_;   // wall normal pressure gradient (y-dependent)
    Real Pbasex_;         // along wall pressure gradient (const)

    // constant terms
    ComplexChebyCoeff Cu_;  // constant
    ComplexChebyCoeff Cw_;
    ComplexChebyCoeff Ct_;
    bool nonzCu_;
    bool nonzCw_;
    bool nonzCt_;

    ComplexChebyCoeff Tk_;
    ComplexChebyCoeff Rtk_;

   private:
    void createILCBaseFlow();
    void initILCConstraint(const FlowField& u);  // method called only at construction
    void createConstants();

    bool baseflow_;
    bool constraint_;
};

// Construct laminar flow profile for given flow parameters.
// [a,b]   == y position of [lower, upper] walls
// [ua,ub] == in-plane speed of [lower, upper] walls
// constraint == is mean pressure gradient fixed, or mean (bulk) velocity?
// dPdx, Ubulk == value of fixed pressure gradient or fixed Ubulk velocity
// Vsuck == suction velocity at walls (asymptotic suction boundary layer)
ChebyCoeff laminarVelocityProfile(Real gamma, Real dPdx, Real Ubulk, Real ua, Real ub, Real a, Real b, int Ny,
                                  ILCFlags flags);

ChebyCoeff linearTemperatureProfile(Real a, Real b, int Ny, ILCFlags flags);

Real hydrostaticPressureGradientX(ILCFlags flags);
ChebyCoeff hydrostaticPressureGradientY(ChebyCoeff Tbase, ILCFlags flags);

}  // namespace chflow
#endif