/**
 * Control parameters for time-integration within the ILC module
 * ILCFlags specifies all relevant parameters for integrating the Oberbeck-Boussinesq equations in doubly periodic channel domains.
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 *
 * Original author: Florian Reetz
 */

#ifndef ILCFLAGS_H
#define ILCFLAGS_H

#include "channelflow/dnsflags.h"
#include "channelflow/utilfuncs.h"

namespace chflow {

/** \brief extension of the DNSFlags class for ILC
 *
 * ILCFlags class, holds all additional parameters for convective shear flows
 */
class ILCFlags : public DNSFlags {
    // Is derived from DNSFlags to keep saving and loading consistent

   public:
    ILCFlags(Real nu = 0.025,  // values are just below primary thermal instability
             Real kappa = 0.025, Real alpha = 1.0, Real grav = 1.0, Real rho_ref = 1.0, Real t_ref = 0.0,
             Real gammax = 0.0, Real gammaz = 0.0, Real ulowerwall = 0.0, Real uupperwall = 0.0, Real wlowerwall = 0.0,
             Real wupperwall = 0.0, Real tlowerwall = 0.5, Real tupperwall = -0.5, Real ystats = 0);
    //     ILCFlags (const ILCFlags& ilcflags);
    //     ILCFlags (const DNSFlags& flags);
    //     ILCFlags (const std::string& filebase);
    ILCFlags(ArgList& args, const bool laurette = false);

    /** \brief The infamous virtual destructor */
    virtual ~ILCFlags() = default;

    Real kappa;
    Real alpha;
    Real grav;
    Real rho_ref;
    Real t_ref;
    Real gammax;
    Real gammaz;

    Real tlowerwall;
    Real tupperwall;

    Real ystats;

    //     Real freefall;
    //     Real Treference;

    cfarray<FieldSymmetry> tempsymmetries;  // restrict temp(t) to these symmetries

    // override DNSFlags::save/load. Both methods include a call to the parent method
    virtual void save(const std::string& savedir = "") const override;
    virtual void load(int taskid, const std::string indir) override;
};

}  // namespace channelflow
#endif  // ILCFLAGS_H