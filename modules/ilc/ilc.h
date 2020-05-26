/**
 * Main interface to handle DNS of ILC
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 *
 * Original author: Florian Reetz
 */

#ifndef ILC_H
#define ILC_H

#include "channelflow/dns.h"
#include "channelflow/dnsalgo.h"
#include "modules/ilc/ilcflags.h"
#include "modules/ilc/obe.h"

namespace chflow {

int field2vector_size(const FlowField& u, const FlowField& temp);

/** \brief Turn the two flowfields for velocity and temperature into one Eigen vector
 * \param[in] u velocity field
 * \param[in] temp temperature field
 * \param[in] x vector for the linear algebra
 *
 * The vectorization of u is analog to the field2vector, temparture is piped entirely
 * into the vector (a single independent dimensions)
 */
void field2vector(const FlowField& u, const FlowField& temp, Eigen::VectorXd& x);

/** \brief Turn  one Eigen vector into the two flowfields for velocity and temperature
 * \param[in] x vector for the linear algebra
 * \param[in] u velocity field
 * \param[in] temp temperature field
 *
 * The vectorization of u is analog to the field2vector, temperature is piped entirely
 * into the vector (a single independent dimension)
 */
void vector2field(const Eigen::VectorXd& x, FlowField& u, FlowField& temp);

/** \brief extension of the DNSFlags class
 *
 * VEDNSFlags class, holds all additional parameters for viscoelastic fluids
 */

/** \brief wrapper class of DNSAlgorithm and ILC
 *
 *
 */
class ILC : public DNS {
   public:
    //     ILC ();
    //     ILC (const ILC & ilc);
    ILC(const std::vector<FlowField>& fields, const ILCFlags& flags);
    //     ILC (const vector<FlowField> & fields, const vector<ChebyCoeff> & base,
    // 	    const ILCFlags & flags);

    virtual ~ILC();

    ILC& operator=(const ILC& ilc);

    //     virtual void advance (vector<FlowField> & fields, int nSteps = 1);
    //
    //     virtual void reset_dt (Real dt);
    //     virtual void printStack () const;

    const ChebyCoeff& Tbase() const;
    const ChebyCoeff& Ubase() const;
    const ChebyCoeff& Wbase() const;

   protected:
    std::shared_ptr<OBE> main_obe_;
    std::shared_ptr<OBE> init_obe_;

    std::shared_ptr<OBE> newOBE(const std::vector<FlowField>& fields, const ILCFlags& flags);
    std::shared_ptr<OBE> newOBE(const std::vector<FlowField>& fields, const std::vector<ChebyCoeff>& base,
                                const ILCFlags& flags);
};

}  // namespace chflow
#endif  // ILC_H