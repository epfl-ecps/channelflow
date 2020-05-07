/**
 * Dynamical System Interface for the Inclined Layer Convection module
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 *
 * Original author: Florian Reetz
 */

#ifndef ILCDSI_H
#define ILCDSI_H

#include <sys/stat.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include "cfbasics/cfvector.h"
#include "channelflow/cfdsi.h"
#include "channelflow/cfmpi.h"
#include "channelflow/chebyshev.h"
#include "channelflow/flowfield.h"
#include "channelflow/periodicfunc.h"
#include "channelflow/symmetry.h"
#include "channelflow/tausolver.h"
#include "channelflow/utilfuncs.h"
#include "modules/ilc/ilc.h"
#include "nsolver/nsolver.h"

using namespace std;

namespace chflow {

enum class ilc_continuationParameter {
    none,
    Ra,
    Pr,
    gx,
    gz,
    gxEps,
    Grav,
    Tref,
    P,
    Uw,
    UwGrav,
    Rot,
    Theta,
    ThArc,
    ThLx,
    ThLz,
    Lx,
    Lz,
    Aspect,
    Diag,
    Vs,
    VsNu,
    VsH,
    H
};

// Real GMRESHookstep_vector (FlowField& u, FlowField& alpha, Real& T, FieldSymmetry& sigma,
//                            PoincareCondition* h,
//                            const nsolver::hookstepSearchFlags& searchflags,
//                            DNSFlags& dnsflags, VEDNSFlags& vednsflags, TimeStep& dt, Real& CFL, Real Unormalize);

std::vector<Real> ilcstats(const FlowField& u, const FlowField& temp, const ILCFlags flags = ILCFlags());

// header for fieldstats
string ilcfieldstatsheader(const ILCFlags flags = ILCFlags());

// header for fieldstats with parameter t
string ilcfieldstatsheader_t(std::string muname, const ILCFlags flags = ILCFlags());
string ilcfieldstats(const FlowField& u, const FlowField& temp, const ILCFlags flags = ILCFlags());
string ilcfieldstats_t(const FlowField& u, const FlowField& temp, Real t, const ILCFlags flags = ILCFlags());
FlowField totalVelocity(const FlowField& velo, const ILCFlags flags);
FlowField totalTemperature(const FlowField& temp, const ILCFlags flags);
Real buoyPowerInput(const FlowField& u, const FlowField& temp, const ILCFlags flags, bool relative = true);
Real dissipation(const FlowField& u, const ILCFlags flags, bool normalize = true, bool relative = true);
Real heatinflux(const FlowField& temp, const ILCFlags flags, bool normalize = true, bool relative = true);
Real heatcontent(const FlowField& ttot, const ILCFlags flags);
Real Nusselt_plane(const FlowField& u, const FlowField& temp, const ILCFlags flags, bool relative = true);

class ilcDSI : public cfDSI {
   public:
    /** \brief default constructor */
    ilcDSI();
    virtual ~ilcDSI() {}

    /** \brief Initialize ilcDSI */
    ilcDSI(ILCFlags& ilcflags, FieldSymmetry sigma, PoincareCondition* h, TimeStep dt, bool Tsearch, bool xrelative,
           bool zrelative, bool Tnormalize, Real Unormalize, const FlowField& u, const FlowField& temp,
           std::ostream* os = &std::cout);

    Eigen::VectorXd eval(const Eigen::VectorXd& x) override;
    Eigen::VectorXd eval(const Eigen::VectorXd& x0, const Eigen::VectorXd& x1,
                         bool symopt) override;  // needed for multishooting
    void save(const Eigen::VectorXd& x, const string filebase, const string outdir = "./",
              const bool fieldsonly = false) override;

    string stats(const Eigen::VectorXd& x) override;
    pair<string, string> stats_minmax(const Eigen::VectorXd& x) override;
    string statsHeader() override;
    void phaseShift(Eigen::VectorXd& x) override;
    void phaseShift(Eigen::MatrixXd& y) override;
    Real extractT(const Eigen::VectorXd& x) override;
    Real extractXshift(const Eigen::VectorXd& x) override;
    Real extractZshift(const Eigen::VectorXd& x) override;

    void makeVectorILC(const FlowField& u, const FlowField& temp, const FieldSymmetry& sigma, const Real T,
                       Eigen::VectorXd& x);
    void extractVectorILC(const Eigen::VectorXd& x, FlowField& u, FlowField& temp, FieldSymmetry& sigma, Real& T);

    /// \name Compute derivatives of the two FlowFields contained in this vector
    Eigen::VectorXd xdiff(const Eigen::VectorXd& a) override;
    Eigen::VectorXd zdiff(const Eigen::VectorXd& a) override;
    Eigen::VectorXd tdiff(const Eigen::VectorXd& a, Real epsDt) override;
    /// \name Handle continuation parameter
    void updateMu(Real mu) override;
    void chooseMuILC(std::string muName);
    void chooseMuILC(ilc_continuationParameter mu);
    string printMu() override;  // document
    void saveParameters(string searchdir) override;
    ilc_continuationParameter s2ilc_cPar(std::string muName);
    string ilc_cPar2s(ilc_continuationParameter cPar);

    // Save real eigenvectors
    void saveEigenvec(const Eigen::VectorXd& x, const string label, const string outdir) override;
    // Save complex conjugate eigenvectors pair
    void saveEigenvec(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const string label1, const string label2,
                      const string outdir) override;

   protected:
    ILCFlags ilcflags_;
    ilc_continuationParameter ilc_cPar_ = ilc_continuationParameter::none;
};

// G(x) = G(u,sigma) = (sigma f^T(u) - u) for orbits
void G(const FlowField& u, const FlowField& temp, Real& T, PoincareCondition* h, const FieldSymmetry& sigma,
       FlowField& Gu, FlowField& Gtemp, const ILCFlags& ilcflags, const TimeStep& dt, bool Tnormalize, Real Unormalize,
       int& fcount, Real& CFL, ostream& os);
void f(const FlowField& u, const FlowField& temp, Real& T, PoincareCondition* h, FlowField& f_u, FlowField& f_temp,
       const ILCFlags& ilcflags_, const TimeStep& dt_, int& fcount, Real& CFL, ostream& os);

}  // namespace chflow

#endif
