/**
 * Channelflow Dynamical System Interface (DSI)
 *
 * Interface to use NSolver's dynamical system methods
 * with the Channelflow Navier-Stokes solver.
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#ifndef CFDSI_H
#define CFDSI_H

#include <sys/stat.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include "cfbasics/cfvector.h"
#include "channelflow/cfmpi.h"
#include "channelflow/chebyshev.h"
#include "channelflow/dns.h"
#include "channelflow/flowfield.h"
#include "channelflow/periodicfunc.h"
#include "channelflow/symmetry.h"
#include "channelflow/tausolver.h"
#include "channelflow/utilfuncs.h"
#include "nsolver/nsolver.h"

namespace chflow {

enum class continuationParameter {
    T,
    Re,
    P,
    Ub,
    Uw,
    ReP,
    Theta,
    ThArc,
    ThLx,
    ThLz,
    Lx,
    Lz,
    Aspect,
    Diag,
    Lt,
    Vs,
    ReVs,
    H,
    HVs,
    Rot
};

Real GMRESHookstep_vector(FlowField& u, FieldSymmetry& sigma, PoincareCondition* h,
                          const NewtonSearchFlags& searchflags, DNSFlags& dnsflags, TimeStep& dt, Real& CFL,
                          Real Unormalize);

// converts the string from "fieldstats" in diffops to a vector of Reals
std::vector<Real> fieldstats_vector(const FlowField& u);

class cfDSI : public DSI {
   public:
    /** \brief default constructor */
    cfDSI();
    virtual ~cfDSI() {}

    /** \brief Initialize cfDSI */
    cfDSI(DNSFlags& dnsflags, FieldSymmetry sigma, PoincareCondition* h, TimeStep dt, bool Tsearch, bool xrelative,
          bool zrelative, bool Tnormalize, Real Unormalize, const FlowField& u, std::ostream* os = &std::cout);

    Eigen::VectorXd eval(const Eigen::VectorXd& x) override;
    Eigen::VectorXd eval(const Eigen::VectorXd& x0, const Eigen::VectorXd& x1, bool symopt) override;
    void save(const Eigen::VectorXd& x, const std::string filebase, const std::string outdir = "./",
              const bool fieldsonly = false) override;
    void saveEigenvec(const Eigen::VectorXd& x, const std::string label, const std::string outdir) override;
    void saveEigenvec(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const std::string label1,
                      const std::string label2, const std::string outdir) override;

    Real DSIL2Norm(const Eigen::VectorXd& x) override;
    std::string stats(const Eigen::VectorXd& x) override;
    std::pair<std::string, std::string> stats_minmax(const Eigen::VectorXd& x) override;
    std::string statsHeader() override;
    void makeVector(const FlowField& u, const FieldSymmetry& sigma, const Real T, Eigen::VectorXd& x);
    void extractVector(const Eigen::VectorXd& x, FlowField& u, FieldSymmetry& sigma, Real& T);
    void toVector(const std::vector<FlowField>& u, const FieldSymmetry& sigma, const Real T, Eigen::VectorXd& x){};

    /// \name Compute derivatives of FlowField corresponding to this vector
    Eigen::VectorXd xdiff(const Eigen::VectorXd& a) override;
    Eigen::VectorXd zdiff(const Eigen::VectorXd& a) override;
    Eigen::VectorXd tdiff(const Eigen::VectorXd& a, Real epsDt) override;

    /// \name Handle continuation parameter
    void updateMu(Real mu) override;
    void chooseMu(std::string muName);
    void chooseMu(continuationParameter mu);
    std::string printMu() override;  // document
    void saveParameters(std::string searchdir) override;
    continuationParameter s2cPar(std::string muName);
    std::string cPar2s(continuationParameter cPar);
    void phaseShift(Eigen::VectorXd& x) override;
    void phaseShift(Eigen::MatrixXd& y) override;
    inline void setPhaseShifts(bool xphasehack, bool zphasehack, bool uUbasehack);
    Real observable(Eigen::VectorXd& x) override;

    Real tph_observable(Eigen::VectorXd& x) override;
    Real extractT(const Eigen::VectorXd& x) override;
    Real extractXshift(const Eigen::VectorXd& x) override;
    Real extractZshift(const Eigen::VectorXd& x) override;

    Real getCFL() const { return CFL_; };
    bool XrelSearch() const override { return xrelative_; };
    bool ZrelSearch() const override { return zrelative_; };
    bool Tsearch() const override { return Tsearch_; };

   protected:
    DNSFlags dnsflags_;
    CfMPI* cfmpi_;
    FieldSymmetry sigma_;
    PoincareCondition* h_;
    TimeStep dt_;
    bool Tsearch_;
    bool xrelative_;
    bool zrelative_;
    Real Tinit_;
    Real axinit_;
    Real azinit_;
    bool Tnormalize_;
    Real Unormalize_;
    int fcount_;
    int Nx_;
    int Ny_;
    int Nz_;
    int Nd_;
    Real Lx_;
    Real Lz_;
    Real ya_;
    Real yb_;
    Real CFL_;
    int uunk_;
    bool xphasehack_;
    bool zphasehack_;
    bool uUbasehack_;

    continuationParameter cPar_ = continuationParameter::T;
};

inline void cfDSI::setPhaseShifts(bool xphasehack, bool zphasehack, bool uUbasehack) {
    xphasehack_ = xphasehack;
    zphasehack_ = zphasehack;
    uUbasehack_ = uUbasehack;
}

void f(const FlowField& u, int N, Real dt, FlowField& f_u, const DNSFlags& flags_, std::ostream& os);

// Versions of f, G, DG that handle Poincare section calculations, additionally.
void f(const FlowField& u, Real& T, PoincareCondition* h, FlowField& fu, const DNSFlags& flags, const TimeStep& dt,
       int& fcount, Real& CFL, std::ostream& os = std::cout);

void G(const FlowField& u, Real& T, PoincareCondition* h, const FieldSymmetry& sigma, FlowField& Gu,
       const DNSFlags& flags, const TimeStep& dt, bool Tnormalize, Real Unormalize, int& fcount, Real& CFL,
       std::ostream& os = std::cout);

}  // namespace chflow

#endif
