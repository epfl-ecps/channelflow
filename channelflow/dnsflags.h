/**
 * Control parameters for time-integration by spectral Navier-Stokes simulator
 * DNSFlags specifies all relevant parameters for integrating Navier-Stokes equation in standard channel domains.
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef CHANNELFLOW_DNSFLAGS_H
#define CHANNELFLOW_DNSFLAGS_H

#include "cfbasics/arglist.h"
#include "cfbasics/cfvector.h"
#include "cfbasics/mathdefs.h"
#include "channelflow/chebyshev.h"
#include "channelflow/flowfield.h"

namespace chflow {

class DNSFlags;

// Enum types for specifying the behavior of DNS, fields of DNSFlags.

enum VelocityScale { WallScale, ParabolicScale };  // BulkScale
enum BaseFlow { ZeroBase, LinearBase, ParabolicBase, LaminarBase, SuctionBase, ArbitraryBase };
enum MeanConstraint { PressureGradient, BulkVelocity };
enum TimeStepMethod { CNFE1, CNAB2, CNRK2, SMRK2, SBDF1, SBDF2, SBDF3, SBDF4 };
enum NonlinearMethod {
    Rotational,
    Convection,
    Divergence,
    SkewSymmetric,
    Alternating,
    Alternating_,
    LinearAboutProfile
};
enum Dealiasing { NoDealiasing, DealiasXZ, DealiasY, DealiasXYZ };
enum Verbosity { Silent, PrintTime, PrintTicks, VerifyTauSolve, PrintAll };

// Urgh, boilerplate code
VelocityScale s2velocityscale(const std::string& s);
BaseFlow s2baseflow(const std::string& s);
MeanConstraint s2constraint(const std::string& s);
TimeStepMethod s2stepmethod(const std::string& s);
NonlinearMethod s2nonlmethod(const std::string& s);
Dealiasing s2dealiasing(const std::string& s);
Verbosity s2verbosity(const std::string& s);

int getBodyforcefromLine(int taskid, std::ifstream& is);

std::string velocityscale2string(VelocityScale vs);
std::string baseflow2string(BaseFlow bf);
std::string constraint2string(MeanConstraint mc);
std::string stepmethod2string(TimeStepMethod ts);
std::string nonlmethod2string(NonlinearMethod nm);
std::string dealiasing2string(Dealiasing d);
std::string verbosity2string(Verbosity v);

std::ostream& operator<<(std::ostream& os, VelocityScale v);
std::ostream& operator<<(std::ostream& os, BaseFlow b);
std::ostream& operator<<(std::ostream& os, MeanConstraint m);
std::ostream& operator<<(std::ostream& os, TimeStepMethod t);
std::ostream& operator<<(std::ostream& os, NonlinearMethod n);
std::ostream& operator<<(std::ostream& os, Dealiasing d);
std::ostream& operator<<(std::ostream& os, Verbosity v);

class BodyForce {
   public:
    BodyForce();

    /** \brief The infamous virtual destructor */
    virtual ~BodyForce() = default;

    Vector operator()(Real x, Real y, Real z, Real t);
    void eval(Real t, FlowField& f);
    virtual void eval(Real x, Real y, Real z, Real t, Real& fx, Real& fy, Real& fz);
    virtual bool isOn(Real t);
};

// Specify the behavior of NSIntegrators by setting fields of DNSFlags.
class DNSFlags {
   public:
    //        type name       default
    DNSFlags(Real nu = 0.0025, Real dPdx = 0.0, Real dPdz = 0.0, Real Ubulk = 0.0, Real Wbulk = 0.0, Real Uwall = 1.0,
             Real ulowerwall = 0.0, Real uupperwall = 0.0, Real wlowerwall = 0.0, Real wupperwall = 0.0,
             Real theta = 0.0, Real Vsuck = 0.0, Real rotation = 0.0, Real t0 = 0.0, Real T = 20.0, Real dT = 1.0,
             Real dt = 0.03125, bool variabledt = true, Real dtmin = 0.001, Real dtmax = 0.2, Real CFLmin = 0.4,
             Real CFLmax = 0.6, Real symmetryprojectioninterval = 100.0, BaseFlow baseflow = LaminarBase,
             MeanConstraint constraint = PressureGradient, TimeStepMethod timestepping = SBDF3,
             TimeStepMethod initstepping = SMRK2, NonlinearMethod nonlinearity = Rotational,
             Dealiasing dealiasing = DealiasXZ, BodyForce* bodyforce = 0, bool taucorrection = true,
             Verbosity verbosity = PrintTicks, std::ostream* logstream = &std::cout);

    DNSFlags(ArgList& args, const bool laurette = false);

    /** \brief The infamous virtual destructor */
    virtual ~DNSFlags() = default;

    bool dealias_xz() const;
    bool dealias_y() const;

    virtual void save(const std::string& outdir = "") const;  // save into file filebase.txt
    virtual void load(int taskid, const std::string indir);
    virtual const std::vector<std::string> getFlagList();

    BaseFlow baseflow;             // utot = u + Ubase(y) ex
    MeanConstraint constraint;     // Enforce const press grad or const bulk vel
    TimeStepMethod timestepping;   // Time-stepping algorithm
    TimeStepMethod initstepping;   // Algorithm for initializing multistep methods
    NonlinearMethod nonlinearity;  // Method of calculating nonlinearity of NS eqn
    Dealiasing dealiasing;         // Use 3/2 rule to eliminate aliasing
    BodyForce* bodyforce;          // Body force, zero if pointer set to 0
    bool taucorrection;            // Remove divergence caused by discretization

    Real nu;                            // Kinematic viscosity nu
    Real Vsuck;                         // suction velocity
    Real rotation;                      // dimensionless rotation around the z-axis
    Real theta;                         // tilt of the domain relative to downstream
    Real dPdx;                          // Constraint value for mean flow: pressure gradient in x
    Real dPdz;                          // Constraint value for mean flow: pressure gradient in z
    Real Ubulk;                         // Constraint value for mean flow: bulk velocity in x
    Real Wbulk;                         // Constraint value for mean flow: bulk velocity in z
    Real Uwall;                         // wall speed downstream
    Real ulowerwall;                    // lower wall speed along x, e.g. -1 for plane couette
    Real uupperwall;                    // upper wall speed along x, e.g. +1 for plane couette
    Real wlowerwall;                    // lower wall speed along z
    Real wupperwall;                    // upper wall speed along z
    Real t0;                            // start time
    Real T;                             // final time
    Real dT;                            // print interval
    Real dt;                            // time step
    bool variabledt;                    // use variable time step
    Real dtmin;                         // lower bound for time step
    Real dtmax;                         // upper bound for time step
    Real CFLmin;                        // lower bound for CFL number
    Real CFLmax;                        // upper bound for CFL number
    int symmetryprojectioninterval;     // Only project onto symmetries at this interval
    Verbosity verbosity;                // Print diagnostics, times, ticks, or nothing
    std::ostream* logstream;            // stream for output
    cfarray<FieldSymmetry> symmetries;  // restrict u(t) to these symmetries

   protected:
    void args2BC(ArgList& args);
    void args2numerics(ArgList& args, const bool laurette = false);
};

std::ostream& operator<<(std::ostream& os, const DNSFlags& flags);

// TimeStep keeps dt between dtmin and dtmax, and CFL between CFLminand CFLmax,
// in such a way that dt*n = dT for some integer n. That's useful if you
// want to plot/save data at regular dT intervals, but use a variable timestep
// dt for efficiency. You can mandate a fixed timestep by setting dtmin==dtmax.
// For example of use, see example codes.

class TimeStep {
   public:
    TimeStep();
    TimeStep(Real dt, Real dtmin, Real dtmax, Real dT, Real CFLmin, Real CFLmax, bool variable = true);
    TimeStep(DNSFlags& flags);

    // If variable, adjust dt to keep CFLmin<=CFL<=CFLmax (soft),
    // and dtmin<=dt<=dtmax (hard). Returns true if dt changes, false otherwise
    bool adjust(Real CFL, bool verbose = true, std::ostream& os = std::cout);
    bool adjustToMiddle(Real CFL, bool verbose = true, std::ostream& os = std::cout);

    // to bring any variable * dt below a maximum
    bool adjust(Real a, Real a_max, bool verbose = true, std::ostream& os = std::cout);
    bool adjustToDesired(Real a, Real a_des, bool verbose = true, std::ostream& os = std::cout);

    // tweak dT and dt to fit T exactly
    bool adjust_for_T(Real T, bool verbose = true, std::ostream& os = std::cout);

    int n() const;    // n*dt == dT
    int N() const;    // N*dT == T
    Real dt() const;  // integration timestep
    Real dtmin() const;
    Real dtmax() const;
    Real dT() const;  // plot/CFL-check interval
    Real T() const;   // total integration time
    Real CFL() const;
    Real CFLmin() const;
    Real CFLmax() const;
    bool variable() const;
    operator Real() const;  // same as dt()

   private:
    int n_;
    int N_;
    Real dt_;
    Real dtmin_;  //
    Real dtmax_;  //
    Real dT_;     // dT_ == n_*dt_, plot interval
    Real T_;      // T_  == N_*dt_, total integration time
    Real CFLmin_;
    Real CFL_;
    Real CFLmax_;
    bool variable_;
};

std::ostream& operator<<(std::ostream& os, const TimeStep& ts);

}  // namespace chflow
#endif
