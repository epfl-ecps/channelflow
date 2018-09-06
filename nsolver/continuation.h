/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */

#ifndef NSOLVER_CONTINUATION_H
#define NSOLVER_CONTINUATION_H

#include "nsolver/dsi.h"
#include "nsolver/newton.h"
#include "nsolver/nsolver.h"

namespace chflow {
class ContinuationFlags {
   public:
    ContinuationFlags();
    ContinuationFlags(ArgList& args);

    bool arclength;            // arclength continuation
    Real initialParamStep;     ///< initial relative increment for quadratic extrapolation in continuation parameter
    bool adjdt;                ///< adjust dt between continuation steps to keep CFL in bounds
    Real s0;                   ///< start value for arclength (arbitrary)
    Real ds0;                  ///< initial increment for arclength continuation
    Real dsmax;                ///< Maximum arclength step
    Real dsmin;                ///< Minimum arclength step
    Real guessErrMin;          ///< minimum error for extrapolated guesses
    Real guessErrMax;          ///< maximum error for extrapolated guesses
    int digits;                ///< number of digits for MuD.asc
    bool orbOut;               // use flag to print max/min of statistics into MuEmax.asc/MuEmin.asc
    bool restartMode;          // Continuation in restart mode
    int maxSteps;              // Maximum number of continuation steps
    Real muRef;                // mu reference
    int initStep = 0;          // takes a non-zero number only when restart mode is on
    int nlastFailedSteps = 0;  // only for restart continuation
    Real guesserr_last;        // last guess error; only for restart continuation
    bool haveTargetMu;         // Abort if a target value of mu is reached
    Real targetMu;             // The target value

    void save(const std::string& outdir = "") const;
    void load(int taskid, const std::string indir);
    const std::vector<std::string> getFlagList();
};

// Real continuation1(DSI& dsi, NewtonAlgorithm& N, const VectorXd& x0, const Real mu0, ContinuationFlags& cflags);

Eigen::VectorXd quadraticInterpolate(cfarray<Eigen::VectorXd>& xn, const cfarray<Real>& mun, Real mu, Real eps = 1e-13);

bool readContinuationInfo(std::string restartdir[3], ContinuationFlags& cflags);

/** Perform hookstep continuation.
 *
 * \param[in] dsiG the DSI object specifying equations via eval() and parameter updates via updateMu()
 * \param[in] newton an object of a class derived from NewtonAlgorithm, provides solve function
 * \param[in] x the three known solution
 * \param[in] mu parameter value for which the solutions were found
 * \param[in] cflags specify parameters for continuation algorithm
 *
 * \return parameter value mu of last successfull search step
 *
 * Intermediate results get saved into directories search-i where i is the step counter.
 * The parameter value, L2Norm and arclength are saved into MuE.asc
 */
Real continuation(DSI& dsiG, Newton& newton, cfarray<Eigen::VectorXd> x, cfarray<Real> mu, ContinuationFlags& cflags);

}  // namespace chflow
#endif  // NSOLVER_CONTINUATION_H
