/**
 * An assortment of convenience functions for channelflow/programs
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef CHANNELFLOW_UTILFUNCS_H
#define CHANNELFLOW_UTILFUNCS_H

#include <unistd.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "channelflow/config.h"

#include "cfbasics/arglist.h"
#include "cfbasics/cfarray.h"

#include "cfbasics/mathdefs.h"
#include "channelflow/diffops.h"
#include "channelflow/nse.h"

namespace chflow {

void WriteProcessInfo(int argc, char* argv[], std::string filename = "processinfo",
                      std::ios::openmode mode = std::ios::out);

void save(const std::string& filebase, Real T, const FieldSymmetry& tau);
void load(const std::string& filebase, Real& T, FieldSymmetry& tau);

// Produce plots of various slices, etc.
void plotfield(const FlowField& u_, const std::string& outdir, const std::string& label, int xstride = 1,
               int ystride = 1, int zstride = 1, int nx = 0, int ny = -1,
               int nz = 0);  // ny=-1 => reset ny = (Ny-1)/2, midplane

void plotspectra(const FlowField& u_, const std::string& outdir, const std::string& label, bool showpadding = false);

// Produce plots of various slices, etc.
void plotxavg(const FlowField& u_, const std::string& outdir, const std::string& label);

// Return a translation that maximizes the s-symmetry of u.
// (i.e. return tau // that minimizes L2Norm(s(tau(u)) - tau(u))).
FieldSymmetry optimizePhase(const FlowField& u, const FieldSymmetry& s, int Nsteps = 10, Real residual = 1e-13,
                            Real damp = 1.0, bool verbose = true, Real x0 = 0.0, Real z0 = 0.0);

// Compute the optimal phase shift between fields u0 and u1 along x
// so that f(ax) = sigma(ax,0) u1 - u0 is minimized. Tolerance applies to ax, not
// to f. nSampling is the number of values for ax at which f is sampled
// before entering the minimization procedure.
Real optPhaseShiftx(const FlowField& u0, const FlowField& u1, Real amin, Real amax, int nSampling = 3,
                    Real tolerance = 1e-3);

// Return a phase shift that will fix the x-phase of a x-traveling wave,
// by phase shifting to get even or odd parity of u(x,0,Lz/2,i) about
// x=Lx/2. axguess is a guess for appropriate phase shift in units ax = xshift/Lx
FieldSymmetry xfixphasehack(FlowField& u, Real axguess = 0.0, int i = 0, parity p = Odd, std::string mode = "std");

// Return a phase shift that will fix the z-phase of a z-traveling wave,
// by z-phase shifting to get even or odd parity of <u_i>_{xy}(z) about
// z=Lz/2. azguess is a guess for appropriate phase shift in units az = zshift/Lz
FieldSymmetry zfixphasehack(FlowField& u, Real azguess = 0.0, int i = 0, parity p = Odd, std::string mode = "std");

void fixuUbasehack(FlowField& u, ChebyCoeff U);

// Return |P(u)|^2/|u|^2 where P(u) is projection onto
// sign ==  1 : s-symmetric subspace.
// sign == -1 : s-antisymmetric subspace.
Real PuFraction(const FlowField& u, const FieldSymmetry& s, int sign);

// Check divergence, dirichlet, normality, and maybe orthogonality of basis
void basischeck(const std::vector<RealProfile>& e, bool orthocheck = false);

void fixDiri(ChebyCoeff& f);
void fixDiriMean(ChebyCoeff& f);
void fixDiriNeum(ChebyCoeff& f);
void fixDiri(ComplexChebyCoeff& f);
void fixDiriMean(ComplexChebyCoeff& f);
void fixDiriNeum(ComplexChebyCoeff& f);

class FieldSeries {
   public:
    FieldSeries();
    FieldSeries(int N);  // Interpolate with (N-1)th-order polynomial

    void push(const FlowField& f, Real t);
    void interpolate(FlowField& f, Real t) const;
    bool full() const;

   private:
    cfarray<Real> t_;
    cfarray<FlowField> f_;
    int emptiness_;  // init val N, decrement in push(), stacks are full at 0
};

// Try to get time from filename, otherwise return 0
// The expected filename format is ..../ {letters} {time} {letters}
Real tFromFilename(const std::string filename);

// Extract t from filenames and return which t is smaller (useful for sorting)
bool comparetimes(const std::string& s0, const std::string& s1);

// Return channlflow version numbers, e.g. 0, 9, 20 for channelflow-0.9.20.
void channelflowVersion(int& major, int& minor, int& update);

// Set dnsflags required to create a baseflow
DNSFlags setBaseFlowFlags(ArgList& args, std::string& Uname, std::string& Wname);

// Creates baseflow with BCs
std::vector<ChebyCoeff> baseFlow(int Ny, Real a, Real b, DNSFlags& flags, std::string Uname, std::string Wname);

}  // namespace chflow
#endif
