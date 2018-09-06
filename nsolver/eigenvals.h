/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */
#ifndef NSOLVER_EIGENVALS_H
#define NSOLVER_EIGENVALS_H

#include "cfbasics/arglist.h"
#include "cfbasics/cfbasics.h"
#include "nsolver/arnoldi.h"
#include "nsolver/dsi.h"
#include "nsolver/lanczos.h"

namespace chflow {

class EigenvalsFlags {
   public:
    EigenvalsFlags();
    EigenvalsFlags(ArgList& args);

    std::ostream* logstream = &std::cout;

    bool isnormal = false;

    int Narnoldi = 100;
    int Nstable = 5;
    bool fixedNs = false;

    Real EPS_kry = 1e-10;
    bool centdiff = false;
    bool orthochk = false;

    std::string duname = "";
    std::string outdir = "./";

    // new flags:
    Real EPS_stab = 1e-06;

    void save(const std::string& outdir = "") const;  // save into file filebase.txt
    void load(int taskid, const std::string indir);
};

class Eigenvals {
   public:
    Eigenvals(ArgList& args);
    Eigenvals(EigenvalsFlags eigenflags);

    void solve(DSI& dsi, const Eigen::VectorXd& x, Eigen::VectorXd& dx, Real T, Real eps);
    void checkConjugacy(const Eigen::VectorXcd& u, const Eigen::VectorXcd& v);
    std::ostream* getLogstream() { return eigenflags.logstream; }

   private:
    EigenvalsFlags eigenflags;
};

std::ostream& operator<<(std::ostream& os, const EigenvalsFlags& flags);

}  // namespace chflow

#endif
