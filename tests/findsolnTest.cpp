/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 *
 * Test based on findsoln.cpp,
 * for any info see channelflow/programs/findsoln.cpp
 */

#include <sys/stat.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include "cfbasics/cfvector.h"
#include "channelflow/cfdsi.h"
#include "channelflow/cfmpi.h"
#include "channelflow/chebyshev.h"
#include "channelflow/dns.h"
#include "channelflow/flowfield.h"
#include "channelflow/periodicfunc.h"
#include "channelflow/symmetry.h"
#include "channelflow/tausolver.h"
#include "nsolver/nsolver.h"

#include <sys/time.h>

using namespace std;
using namespace chflow;
int main(int argc, char* argv[]) {
    int failure = 0;
    bool verbose = true;
    timeval start, end;
    gettimeofday(&start, 0);
    cfMPI_Init(&argc, &argv);
    {
        int taskid = mpirank();
        if (taskid == 0) {
            cerr << "findsolnTest: " << flush;
            if (verbose) {
                cout << "\n====================================================" << endl;
                cout << "findsolnTest\n\n";
            }
        }
        string purpose(
            "Newton-Krylov-hookstep search for (relative) equilibrium or periodic orbit of plane Couette flow");
        ArgList args(argc, argv, purpose);
        nsolver::hookstepSearchFlags srchflags;
        const bool eqb =
            args.getflag("-eqb", "--equilibrium", "search for equilibrium or relative equilibrium (trav wave)");
        const bool orb =
            args.getflag("-orb", "--periodicorbit", "search for periodic orbit or relative periodic orbit");
        const bool poincsrch = args.getflag("-poinc", "--poincare",
                                            "(relative) periodic orbit search constrained to I-D=0 Poincare section");
        srchflags.xrelative =
            args.getflag("-xrel", "--xrelative", "search over x phase shift for relative orbit or eqb");
        srchflags.zrelative =
            args.getflag("-zrel", "--zrelative", "search over z phase shift for relative orbit or eqb");

        DNSFlags dnsflags;
        TimeStep dt;
        args2dnsflags(args, dnsflags, dt);

        /***/
        Real T = args.getreal("-T", "--maptime", 20.0, "initial guess for orbit period or time of eqb/reqb map f^T(u)");
        const string sigstr =
            args.getstr("-sigma", "--sigma", "", "file containing symmetry of relative solution (default == identity)");

        srchflags.epsSearch = args.getreal("-es", "--epsSearch", 1e-13, "stop search if L2Norm(s f^T(u) - u) < epsEQB");
        srchflags.epsKrylov = args.getreal("-ek", "--epsKrylov", 1e-14, "min. condition # of Krylov vectors");
        srchflags.epsDx = args.getreal("-edx", "--epsDxLinear", 1e-7, "relative size of dx to x in linearization");
        srchflags.epsDt = args.getreal("-edt", "--epsDtLinear", 1e-5, "size of dT in linearization of f^T about T");
        srchflags.epsGMRES =
            args.getreal("-eg", "--epsGMRES", 1e-3, "stop GMRES iteration when Ax=b residual is < this");
        srchflags.epsGMRESf =
            args.getreal("-egf", "--epsGMRESfinal", 0.05, "accept final GMRES iterate if residual is < this");
        srchflags.centdiff = args.getflag("-cd", "--centerdiff", "centered differencing to estimate differentials");

        srchflags.Nnewton = args.getint("-Nn", "--Nnewton", 20, "max number of Newton steps ");
        srchflags.Ngmres = args.getint("-Ng", "--Ngmres", 120, "max number of GMRES iterations per restart");
        srchflags.Nhook = args.getint("-Nh", "--Nhook", 20, "max number of hookstep iterations per Newton");

        srchflags.delta = args.getreal("-d", "--delta", 0.01, "initial radius of trust region");
        srchflags.deltaMin =
            args.getreal("-dmin", "--deltaMin", 1e-12, "stop if radius of trust region gets this small");
        srchflags.deltaMax = args.getreal("-dmax", "--deltaMax", 0.1, "maximum radius of trust region");
        srchflags.deltaFuzz = args.getreal("-df", "--deltaFuzz", 1e-06, "accept steps within (1+/-deltaFuzz)*delta");
        srchflags.lambdaMin = args.getreal("-lmin", "--lambdaMin", 0.2, "minimum delta shrink rate");
        srchflags.lambdaMax = args.getreal("-lmax", "--lambdaMax", 1.5, "maximum delta expansion rate");
        srchflags.lambdaRequiredReduction = 0.5;  // when reducing delta, reduce by at least this factor.
        srchflags.improvReq = args.getreal("-irq", "--improveReq", 1e-3,
                                           "reduce delta and recompute hookstep if improvement "
                                           " is worse than this fraction of what we'd expect from gradient");
        srchflags.improvOk = args.getreal("-iok", "--improveOk", 0.10,
                                          "accept step and keep same delta if improvement "
                                          "is better than this fraction of quadratic model");
        srchflags.improvGood = args.getreal("-igd", "--improveGood", 0.75,
                                            "accept step and increase delta if improvement "
                                            "is better than this fraction of quadratic model");
        srchflags.improvAcc = args.getreal("-iac", "--improveAcc", 0.10,
                                           "recompute hookstep with larger trust region if "
                                           "improvement is within this fraction quadratic prediction.");
        // srchflags.unormalize= args.getreal ("-un", "--unormalize", 0, "normalize residual by
        // 1/(L2Norm3d(f^T(u))-unormalize), to keep away from u=0 laminar soln");

        const int nproc0 =
            args.getint("-np0", "--nproc0", 0, "number of MPI-processes for transpose/number of parallel ffts");
        const int nproc1 = args.getint("-np1", "--nproc1", 0, "number of MPI-processes for one fft");

        srchflags.outdir = "findsolnTest_output/";
        const bool normal = args.getflag("-n", "--normalize", "return |u-v|/sqrt(|u||v|)");
        const Real tol = args.getreal("-t", "--tolerance", 1.0e-5, "max distance allowed for the test to pass");
        const string uname = "data/eq.h5";

        fftw_loadwisdom();

        if (eqb && !orb)
            srchflags.solntype = Equilibrium;
        else if (orb && !eqb)
            srchflags.solntype = PeriodicOrbit;
        else {
            cerr << "Please choose either -eqb or -orb option to search for (relative) equilibrium or (relative) "
                    "periodic orbit"
                 << endl;
            exit(1);
        }
        args.check();
        args.save("./");
        WriteProcessInfo();
        PoincareCondition* hpoincare = poincsrch ? new DragDissipation() : 0;

        FieldSymmetry sigma;
        if (sigstr.length() != 0)
            sigma = FieldSymmetry(sigstr);
        sigma *= 1.001;  // Perturb sigma

        CfMPI* cfmpi = new CfMPI(nproc0, nproc1);
        FlowField u(uname, cfmpi);
        u.optimizeFFTW(FFTW_PATIENT);
        fftw_savewisdom();
        u = FlowField(uname, cfmpi);
        u *= 1.001;  // Perturb field for test
        Real CFL;
        dnsflags.verbosity = Silent;
        Real ubulk = Re(u.profile(0, 0, 0)).mean();
        Real wbulk = Re(u.profile(0, 0, 2)).mean();
        if (abs(ubulk) < 1e-15)
            ubulk = 0.0;
        if (abs(wbulk) < 1e-15)
            wbulk = 0.0;

        ChebyCoeff Ubase = laminarProfile(dnsflags.nu, dnsflags.constraint, dnsflags.dPdx, dnsflags.Ubulk - ubulk,
                                          dnsflags.Vsuck,  // TobiasHack
                                          u.a(), u.b(), dnsflags.ulowerwall, dnsflags.uupperwall, u.Ny());
        Ubase.save("Ubase");
        ofstream logstream;
        if (u.taskid() == 0 && verbose) {
            srchflags.logstream = &cout;
            dnsflags.logstream = &cout;
        }
        ostream* os = srchflags.logstream;  // a short name for ease of use
        if (u.taskid() == 0 && verbose) {
            if (dnsflags.symmetries.length() > 0) {
                *os << "Restricting flow to invariant subspace generated by symmetries" << endl;
                *os << dnsflags.symmetries << endl;
            }

            *os << "Working directory == " << pwd() << endl;
            *os << "Command-line args == ";
            for (int i = 0; i < argc; ++i)
                *os << argv[i] << ' ';
            *os << endl;

            *os << setprecision(16);
            *os << " 1/nu == " << 1 / dnsflags.nu << " (pseudo-Reynolds)" << endl;
            *os << "   nu == " << dnsflags.nu << endl;
            *os << "sigma == " << sigma << endl;
            *os << "    T == " << T << endl;
            *os << "   dt == " << dt << endl;
            *os << "DNSFlags == " << dnsflags << endl << endl;
            *os << setprecision(8);
        }

        mkdir(srchflags.outdir);
        Real residual = GMRESHookstep_vector(u, T, sigma, hpoincare, srchflags, dnsflags, dt, CFL, 0);

        if (u.taskid() == 0)
            *os << "Final search residual is " << residual << endl;

        gettimeofday(&end, 0);
        Real sec = (Real)(end.tv_sec - start.tv_sec);
        Real ms = (((Real)end.tv_usec) - ((Real)start.tv_usec));
        Real timeused = sec + ms / 1000000.;
        if (u.taskid() == 0 && verbose)
            cout << "duration for this findsoln run: " << timeused << endl;

        // Test result against known equilibrium
        FlowField u_ref("data/eq.h5", cfmpi);

        // FlowField u_best("findsolnTest_output/xbest.h5", cfmpi);

        Real nrm = normal ? 1.0 / sqrt(L2Norm(u) * L2Norm(u_ref)) : 1;
        double l2d = nrm * L2Dist(u, u_ref);

        if (u.taskid() == 0) {
            if (l2d > tol) {
                if (verbose)
                    cout << endl << "Final L2Dist: " << l2d << " > tol = " << tol << endl;
                cerr << "\t** FAIL **" << endl;
                cout << "\t** FAIL **" << endl;
                failure = 1;
            } else {
                if (verbose)
                    cout << endl << "Final L2Dist: " << l2d << endl;
                cerr << "\t   pass   " << endl;
                cout << "\t   pass   " << endl;
            }
        }

        fftw_savewisdom();
        delete cfmpi;
    }
    cfMPI_Finalize();
    return failure;
}
