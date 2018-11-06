/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */

#include <iostream>
#include "cfbasics/cfbasics.h"
#include "channelflow/cfdsi.h"
#include "channelflow/flowfield.h"
#include "nsolver/nsolver.h"

using namespace std;
using namespace Eigen;
using namespace chflow;

int main(int argc, char* argv[]) {
    cfMPI_Init(&argc, &argv);
    {
        ArgList args(argc, argv, "parametric continuation of invariant solution");

        ContinuationFlags cflags(args);
        cflags.save();

        unique_ptr<Newton> N;
        bool Rxsearch, Rzsearch, Tsearch;
        NewtonSearchFlags searchflags(args);
        searchflags.save();
        N = unique_ptr<Newton>(new NewtonAlgorithm(searchflags));

        Rxsearch = searchflags.xrelative;
        Rzsearch = searchflags.zrelative;
        Tsearch = searchflags.solntype == PeriodicOrbit ? true : false;
        const bool Tnormalize = (Tsearch || searchflags.laurette) ? false : true;

        Real diagonal = 0.0;  // sqrt(Lx^2 + Lz^2)
        int W = 24;
        const Real EPSILON = 1e-13;

        DNSFlags dnsflags(args, searchflags.laurette);
        TimeStep dt(dnsflags);
        dnsflags.save();

        args.section("Program options");
        const string muname = args.getstr(
            "-cont", "--continuation", "",
            "continuation parameter, one of [Re P Ub Uw ReP Theta ThLx ThLz Lx Lz Aspect Diag Lt Vs ReVs H HVs Rot]");
        const string sigmastr =
            args.getstr("-sigma", "--sigma", "", "file containing sigma of sigma f^T(u) - u = 0 (default == identity)");
        const Real Unormalize = args.getreal("-un", "--unormalize", 0.0, "lower bound in energy for search");
        Real Lxtarg = args.getreal("-Lxtarg", "--LxTarget", 2 * pi, "with -cont Ltarg, aim for this value of Lx");
        Real Lztarg = args.getreal("-Lztarg", "--LzTarget", pi, "with -cont Ltarg, aim for this value of Lz");
        const bool xphasehack =
            args.getflag("-xphhack", "--xphasehack",
                         "fix x phase so that u(x,0,Lz/2,0) - mean = 0 at x=Lx/2 (HACK for u UNSYMM in x)");
        const bool zphasehack =
            args.getflag("-zphhack", "--zphasehack",
                         "fix z phase so that     d/dz <u>_{xy}(z) = 0 at z=Lz/2 (HACK for u UNSYMM in z)");
        const bool uUbasehack =
            args.getflag("-uUbhack", "--uUbasehack", "fix u/Ubase decomposition so that <du/dy> == 0 at walls.");
        const int nproc0 =
            args.getint("-np0", "--nproc0", 0, "number of MPI-processes for transpose/number of parallel ffts");
        const int nproc1 = args.getint("-np1", "--nproc1", 0, "number of MPI-processes for one fft");

        // check if invariant solution is relative
        FieldSymmetry givenSigma;
        if (sigmastr.length() != 0)
            givenSigma = FieldSymmetry(sigmastr);
        bool relative = Rxsearch || Rzsearch || !givenSigma.isIdentity();

        bool restart = cflags.restartMode;

        string uname(""), restartdir[3];
        if (restart) {
            bool solutionsAvail = readContinuationInfo(restartdir, cflags);

            if (!solutionsAvail) {
                restartdir[0] = args.getpath(1, "<string>", "directory containing solution 1");
                restartdir[1] = args.getpath(2, "<string>", "directory containing solution 2");
                restartdir[2] = args.getpath(3, "<string>", "directory containing solution 3");
            }

        } else {
            uname = args.getstr(1, "<flowfield>", "initial solution from which to start continuation");
        }
        args.check();

        if (muname == "") {
            cerr << "Please choose --continuation with one of [Re P Ub Uw ReP Theta ThLx ThLz Lx Lz Aspect Diag Lt Vs "
                    "ReVs H HVs Rot]"
                 << endl;
            exit(1);
        }
        args.save();
        WriteProcessInfo(argc, argv);

        CfMPI* cfmpi = &CfMPI::getInstance(nproc0, nproc1);

        FlowField u[3];
        FieldSymmetry sigma[3];
        cfarray<Real> mu(3);
        Real T[3];
        unique_ptr<cfDSI> dsi;
        dsi = unique_ptr<cfDSI>(new cfDSI());

        if (restart) {
            cout << "Restarting from previous solutions. Please be aware that the DNSFlags "
                 << "from the corresponding directories will overwrite any specified command line parameters!" << endl;
            for (int i = 0; i < 3; ++i) {
                u[i] = FlowField(restartdir[i] + "ubest", cfmpi);
                if (i == 0) {
                    printout("Optimizing FFTW...", false);
                    u[i].optimizeFFTW(FFTW_PATIENT);
                    fftw_savewisdom();
                    printout("done");
                    u[i] = FlowField(restartdir[i] + "ubest", cfmpi);
                }
                if (relative)
                    sigma[i] = FieldSymmetry(restartdir[i] + "sigmabest.asc");
                load(mu[i], restartdir[i] + "mu.asc");
                if (Tsearch)
                    load(T[i], restartdir[i] + "Tbest.asc");
                else
                    T[i] = dnsflags.T;
            }
            cout << "Loading dnsflags from " + restartdir[0] + "dnsflags.txt, neglecting command line switches."
                 << endl;
            dnsflags.load(cfmpi->taskid(), restartdir[0]);
            dt = TimeStep(dnsflags);

            // Some consistency checks of the initial solutions for a continuation in restart mode
            if (!relative && (sigma[0] != sigma[1] || sigma[1] != sigma[2]))
                cferror("loadSolutions error : initial symmetries should be equal for non-relative searchs");
            if (muname == "Aspect") {
                // Check that diagonal of u[2]'s is same as u[0]. Rescale Lx,Lz keeping aspect ratio
                // constant if diagonals are different. Normally reloaded aspect-continued data will
                // have constant diagonals, but do check and fix in order to restart data with some
                // small fuzz in diagonal.
                diagonal = pythag(u[1].Lx(), u[1].Lz());
                for (int n = 2; n >= 0; n -= 2) {
                    if (abs(diagonal - pythag(u[n].Lx(), u[n].Lz())) >= EPSILON) {
                        cout << "Diagonal of u[" << n << "] != diagonal of u[1]. Rescaling Lx,Lz...." << endl;
                        Real alpha_n = atan(1 / mu[n]);
                        u[n].rescale(diagonal * cos(alpha_n), diagonal * sin(alpha_n));
                    }
                }
            } else if (muname == "Diag") {
                // Check that aspect ratio of u[2]'s is same u[0]. Rescale Lx,Lz keeping diagonal
                // constant if aspect ratios differ. Normally reloaded diagonl-continued data will
                // have constant aspect ratio, but do check and fix in order to restart data with some
                // small fuzz in aspect ratio.
                Real aspect = u[1].Lx() / u[1].Lz();
                Real alpha = atan(1.0 / (u[1].Lx() / u[1].Lz()));
                for (int n = 2; n >= 0; n -= 2) {
                    if (abs(aspect - u[n].Lx() / u[n].Lz()) >= EPSILON) {
                        cout << "Aspect ratio of u[" << n << "] != aspect ratio of u[1]. Rescaling Lx,Lz...." << endl;
                        Real diagonal_n = pythag(u[n].Lx(), u[n].Lz());
                        u[n].rescale(diagonal_n * cos(alpha), diagonal_n * sin(alpha));
                    }
                }
            } else if (muname == "Lt") {
                // Check that all solutions are colinear with (u[1].Lx, u[1].Lz) and (Lxtarg,Lztarg).
                // If not, rescale u[2] and u[0] Lx,Lz so that they are
                Real phi = atan((Lztarg - u[1].Lz()) / (Lxtarg - u[1].Lx()));
                for (int i = 2; i >= 0; i -= 2) {
                    if (abs(atan((Lztarg - u[i].Lz()) / (Lxtarg - u[i].Lx())) - phi) >= EPSILON) {
                        cout << "u[" << i
                             << "] is not colinear with (u[1].Lx, u[1].Lz) and (Lxtarg,Lztarg). Rescaling to fix."
                             << endl;
                        u[i].rescale(Lxtarg - mu[i] * cos(phi), Lztarg - mu[i] * sin(phi));
                    }
                }
            }
            cout << endl << "loaded the following data..." << endl;
        } else {  // not a restart
            // Compute initial data points for extrapolation from perturbations of given solution
            u[1] = FlowField(uname, cfmpi);
            project(dnsflags.symmetries, u[1], "initial value u", cout);
            fixdivnoslip(u[1]);

            u[2] = u[1];
            printout("Optimizing FFTW...", false);
            u[2].optimizeFFTW(FFTW_PATIENT);  // Overwrites u[2]
            printout("done");
            fftw_savewisdom();
            u[2] = u[1];
            u[0] = u[1];

            if (sigmastr.length() != 0)
                sigma[0] = sigma[1] = sigma[2] = givenSigma;
            T[0] = T[1] = T[2] = dnsflags.T;

            // begin superfluous output
            Real phi, Ltarget;
            if (muname == "Lt") {
                phi = atan((Lztarg - u[1].Lz()) / (Lxtarg - u[1].Lx()));
                Ltarget = spythag(u[1].Lx(), Lxtarg, u[1].Lz(), Lztarg, phi);
            } else {
                Lxtarg = u[1].Lx();
                Lztarg = u[1].Lz();
                phi = 0.0;
                Ltarget = 0.0;
            }
            cout << "Some geometrical parameters. Phi and Ltarg are used for continuation along line in 2d (Lx,Lz) "
                    "plane."
                 << endl;
            cout << "  phi == " << phi << endl;
            cout << "Ltarg == " << Ltarget << endl;
            cout << "distance  == " << pythag(Lxtarg - u[1].Lx(), Lztarg - u[1].Lz()) << endl;
            cout << "u[1].Lx() == " << u[1].Lx() << endl;
            cout << "   Lxtarg == " << Lxtarg << endl;
            cout << "u[1].Lz() == " << u[1].Lz() << endl;
            cout << "   Lztarg == " << Lztarg << endl;

            cout << "Lxtarg - Ltarg cos(phi) == " << (Lxtarg - Ltarget * cos(phi)) << endl;
            cout << "Lztarg - Ltarg sin(phi) == " << (Lztarg - Ltarget * sin(phi)) << endl;
            cout << "dt_Lx == " << dt.dt() / u[1].Lx() << endl;

            cout << "set up the following initial data..." << endl;
        }
        cout << setw(4) << "i" << setw(W) << "T" << setw(W) << setw(W) << "L2Norm(u)" << setw(W) << "sigma" << endl;
        for (int i = 2; i >= 0; --i) {
            Real l2normui = L2Norm(u[i]);
            cout << setw(4) << i << setw(W) << T[i] << setw(W) << l2normui << setw(W) << sigma[i] << setw(W) << endl;
        }
        // end superfluous output

        dsi = unique_ptr<cfDSI>(new cfDSI(dnsflags, sigma[0], 0, dt, Tsearch, Rxsearch, Rzsearch, Tnormalize,
                                          Unormalize, u[0], N->getLogstream()));

        cout << setprecision(8);
        printout("Working directory == " + pwd());
        printout("Command-line args == ");
        dnsflags.verbosity = Silent;
        for (int i = 0; i < argc; ++i)
            cout << argv[i] << ' ';
        cout << endl;
        cout << "sigma == " << sigma[0] << endl;
        cout << "T     == " << T[0] << endl;
        cout << "dPdx  == " << dnsflags.dPdx << endl;
        cout << "dt    == " << dt.dt() << endl;
        cout << "DNSFlags == " << dnsflags << endl << endl;

        dsi->setPhaseShifts(xphasehack, zphasehack, uUbasehack);
        dsi->chooseMu(muname);
        if (!restart) {
            mu[1] = dsi->mu();
            mu[0] = mu[1] + cflags.initialParamStep;
            mu[2] = mu[1] - cflags.initialParamStep;
        }
        if (muname == "Aspect" || muname == "Diag") {
            cout << "aspect ratio || diagonal continuation : " << endl;
            cout << setprecision(15);
            cout.setf(std::ios::left);
            cout << setw(4) << "n" << setw(20) << muname << setw(20) << "aspect" << setw(20) << "diagonal" << endl;
            for (int n = 2; n >= 0; --n)
                cout << setw(4) << n << setw(20) << mu[n] << setw(20) << u[n].Lx() / u[n].Lz() << setw(20)
                     << pythag(u[n].Lx(), u[n].Lz()) << endl;
            cout.unsetf(std::ios::left);
        }

        if (muname == "Lt") {
            cout << "Lx,Lz target continuation : " << endl;
            cout << setprecision(15);
            cout.setf(std::ios::left);
            cout << setw(4) << "n" << setw(20) << muname << setw(20) << "Lx" << setw(20) << "Lz" << endl;
            for (int i = 2; i >= 0; --i)
                cout << setw(4) << i << setw(20) << mu[i] << setw(20) << u[i].Lx() << setw(20) << u[i].Lz() << endl;
            cout.unsetf(std::ios::left);
        }

        cfarray<VectorXd> x(3);
        for (int i = 0; i <= 2; ++i) {
            dsi->updateMu(mu[i]);
            dsi->makeVector(u[i], sigma[i], T[i], x[i]);
        }

        int Nunk = x[0].rows();
        int Nunk_total = Nunk;
#ifdef HAVE_MPI
        MPI_Allreduce(&Nunk, &Nunk_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
        cout << Nunk_total << " unknowns" << endl;

        Real muFinal = continuation(*dsi, *N, x, mu, cflags);
        cout << "Final mu is " << muFinal << endl;
    }
    cfMPI_Finalize();

    return 0;
}
