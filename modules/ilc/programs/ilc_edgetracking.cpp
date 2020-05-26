/**
 * Edgetracking program, like the program for pure shear flows.
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 *
 * Original author: Florian Reetz
 */

#include <string.h>
#include <fstream>
#include <iostream>
#include "channelflow/dns.h"
#include "channelflow/flowfield.h"
#include "channelflow/poissonsolver.h"
#include "channelflow/utilfuncs.h"
#include "modules/ilc/ilcdsi.h"

#include <sys/time.h>

using namespace std;
using namespace chflow;

enum Attractor { NoAttractor, Laminar, Turbulent };

const string tab = "\t";

template <class T>
void push(vector<T>& a, T b) {
    int l = a.size();
    for (int i = 0; i < l - 1; ++i) {
        a[i] = a[i + 1];
    }
    a[l - 1] = b;
}

class AttractorFlags {
   public:
    AttractorFlags(Real EcfH = 0, Real EcfL = 0, Real L2H = 0, Real L2L = 0,
                   //             Real EnstH = 0,
                   //             Real EnstL = 0,
                   bool normToHeight = 0, int tMaxDecay = 0);

    // Threshold values
    Real EcfH;
    Real EcfL;
    Real L2H;
    Real L2L;
    //         Real EnstH;
    //         Real EnstL;
    bool normToHeight;
    int tMaxDecay;
};

class EdgetrackingFlags {
   public:
    EdgetrackingFlags();
    Real tLastWrite;
    Real epsilonAdvance;
    Real epsilonBisection;
    Real lambdaStep;
    Real nBisectTrajectories;
    Real tMinAttractor;
    Real tMaxAttractor;
    bool verifyLH;
    bool saveMinima;
    Real saveInterval;
    bool saveUH;
    bool saveUL;
    bool keepUL;
    bool checkConvergence;
    bool returnFieldAtMinEcf;
    string savedir;
    string directoryEnergy;
    string directoryBisections;
    string directoryFF;
    string directoryMinima;
    string filenameEnergyL;
    string filenameEnergyH;
    string filenameTB;
};

class ConvergenceData {
   public:
    ConvergenceData();
    vector<Real> EcfMax;
    vector<Real> EcfMin;
    vector<int> tEcfMin;
};

int EdgeStateTracking(FlowField& uL, FlowField& uH, FlowField& tempL, FlowField& tempH,
                      const EdgetrackingFlags& etflags, const AttractorFlags& aflags, const TimeStep& dt,
                      const ILCFlags& ilcflags, ostream& os);

void CreateFiles(const EdgetrackingFlags& etflags);
Real Bisection(FlowField& uL, FlowField& uH, FlowField& tempL, FlowField& tempH, Real t0,
               const EdgetrackingFlags& etflags, const AttractorFlags& aflags, const TimeStep& dt,
               const ILCFlags& ilcflags, ostream& os = std::cout);  // Return: lambdaH
bool AdvanceLH(FlowField& uL0, FlowField& uH0, FlowField& tempL0, FlowField& tempH0, int& t0,
               const EdgetrackingFlags& etflags, const TimeStep& dt, const ILCFlags& flags,
               const AttractorFlags& aflags, ostream& os, ConvergenceData& cd);
void ChooseNewLH(const FlowField& u0, const FlowField& temp0, Real& lambdaL, Real& lambdaH, const Real t,
                 const EdgetrackingFlags& etflags, const AttractorFlags& aflags, const TimeStep& dt,
                 const ILCFlags& ilcflags, ostream& os = std::cout);
void SaveForContinuation(const FlowField& uL, const FlowField& uH, const FlowField& tempL, const FlowField& tempH,
                         const string dirBisections, Real t);

Attractor CalcAttractor(const FlowField& u0, const FlowField& temp0, FlowField& returnUField, FlowField& returnTField,
                        const AttractorFlags& aflags, const TimeStep& dt, const ILCFlags& flags,
                        const string energyfile, const string saveFinalFF = "", const int tMin = 0,
                        const int tMax = 10000, bool verbose = false, const int t0 = 0, std::ostream& os = std::cout);
Attractor CalcAttractor(const FlowField& u0, const FlowField& temp0, const AttractorFlags& aflags, const TimeStep& dt,
                        const ILCFlags& flags, const string energyfile, const string saveFinalFF = "",
                        const int tMin = 0, const int tMax = 10000, bool verbose = false, const int t0 = 0,
                        std::ostream& os = std::cout);
Attractor attractor(const FlowField& u, const AttractorFlags& aflags);  // attractor is only defined on velocity field
Attractor attractor(const FlowField& u, int& tDecay, Real lastEcf, const AttractorFlags& aflags);
std::string a2s(Attractor a);

/** Relative L2-Difference between two flowfields */
Real RelL2Diff(const FlowField& u1, const FlowField& u2) { return L2Dist(u1, u2) / (L2Norm(u1) + L2Norm(u2)); }

int main(int argc, char* argv[]) {
    cfMPI_Init(&argc, &argv);
    {
        WriteProcessInfo(argc, argv);
        ArgList args(argc, argv, "Find edge state in ILC");

        ILCFlags ilcflags(args);
        TimeStep dt(ilcflags);
        ilcflags.verbosity = Silent;

        args.section("Program options");
        const bool cont = args.getflag("-c", "--continue", "continue from a previous calculation");

        AttractorFlags aflags;
        aflags.EcfH =
            args.getreal("-EcfH", "--EcfHigh", 0, "threshold in cross flow energy to consider flowfield turbulent");
        aflags.EcfL =
            args.getreal("-EcfL", "--EcfLow", 0, "threshold in cross flow energy to consider flowfield laminar");
        aflags.L2H = args.getreal("-L2H", "--L2High", 0, "threshold in L2Norm to consider flowfield turbulent");
        aflags.L2L = args.getreal("-L2L", "--L2Low", 0, "threshold in L2Norm to consider flowfield laminar");
        aflags.normToHeight =
            args.getflag("-n2h", "--normToHeight", "norm threshold values to height of box (instead of 1)");
        aflags.tMaxDecay =
            args.getint("-maxDcy", "--tMaxDecay", 0,
                        "Consider a flowfield as laminar if it monotonically decays for that many timeuints");

        EdgetrackingFlags etflags;
        etflags.tMaxAttractor = args.getreal("-tMaxAtt", "--tMaxAttractor", 10000,
                                             "maximum time for waiting that flowfield reaches attractor");
        etflags.tMinAttractor =
            args.getreal("-tMinAtt", "--tMinAttractor", 0, "minimum time for waiting that flowfield reaches attractor");
        etflags.epsilonAdvance = args.getreal("-epsA", "--epsilonAdvance", 1e-4,
                                              "maximal separation of flowfields before starting new bisectioning");
        etflags.epsilonBisection =
            args.getreal("-epsB", "--epsilonBisection", 1e-6, "maximal separation of flowfields after bisectioning");
        etflags.nBisectTrajectories =
            args.getint("-nBis", "--nBisecTrajectories", 1, "Minimium number of trajectories for each bisection");
        const bool chooseLHfirst = args.getflag(
            "-cLH", "--chooseLHfirst",
            "start edgetracking by choosing uH and uL, i.e. if you are not sure to have a turbulent field");
        etflags.lambdaStep =
            args.getreal("-lS", "--lambdaStep", 0.1, "adjust lambda by this value when looking for new laminar state");
        Real lambdaH = args.getreal("-lH", "--lambdaH", 1, "initial value of lambdaH (e.g. for continuation)");
        Real lambdaL = args.getreal("-lL", "--lambdaL", 0, "initial value of lambdaL (e.g. for continuation)");
        etflags.keepUL = args.getflag("-keepUL", "--keepUL", "keep UL instead of UH");
        etflags.verifyLH =
            args.getflag("-vrfyLH", "--verifyLH",
                         "keep on iterating uH and uL after they are separated to verify they are laminar/turbulent");
        etflags.saveInterval = args.getint("-s", "--saveInterval", 0, "save flowfield(s) to disk every s time units");
        etflags.saveUL = args.getbool("-saveUL", "--saveUL", false, "save uL");
        etflags.saveUH = args.getbool("-saveUH", "--saveUH", true, "save uH");
        etflags.saveMinima = args.getbool("-saveMin", "--saveMinima", false, "save at minima of cross flow energy");
        const int nproc0 =
            args.getint("-np0", "--nproc0", 0, "number of MPI-processes for transpose/number of parallel ffts");
        const int nproc1 = args.getint("-np1", "--nproc1", 0, "number of MPI-processes for one fft");
        const string logfile = args.getstr("-log", "--logfile", "stdout", "output log (filename or \"stdout\")");
        etflags.savedir = args.getstr("-sd", "--savedir", "./", "path to save directory");
        const string u_ifname = (cont) ? "" : args.getstr(2, "initial flowfield", "initial velocity flowfield");
        const string temp_ifname = (cont) ? "" : args.getstr(1, "initial flowfield", "initial temperature flowfield");
        etflags.directoryEnergy = etflags.savedir + "energy/";
        etflags.directoryBisections = etflags.savedir + "bisections/";
        etflags.directoryFF = etflags.savedir + "flowfields/";
        etflags.directoryMinima = etflags.savedir + "minima/";
        etflags.filenameEnergyL = etflags.savedir + "energyL";
        etflags.filenameEnergyH = etflags.savedir + "energyH";
        etflags.filenameTB = etflags.savedir + "tBisections";

        args.check();
        args.save();
        ilcflags.save(etflags.savedir);

        CfMPI* cfmpi = &CfMPI::getInstance(nproc0, nproc1);
        etflags.tLastWrite = 0;

        ofstream logstream;
        if (logfile == "stdout" || logfile == "cout") {
            ilcflags.logstream = &cout;
        } else {
            if (cfmpi->taskid() == 0)
                logstream.open((etflags.savedir + logfile).c_str());
            ilcflags.logstream = &logstream;
        }
        ostream& os = (ostream&)*ilcflags.logstream;

        printout("Edge state tracking algorithm", os);
        printout("Working directory == " + pwd(), os);
        printout("Command-line args == ", false, os);
        for (int i = 0; i < argc; ++i)
            printout(string(argv[i]) + " ", false, os);
        stringstream sstr;
        sstr << ilcflags;
        printout("\nILCFlags: " + sstr.str(), os);

        if ((ilcflags.symmetries.length() > 0) || (ilcflags.tempsymmetries.length() > 0)) {
            printout("Restricting flow to invariant subspace generated by symmetries", os);
            stringstream sstr2u, sstr2t;
            sstr2u << "Velocity: " << ilcflags.symmetries;
            sstr2t << "Temperature: " << ilcflags.tempsymmetries;
            printout(sstr2u.str(), os);
            printout(sstr2t.str(), os);
        }

        printout("", os);  // newline

        // ******** Preparations ********
        Real t = 0;
        FlowField u, uL, uH;
        FlowField temp, tempL, tempH;

        fftw_loadwisdom();
        if (!cont) {
            CreateFiles(etflags);
            u = FlowField(u_ifname, cfmpi);
            temp = FlowField(temp_ifname, cfmpi);
            if (temp.Nd() != 1)
                cferror("Error initializing temp field: number of dimensions must be 1!");
            uH = u;  // temp for calculating projection error and optimizing FFTW
            if (ilcflags.symmetries.length() > 0) {
                printout("Projecting Velocity FlowField to invariant subspace generated by symmetries", os);
                stringstream sstr2;
                sstr2 << ilcflags.symmetries;
                printout(sstr2.str(), os);
                u.project(ilcflags.symmetries);
                printout("Projection error == " + r2s(L2Dist(u, uH)));
            }
            tempH = temp;  // tmp for calculating projection error and optimizing FFTW
            if (ilcflags.tempsymmetries.length() > 0) {
                printout("Projecting Temperature FlowField to invariant subspace generated by symmetries", os);
                stringstream sstr2;
                sstr2 << ilcflags.tempsymmetries;
                printout(sstr2.str(), os);
                temp.project(ilcflags.tempsymmetries);
                printout("Projection error == " + r2s(L2Dist(temp, tempH)));
            }
            uH.optimizeFFTW(FFTW_PATIENT);
            tempH.optimizeFFTW(FFTW_PATIENT);

            fftw_savewisdom();
            if (chooseLHfirst) {
                ChooseNewLH(u, temp, lambdaL, lambdaH, 0., etflags, aflags, dt, ilcflags, os);
            }
            uH = u;
            uH *= lambdaH;
            uL = u;
            uL *= lambdaL;
            tempH = temp;
            tempH *= lambdaH;
            tempL = temp;
            tempL *= lambdaL;
        } else {
            ifstream Tlast(etflags.filenameTB);
            while (Tlast >> t)
                ;
            ilcflags.t0 = t;
            string fffileH = etflags.directoryBisections + "uH" + r2s(t);
            string fffileL = etflags.directoryBisections + "uL" + r2s(t);
            string tempfileH = etflags.directoryBisections + "tempH" + r2s(t);
            string tempfileL = etflags.directoryBisections + "tempL" + r2s(t);
            printout("Loading flowfields from " + fffileL + " and " + fffileH, os);
            uH = FlowField(fffileH, cfmpi);
            uH.project(ilcflags.symmetries);
            uL = FlowField(fffileL, cfmpi);
            uL.project(ilcflags.symmetries);
            tempH = FlowField(tempfileH, cfmpi);
            tempH.project(ilcflags.tempsymmetries);
            tempL = FlowField(tempfileL, cfmpi);
            tempL.project(ilcflags.tempsymmetries);
            u = uH;
            uH.optimizeFFTW(FFTW_PATIENT);
            uH = u;

            // Read file energyH and try to extract last time when something was saved (it's the first number in the
            // last not-empty line not beginning with #)
            ifstream fin(etflags.filenameEnergyH.c_str());
            if (fin) {
                string s1;
                istringstream s2;
                while (getline(fin, s1)) {
                    if (s1 != "" && s1.find("#") == string::npos)
                        s2.str(s1);
                }
                s2 >> etflags.tLastWrite;
                s1 = "Found " + i2s((int)etflags.tLastWrite) + " as last time something was written to energyH";
                printout(s1, os);
            }
            if (etflags.tLastWrite >= ilcflags.T - 1) {
                cferror("Already over time limit");
            }
        }

        EdgeStateTracking(uL, uH, tempL, tempH, etflags, aflags, dt, ilcflags, os);
    }
    cfMPI_Finalize();
    return 0;
}

/**
The actual edge state tracking algorithm
*/
int EdgeStateTracking(FlowField& uL_, FlowField& uH_, FlowField& tempL_, FlowField& tempH_,
                      const EdgetrackingFlags& etflags, const AttractorFlags& aflags, const TimeStep& dt,
                      const ILCFlags& ilcflags, ostream& os) {
    int t = ilcflags.t0;
    FlowField u, uL(uL_), uH(uH_);
    FlowField temp, tempL(tempL_), tempH(tempH_);

    printout("Starting edge tracking at t=" + i2s((int)t), os);

    // Main timestepping loop
    bool finished = false;
    ConvergenceData convergencedata;
    while (!finished && t <= ilcflags.T) {
        // Step 1: Bisectioning
        printout("\nt = " + i2s((int)t) + ": Starting Bisection");
        // Write time to file
        if (uL_.taskid() == 0) {
            ofstream fT((etflags.filenameTB).c_str(), ios::app);
            fT << t << endl;
            fT.close();
        }

        Bisection(uL, uH, tempL, tempH, t, etflags, aflags, dt, ilcflags, os);

        // Step 2: Advance FlowFields
        printout("t = " + i2s((int)t) + ": Advancing flowfields");
        bool finished = AdvanceLH(uL, uH, tempL, tempH, t, etflags, dt, ilcflags, aflags, os, convergencedata);
        if (!finished) {
            if (etflags.keepUL) {
                u = uL;
                temp = tempL;
            } else {
                u = uH;
                temp = tempH;
            }

            // Step 3: Choose new Flowfields
            Real lambdaL = 1, lambdaH = 1;
            ChooseNewLH(u, temp, lambdaL, lambdaH, t, etflags, aflags, dt, ilcflags, os);
            uL = u;
            uL *= lambdaL;
            uH = u;
            uH *= lambdaH;
            tempL = temp;
            tempL *= lambdaL;
            tempH = temp;
            tempH *= lambdaH;
        }
    }

    return 0;
}

/** Create files for energy and bisection times, if not continuing.
 */
void CreateFiles(const EdgetrackingFlags& etflags) {
    // Create files
    int taskid = 0;
#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
#endif
    if (taskid == 0) {
        ios::openmode mode = ios::out;
        mkdir(etflags.directoryEnergy);
        mkdir(etflags.directoryBisections);
        if (etflags.saveInterval != 0)
            mkdir(etflags.directoryFF);
        if (etflags.saveMinima)
            mkdir(etflags.directoryMinima);
        ofstream fL((etflags.filenameEnergyL).c_str(), mode);
        ofstream fH((etflags.filenameEnergyH).c_str(), mode);
        fL << ilcfieldstatsheader_t("t") << endl;
        fH << ilcfieldstatsheader_t("t") << endl;
        fL.close();
        fH.close();

        // Bisection times
        ofstream fT((etflags.filenameTB).c_str(), mode);
        fT.close();
    }
}

/** Find the edge of chaos between uL and uH.
 * @param uL lower flowfield, known to decay
 * @param uH upper flowfield, known to become turbulent
 * @param t0 time, only for filenames
 */
Real Bisection(FlowField& uL, FlowField& uH, FlowField& tempL, FlowField& tempH, Real t0,
               const EdgetrackingFlags& etflags, const AttractorFlags& aflags, const TimeStep& dt,
               const ILCFlags& ilcflags, ostream& os) {
    Real lambdaL = 0, lambdaH = 1;
    Real l2d = RelL2Diff(uL, uH);
    int i = 0;
    FlowField u0(uH);

    printout(
        "Initial L2Diff ( =L2(uH-uL)/(L2(uL)+L2(uH)) ) is " + r2s(l2d) + " and eps is " + r2s(etflags.epsilonBisection),
        os);
    int nTraj = etflags.nBisectTrajectories;

    SaveForContinuation(uL, uH, tempL, tempH, etflags.directoryBisections, t0);
    while (l2d > etflags.epsilonBisection) {
        i++;
        if (nTraj == 1) {
            Real lambda = (lambdaL + lambdaH) / 2.;
            FlowField uT = uH;
            uT += uL;
            uT *= 0.5;
            FlowField tempT = tempH;
            tempT += tempL;
            tempT *= 0.5;

            // Calculate attractor of ulambda and update lambaL or lambdaH accordingly
            string f = "";
            Attractor a = CalcAttractor(uT, tempT, aflags, dt, ilcflags,
                                        etflags.directoryEnergy + "e_bisection_t" + i2s(int(t0)) + "_i" + i2s(i), "",
                                        etflags.tMinAttractor, etflags.tMaxAttractor, false, t0);
            if (a == Turbulent) {
                lambdaH = lambda;
                uH = uT;
                tempH = tempT;
                f = "turbulent";
            } else if (a == Laminar) {
                lambdaL = lambda;
                uL = uT;
                tempL = tempT;
                f = "laminar";
            } else {
                cferror("Bisection did not converge");
            }
            l2d = RelL2Diff(uL, uH);
            printout(
                "Bisection t=" + i2s(t0) + ", i=" + FillSpaces(i, 3) + ": Found " + f + ", new L2Diff is " + r2s(l2d),
                os);
        } else {
            // Take several steps for each "bisection"
            // i.e. u = (1-alpha) uL + alpha uH
            // abort, once u is turbulent and set uL to the alpha-value _two_ steps lower
            Real alphaStep = 1. / (nTraj + 1);

            Attractor a = Laminar;
            int j = 0;
            Real alpha = 0;
            FlowField uT, tempT;
            while (a == Laminar && j <= nTraj + 10) {
                j++;
                // uT = (1-alpha)*uL + alpha*uH
                alpha = j * alphaStep;
                uT = uL;
                uT *= (1. - alpha) / alpha;
                uT += uH;
                uT *= alpha;
                tempT = tempL;
                tempT *= (1. - alpha) / alpha;
                tempT += tempH;
                tempT *= alpha;
                a = CalcAttractor(
                    uT, tempT, aflags, dt, ilcflags,
                    etflags.directoryEnergy + "e_bisection_t" + i2s(int(t0)) + "_i" + i2s(i) + "_j" + i2s(j), "",
                    etflags.tMinAttractor, etflags.tMaxAttractor, false, t0);
                string f = "laminar";
                if (a == Turbulent)
                    f = "turbulent";
                printout("Bisection t=" + i2s(t0) + ", i=" + FillSpaces(i, 3) + ", j=" + FillSpaces(j, 3) +
                             ", alpha=" + r2s(alpha) + ": Found " + f,
                         os);
            }
            if (a == Laminar)
                cferror("Did not find turbulent state, aborting");

            if (j >= 3) {
                // uL = (1-alpha[j-2])*uL + alpha[j-2]*uH
                Real alphaL = (j - 2) * alphaStep;
                uL *= (1. - alphaL) / alphaL;
                uL += uH;
                uL *= alphaL;
                tempL *= (1. - alphaL) / alphaL;
                tempL += tempH;
                tempL *= alphaL;
            }

            uH = uT;  // because uT is the FlowField we know to become turbulent
            tempH = tempT;

            l2d = RelL2Diff(uL, uH);
            printout("Bisection t=" + i2s(t0) + ", i=" + FillSpaces(i, 3) + ": lL=" + r2s(lambdaL) +
                         ", lH=" + r2s(lambdaH) + ", new L2diff is " + r2s(l2d) + "\n",
                     os);
        }
        SaveForContinuation(uL, uH, tempL, tempH, etflags.directoryBisections, t0);
    }

    printout("Bisection converged, L2Diff = " + r2s(l2d), os);
    printout("    lambdaL = " + r2s(lambdaL) + ", lambdaH = " + r2s(lambdaH), os);
    printout(
        "    l2(uH)/l2(u0) = " + r2s(L2Norm(uH) / L2Norm(u0)) + ", l2(uL)/l2(u0) = " + r2s(L2Norm(uL) / L2Norm(u0)),
        os);
    return lambdaH;
}

/** Advance uL and uH.
 * Advance till L2Diff(uL,uH)/L2Norm(uH)>c_epsilon or t > c_T. Check if they ar really turbulent/laminar if verifyLH is
 * active
 *
 * @param uL0 lower flowfield, iterated while not stateIsOnEdge
 * @param uH0 higher flowfield, iterated while not stateIsOnEdge
 * @param t time when advancing starts
 *
 * @return t, final time
 */
bool AdvanceLH(FlowField& uL0, FlowField& uH0, FlowField& tempL0, FlowField& tempH0, int& t,
               const EdgetrackingFlags& etflags, const TimeStep& dt, const ILCFlags& flags,
               const AttractorFlags& aflags, ostream& os, ConvergenceData& cd) {
    // Create TimeStep, pressureField and DNS objects
    //     Real t = t0; // Actual time
    int t0 = t;

    TimeStep dtL = dt;
    TimeStep dtH = dt;
    vector<FlowField> fieldsH(3);
    vector<FlowField> fieldsL(3);
    fieldsL[0] = uL0;     // is uL
    fieldsH[0] = uH0;     // is uH
    fieldsL[1] = tempL0;  // is tempL
    fieldsH[1] = tempH0;  // is tempH
    fieldsL[2] =
        FlowField(uH0.Nx(), uH0.Ny(), uH0.Nz(), 1, uH0.Lx(), uH0.Lz(), uH0.a(), uH0.b(), fieldsL[0].cfmpi());  // is qL
    fieldsH[2] =
        FlowField(uH0.Nx(), uH0.Ny(), uH0.Nz(), 1, uH0.Lx(), uH0.Lz(), uH0.a(), uH0.b(), fieldsH[0].cfmpi());  // is qH
    ILC dnsH(fieldsH, flags);
    ILC dnsL(fieldsL, flags);
    ChebyCoeff ubase(laminarProfile(flags, uH0.a(), uH0.b(), uH0.Ny()));
    ChebyCoeff wbase(uH0.Ny(), uH0.a(), uH0.b());
    PressureSolver psolver(uH0, ubase, wbase, flags.nu, flags.Vsuck, flags.nonlinearity);

    psolver.solve(fieldsH[2], fieldsH[0]);
    psolver.solve(fieldsL[2], fieldsL[0]);

    // Open Files
    ofstream fL, fH;
    if (uH0.taskid() == 0) {
        fL.open((etflags.filenameEnergyL).c_str(), ios::app);
        fH.open((etflags.filenameEnergyH).c_str(), ios::app);
    }

    // Advance uH and uL
    Real t2 = t;                                  // Internal time while verifying
    Real tAttractor = etflags.tMaxAttractor + t;  // Maximum time to verify
    bool turbulent = false, laminar = false, stateIsOnEdge = true;
    Real ecf0 = 0, ecf1 = 0, ecf2 = 0;  // ecf(uH, t - 0,1,2)
    vector<Real> ecf;
    for (int i = 0; i < 100; ++i)
        ecf.push_back(0);
    while ((!turbulent || !laminar) && t <= flags.T && (stateIsOnEdge || (etflags.verifyLH && t2 <= tAttractor))) {
        t2++;
        // Advance uH
        if (!turbulent) {
            dnsH.advance(fieldsH, dtH.n());
            Real cflH = dnsH.CFL(fieldsH[0]);
            if (dtH.adjust(cflH, false)) {
                dnsH.reset_dt(dtH);
            }
        }

        // Advance uL
        if (!laminar) {
            dnsL.advance(fieldsL, dtL.n());
            Real cflL = dnsL.CFL(fieldsL[0]);
            if (dtL.adjust(cflL, false)) {
                dnsL.reset_dt(dtL);
            }
        }

        if (stateIsOnEdge) {
            Real l2d = RelL2Diff(fieldsL[0], fieldsH[0]);
            if (l2d < etflags.epsilonAdvance) {
                t++;
                if (t > etflags.tLastWrite) {  // prevent writing the same stuff twice
                    printout(ilcfieldstats_t(fieldsH[0], fieldsH[1], t), fH);
                    printout(ilcfieldstats_t(fieldsL[0], fieldsL[1], t), fL);
                }
                ecf2 = ecf1;
                ecf1 = ecf0;
                ecf0 = Ecf(fieldsH[0]);
                bool ecfMinimum = (ecf1 < ecf2) && (ecf1 <= ecf0);
                bool ecfMaximum = (ecf1 > ecf2) && (ecf1 >= ecf0);

                // Check whether we are converged to a (relative) equilibrium where ecf doesn't change anymore
                if (etflags.checkConvergence) {
                    bool eq = true;
                    int l = ecf.size();
                    ecf[t % l] = ecf0;
                    for (int i = 0; i < l; ++i) {
                        if (ecf[i] < 1e-16 || abs(ecf[i] - ecf[0]) / ecf[0] > etflags.epsilonBisection) {
                            eq = false;
                            break;
                        }
                    }
                    if (eq) {
                        printout("Ecf is constant -- converged to a (relative) equilibrium", os);
                        return true;
                    }

                    // Check whether we are converged to a periodic orbit (somewhat more subtle)
                    // We consider a state as periodic if either 4 minima and 4 maxima in a row occur with same distance
                    // and same values or when maxima,minima 1,3 and 2,4 are the same and occur at the same distance
                    if (ecfMaximum)
                        push(cd.EcfMax, ecf0);
                    if (ecfMinimum) {
                        push(cd.EcfMin, ecf0);
                        push(cd.tEcfMin, t);

                        Real eps = 1e-3;
                        if (abs(cd.EcfMax[0] - cd.EcfMax[2]) / ecf0 < eps &&
                            abs(cd.EcfMax[1] - cd.EcfMax[3]) / ecf0 < eps &&
                            abs(cd.EcfMin[0] - cd.EcfMin[2]) / ecf0 < eps &&
                            abs(cd.EcfMin[1] - cd.EcfMin[3]) / ecf0 < eps &&
                            abs((cd.tEcfMin[3] - cd.tEcfMin[1]) - (cd.tEcfMin[2] - cd.tEcfMin[0])) < 5) {
                            printout("Converged to a periodic orbit", os);
                            return true;
                        }
                    }
                }

                // Save fields
                bool saveIntervalReached = etflags.saveInterval != 0 && t % (int)etflags.saveInterval == 0;
                if (saveIntervalReached) {
                    if (etflags.saveUH) {
                        fieldsH[0].save(etflags.directoryFF + "uH" + i2s((int)t));
                        fieldsH[1].save(etflags.directoryFF + "tH" + i2s((int)t));
                    }
                    if (etflags.saveUL) {
                        fieldsL[0].save(etflags.directoryFF + "uL" + i2s((int)t));
                        fieldsL[1].save(etflags.directoryFF + "tL" + i2s((int)t));
                    }
                }
                if (ecfMinimum && etflags.saveMinima) {
                    if (etflags.saveUH) {
                        fieldsH[0].save(etflags.directoryMinima + "uH" + i2s((int)t));
                        fieldsH[1].save(etflags.directoryMinima + "tH" + i2s((int)t));
                    }
                    if (etflags.saveUL) {
                        fieldsL[0].save(etflags.directoryMinima + "uL" + i2s((int)t));
                        fieldsL[1].save(etflags.directoryMinima + "tL" + i2s((int)t));
                    }
                }
                uL0 = fieldsL[0];
                uH0 = fieldsH[0];
                tempL0 = fieldsL[1];
                tempH0 = fieldsH[1];
            } else {
                // The flowfields are separated too far
                // Use state when this occurs for the first time for the next bisectioning
                // and keep on iterating to verify they become turbulent/laminar
                printout("L2(uL) " + r2s(L2Norm(fieldsL[0])), os);
                printout("L2(uH) " + r2s(L2Norm(fieldsH[0])), os);
                stateIsOnEdge = false;
                printout("Flowfields separated by " + r2s(RelL2Diff(fieldsL[0], fieldsH[0])), os);
                printout("Advanced flowfields to t = " + i2s((int)t), os);
                printout("", os);  // newline
            }
        }

        // Check whether flowfields are laminar/turbulent
        if (!turbulent && attractor(fieldsH[0], aflags) == Turbulent) {
            turbulent = true;
            printout("uH is turbulent at t = " + i2s((int)t2), os);
            if (!etflags.verifyLH) {
                cferror("uH is turbulent, check thresholds");
            }
            if (t2 - t0 <= 2) {
                cferror("t2 - t0 = " + r2s(t2 - t0) + ", aborting\nuH is turbulent, check your thresholds!");
            }
        }
        if (!laminar && attractor(fieldsL[0], aflags) == Laminar) {
            laminar = true;
            printout("uL is laminar at t = " + i2s((int)t2), os);
            if (!etflags.verifyLH) {
                cferror("uL is laminar, check thresholds");
            }
            if (t2 - t0 <= 2) {
                cferror("t2 - t0 = " + r2s(t2 - t0) + ", aborting\nuL is laminar, check your thresholds!");
            }
        }
    }

    // Abort if the flowfields did not reach the expected state
    if (!laminar && etflags.verifyLH) {
        cferror("verifyLH: uL does not become laminar");
    }
    if (!turbulent && etflags.verifyLH) {
        cferror("verifyLH: uH does not become turbulent");
    }

    // Close files
    if (fieldsH[0].taskid() == 0) {
        fH.close();
        fL.close();
    }

    if (t == flags.T)
        return true;
    return false;
}

/** Choose two flowfields such that uL becomes laminar and uH turbulent.
 * @param u0 flowfield to scale up/down
 * @param lambdaL return reference for lambdaL
 * @param lambdaH return reference for lambdaH
 */
void ChooseNewLH(const FlowField& u0, const FlowField& temp0, Real& lambdaL, Real& lambdaH, const Real t,
                 const EdgetrackingFlags& etflags, const AttractorFlags& aflags, const TimeStep& dt,
                 const ILCFlags& ilcflags, ostream& os) {
    assert(lambdaL <= lambdaH);
    printout("Choosing new laminar/turbulent flowfields", os);
    bool equalLambda = (lambdaH == lambdaL) ? true : false;

    FlowField u = u0;
    FlowField temp = temp0;
    Attractor AH = NoAttractor, AL = NoAttractor;
    // Choose turbulent field
    while (AH != Turbulent) {
        u = u0;
        u *= lambdaH;
        temp = temp0;
        temp *= lambdaH;
        AH = CalcAttractor(u, temp, aflags, dt, ilcflags,
                           etflags.directoryEnergy + "e_chooseLH_t" + i2s(int(t)) + "_lambdaH" + r2s(lambdaH), "",
                           etflags.tMinAttractor, etflags.tMaxAttractor, false, t);

        printout("FlowField with lambda = " + r2s(lambdaH) + " is " + a2s(AH), os);
        if (AH == Laminar) {
            lambdaL = lambdaH;
            lambdaH += etflags.lambdaStep;
            AL = Laminar;
        } else if (AH == NoAttractor) {
            cferror("flowfield is ambigous");
        }
    }
    // Choose laminar field
    while (AL != Laminar) {
        if (equalLambda) {
            lambdaL -= etflags.lambdaStep;
            equalLambda = false;
        }
        u = u0;
        u *= lambdaL;
        temp = temp0;
        temp *= lambdaL;
        //         AL = Attractor ( u, "_chooseLH_t" + i2s(int(t)) + "_lambdaL" + r2s(lambdaL)  );
        AL = CalcAttractor(u, temp, aflags, dt, ilcflags,
                           etflags.directoryEnergy + "e_chooseLH_t" + i2s(int(t)) + +"_lambdaL" + r2s(lambdaL), "",
                           etflags.tMinAttractor, etflags.tMaxAttractor, false, t);

        printout("FlowField with lambda = " + r2s(lambdaL) + " is " + a2s(AL), os);
        if (AL == Turbulent) {
            lambdaH = lambdaL;
            lambdaL -= etflags.lambdaStep;
        } else if (AL == NoAttractor) {
            cferror("flowfield is ambigous");
        }
    }
    printout("Finished choosing uH, uL -- lambdaH = " + r2s(lambdaH) + ", lambdaL = " + r2s(lambdaL), os);
    FlowField uL(u0);
    uL *= lambdaL;
    FlowField uH(u0);
    uH *= lambdaH;
    FlowField tempL(temp0);
    tempL *= lambdaL;
    FlowField tempH(temp0);
    tempH *= lambdaH;
    SaveForContinuation(uL, uH, tempL, tempH, etflags.directoryBisections, t);
}

/** Save all data needed for continuation to file 'continue'
 */
void SaveForContinuation(const FlowField& uL, const FlowField& uH, const FlowField& tempL, const FlowField& tempH,
                         const string directoryBisections, Real t) {
    string ffnameH = directoryBisections + "uH" + i2s((int)t);
    string ffnameL = directoryBisections + "uL" + i2s((int)t);
    uH.save(ffnameH);
    uL.save(ffnameL);

    string tffnameH = directoryBisections + "tH" + i2s((int)t);
    string tffnameL = directoryBisections + "tL" + i2s((int)t);
    tempH.save(tffnameH);
    tempL.save(tffnameL);

    fftw_savewisdom();
}

EdgetrackingFlags::EdgetrackingFlags()
    : tLastWrite(0),
      epsilonAdvance(1e-4),
      epsilonBisection(1e-6),
      lambdaStep(0.1),
      nBisectTrajectories(1),
      tMinAttractor(0),
      tMaxAttractor(10000),
      verifyLH(false),
      saveMinima(false),
      saveInterval(0),
      saveUH(true),
      saveUL(false),
      keepUL(false),
      checkConvergence(false),
      returnFieldAtMinEcf(false),
      savedir("./"),
      directoryEnergy("energy/"),
      directoryBisections("bisections/"),
      directoryFF("flowfields/"),
      directoryMinima("minima/"),
      filenameEnergyL("energyL"),
      filenameEnergyH("energyH"),
      filenameTB("tBisections") {}

ConvergenceData::ConvergenceData() {
    for (int i = 0; i < 4; ++i) {
        EcfMin.push_back(0);
        EcfMax.push_back(0);
        tEcfMin.push_back(0);
    }
}

/** Calculate the attractor of a given flowfield.
 * The decicision whether a flowfield is laminar or turbulent is made based upon the parameters in the configfile.
 */
Attractor CalcAttractor(const FlowField& u0, const FlowField& temp0, const AttractorFlags& aflags, const TimeStep& dt_,
                        const ILCFlags& flags, const string energyfile, const string saveFinalFF, const int tMin,
                        const int tMax, bool verbose, const int t0, ostream& os) {
    FlowField u(u0);
    FlowField temp(temp0);
    return CalcAttractor(u0, temp0, u, temp, aflags, dt_, flags, energyfile, saveFinalFF, tMin, tMax, verbose, t0, os);
}

Attractor CalcAttractor(const FlowField& u0, const FlowField& temp0, FlowField& u, FlowField& temp,
                        const AttractorFlags& aflags, const TimeStep& dt_, const ILCFlags& flags,
                        const string energyfile, const string saveFinalFF, const int tMin, const int tMax, bool verbose,
                        const int t0, ostream& os) {
    u = u0;
    temp = temp0;
    // Check what data to save and create files

    // Advance flowfield till it reaches an attractor or time limit
    TimeStep dt = dt_;
    FlowField q(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());
    vector<FlowField> fields = {u, temp, q};
    ILC dns(fields, flags);
    ChebyCoeff ubase = laminarProfile(flags, u.a(), u.b(), u.Ny());
    ChebyCoeff wbase = ChebyCoeff(u.Ny(), u.a(), u.b());
    PressureSolver psolver(u, ubase, wbase, flags.nu, flags.Vsuck, flags.nonlinearity);
    psolver.solve(fields[2], fields[0]);
    ofstream f;
    bool saveEnergy = (energyfile != "");
    if (saveEnergy) {
        if (u0.taskid() == 0)
            f.open(energyfile.c_str(), ios::out);
        printout(ilcfieldstatsheader_t("t"), f);
        printout(ilcfieldstats_t(u, temp, t0), f);
    }
    Attractor result = NoAttractor;
    int t = 0;
    int tDecay = 0;  // timeunits that flowfield is monotonically decaying
    while (result == NoAttractor && t < tMax) {
        t++;
        Real lastEcf = Ecf(fields[0]);  // for calculating decay time
        dns.advance(fields, dt.n());

        Real cfl = dns.CFL(fields[0]);
        if (dt.adjust(cfl, verbose, os)) {
            dns.reset_dt(dt);
            if (verbose)
                printout("Adjusting dt to " + r2s(dt.dt()) + ", CFL is " + r2s(cfl), os);
        }

        if (saveEnergy) {
            printout(ilcfieldstats_t(fields[0], fields[1], t + t0), f);
        }

        // Check if FF is turbulent or laminar
        if (t > tMin) {
            result = attractor(fields[0], tDecay, lastEcf, aflags);
        }
    }
    // printout("", os);
    if (saveFinalFF != "")
        fields[0].save(saveFinalFF.c_str());
    if (saveEnergy && fields[0].taskid() == 0)
        f.close();
    return result;
}

AttractorFlags::AttractorFlags(Real EcfH_, Real EcfL_, Real L2H_, Real L2L_, bool n2h, int tMaxDecay_)
    : EcfH(EcfH_),
      EcfL(EcfL_),
      L2H(L2H_),
      L2L(L2L_),
      //             EnstH( EnstH_ ),
      //             EnstL( EnstL_ ),
      normToHeight(n2h),
      tMaxDecay(tMaxDecay_) {}

Attractor attractor(const FlowField& u, const AttractorFlags& a) {
    int t = 0;
    return attractor(u, t, 0., a);
}

Attractor attractor(const FlowField& u, int& tDecay, Real lastEcf, const AttractorFlags& a) {
    const bool useEcf = (a.EcfH > 1e-12 && a.EcfL > 1e-12);
    //     const bool useEnst = ( a.EnstH > 1e-12 && a.EnstL > 1e-12 );
    const bool useL2 = (a.L2H > 1e-12 && a.L2L > 1e-12);

    if (not(useEcf || useL2)) {
        cferror("No valid attractor settings specified\n");
    }

    // A FlowField has reached an attractor if all conditions are true
    bool isLaminar = true, isTurbulent = true;
    Real val = 0;
    if (useEcf) {
        val = Ecf(u);
        if (a.normToHeight)
            val /= (u.b() - u.a());
        if (val < a.EcfH)
            isTurbulent = false;
        if (val > a.EcfL)
            isLaminar = false;
    }
    if (useL2) {
        val = L2Norm(u);
        if (a.normToHeight)
            val /= sqrt((u.b() - u.a()));
        if (val < a.L2H)
            isTurbulent = false;
        if (val > a.L2L)
            isLaminar = false;
    }
    //     if ( useEnst ) {
    //         val = Enstrophy(u);
    //    if ( a.normToHeight ) val /= sqrt((u.b() - u.a()));
    //         if ( val < a.EnstH )
    //             isTurbulent = false;
    //         if ( val > a.EnstL )
    //             isLaminar = false;
    //     }

    if (a.tMaxDecay > 0) {
        if (Ecf(u) <= lastEcf)
            tDecay++;
        else
            tDecay = 0;

        if (tDecay > a.tMaxDecay)
            isLaminar = true;
    }

    if (isTurbulent && isLaminar) {
        cferror("FlowField is both laminar and turbulent, exiting");
    } else if (isTurbulent) {
        return Turbulent;
    } else if (isLaminar) {
        return Laminar;
    }
    return NoAttractor;
}

string a2s(Attractor a) {
    if (a == Turbulent)
        return "turbulent";
    if (a == Laminar)
        return "laminar";
    return "neither laminar nor turbulent";
}
