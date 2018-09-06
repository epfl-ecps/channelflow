/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */
#include "newtonalgorithm.h"

#include "nsolver/config.h"  // for interfacing from other programs
#ifdef HAVE_MPI
#include "mpi.h"
#endif
using namespace std;

using namespace Eigen;

namespace chflow {

NewtonSearchFlags::NewtonSearchFlags(SolutionType solntype_,
                                     bool xrelative_,  // new
                                     bool zrelative_,  // new
                                     Real epsSearch_, Real epsKrylov_, Real epsDx_, Real epsDt_, Real epsSolver_,
                                     Real epsSolverF_, bool centdiff_, int Nnewton_, int Nsolver_, int Nhook_,
                                     Real delta_, Real deltaMin_, Real deltaMax_, Real deltaFuzz_, Real lambdaMin_,
                                     Real lambdaMax_, Real lambdaRequiredReduction_, Real improvReq_, Real improvOk_,
                                     Real improvGood_, Real improvAcc_, int lBiCGStab_, int nShot_, bool fixtphase_,
                                     Real TRef_, Real axRef_, Real azRef_, Real gRatio, string outdir_,
                                     ostream* logstream_, bool laurette_)
    : solntype(solntype_),
      xrelative(xrelative_),
      zrelative(zrelative_),
      epsSearch(epsSearch_),
      epsKrylov(epsKrylov_),
      epsDx(epsDx_),
      epsDt(epsDt_),
      epsSolver(epsSolver_),
      epsSolverF(epsSolverF_),
      centdiff(centdiff_),
      Nnewton(Nnewton_),
      Nsolver(Nsolver_),
      Nhook(Nhook_),
      delta(delta_),
      deltaMin(deltaMin_),
      deltaMax(deltaMax_),
      deltaFuzz(deltaFuzz_),
      lambdaMin(lambdaMin_),
      lambdaMax(lambdaMax_),
      lambdaRequiredReduction(lambdaRequiredReduction_),
      improvReq(improvReq_),
      improvOk(improvOk_),
      improvGood(improvGood_),
      improvAcc(improvAcc_),
      lBiCGStab(lBiCGStab_),
      nShot(nShot_),
      fixtphase(fixtphase_),
      TRef(TRef_),
      axRef(axRef_),
      azRef(azRef_),
      gRatio(gRatio),
      outdir(outdir_),
      logstream(logstream_),
      laurette(laurette_) {}

NewtonSearchFlags::NewtonSearchFlags(ArgList& args) : lambdaRequiredReduction(0.5) {
    args.section("Newton algorithm options");
    bool searcheq = args.getflag("-eqb", "--equilibrium", "search for fixed point or traveling wave");
    bool searchorb = args.getflag("-orb", "--orbit", "search for periodic or relative periodic orbit");
    xrelative =
        args.getflag("-xrel", "--xrelative", "search for traveling wave or relative periodic orbit with shift in x");
    zrelative =
        args.getflag("-zrel", "--zrelative", "search for traveling wave or relative periodic orbit with shift in z");
    laurette = args.getflag("-L", "--Laurette", "use Laurette's method for finding fixed points and travelling waves");

    string solverString = args.getstr(
        "-solver", "--solver", "gmres",
        "method used for solving linear system of equations, options are 'eigen', 'gmres', 'fgmres' or 'bicgstab'");
    string optString =
        args.getstr("-opt", "--optimization", "hookstep",
                    "method used for optimizing the Newton update, options are 'none', 'linear' or 'hookstep'");
    solver = string2solver(solverString);
    optimization = string2optimization(optString);

    epsSearch = args.getreal("-es", "--epsSearch", 1e-13, "stop search if L2Norm(s f^T(u) - u) < epsSearch");
    epsDx = args.getreal("-edx", "--epsDx", 1e-7, "relative size of dx to x in linearization");
    epsDt = args.getreal("-edt", "--epsDt", 1e-5, "size of dT in linearization of f^T around T = 0");
    centdiff = args.getflag("-cd", "--centerdiff", "set centered differencing for linear approximation of Jacobian");
    Nnewton = args.getint("-Nn", "--Nnewton", 20, "maximum number of Newton steps ");

    nShot = args.getint("-nShot", "--Nshots", 1, "number of shots for multishooting in Newton");
    fixtphase = args.getflag("-tph", "--fixtphase",
                             "fix t-phase on an orbit, such that time variation of a chosen observable is 0");
    TRef = args.getreal("-TRef", "--TReference", 1.0, "reference value of T for preconditioning");
    axRef = args.getreal("-axRef", "--axReference", 1.0, "reference value of ax for preconditioning");
    azRef = args.getreal("-azRef", "--azReference", 1.0, "reference value of az for preconditioning");

    outdir = args.getpath("-od", "--outdir", "./", "output directory");

    args.section("Newton solver options");
    epsKrylov = args.getreal("-ek", "--epsKrylov", 1e-14, "minimum condition number of Krylov vectors");
    epsSolver = args.getreal("-eg", "--epsSolver", 1e-3, "stop GMRES iteration when Ax = b residual is < epsSolver");
    epsSolverF =
        args.getreal("-egf", "--epsSolverFinal", 0.05, "accept final GMRES iterate if residual is < epsSolverF");
    Nsolver = args.getint("-Ng", "--Ngmres", 500, "maximum number of GMRES iterations per Newton iteration");

    gRatio = args.getreal("-gr", "--gRatio", 10.0,
                          "minimum convergence rate (Gx_prev/Gx) for a Newton iteration to expand the Krylov subspace "
                          "for the next iterations");

    lBiCGStab = args.getint("-lBiCGS", "--lBiCGStab", 2, "subspace dimension l for BiCGstab(l)");

    args.section("Newton optimization options");
    Nhook = args.getint("-Nh", "--Nhook", 20, "maximum number of hookstep iterations per Newton iteration");
    delta = args.getreal("-d", "--delta", 0.01, "initial radius of trust region OR minimum linear step size");
    deltaMin = args.getreal("-dmin", "--deltaMin", 1e-12, "stop if radius of trust region is < deltaMin");
    deltaMax = args.getreal("-dmax", "--deltaMax", 0.1, "maximum radius of trust region");
    deltaFuzz = args.getreal("-df", "--deltaFuzz", 1e-6, "accept steps within (1 +/- deltaFuzz)*delta");
    lambdaMin = args.getreal("-lmin", "--lambdaMin", 0.2, "minimum shrink rate of delta");
    lambdaMax = args.getreal("-lmax", "--lambdaMax", 1.5, "maximum expansion rate of delta");
    improvReq = args.getreal("-irq", "--improveReq", 1e-3,
                             "reduce delta and recompute hookstep if improvement is worse than this fraction of what "
                             "we'd expect from gradient");
    improvOk = args.getreal(
        "-iok", "--improveOk", 0.10,
        "accept step and keep the same delta if improvement is better than this fraction of quadratic model");
    improvGood =
        args.getreal("-igd", "--improveGood", 0.75,
                     "accept step and increase delta if improvement is better than this fraction of quadratic model");
    improvAcc = args.getreal(
        "-iac", "--improveAcc", 0.10,
        "recompute hookstep with a larger trust region if improvement is within this fraction of quadratic prediction");

    if (!(searcheq ^ searchorb) && !args.helpmode()) {
        cferror("Please specify the type of the solution you are searching with either '-eqb' or '-orb'");
    }

    logstream = &cout;
    solntype = searchorb ? PeriodicOrbit : Equilibrium;
    if (laurette && solntype == PeriodicOrbit) {
        cerr << " You cannot use Laurette algorithm for finding periodic orbits." << endl;
        exit(1);
    }
}

SolverMethod NewtonSearchFlags::string2solver(string s) {
    if (s == "eigen")
        return SolverEigen;
    else if (s == "gmres")
        return SolverGMRES;
    else if (s == "fgmres")
        return SolverFGMRES;
    else if (s == "bicgstab")
        return SolverBiCGStab;
    throw invalid_argument("NewtonSearchFlags::string2solver(): solver '" + s +
                           "' is not known. Possible values are 'eigen', 'bicgstab', 'gmres' or 'fgmres' ");
}

string NewtonSearchFlags::solver2string() const {
    if (solver == SolverEigen)
        return "eigen";
    else if (solver == SolverGMRES)
        return "gmres";
    else if (solver == SolverFGMRES)
        return "fgmres";
    else if (solver == SolverBiCGStab)
        return "bicgstab";
    throw invalid_argument("NewtonSearchFlags::solver2string(): solver is not known");
}

OptimizationMethod NewtonSearchFlags::string2optimization(string s) {
    if (s == "none")
        return None;
    else if (s == "linear")
        return Linear;
    else if (s == "hookstep")
        return Hookstep;
    throw invalid_argument("NewtonSearchFlags::string2optimization(): optimization method '" + s + "' is not known.");
}

string NewtonSearchFlags::optimization2string() const {
    if (optimization == None)
        return "none";
    else if (optimization == Linear)
        return "linear";
    else if (optimization == Hookstep)
        return "hookstep";
    throw invalid_argument("NewtonSearchFlags::optimization2string(): optimization method is not known.");
}

SolutionType NewtonSearchFlags::string2solntype(string s) const {
    if (s == "orbit")
        return PeriodicOrbit;
    else if (s == "eqb")
        return Equilibrium;
    throw invalid_argument("NewtonSearchFlags::solntype2string(): optimization method is not known.");
}

string NewtonSearchFlags::solntype2string() const {
    if (solntype == PeriodicOrbit)
        return "orbit";
    else if (solntype == Equilibrium)
        return "eqb";
    throw invalid_argument("NewtonSearchFlags::solntype2string(): optimization method is not known.");
}

void NewtonSearchFlags::save(const string& outdir) const {
    if (mpirank() == 0) {
        string filename = appendSuffix(outdir, "newtonflags.txt");
        ofstream os(filename.c_str());
        if (!os.good())
            cferror("NewtonSearchFlags::save(outdir) :  can't open file " + filename);
        os.precision(16);
        os.setf(ios::left);

        os << setw(REAL_IOWIDTH) << solver2string() << "  %solver\n"
           << setw(REAL_IOWIDTH) << optimization2string() << "  %optimization\n"
           << setw(REAL_IOWIDTH) << solntype2string() << "  %orbit\n"
           << setw(REAL_IOWIDTH) << xrelative << "  %xrelative\n"
           << setw(REAL_IOWIDTH) << zrelative << "  %zrelative\n"
           << setw(REAL_IOWIDTH) << epsSearch << "  %epsSearch\n"
           << setw(REAL_IOWIDTH) << epsKrylov << "  %epsKrylov\n"
           << setw(REAL_IOWIDTH) << epsDx << "  %epsDx\n"
           << setw(REAL_IOWIDTH) << epsDt << "  %epsDt\n"
           << setw(REAL_IOWIDTH) << epsSolver << "  %epsSolver\n"
           << setw(REAL_IOWIDTH) << epsSolverF << "  %epsSolverFinal\n"
           << setw(REAL_IOWIDTH) << lBiCGStab << "  %lBiCGStab\n"
           << setw(REAL_IOWIDTH) << gRatio << "  %gRatio\n"
           << setw(REAL_IOWIDTH) << centdiff << "  %centdiff\n"
           << setw(REAL_IOWIDTH) << Nnewton << "  %Nnewton\n"
           << setw(REAL_IOWIDTH) << Nsolver << "  %Nsolver\n"
           << setw(REAL_IOWIDTH) << Nhook << "  %Nhook\n"
           << setw(REAL_IOWIDTH) << nShot << "  %Nshot\n"
           << setw(REAL_IOWIDTH) << fixtphase << "  %fixtphase\n"
           << setw(REAL_IOWIDTH) << TRef << "  %TReference\n"
           << setw(REAL_IOWIDTH) << axRef << "  %axReference\n"
           << setw(REAL_IOWIDTH) << azRef << "  %azReference\n"
           << setw(REAL_IOWIDTH) << delta << "  %delta\n"
           << setw(REAL_IOWIDTH) << deltaMin << "  %deltaMin\n"
           << setw(REAL_IOWIDTH) << deltaMax << "  %deltaMax\n"
           << setw(REAL_IOWIDTH) << deltaFuzz << "  %deltaFuzz\n"
           << setw(REAL_IOWIDTH) << lambdaMin << "  %lambdaMin\n"
           << setw(REAL_IOWIDTH) << lambdaMax << "  %lambdaMax\n"
           << setw(REAL_IOWIDTH) << improvReq << "  %improvReq\n"
           << setw(REAL_IOWIDTH) << improvOk << "  %improveOk\n"
           << setw(REAL_IOWIDTH) << improvGood << "  %improveGood\n"
           << setw(REAL_IOWIDTH) << improvAcc << "  %improveAcc\n"
           << setw(REAL_IOWIDTH) << laurette << "  %laurette\n";
        os.unsetf(ios::left);
    }
}

void NewtonSearchFlags::load(int taskid, const string indir) {
    ifstream is;
    if (taskid == 0) {
        is.open(indir + "newtonflags.txt");
        if (!is.good()) {
            cout << "    NewtonSearchflags::load(taskid, indir): can't open file " + indir + "newtonflags.txt" << endl;
            return;
        }
        if (!checkFlagContent(is, getFlagList())) {
            cerr << " NewtonSearchFlags::load(taskid, indir): the order of variables in the file is not what we expect "
                    "!!"
                 << endl;
            exit(1);
        }
    }
    solver = string2solver(getStringfromLine(taskid, is));
    optimization = string2optimization(getStringfromLine(taskid, is));
    solntype = string2solntype(getStringfromLine(taskid, is));
    xrelative = getIntfromLine(taskid, is);
    zrelative = getIntfromLine(taskid, is);
    epsSearch = getRealfromLine(taskid, is);
    epsKrylov = getRealfromLine(taskid, is);
    epsDx = getRealfromLine(taskid, is);
    epsDt = getRealfromLine(taskid, is);
    epsSolver = getRealfromLine(taskid, is);
    epsSolverF = getRealfromLine(taskid, is);
    lBiCGStab = getIntfromLine(taskid, is);
    gRatio = getRealfromLine(taskid, is);
    centdiff = getIntfromLine(taskid, is);
    Nnewton = getIntfromLine(taskid, is);
    Nsolver = getIntfromLine(taskid, is);
    Nhook = getIntfromLine(taskid, is);
    nShot = getIntfromLine(taskid, is);
    fixtphase = getIntfromLine(taskid, is);
    TRef = getRealfromLine(taskid, is);
    axRef = getRealfromLine(taskid, is);
    azRef = getRealfromLine(taskid, is);
    delta = getRealfromLine(taskid, is);
    deltaMin = getRealfromLine(taskid, is);
    deltaMax = getRealfromLine(taskid, is);
    lambdaMin = getRealfromLine(taskid, is);
    lambdaMax = getRealfromLine(taskid, is);
    improvReq = getRealfromLine(taskid, is);
    improvOk = getRealfromLine(taskid, is);
    improvGood = getRealfromLine(taskid, is);
    improvAcc = getRealfromLine(taskid, is);
    laurette = getIntfromLine(taskid, is);
}

const vector<string> NewtonSearchFlags::getFlagList() {
    const vector<string> flagList = {
        "%solver",      "%optimization", "%orbit",       "%xrelative",      "%zrelative", "%epsSearch",  "%epsKrylov",
        "%epsDx",       "%epsDt",        "%epsSolver",   "%epsSolverFinal", "%lBiCGStab", "%gRatio",     "%centdiff",
        "%Nnewton",     "%Nsolver",      "%Nhook",       "%Nshot",          "%fixtphase", "%TReference", "%axReference",
        "%azReference", "%delta",        "%deltaMin",    "%deltaMax",       "%deltaFuzz", "%lambdaMin",  "%lambdaMax",
        "%improvReq",   "%improveOk",    "%improveGood", "%improveAcc",     "%laurette"};
    return flagList;
}

NewtonAlgorithm::NewtonAlgorithm(NewtonSearchFlags searchflags_)
    : Newton(searchflags_.logstream, searchflags_.outdir, searchflags_.epsSearch),
      searchflags(searchflags_),
      os(0),
      fcount_newton_(0),
      fcount_opt_(0),
      gmres_(),
      delta_(0.0),
      rx_(0.0) {
    os = searchflags.logstream;
    *msDSI_ = MultishootingDSI(searchflags.nShot, searchflags.solntype == PeriodicOrbit, searchflags.xrelative,
                               searchflags.zrelative, searchflags.TRef, searchflags.axRef, searchflags.azRef,
                               searchflags.fixtphase);
}

void NewtonAlgorithm::setLogstream(ostream* os_) {
    Newton::setLogstream(os_);
    searchflags.logstream = os_;
    os = os_;
}

void NewtonAlgorithm::setOutdir(string od) {
    Newton::setOutdir(od);
    searchflags.outdir = od;
}

VectorXd NewtonAlgorithm::solve(DSI& dsiG, const VectorXd& y0, Real& gx) {
    if (searchflags.optimization == Hookstep && searchflags.solver != SolverGMRES) {
        cerr << "NewtonAlgorithm::solve(): The Hookstep optimization must be used together with the GMRES solver"
             << endl;
        exit(1);
    }

    success = false;

    VectorXd x0;
    if (!(msDSI_->isDSIset())) {
        msDSI_->setDSI(dsiG, y0.rows());
        x0 = msDSI_->makeMSVector(y0);
    } else {
        if (msDSI_->isVecMS(y0.rows(), AC->use()))
            x0 = y0;
        else {
            cerr << "Multishooting class is already set, but the size of the multishooting vector is not equal to the "
                    "length of the unknown vector !!!"
                 << endl;
            exit(1);
        }
    }

    int taskid = mpirank();

    // Need to construct orthogonality constraints
    bool Rxsearch = searchflags.xrelative;
    bool Rzsearch = searchflags.zrelative;
    // TODO: this breaks poincare search, doesn't it?
    bool Tsearch = searchflags.solntype == PeriodicOrbit ? true : false;

    /***/ int Nunk = x0.rows();                         // # of unknowns in x
    const int Northog = Tsearch + Rxsearch + Rzsearch;  // # of orthogonality constraints
    const int Nconstr = Northog + AC->use();            // # of constraints, orthog and arclength
    const int Nconstr0 = (taskid == 0) ? Nconstr : 0;
    const int uunk = Nunk - Nconstr;  // # of phyiscal variables in x
                                      //   const int xac = Nunk - 1;
    int Nunk_total = Nunk;

#ifdef HAVE_MPI
    MPI_Allreduce(&Nunk, &Nunk_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    MatrixXd E(1, 1);  // contains orthogonality constraints
    if (Northog != 0) {
        E = MatrixXd(Nunk, Northog);
        E.setZero();
    }

    fcount_newton_ = 0;
    fcount_opt_ = 0;

    // get short names for the following heavily used searchflags fields
    //   ostream* os = searchflags.logstream;
    delta_ = searchflags.delta;
    const Real epsSearch = searchflags.epsSearch;
    const Real epsDx = searchflags.epsDx;
    int NSolver = (Nunk_total < searchflags.Nsolver) ? Nunk_total : searchflags.Nsolver;
    const int Nnewton = searchflags.Nnewton;
    const bool centdiff = searchflags.centdiff;

    // ==============================================================
    // Initialize Newton iteration
    *os << "Computing G(x)" << endl;
    VectorXd x(Nunk);
    for (int i = 0; i < x0.rows(); ++i)
        x(i) = x0(i);

    VectorXd Gx = evalWithAC(x, fcount_newton_);  // 0 ==> current best guess

    string filebase = "best";

    // From here on, scale/unscale x when interfacing with the external world
    // rescale(x,xscale);

    //   VectorXd      dx ( Nunk ); // temporary for increments in unknowns
    VectorXd DG_dx(Nunk);  // "DG*dx" = 1/eps (G(x + eps dx)-G(x))
    VectorXd dxN(Nunk);    // The Newton step
    VectorXd dxOpt(Nunk);  // The hookstep
    VectorXd GxOpt(Nunk);  // G(xH);
                           //   VectorXd      xH ( Nunk ); // x plus dxH
                           //   setToZero ( dx );
    setToZero(DG_dx);
    setToZero(dxN);
    setToZero(dxOpt);
    setToZero(GxOpt);
    //   setToZero ( xH );

    //   Real gx = 0.0;        // g(u,T) = L2Norm(G(x))
    Real init_rx = 0.0;
    Real prev_rx = 0.0;
    Real init_gx = 0.0;
    Real prev_gx = 0.0;
    Real solverResidual = 1.0;
    Real dxOptNorm = 0.0;  // norm of hookstep
    Real dxNnorm = 0.0;    // norm of newton step
    Real xnorm = 0.0;
    bool expand_Krylov = false;

    std::ofstream osconv;
    if (taskid == 0)
        osconv.open((searchflags.outdir + "convergence.asc").c_str());
    osconv.setf(ios::left);
    osconv << setw(14) << "%-L2Norm(Gx)" << setw(14) << "rx" << setw(14) << "delta";
    osconv << setw(14) << "L2Norm(x)" << setw(14) << "L2Norm(u)";
    if (Tsearch)
        osconv << setw(14) << "T";
    if (Rxsearch)
        osconv << setw(14) << "ax";
    if (Rzsearch)
        osconv << setw(14) << "az";
    osconv << setw(14) << "L2Norm(dxN)"  // TobiasHack
           << setw(14) << "L2Norm(dxOpt)" << setw(14) << "SolverRes" << setw(10) << "ftotal" << setw(10) << "fnewt"
           << setw(10) << "fopt";
    osconv << endl;

    *os << endl;
    for (int nNewton = 0; nNewton <= Nnewton; ++nNewton) {
        *os << "========================================================" << endl;
        *os << "Newton iteration number " << nNewton << endl;
        // Compute quantities that will be used multiple times
        dxNnorm = L2Norm(dxN, Nconstr0);
        dxOptNorm = L2Norm(dxOpt, Nconstr0);
        xnorm = L2Norm(x, Nconstr0);
        Real unorm = msDSI_->DSIL2Norm(x);
        rx_ = 0.5 * L2Norm2(Gx);
        gx = L2Norm(Gx);

        if (nNewton == 0) {
            init_rx = rx_;
            prev_rx = rx_;
            init_gx = gx;
            prev_gx = gx;
        }
        Real l2dist = L2Dist(x, x0);

        *os << "Current state of Newton iteration:" << endl;
        *os << "   fcount_newton   == " << fcount_newton_ << endl;
        *os << "   fcount_optimiza == " << fcount_opt_ << endl;
        *os << "   L2Norm(x)       == " << xnorm << endl;
        *os << "   L2Norm(dxN)     == " << dxNnorm << endl;
        *os << "   L2Norm(dxOpt)   == " << dxOptNorm << endl;
        *os << "   L2Dist(x,x0)    == " << l2dist << endl;
        *os << "gx == L2Norm(G(x)) : " << endl;
        *os << "   initial  gx == " << init_gx << endl;
        *os << "   previous gx == " << prev_gx << endl;
        *os << "   current  gx == " << gx << endl;
        *os << "rx == 1/2 L2Norm2(G(x)) : " << endl;
        *os << "   initial  rx == " << init_rx << endl;
        *os << "   previous rx == " << prev_rx << endl;
        *os << "   current  rx == " << rx_ << endl;
        *os << "         delta == " << delta_ << endl;

        osconv.setf(ios::left);
        osconv << setw(14) << gx;
        osconv.setf(ios::left);
        osconv << setw(14) << rx_;
        osconv.setf(ios::left);
        osconv << setw(14) << delta_;
        osconv.setf(ios::left);
        osconv << setw(14) << xnorm;
        osconv.setf(ios::left);
        osconv << setw(14) << unorm;
        if (Tsearch)
            osconv << setw(14) << msDSI_->extractT(x);
        if (Rxsearch)
            osconv << setw(14) << msDSI_->extractXshift(x);
        if (Rzsearch)
            osconv << setw(14) << msDSI_->extractZshift(x);
        osconv.setf(ios::left);
        osconv << setw(14) << dxNnorm;
        osconv.setf(ios::left);
        osconv << setw(14) << dxOptNorm;
        osconv.setf(ios::left);
        osconv << setw(14) << solverResidual;
        osconv.setf(ios::left);
        osconv << setw(10) << fcount_newton_ + fcount_opt_;
        osconv.setf(ios::left);
        osconv << setw(10) << fcount_newton_;
        osconv.setf(ios::left);
        osconv << setw(10) << fcount_opt_;
        osconv << endl;

        msDSI_->save(x, filebase, searchflags.outdir);

        if (gx < epsSearch) {
            *os << "Newton search converged. Breaking." << endl;
            *os << "L2Norm(G(x)) == " << gx << " < "
                << "epsSearch"
                << " == " << epsSearch << endl;
            success = true;
            return x;
        } else if (nNewton == Nnewton) {
            *os << "Reached maximum number of Newton steps. Breaking." << endl;
            return x;
        }

        if (prev_gx / gx > searchflags.gRatio && nNewton >= 1)
            expand_Krylov = true;

        prev_rx = rx_;
        prev_gx = gx;

        if (Tsearch) {
            E.col(0) = msDSI_->tdiff(x0, searchflags.epsDt);
        }
        if (Rxsearch) {
            E.col(Tsearch) = msDSI_->xdiff(x0);
        }
        if (Rzsearch) {
            E.col(Tsearch + Rxsearch) = msDSI_->zdiff(x0);
        }

        if (searchflags.solver == SolverEigen) {
            // Use eigen to solve the linear system.
            // First compute the full Jacobian via finite differences, then solve the system with eigens dense solvers.
            *os << "Building Jacobi matrix..." << flush;
            MatrixXd J = jacobi(x, epsDx, centdiff, fcount_newton_);
            *os << "done" << endl;

            *os << "Solving linear system..." << flush;
            //       dx = J.householderQr().solve (fx);
            dxN = J.fullPivHouseholderQr().solve(Gx);
            dxN *= -1;
            *os << "done" << endl;

            solverResidual = L2Norm((VectorXd)(J * dxN + Gx)) / gx;
            *os << "Final residual of solver is " << solverResidual << endl;

        } else if (searchflags.solver == SolverBiCGStab) {
            Real epsJ = epsDx;
            bool cd = centdiff;
            int fcount_tmp = fcount_newton_;
            Rn2Rnfunc A = [&dsiG, &x, &Gx, epsJ, cd](const VectorXd& v) {
                int fcount = 0;
                return dsiG.Jacobian(x, v, Gx, epsJ, cd, fcount);
            };

            // iteratively solve the system with Bi-conjucate gradient stabilized method
            BiCGStabL<VectorXd> bicgstab(A, Gx, searchflags.lBiCGStab, NSolver);
            int nSolver;
            for (nSolver = 0; nSolver < NSolver; nSolver++) {
                *os << "Newt,BiCGStab(" << bicgstab.l() << ") step: " << nNewton << "," << nSolver << " ";
                bicgstab.iterate();
                *os << " residual = " << bicgstab.residual() << endl;
                solverResidual = bicgstab.residual();
                if (bicgstab.residual() < searchflags.epsSolver) {
                    break;
                }
            }
            *os << "Took " << nSolver << " BiCGstab steps, final residual is " << bicgstab.residual() << endl;
            dxN = -bicgstab.solution();
            fcount_newton_ = fcount_tmp;

        } else if (searchflags.solver == SolverGMRES) {
            // iteratively solve with GMRES

            // Set up RHS vector b = -G(x)
            VectorXd b(Nunk);  // RHS vector
            setToZero(b);
            b = Gx;
            b *= -1.0;
            gmres_ = unique_ptr<GMRES>(new GMRES(b, NSolver, searchflags.epsKrylov));
            // ===============================================================
            // GMRES iteration to solve DG(x) dx = -G(x)
            for (int n = 0; n < NSolver; ++n) {
                *os << "Newt,GMRES == " << nNewton << ',' << n << ", " << flush;

                // Compute v = Ab in Arnoldi iteration terms, where b is Q.column(n)
                // In Navier-Stokes terms, the main quantity to compute is
                // DG dx = 1/e (G(u + e du, sigma + e dsigma, T + e dT) - G(u,sigma,T)) for e << 1
                VectorXd q = gmres_->testVector();

                // Compute Df dx = 1/e (f(x+e*q) - f(x))
                VectorXd Lq = jacobianWithAC(x, q, Gx, epsDx, centdiff, fcount_newton_);
                //       VectorXd Lq = dsiG.Jacobian( x, q, Gx, epsDx, centdiff, arclengthConstraint.use(),
                //       fcount_newton );

                // Enforce orthogonality conditions
                // i.e. for e.g. xrel: Lq(xunk) = dG/dx dot q
                VectorXd e(Nunk);
                int tph = (msDSI_->tph()) ? 1 : 0;
                for (int i = tph; i < Northog; ++i) {
                    e = E.col(i);
                    Real res = L2IP(q, e);  // L2IP(Lq,e)
                    if (taskid == 0) {
                        Lq(uunk + i) = res;
                    }
                }

                gmres_->iterate(Lq);
                solverResidual = gmres_->residual();
                *os << " res == " << solverResidual << endl;

                if (solverResidual < searchflags.epsSolver) {
                    *os << "GMRES converged. Breaking." << endl;
                    dxN = gmres_->solution();
                    break;
                } else if (n == NSolver - 1 && solverResidual < searchflags.epsSolverF) {
                    *os << "GMRES has not converged, but the final iterate is acceptable. Breaking." << endl;
                    dxN = gmres_->solution();
                    break;
                } else if (n == NSolver - 1) {
                    *os << "GMRES failed to converge. Returning best answer so far." << endl;
                    *os << "Residual gx: " << gx << endl;
                    return x;
                }
            }  // end GMRES iteration

        } else if (searchflags.solver == SolverFGMRES) {
            // iteratively solve with Flexible GMRES

            // Set up RHS vector b = -G(x)
            VectorXd b(Nunk);  // RHS vector
            setToZero(b);
            b = Gx;
            b *= -1.0;

            MatrixXd Aq_p, q_p;
            if (expand_Krylov) {
                // Read vectors building Krylov subspace of the previous iteration to expand it for the next iteration
                q_p = fgmres_->Zn();
                Aq_p = fgmres_->AZn();
            }

            fgmres_ = unique_ptr<FGMRES>(new FGMRES(b, NSolver, searchflags.epsKrylov));
            // ===============================================================
            // FGMRES iteration to solve DG(x) dx = -G(x)
            for (int n = 0; n < NSolver; ++n) {
                VectorXd q;

                if (n == 0 && expand_Krylov) {  // First project vectors of the previous Krylov subspace on b
                    *os << endl << " <<< Projection of Krylov subspace of previous Newton steps on b >>>" << endl;

                    for (int k = 0; k < q_p.cols(); k++)
                        fgmres_->iterate(q_p.col(k), Aq_p.col(k));

                    *os << "Krylov subspace is already " << q_p.cols() << " dimensional" << endl;

                    q = fgmres_->b();

                } else  // then build on the krylov subspace to lower the error below a certain threshold
                    q = fgmres_->testVector();

                *os << "Newt,GMRES == " << nNewton << ',' << n << ", " << flush;

                // Compute Df dx = 1/e (f(x+e*q) - f(x))
                VectorXd Lq = jacobianWithAC(x, q, Gx, epsDx, centdiff, fcount_newton_);

                // Enforce orthogonality conditions
                // i.e. for e.g. xrel: Lq(xunk) = dG/dx dot q
                VectorXd e(Nunk);
                int tph = (msDSI_->tph()) ? 1 : 0;
                for (int i = tph; i < Northog; ++i) {
                    e = E.col(i);
                    Real res = L2IP(q, e);  // L2IP(Lq,e)
                    if (taskid == 0) {
                        Lq(uunk + i) = res;
                    }
                }

                fgmres_->iterate(q, Lq);
                solverResidual = fgmres_->residual();
                *os << " res == " << solverResidual << endl;

                if (solverResidual < searchflags.epsSolver) {
                    *os << "Flexible FGMRES converged. Breaking." << endl;
                    dxN = fgmres_->solution();
                    // gmres_->printAnb();
                    break;
                } else if (n == NSolver - 1 && solverResidual < searchflags.epsSolverF) {
                    *os << "Flexible FGMRES has not converged, but the final iterate is acceptable. Breaking." << endl;
                    dxN = fgmres_->solution();
                    break;
                } else if (n == NSolver - 1) {
                    *os << "Flexible FGMRES failed to converge. Returning best answer so far." << endl;
                    *os << "Residual gx: " << gx << endl;
                    return x;
                }
            }  // end FGMRES iteration

        } else {
            throw runtime_error(
                "NewtonAlgorithm::solve(): solver is not known. Possible values are 'eigen', 'bicgstab', 'gmres' or "
                "'fgmres' ");
        }

        dxNnorm = L2Norm(dxN, Nconstr0);

        // if(AC.use()) //confine the Newton update to the sphere defined by the pseudo arclength
        // dxN *= (AC.ds()/AC.arclength(x+dxN));

        if (AC->use())  // confine the Newton update to the sphere defined by the pseudo arclength
            dxN = AC->yLast() + (x + dxN - AC->yLast()) * (AC->ds() / AC->arclength(x + dxN)) - x;

        // optimize Newton update dxN->dxOpt
        int optfail = 1;
        dxOpt = dxN;
        if (searchflags.optimization == None) {
            if (AC->use()) {
                *os << "Running without convergence check. Use Newton step optimization for this check." << endl;
            }
            GxOpt = evalWithAC(x + dxOpt, fcount_newton_);  // Gx = f(x)
        } else if (searchflags.optimization == Linear) {
            GxOpt = Gx;
            if (AC->use()) {
                optfail = convergenceCheckAC(dxOpt, GxOpt, x);
            } else {
                optfail = linear(dxOpt, GxOpt, x);
            }
            if (optfail)
                return x;
        } else if (searchflags.optimization == Hookstep) {
            if (searchflags.solver == SolverGMRES) {
                GxOpt = Gx;
                if (AC->use()) {
                    *os << "Hookstep optimization is currently not implemented with Arclength Continuation." << endl;
                    optfail = convergenceCheckAC(dxOpt, GxOpt, x);
                } else {
                    VectorXd b(Gx);  // RHS vector
                    b *= -1.0;
                    optfail = hookstep(dxOpt, GxOpt, x, b);
                }
                if (optfail)
                    return x;
            }
        } else {
            throw runtime_error(
                "NewtonAlgorithm::solve(): optimization is not known. Possible values are 'none', 'linear' or "
                "'hookstep'");
        }

        *os << endl;
        *os << "Taking best step and continuing Newton iteration." << endl;
        x += dxOpt;
        Gx = GxOpt;
        xnorm = L2Norm(x, Nconstr0);
        dxOptNorm = L2Norm(dxOpt, Nconstr0);
        rx_ = 0.5 * L2Norm2(Gx);

        *os << "L2Norm(G(x)) == " << L2Norm(Gx) << endl;
        *os << "rx           == " << rx_ << endl;
        *os << "L2Norm(dxN)  == " << L2Norm(dxN, Nconstr0) << endl;
        *os << "L2Norm(dxOpt)== " << dxOptNorm << endl;
    }
    return x;
}

MatrixXd NewtonAlgorithm::jacobi(const VectorXd& x, const Real epsilon, const bool centerdiff, int& fcount) {
    // Sajjad: I think this function works only in serial; FIXME

    const int N = x.size();
    MatrixXd J(N, N);  // Jacobi Matrix
    VectorXd xe(N);    // x+eps*e_i
    VectorXd df_dxi;   // df/dx_i
    const VectorXd fx = evalWithAC(x, fcount);
    fcount++;
    VectorXd fxe(N);
    for (int i = 0; i < N; ++i) {
        xe.setZero();
        xe(i) += 1;
        df_dxi = jacobianWithAC(x, xe, fx, epsilon, centerdiff, fcount);

        for (int j = 0; j < N; ++j) {
            J(j, i) = df_dxi(j);
        }
    }

    return J;
}

VectorXd hookstepSearch(DSI& dsiG, const VectorXd& x0, const NewtonSearchFlags& searchflags, Real& gx) {
    NewtonAlgorithm Newt(searchflags);
    return Newt.solve(dsiG, x0, gx);
}

int NewtonAlgorithm::convergenceCheckAC(const VectorXd& dxN, VectorXd& Gx, const VectorXd& x) {
    assert(AC->use());
    *os << "Optimized Newton algorithm with AC: Computing residual to check convergence..." << flush;
    Real gxlast = L2Norm(Gx);
    Gx = evalWithAC(x + dxN, fcount_opt_);
    Real gx = L2Norm(Gx);
    *os << ": gx = " << gx;
    if (gx > gxlast) {
        *os << " > " << gxlast << endl;
        *os << "The error is increasing if we take the Newton step. Newton step optimization leaves step size "
               "correction to arclength continuation."
            << endl;
        return 1;
    } else {
        *os << " < " << gxlast << endl;
        *os << "The residual is decreasing if we take this AC-Newton step." << endl;
    }
    return 0;
}

int NewtonAlgorithm::linear(VectorXd& dxOpt, VectorXd& GxOpt, const VectorXd& x) {
    // Find optimal step size by linear interpolation (code by Tobias K.)

    // Helper function for computing residual of x + d*dx as a function of d
    std::function<Real(Real)> residual = [&x, &dxOpt, this](Real d) {
        VectorXd edx = d * dxOpt;
        *os << "Computing residual for d = " << d << ": " << flush;
        Real res = L2Norm(evalWithAC(x + edx, fcount_opt_));
        *os << ", gx = " << res << endl;
        return res;
    };

    Real d = 1;
    Real gxlast = L2Norm(GxOpt);
    Real gx = residual(d);
    if (gx > gxlast) {
        *os << "The error is increasing if we take the Newton step. Try decreasing step size until we are improving."
            << endl;
        while (gx > gxlast) {
            *os << "Error of Newton step is " << gx << ", increasing! Halving step size." << endl;
            d /= 2;
            gx = residual(d);
        }

        if (d < searchflags.delta) {
            *os << "NewtonSimple: Newton step is increasing and we can't decrease stepsize below searchflags.delta. "
                   "Returning best solution so far."
                << endl;
            return 1;
        }
        // We needed to decrease d in order to have a good step. This indicates that our Newton search was not optimal
        // to start with. So let's optimize for the step size!
        Brent b(residual, d, gx, 0, gxlast, 2 * d, residual(2 * d));
        d = b.minimize(30, 1e-6, 1e-6);
    } else {  // gx < gxlast
        *os << "The residual is decreasing if we take the Newton step, but let's try a step of half the size anyways."
            << endl;
        Real gx2 = residual(0.5);

        if (gx < gx2) {
            *os << "The original step is better, so we keep this and continue." << endl;
        } else {
            *os << "The reduced step is better than the original. Try to find the optimum step size." << endl;
            Brent b(residual, 0.5, gx2, 0, gxlast, 1, gx);
            d = b.minimize(30, 1e-6, 1e-6);
        }
    }  // gx > gxlast

    *os << "Computing gx for Newton step " << flush;
    dxOpt *= d;
    GxOpt = evalWithAC(x + dxOpt, fcount_opt_);  // Gx = f(x)

    return 0;
}

int NewtonAlgorithm::hookstep(VectorXd& dxH, VectorXd& GxH, const VectorXd& x, const VectorXd& b) {
    // Redefine what was done in solve and is needed here
    int taskid = mpirank();
    // Need to construct orthogonality constraints
    bool Rxsearch = searchflags.xrelative;
    bool Rzsearch = searchflags.zrelative;
    // TODO: this breaks poincare search, doesn't it?
    bool Tsearch = searchflags.solntype == PeriodicOrbit ? true : false;
    const Real epsDx = searchflags.epsDx;
    const bool centdiff = searchflags.centdiff;

    /***/ int Nunk = b.rows();                          // # of unknowns in x
    const int Northog = Tsearch + Rxsearch + Rzsearch;  // # of orthogonality constraints
    const int Nconstr = Northog + AC->use();            // # of constraints, orthog and arclength
    const int Nconstr0 = (taskid == 0) ? Nconstr : 0;
    const int xac = Nunk - 1;

    VectorXd dxN(dxH);
    VectorXd Gx(GxH);

    VectorXd xH(Nunk);  // x plus dxH
    setToZero(xH);
    // ==================================================================
    // Hookstep algorithm
    int Nk = 0;
    MatrixXd Hn;
    // MatrixXd Qn1;
    const MatrixXd& Q = gmres_->Q();
    JacobiSVD<MatrixXd> svd;

    MatrixXd V;
    MatrixXd U;
    VectorXd D;

    RowVectorXd bQn1;
    VectorXd bh(1);

    *os << "------------------------------------------------" << endl;
    *os << "Beginning hookstep calculations." << endl;

    Nk = gmres_->n();  // Krylov dimension
    Hn = gmres_->Hn();
    // Qn1 = gmres_->Qn1();
    // gmres_->resetQ();
    svd = JacobiSVD<MatrixXd>(Hn, ComputeThinU | ComputeThinV);

    V = svd.matrixV();
    U = svd.matrixU();
    D = svd.singularValues();

    // Manual multiplication of
    // ColumnVector btmp = Qn1.transpose() * b; // Nk+1 x Nunk  times  Nunk x 1 == Nk+1 x 1
    VectorXd btmp(Nk + 1);
    for (int i = 0; i < Nk + 1; ++i) {
        Real sum = 0.0;
        for (int j = 0; j < Nunk; ++j)
            sum += Q(j, i) * b(j);
#ifdef HAVE_MPI
        Real tmp = sum;
        MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
        btmp(i) = sum;
    }

    // Manual multiplication of
    // ColumnVector bh = U.transpose()*btmp;  // Nk x Nk+1 times Nk+1 x 1 == Nk x 1
    bh.resize(Nk, 1);
    for (int i = 0; i < Nk; ++i) {
        Real sum = 0.0;
        for (int j = 0; j < Nk + 1; ++j)
            sum += U(j, i) * btmp(j);
        bh(i) = sum;
    }
    //    bQn1 = b.transpose() *Qn1;
    //    bh = Ut*bQn1.transpose();

    int hookcount = 0;

    // Hookstep algorithm requires iterated tweaking. Remember the best hookstep
    // with these variables, so we can revert to it if necessary.
    bool have_backup = false;
    Real backup_dxHnorm = 0.0;
    Real backup_rH = 0.0;
    Real backup_delta = 0.0;
    VectorXd backup_dxH;
    VectorXd backup_GxH;
    //    Outcome outcome = do_nothing;
    bool hookstep_equals_newtonstep = true;

    Real deltaMaxLocal = searchflags.deltaMax;

    // Find a good trust region and the optimal hookstep in it.
    while (true) {
        *os << "-------------------------------------------" << endl;
        *os << "Hookstep number " << hookcount << endl;
        *os << "delta == " << delta_ << endl;
        hookcount++;
        Real dxn = L2Norm(dxN, Nconstr0);
        if (dxn <= delta_) {
            *os << "Newton step is within trust region: " << endl;
            *os << "L2Norm(dxN) == " << dxn << " <= " << delta_ << " == delta" << endl;

            hookstep_equals_newtonstep = true;
            dxH = dxN;
        } else {
            *os << "Newton step is outside trust region: " << endl;
            *os << "L2Norm(dxN) == " << dxn << " > " << delta_ << " == delta" << endl;
            *os << "Calculate hookstep dxH(mu) with radius L2Norm(dxH(mu)) == delta" << endl;
            hookstep_equals_newtonstep = false;

            // This for-loop determines the hookstep. Search for value of mu that
            // produces ||dxHmu|| == delta. That provides optimal reduction of
            // quadratic model of residual in trust region norm(dxHmu) <= delta.
            VectorXd dxHmu(Nk);  // dxH(mu) : Hookstep dxH as a function of parameter
            Real ndxHmu = 0;     // norm of dxHmu
            Real mu = 0;         // a parameter we search over to find dxH with radius delta

            // See Dennis and Schnabel for this search-over-mu algorithm
            for (int hookSearch = 0; hookSearch < searchflags.Nhook; ++hookSearch) {
                for (int i = 0; i < Nk; ++i) {
                    dxHmu(i) = bh(i) / (D(i) + mu);  // research notes
                }
                ndxHmu = L2Norm(dxHmu);
                Real Phi = ndxHmu * ndxHmu - delta_ * delta_;
                *os << "mu, L2Norm(dxH(mu)) == " << mu << ", " << ndxHmu << endl;
                // Found satisfactory value of mu and thus dxHmu and dxH s.t. |dxH| < delta
                if (ndxHmu < delta_ ||
                    (ndxHmu > (1 - searchflags.deltaFuzz) * delta_ && ndxHmu < (1 + searchflags.deltaFuzz) * delta_)) {
                    break;  // FIXME: this break was introduced by Tobias. I think, it is in place here, but I'm not
                            // sure. So please remove it, if I'm wrong.
                }
                // Update value of mu and try again. Update rule is a Newton search for
                // Phi(mu)==0 based on a model of form a/(b+mu) for norm(sh(mu)). See
                // Dennis & Schnabel.
                else if (hookSearch < searchflags.Nhook - 1) {
                    Real PhiPrime = 0.0;
                    for (int i = 0; i < Nk; ++i) {
                        Real di_mu = D(i) + mu;
                        Real bi = bh(i);
                        PhiPrime -= 2 * bi * bi / (di_mu * di_mu * di_mu);
                    }
                    mu -= (ndxHmu / delta_) * (Phi / PhiPrime);
                } else {
                    *os << "Couldn't find solution of hookstep optimization eqn Phi(mu)==0" << endl;
                    *os << "This shouldn't happen. It indicates an error in the algorithm." << endl;
                    *os << "Returning best answer so far\n";
                    return 1;
                }
            }  // search over mu for norm(s) == delta
            Real dxhmu = L2Norm(dxHmu, Nconstr0);
            *os << "Found hookstep of proper radius" << endl;
            *os << "ndxHmu == " << dxhmu << " ~= " << delta_ << " == delta " << endl;
            dxH = Q.block(0, 0, Nunk, Nk) * (V * dxHmu);
        }  // end else clause for Newton step outside trust region

        // Compute
        // (1) actual residual from evaluation of
        //     rH == r(x+dxH) act  == 1/2 (G(x+dxH), G(x+dxH))
        //
        // (2) predicted residual from quadratic model of r(x)
        //     rP == r(x+dx) pred  == 1/2 (G(x), G(x)) + (G(x), DG dx)
        //
        // (3) slope of r(x)
        //     dr/dx == (r(x + dx) - r(x))/|dx|
        //           == (G(x), DG dx)/|dx|
        // where ( , ) is FlowField inner product V2IP, which equals L2 norm of vector rep

        // (1) Compute actual residual of step, rH
        *os << "Computing residual of hookstep dxH" << endl;

        xH = x;
        xH += dxH;
        *os << "L2Norm(dxH) == " << L2Norm(dxH, Nconstr0) << endl;

        GxH = evalWithAC(xH, fcount_opt_);
        if (AC->use() && taskid == 0)
            GxH(xac) = 0;
        //       GxH = dsiG.eval ( xH );

        *os << endl;
        Real rH = 0.5 * L2Norm2(GxH);  // actual residual of hookstep
        Real Delta_rH = rH - rx_;      // improvement in residual
        *os << "r(x), r(x+dxH) == " << rx_ << ", " << rH << endl;

        // (2) and (3) Compute quadratic model and slope
        *os << "Computing local quadratic model of residual" << endl;

        //       VectorXd DG_dx = dsiG.Jacobian( x, dxH, Gx, epsDx, centdiff, fcount_hookstep );
        VectorXd DG_dx = jacobianWithAC(x, dxH, Gx, epsDx, centdiff, fcount_opt_);
        if (AC->use() && taskid == 0)
            DG_dx(xac) = 0;
        *os << endl;
        // Local quadratic and linear models of residual, based on Taylor exp at current position
        // Quadratic model of r(x+dx): rQ == 1/2 |G(x) + DG dx|^2
        // Linear  model   of r(x+dx): rL == 1/2 (G(x), G(x)) + (G(x), DG dx)

        Real Delta_rL = L2IP(Gx, DG_dx);  // rL - 1/2 (G(x), G(x)) == (G(x), DG dx)

        Real rL = rx_ + Delta_rL;  // rL == 1/2 (G(x), G(x)) + (G(x), DG dx)
        if (rL >= rx_) {
            *os << "error : local linear model of residual is increasing, indicating\n"
                << "        that the solution to the Newton equations is inaccurate\n";

            if (searchflags.centdiff == true) {
                *os << "Returning best answer so far." << endl;
                return 1;
            } else {
                *os << "Trying local linear model again, using centered finite differencing" << endl;
                //           DG_dx = dsiG.Jacobian ( x, dxH, Gx, epsDx, true, fcount_hookstep );
                DG_dx = jacobianWithAC(x, dxH, Gx, epsDx, true, fcount_opt_);
                if (AC->use() && taskid == 0)
                    DG_dx(xac) = 0;

                Delta_rL = L2IP(Gx, DG_dx);  // rL - 1/2 (G(x),G(x)) == (G(x), DG dx)
                rL = rx_ + Delta_rL;         // rL == 1/2 (G(x),G(x)) + (G(x), DG dx)

                if (rL >= rx_) {
                    *os << "error : centered differencing didn't help\n"
                        << "Returning best answer so far." << endl;
                    return 1;
                }
                // if we get here, centered differencing does help, and we can continue
            }
        }
        DG_dx += Gx;
        Real rQ = 0.5 * L2Norm2(DG_dx);  // rQ == 1/2 |G(x) + DG dx|^2
        Real Delta_rQ = rQ - rx_;        // rQ == 1/2 (G(x),G(x)) - 1/2 |G(x) + DG dx|^2
        Real dxHnorm = L2Norm(dxH, Nconstr0);

        // ===========================================================================
        *os << "Determining what to do with current Newton/hookstep and trust region" << endl;

        // Coefficients for non-local quadratic model of residual, based on r(x), r'(x) and r(x+dx)
        // dr/dx == (G(x), DG dx)/|dx|
        Real drdx = Delta_rL / dxHnorm;

        // Try to minimize a quadratic model of the residual
        // r(x+dx) = r(x) + r' |dx| + 1/2 r'' |dx|^2
        Real lambda = -0.5 * drdx * dxHnorm / (rH - rx_ - drdx * dxHnorm);

        // Compare the actual reduction in residual to the quadratic model of residual.
        // How well the model matches the actual reduction will determine, later on, how and
        // when the radius of the trust region should be changed.
        Real Delta_rH_req = searchflags.improvReq * drdx * dxHnorm;  // the minimum acceptable change in residual
        Real Delta_rH_ok = searchflags.improvOk * Delta_rQ;  // acceptable, but reduce trust region for next newton step
        Real Delta_rH_good =
            searchflags.improvGood * Delta_rQ;  // acceptable, keep same trust region in next newton step
        Real Delta_rQ_acc =
            abs(searchflags.improvAcc * Delta_rQ);  // for accurate models, increase trust region and recompute hookstep
        Real Delta_rH_accP = Delta_rQ + Delta_rQ_acc;  //   upper bound for accurate predictions
        Real Delta_rH_accM = Delta_rQ - Delta_rQ_acc;  //   lower bound for accurate predictions

        // Characterise change in residual Delta_rH
        ResidualImprovement improvement;  // fine-grained characterization

        // Place improvement in contiguous spectrum: Unacceptable > Poor > Ok > Good.
        if (Delta_rH > Delta_rH_req)
            improvement = Unacceptable;  // not even a tiny fraction of linear prediction
        else if (Delta_rH > Delta_rH_ok)
            improvement = Poor;  // worse than small fraction of quadratic prediction
        else if (Delta_rH < Delta_rH_ok && Delta_rH > Delta_rH_good)
            improvement = Ok;
        else {
            improvement = Good;  // not much worse or better than large fraction of prediction
            if (Delta_rH_accM <= Delta_rH && Delta_rH <= Delta_rH_accP)
                improvement = Accurate;  // close to quadratic prediction
            else if (Delta_rH < Delta_rL)
                improvement = NegaCurve;  // negative curvature in r(|s|) => try bigger step
        }
        const int w = 13;
        *os << "rx       == " << setw(w) << rx_ << " residual at current position" << endl;
        *os << "rH       == " << setw(w) << rH << " residual of newton/hookstep" << endl;
        *os << "Delta_rH == " << setw(w) << Delta_rH << " actual improvement in residual from newton/hookstep" << endl;
        *os << endl;
        if (improvement == Unacceptable)
            *os << "Delta_rH == " << setw(w) << Delta_rH << " ------> Unacceptable <------ " << endl;
        *os << "            " << setw(w) << Delta_rH_req << " lower bound for acceptable improvement" << endl;
        if (improvement == Poor)
            *os << "Delta_rH == " << setw(w) << Delta_rH << " ------> Poor <------" << endl;
        *os << "            " << setw(w) << Delta_rH_ok << " upper bound for ok improvement." << endl;
        if (improvement == Ok)
            *os << "Delta_rH == " << setw(w) << Delta_rH << " ------> Ok <------" << endl;
        *os << "            " << setw(w) << Delta_rH_good << " upper bound for good improvement." << endl;
        if (improvement == Good)
            *os << "Delta_rH == " << setw(w) << Delta_rH << " ------> Good <------" << endl;
        *os << "            " << setw(w) << Delta_rH_accP << " upper bound for accurate prediction." << endl;
        if (improvement == Accurate)
            *os << "Delta_rH == " << setw(w) << Delta_rH << " ------> Accurate <------" << endl;
        *os << "            " << setw(w) << Delta_rH_accM << " lower bound for accurate prediction." << endl;
        *os << "            " << setw(w) << Delta_rL << " local linear model of improvement" << endl;
        if (improvement == NegaCurve)
            *os << "Delta_rH == " << setw(w) << Delta_rH << " ------> NegativeCurvature <------" << endl;

        bool recompute_hookstep = false;

        *os << "lambda       == " << lambda
            << " is the reduction/increase factor for delta suggested by quadratic model" << endl;
        *os << "lambda*delta == " << lambda * delta_ << " is the delta suggested by quadratic model" << endl;
        // See comments at end for outline of this control structure

        // Following Dennis and Schnable, if the Newton step lies within the trust region,
        // reset trust region radius to the length of the Newton step, and then adjust it
        // further according to the quality of the residual.
        // if (hookstep_equals_newtonstep)
        // delta = dxHnorm;

        // CASE 1: UNACCEPTABLE IMPROVEMENT
        // Improvement is so bad (relative to gradient at current position) that we'll
        // in all likelihood get better results in a smaller trust region.
        if (improvement == Unacceptable) {
            *os << "Improvement is unacceptable." << endl;

            if (have_backup) {
                *os << "But we have a backup step that was acceptable." << endl;
                *os << "Revert to backup step and backup delta, take the step, and go to next Newton-GMRES iteration."
                    << endl;
                dxH = backup_dxH;
                GxH = backup_GxH;
                rH = backup_rH;
                dxHnorm = backup_dxHnorm;
                delta_ = backup_delta;

                recompute_hookstep = false;
            } else {
                *os << "No backup step is available." << endl;
                *os << "Reduce trust region by minimizing local quadratic model and recompute hookstep." << endl;
                deltaMaxLocal = delta_;
                lambda = adjustLambda(lambda, searchflags.lambdaMin, searchflags.lambdaRequiredReduction, *os);
                delta_ = adjustDelta(delta_, lambda, searchflags.deltaMin, searchflags.deltaMax, *os);
                *os << "delta new == " << delta_ << endl;
                if (delta_ < searchflags.deltaMin) {
                    *os << "delta min == " << searchflags.deltaMin << endl;
                    *os << "Reduced delta is below deltaMin. That's too small, so reset delta to deltaMin" << endl;
                    delta_ = searchflags.deltaMin;
                    *os << "delta new == " << delta_ << endl;
                    return 1;
                } else if (delta_ > dxHnorm) {
                    *os << "That delta is still bigger than the Newton step." << endl;
                    // *os << "This shouldn't happen. Control structure needs review." << endl;
                    *os << "Reducing delta to half the length of the Newton step." << endl;
                    *os << "  old delta == " << delta_ << endl;
                    delta_ = 0.5 * dxHnorm;
                    *os << "  new delta == " << delta_ << endl;
                    if (delta_ < searchflags.deltaMin) {
                        *os << "But that is less than deltaMin == " << searchflags.deltaMin << endl;
                        *os << "I am afraid we have to stop now. Returning best solution." << endl;
                        return 1;
                    }
                }
                // delta_has_decreased = true;
                recompute_hookstep = true;
            }
        }
        // CASE 2: EXCELLENT IMPROVEMENT AND ROOM TO GROW
        // Improvement == Accurate or Negacurve means we're likely to get a significant improvement
        // by increasing trust region. Increase trust region by a fixed factor (rather than quadratic
        // model) so that trust-region search is monotonic.
        // Exceptions are
        //   have_backup && backup_rH < rH -- residual is getting wrose as we increase delta
        //   hookstep==newtonstep          -- increasing trust region won't change answer
        //   delta>=deltamax               -- we're already at the largest allowable trust region
        else if ((improvement == NegaCurve || improvement == Accurate) && !(have_backup && backup_rH < rH) &&
                 !hookstep_equals_newtonstep && !(delta_ >= searchflags.deltaMax)) {
            *os << "Improvement is " << improvement << endl;

            // lambda = adjustLambda(lambda, searchflags.lambdaMin, searchflags.lambdaMax);
            Real new_delta =
                adjustDelta(delta_, searchflags.lambdaMax, searchflags.deltaMin, searchflags.deltaMax, *os);

            if (new_delta < deltaMaxLocal) {
                *os << "Continue adjusting trust region radius delta because" << endl;
                *os << "Suggested delta has room: new_delta < deltaMaxLocal " << new_delta << " < " << deltaMaxLocal
                    << endl;
                if (have_backup)
                    *os << "And residual is improving:     rH < backup_sS     " << rH << " < " << backup_rH << endl;
                *os << "Increase delta and recompute hookstep." << endl;
                have_backup = true;
                backup_dxH = dxH;
                backup_GxH = GxH;
                backup_rH = rH;
                backup_dxHnorm = dxHnorm;
                backup_delta = delta_;

                recompute_hookstep = true;
                *os << " old delta == " << delta_ << endl;
                delta_ = new_delta;
                *os << " new delta == " << delta_ << endl;

            } else {
                delta_ = deltaMaxLocal;
                *os << "Stop adjusting trust region radius delta and take step because the new delta" << endl;
                *os << "reached a local limit:  new_delta >= deltaMaxLocal " << new_delta << " >= " << deltaMaxLocal
                    << endl;
                *os << "Reset delta to local limit and go to next Newton iteration" << endl;
                *os << "  delta == " << delta_ << endl;
                recompute_hookstep = false;
            }
        }

        // CASE 3: MODERATE IMPROVEMENT, NO ROOM TO GROW, OR BACKUP IS BETTER
        // Remaining cases: Improvement is acceptable: either Poor, Ok, Good, or
        // {Accurate/NegaCurve and (backup is superior || Hookstep==NewtonStep || delta>=deltaMaxLocal)}.
        // In all these cases take the current step or backup if better.
        // Adjust delta according to accuracy of quadratic prediction of residual (Poor, Ok, etc).
        else {
            *os << "Improvement is " << improvement << " (some form of acceptable)." << endl;
            *os << "Stop adjusting trust region and take a step because" << endl;
            *os << "Improvement is merely Poor, Ok, or Good : "
                << (improvement == Poor || improvement == Ok || improvement == Good) << endl;
            *os << "Newton step is within trust region      : " << hookstep_equals_newtonstep << endl;
            *os << "Backup step is better                   : " << (have_backup && backup_rH < rH) << endl;
            *os << "Delta has reached local limit           : " << (delta_ >= searchflags.deltaMax) << endl;
            recompute_hookstep = false;

            // Backup step is better. Take it instead of current step.
            if (have_backup && backup_rH < rH) {
                *os << "Take backup step and set delta to backup value." << endl;
                dxH = backup_dxH;
                GxH = backup_GxH;
                *os << "  old delta == " << delta_ << endl;
                dxHnorm = backup_dxHnorm;
                delta_ = backup_delta;
                *os << "  new delta == " << delta_ << endl;
            }

            // Current step is best available. Take it. Adjust delta according to Poor, Ok, etc.
            // Will start new monotonic trust-region search in Newton-hookstep, so be more
            // flexible about adjusting delta and use quadratic model.
            else {
                if (have_backup) {
                    *os << "Take current step and keep current delta, since it's produced best result." << endl;
                    *os << "delta == " << delta_ << endl;
                    // Keep current delta because current step was arrived at through a sequence of
                    // adjustments in delta, and this value yielded best results.
                } else if (improvement == Poor) {
                    *os << "Take current step and reduce delta, because improvement is poor." << endl;
                    lambda = adjustLambda(lambda, searchflags.lambdaMin, 1.0, *os);
                    delta_ = adjustDelta(delta_, lambda, searchflags.deltaMin, searchflags.deltaMax, *os);
                    if (delta_ < searchflags.deltaMin) {
                        *os << "    delta == " << delta_ << endl;
                        *os << "delta min == " << searchflags.deltaMin << endl;
                        *os << "Sorry, can't go below deltaMin. Returning current best solution" << endl;
                        return 1;
                    }
                } else if (improvement == Ok || hookstep_equals_newtonstep || (delta_ >= searchflags.deltaMax)) {
                    *os << "Take current step and leave delta unchanged, for the following reasons:" << endl;
                    *os << "improvement            : " << improvement << endl;
                    *os << "|newtonstep| <= delta  : " << (hookstep_equals_newtonstep ? "true" : "false") << endl;
                    *os << "delta >= deltaMax      : " << (delta_ >= searchflags.deltaMax ? "true" : "false") << endl;
                    // *os << "delta has decreased    : " << (delta_has_decreased ? "true" : "false") << endl;
                    *os << "delta == " << delta_ << endl;
                } else {  // improvement == Good, Accurate, or NegaCurve and no restriction on increasing apply
                    *os << "Take step and increase delta, if there's room." << endl;
                    lambda = adjustLambda(lambda, 1.0, searchflags.lambdaMax, *os);
                    delta_ = adjustDelta(delta_, lambda, searchflags.deltaMin, searchflags.deltaMax, *os);
                }
            }
        }
        // *os << "Recompute hookstep ? " << (recompute_hookstep ? "yes" : "no") << endl;

        if (recompute_hookstep)
            continue;
        else
            break;

    }  // matches while (true) for finding good trust region and optimal hookstep within it

    return 0;
}

// The following is meant as a guide to the structure and notation
// of the above code. See Divakar Viswanath, "Recurrent motions
// within plane Couette turbulence", J. Fluid Mech. 580 (2007),
// http://arxiv.org/abs/physics/0604062 for a higher-level presentation
// of the algorithm.
//
// I've biased the notation towards conventions from fluids and from
// Predrag Cvitanovic's ChaosBook, and away from Dennis & Schnabel and
// Trefethen. That is, f stands for the finite-time map of plane Couette
// dynamics, not the residual being minimized.
//
// Note: in code, dt stands for the integration timestep and
//                dT stands for Newton-step increment of period T
// But for beauty's sake I'm using t and dt for the period and
// its Newton-step increment in the following comments.
//
// Let
//     f^t(u) be the time-t map of Navier-Stokes computed by DNS.
//     sigma    be a symmetry of the flow
//
// We seek solutions of the equation
//
//    G(u,sigma,t) = sigma f^t(u) - u = 0                       (1)
//
// t can be a fixed or a free variable, as can sigma. For plane
// Couette the potential variables in sigma are ax,az in the
// phase shifts x -> x + ax Lx and z -> z + az Lz.
//
// Let x be the vector of unknown real numbers being sought.
// For simplicity's sake in this exposition I will assume that
// we're looking for a relative periodic orbit, so that the
// period t of the orbit and the phase shifts ax and az are
// unknowns. Then x = (u, sigma, t). Of course by this I really
// that x is the finite set of real-valued variables that
// parameterize the discrete representation of u (e.g. the
// real and imaginary parts of the spectral coefficients),
// together with the real-valued variables that parameterize
// sigma, and the real number t. To be perfectly pedantic (and
// because it will come in handy later), let there be M
// independent real numbers {u_m} in the discrete representation
// of u, and let
//
//    {u_m} = P(u),  u_m = P_m(u)                           (2)
//
// represent the map between continuous fields u and the discrete
// representation {u_m}. Then x = (u, sigma, t) is an M+3 dimensional
// vector
//
//    x_m = (u_1, u_2, ..., u_M, ax, az, t)
//
// and the discrete equation to be solved is
//
//   G(x_m) = 0                                               (3)
//
// where G is a vector-valued function of dimension M.
//
//   G_m(x_m)   = P_m(sigma f^t(u) - u)                         (4)
//
// At this point we have three fewer equations in (4) than unknowns
// in x because the equivariance of the flow means that solutions
// are indeterminate under time and phase shifts.
//
// The Newton algorithm for solving G(x) = 0 is
//
// Let x* = (u*,sigma*,t*) be the solution, i.e. G(x*) = G(u*,sigma*, t*) = 0.
// Suppose we start with an initial guess x near x*.
// Let
//     x* = x + dxN       (dxN is the Newton step)           (5)
//     u* = u + duN
// sigma* = sigma + dsigmaN
//     t* = t + dtN
//
// Then G(x*) = G(x + dxN)
//          0 = G(x) + DG(x) dxN + O(|dxN|^2)
//
// Drop h.o.t and solve the Newton equation
//
//    DG(x) dxN = -G(x)                                     (6)
//
// Because DG is a M x (M+3) matrix, we have to supplement this
// system with three constraint equations to have a unique solution
// dxN.
//
//   (duN, du/dt) = 0                                       (7)
//   (duN, du/dx) = 0
//   (duN, du/dz) = 0
//
// ( , ) here signifies an inner product, so these are orthogonality
// constraints preventing the Newton step from going in the directions
// of the time, x, and z equivariance. That forms an (M+3) x (M+3)
// dimensional system
//
//    A dxN = b                                             (8)
//
//  where the first M rows are given by (6) and the last three by (7).
//
//
// Since M is very large, we will use the iterative GMRES algorithm
// to find an approximate solution dxN to the (M+3)x(M+3) system A dxN = b.
// GMRES requires multiple calculations A dx for test values of dx. The
// left-hand side of (6) can be approximated with finite differencing:
//
//   DG(x) dx = 1/e (G(x + e dx) - G(x)) where e |dx| << 1.
//            = 1/e P[(sigma+dsigma) f^{t+dt}(u+du) - (u+du) - (sigma f^{t}(u) - u)]
//            = 1/e P[(sigma+dsigma) f^{t+dt}(u+du) - sigma f^{t}(u) - du]
//
// And the left-hand sides of (7) are a simple evaluation of an inner
// product. For details of the GMRES algorithm, see Trefethen and Bau.

// That gives us the Newton step dxN. However if we are too far from
// the solution x*, the linearization inherent in the Newton algorithm
// will not be accurate. At this point we switch from the pure Newton
// algorithm to a constrained minimization algorithm called the
// "hookstep", specially adapted to work with GMRES.

// Hookstep algorithm: If the Newton step dxN does not decrease
// the residual || G(x+dxN) || sufficiently, we obtain a smaller step
// dxH by minimizing || A dxH - b ||^2 subject to ||dxH||^2 <= delta^2,
// rather using the Newton step, which solves A dx = b.

// GMRES-Hookstep algorithm: Since A is very high-dimensional, we
// further constrain the hookstep dxH to lie within the Krylov subspace
// obtained from the solution of the Newton step equations. I.e.
// we minimize the norm of the projection of A dxH - b onto the Krylov
// subspace. The nth GMRES iterate gives matrices Qn, Qn1, and Hn such
// that
//
//   A Qn = Qn1 Hn                                              (9)
//
// where Qn is M x n, Qn1 is M x (n+1), and Hn is (n+1) x n.
// Projection of A dx - b on (n+1)th Krylov subspace is Qn1^T (A dx - b)
// Confine dx to nth Krylov subspace: dxH = Qn sn, where sn is n x 1.
// Then minimize
//
//   || Qn1^T (A dxH - b) || = || Qn1^T A Qn sn - Qn1^T b ||
//                          = || Qn1^T Qn1 Hn sn - Qn1^T b ||
//                          = || Hn sn - Qn1^T b ||            (10)
//
// subject to ||dxH||^2 = || sn ||^2 <= delta^2. Note that the quantity in
// the norm on the right-hand side of (10) is only n-dimensional, and n is
// small (order 10 or 100) compared to M (10^5 or 10^6).
//
// Do minimization of the RHS of (10) via SVD of Hn = U D V^T.
//   || Hn sn - Qn1^T b || = || U D V^T sn - Qn1^T b ||
//                         = || D V^T sn - U^T Qn1^T b ||
//                         = || D sh - bh ||
//
// where sh = V^T sn and bh = U^T Qn1^T b. Now we need to
// minimize || D sh - bh ||^2 subject to ||sh||^2 <= delta^2
//
// From Divakar Viswanath (personal communication):
//
// Since D is diagonal D_{i,i} = d_i, the constrained minimization problem
// can be solved easily with Lagrange multipliers. The solution is
//
//   sh_i = (bh_i d_i)/(d^2_i + mu)
//
// for the mu such that ||sh||^2 = delta^2. The value of mu can be
// found with a 1d Newton search for the zero of
//
//   Phi(mu) = ||sh(mu)||^2 - delta^2
//
// A straight Newton search a la mu -= Phi(mu)/Phi'(mu) is suboptimal
// since Phi'(mu) -> 0 as mu -> infty with Phi''(mu) > 0 everywhere, so
// we use a slightly modified update rule for the Newton search over mu.
// Please refer to Dennis and Schnabel regarding that. Then, given the
// solution sh(mu), we compute the hookstep solution s from
//
//   dxH = Qn sn = Qn V sh
//
//
// How do we know a good value for delta? Essentially, by comparing
// the reduction in residual obtained by actually taking the hookstep
// dxH to the reduction predicted by the linearized model  G(x+dxH) =
// G(x) + DG(x) dxH. If the reduction is accurate but small, we increase
// delta by small steps until the reduction is marginally accurate but larger.
// If the reduction is poor we reduce delta.
//
// (Note: the trust-region optimization is actually performed in terms of the
// squared residual, so the comments in the code refer to a quadratic rather than
// linear model of the residual.)
//
// The heuristics associated with adjusting the trust region are fairly complex.
// Dennis and Schnabel has a decent description, but it took quite a bit of effort
// to translate that description into an algorithm. Rather than attempt to
// translate the algorithm back into prose, I recommend you read Dennis and
// Schnabel and then refer to the code and comments. This portion of findorbit.cpp
// is very liberally commented.

// References:
//
// Divakar Viswanath, "Recurrent motions within plane Couette turbulence",
// J. Fluid Mech. 580 (2007), http://arxiv.org/abs/physics/0604062
//
// Lloyd N. Trefethen and David Bau, "Numerical Linear Algebra", SIAM,
// Philadelphia, 1997.
//
// Dennis and Schnabel, "Numerical Methods for Unconstrained Optimization
// and Nonlinear Equations", Prentice-Hall Series in Computational Mathematics,
// Englewood Cliffs, New Jersey, 1983.
//

// ======================================================================

}  // namespace chflow
