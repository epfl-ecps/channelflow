/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */

#include "nsolver/config.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "cfbasics/cfbasics.h"
#include "nsolver/continuation.h"

#include <stdexcept>
using namespace std;

using namespace Eigen;

namespace chflow {

ContinuationFlags::ContinuationFlags()
    : arclength(false),
      initialParamStep(1e-04),
      s0(0.0),
      ds0(0.0),
      dsmax(0.2),
      dsmin(1e-5),
      guessErrMin(1e-5),
      guessErrMax(2e-5),
      digits(8),
      orbOut(false),
      restartMode(false),
      maxSteps(100),
      muRef(0.0),
      haveTargetMu(false),
      targetMu(0) {}

ContinuationFlags::ContinuationFlags(ArgList& args) {
    args.section("Continuation options");
    arclength = args.getflag("-al", "--arclength", "use arclength continuation instead of parametric continuation");
    initialParamStep = args.getreal("-dmu", "--dmu", 1e-4,
                                    "initial relative increment for quadratic extrapolation in continuation parameter");
    s0 = args.getreal("-s0", "--s0", 0.0, "start value for arclength (arbitrary)");
    ds0 = args.getreal("-ds0", "--ds0", 0.0, "initial arclength increment for continuation");
    dsmin = args.getreal("-dsmin", "--dsmin", 1e-6, "minimum arclength increment");
    dsmax = args.getreal("-dsmax", "--dsmax", 0.10, "maximum arclength increment");
    guessErrMin = args.getreal("-errmin", "--errmin", 1e-5, "minimum error for extrapolated guesses");
    guessErrMax = args.getreal("-errmax", "--errmax", 2e-5, "maximum error for extrapolated guesses");
    digits = args.getint("-dg", "--digits", 8, "number of precision digits for MuD.asc");
    orbOut = args.getflag("-orbOut", "--orbitOutput",
                          "calculate and write minimum and maximum of statstics over a period of an orbit");
    restartMode = args.getflag("-r", "--restart", "restart continuation from three last searches");
    maxSteps = args.getint("-ns", "--nSearch", 100, "maximum number of search steps");
    muRef = args.getreal("-muref", "--muReference", 0, "reference value of mu in computing arclength");
    haveTargetMu = args.getflag("-targ", "--target", "abort when continuation reaches the target value of mu");
    targetMu = args.getreal("-targMu", "--targetMu", 0, "target value of mu");
}

void ContinuationFlags::save(const string& outdir) const {
    if (mpirank() == 0) {
        string filename = appendSuffix(outdir, "continueflags.txt");
        ofstream os(filename.c_str());
        if (!os.good())
            cferror("ContinuationFlags::save(outdir) :  can't open file " + filename);
        os.precision(16);
        os.setf(ios::left);

        os << setw(REAL_IOWIDTH) << arclength << "  %arclength\n"
           << setw(REAL_IOWIDTH) << initialParamStep << "  %dmu\n"
           << setw(REAL_IOWIDTH) << s0 << "  %s0\n"
           << setw(REAL_IOWIDTH) << ds0 << "  %ds0\n"
           << setw(REAL_IOWIDTH) << dsmin << "  %dsmin\n"
           << setw(REAL_IOWIDTH) << dsmax << "  %dsmax\n"
           << setw(REAL_IOWIDTH) << guessErrMin << "  %errmin\n"
           << setw(REAL_IOWIDTH) << guessErrMax << "  %errmax\n"
           << setw(REAL_IOWIDTH) << digits << "  %digits\n"
           << setw(REAL_IOWIDTH) << orbOut << "  %orbitOutput\n"
           << setw(REAL_IOWIDTH) << maxSteps << "  %nSearch\n"
           << setw(REAL_IOWIDTH) << muRef << "  %muReference\n"
           << setw(REAL_IOWIDTH) << haveTargetMu << "  %target\n"
           << setw(REAL_IOWIDTH) << targetMu << "  %targetMu\n";
        os.unsetf(ios::left);
    }
}

void ContinuationFlags::load(int taskid, const string indir) {
    ifstream is;
    if (taskid == 0) {
        is.open(indir + "continueflags.txt");
        if (!is.good()) {
            cout << "    Continuationflags::load(taskid, indir): can't open file " + indir + "continueflags.txt"
                 << endl;
            return;
        }
        if (!checkFlagContent(is, getFlagList())) {
            cerr << " ContinuationFlags::load(taskid, indir): the order of variables in the file is not what we expect "
                    "!!"
                 << endl;
            exit(1);
        }
    }
    arclength = getIntfromLine(taskid, is);
    initialParamStep = getRealfromLine(taskid, is);
    s0 = getRealfromLine(taskid, is);
    ds0 = getRealfromLine(taskid, is);
    dsmin = getRealfromLine(taskid, is);
    dsmax = getRealfromLine(taskid, is);
    guessErrMin = getRealfromLine(taskid, is);
    guessErrMax = getRealfromLine(taskid, is);
    digits = getIntfromLine(taskid, is);
    orbOut = getIntfromLine(taskid, is);
    maxSteps = getIntfromLine(taskid, is);
    muRef = getRealfromLine(taskid, is);
    haveTargetMu = getIntfromLine(taskid, is);
    targetMu = getRealfromLine(taskid, is);
}

const vector<string> ContinuationFlags::getFlagList() {
    const vector<string> flagList = {"%arclength", "%dmu",         "%s0",     "%ds0",     "%dsmin",
                                     "%dsmax",     "%errmin",      "%errmax", "%digits",  "%orbitOutput",
                                     "%nSearch",   "%muReference", "%target", "%targetMu"};
    return flagList;
}

VectorXd quadraticInterpolate(cfarray<VectorXd>& xn, const cfarray<Real>& mun, Real mu, Real eps) {
    if (xn.length() != 3 || mun.length() != 3) {
        stringstream serr;
        serr << "error in quadraticInterpolate(cfarray<VectorXd>& un, cfarray<Real>& mun, Real mu, Real eps)\n";
        serr << "xn.length() != 3 || mun.length() !=3\n";
        serr << "xn.length()  == " << xn.length() << '\n';
        serr << "mun.length() == " << mun.length() << '\n';
        serr << "exiting" << endl;
        cferror(serr.str());
    }
    VectorXd x(xn[0]);
    cfarray<Real> fn(3);
    for (int i = 0; i < xn[0].rows(); ++i) {
        for (int n = 0; n < 3; ++n)
            fn[n] = xn[n](i);
        x(i) = isconst(fn, eps) ? fn[0] : quadraticInterpolate(fn, mun, mu);
    }
    return x;
}

bool readContinuationInfo(string restartdir[3], ContinuationFlags& cflags) {
    ifstream mud("MuD.asc");

    if (!mud.good()) {
        cout << "Restart mode: MuD doesn't exist; trying to restart from specified directories in command line !"
             << endl;
        cflags.restartMode = false;
        return false;
    }

    string temp_word, temp_line;

    int lineCount = 1, columnCount = 0;
    int guesserr_n = 0, directory_n = 0;  // guesserr and directory columns number

    getline(mud, temp_line);  // Read the first line
    stringstream line(temp_line);
    while (line >> temp_word) {  // Find guesserr_n and directory_n
        columnCount++;
        if (temp_word == "guesserr")
            guesserr_n = columnCount;
        if (temp_word == "directory")
            directory_n = columnCount;
    }

    while (getline(mud, temp_line))  // Find number of lines in MuD
        lineCount++;

    if (lineCount < 4 || guesserr_n == 0 || directory_n == 0) {
        cout << "Restart mode: MuD exists but does not give enough/correct information; trying to restart from "
                "specified directories in command line !"
             << endl;
        cflags.restartMode = false;
        return false;
    }

    mud.clear();
    mud.seekg(0, ios::beg);

    for (int i = 0; i < lineCount - 2; i++)  // Go to the third line from below
        getline(mud, temp_line);

    line.clear();
    line.str(temp_line);
    for (int i = 0; i < directory_n; i++)  // Go to the directory column of the third line from below
        line >> temp_word;
    restartdir[2] = temp_word;

    getline(mud, temp_line);  // Read the second line from below
    line.clear();
    line.str(temp_line);
    for (int i = 0; i < directory_n; i++)  // Go to the directory column of the second line from below
        line >> temp_word;
    restartdir[1] = temp_word;

    getline(mud, temp_line);  // Read the last line in MuD
    line.clear();
    line.str(temp_line);
    for (int i = 0; i < guesserr_n - 1; i++)  // Go to the guesserr column of the last line
        line >> temp_word;
    line >> cflags.guesserr_last;  // Read the last guess error
    line.clear();
    line.str(temp_line);
    for (int i = 0; i < directory_n; i++)  // Go to the directory column of the last line
        line >> temp_word;
    restartdir[0] = temp_word;

    int lastSearchNumber;
    if (temp_word == "./initial-0/")
        lastSearchNumber = -1;
    else {
        stringstream lastSearchDirectory(temp_word);
        lastSearchDirectory.seekg(7);  // pass search- to get to the number
        lastSearchDirectory >> lastSearchNumber;
    }

    mud.close();

    // Find number of the last search folder and the number of failed searches after the last succesful search
    int currentSearchNumber = lastSearchNumber + 1;
    ifstream failedSearch("failures/search-" + i2s(currentSearchNumber) + "/mu.asc");

    while (failedSearch.good()) {
        failedSearch.close();
        currentSearchNumber++;
        failedSearch.open("failures/search-" + i2s(currentSearchNumber) + "/mu.asc");
    }

    cflags.initStep = currentSearchNumber;
    cflags.nlastFailedSteps = currentSearchNumber - lastSearchNumber - 1;

    return true;
}

Real continuation(DSI& dsiG, Newton& newton, cfarray<VectorXd> y, cfarray<Real> mu, ContinuationFlags& cflags) {
    assert(y.length() == 3);
    assert(mu.length() == 3);

    MultishootingDSI* msDSI = newton.getMultishootingDSI();
    cfarray<VectorXd> x(3);
    if (!(msDSI->isDSIset())) {
        msDSI->setDSI(dsiG, y[0].rows());
        for (int i = 0; i < 3; i++) {
            dsiG.updateMu(mu[i]);
            x[i] = msDSI->makeMSVector(y[i]);
        }
    } else {
        if (msDSI->isVecMS(y[0].rows(), false) && msDSI->isVecMS(y[1].rows(), false) &&
            msDSI->isVecMS(y[2].rows(), false)) {
            x[0] = y[0];
            x[1] = y[1];
            x[2] = y[2];
        } else
            cerr << "Multishooting class is already set, but the size of the multishooting vector is not equal to the "
                    "length of the unknown vector !!!"
                 << endl;
    }

    int taskid = mpirank();  // to put mu in the vector for arclength continuation
    int fcount = 0;
    int N = x[0].size();
    if (taskid == 0)
        N++;  // holds mu
    ArclengthConstraint AC;
    newton.setArclengthConstraint(&AC);

    Real munorm = (abs(cflags.muRef) < 1e-12) ? abs(mu[1]) : cflags.muRef;
    if (abs(munorm) < 1e-12)
        munorm = 1;

    int W = 24;
    cout << setprecision(8);
    cfarray<Real> obs(3);  // observable
    cfarray<Real> s(3);    // (mu,obs) arclengths
    cfarray<Real> res(3);  // residuals
    ofstream dos;
    ofstream eos;
    ofstream eos_min;
    ofstream eos_max;

    // Creating MuE and MuD
    if (cflags.restartMode) {
        openfile(dos, "MuD.asc", ios::app);
        if (cflags.orbOut) {
            openfile(eos_min, "MuE_min.asc", ios::app);
            openfile(eos_max, "MuE_max.asc", ios::app);
        } else {
            openfile(eos, "MuE.asc", ios::app);
        }
    } else {
        openfile(dos, "MuD.asc");
        stringstream mudirheader;
        mudirheader << setw(W) << (string("%") + dsiG.printMu()) << setw(W) << "obs";
        if (dsiG.Tsearch())
            mudirheader << setw(W) << "T";
        if (dsiG.XrelSearch())
            mudirheader << setw(W) << "ax";
        if (dsiG.ZrelSearch())
            mudirheader << setw(W) << "az";
        mudirheader << setw(W) << "s" << setw(W) << "guesserr" << setw(W) << "error"
                    << "     directory";
        dos << mudirheader.str() << endl;
        if (cflags.orbOut) {
            openfile(eos_min, "MuE_min.asc");
            openfile(eos_max, "MuE_max.asc");
            eos_min << dsiG.statsHeader() << endl;
            eos_max << dsiG.statsHeader() << endl;
        } else {
            openfile(eos, "MuE.asc");
            eos << dsiG.statsHeader() << endl;
        }
    }
    dos << setprecision(cflags.digits);

    // Initial steps
    if (cflags.restartMode) {
        for (int i = 2; i >= 0; --i) {
            dsiG.updateMu(mu[i]);
            obs[i] = msDSI->observable(x[i]);
        }
    } else {
        // find solutions for initial data
        cout << "Initial mu           = " << mu << endl;
        for (int i = 2; i >= 0; --i) {
            cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;
            string searchdir = "initial-" + i2s(i) + "/";
            mkdir(searchdir);
            ofstream logstream((searchdir + "findsoln.log").c_str());
            newton.setLogstream(&logstream);
            newton.setOutdir(searchdir);
            dsiG.setOs(&logstream);
            dsiG.updateMu(mu[i]);
            dsiG.saveParameters(searchdir);
            save(mu[i], (searchdir + "mu.asc").c_str());
            cout << "computing solution for " << dsiG.printMu() << " == " << mu[i] << " in dir " << searchdir << "..."
                 << endl;

            x[i] = newton.solve(dsiG, x[i], res[i]);
            if (newton.getConvergence() == false) {
                string message = "initial search " + i2s(i) + "did not converge, aborting.";
                cferror(message);
            }

            obs[i] = msDSI->observable(x[i]);
            msDSI->phaseShift(x[i], false);
            msDSI->save(x[i], "best", searchdir);

            cout << dsiG.printMu() << " == " << mu[i] << "   obs == " << obs[i] << "   searchresidual == " << res[i]
                 << endl;

            stringstream mudirdata;
            mudirdata << setprecision(cflags.digits) << setw(W) << mu[i] << setw(W) << obs[i];
            if (dsiG.Tsearch())
                mudirdata << setw(W) << msDSI->extractT(x[i]);
            if (dsiG.XrelSearch())
                mudirdata << setw(W) << msDSI->extractXshift(x[i]);
            if (dsiG.ZrelSearch())
                mudirdata << setw(W) << msDSI->extractZshift(x[i]);
            mudirdata << setw(W) << 0.0 << setw(W) << 0.0 << setw(W) << res[i] << "  ./initial-" << i << "/";
            dos << mudirdata.str() << endl;
            if (cflags.orbOut) {
                pair<string, string> minmax;
                minmax = msDSI->stats_minmax(x[i]);
                eos_min << minmax.first << endl;
                eos_max << minmax.second << endl;
            } else {
                eos << msDSI->stats(x[i]) << endl;
            }
        }
    }

    // Form independant variable
    Real ds;
    Real obsnorm = abs(obs[1]);
    cout << "     s0 == " << cflags.s0 << endl;
    cout << " munorm == " << munorm << endl;
    for (int i = 2; i >= 0; --i)
        cout << "  mu[" << i << "] == " << mu[i] << endl;
    cout << "obsnorm == " << obsnorm << endl;
    for (int i = 2; i >= 0; --i)
        cout << " obs[" << i << "] == " << obs[i] << endl;

    if (cflags.arclength) {
        s[2] = cflags.s0 - sqrt(L2Dist2(x[2], x[1]) + pow((mu[2] - mu[1]) / munorm, 2));
        s[1] = cflags.s0;
        s[0] = cflags.s0 + sqrt(L2Dist2(x[0], x[1]) + pow((mu[0] - mu[1]) / munorm, 2));
        ds = (cflags.ds0 != 0) ? cflags.ds0 : sqrt(L2Dist2(x[0], x[1]) + pow((mu[0] - mu[1]) / munorm, 2));

    } else {
        s[2] = cflags.s0 - pythag((obs[2] - obs[1]) / obsnorm, (mu[2] - mu[1]) / munorm);
        s[1] = cflags.s0;
        s[0] = cflags.s0 + pythag((obs[0] - obs[1]) / obsnorm, (mu[0] - mu[1]) / munorm);
        ds = (cflags.ds0 != 0) ? cflags.ds0 : pythag((obs[0] - obs[1]) / obsnorm, (mu[0] - mu[1]) / munorm);
    }

    for (int i = 2; i >= 0; --i)
        cout << "   s[" << i << "] == " << s[i] << endl;
    cout << "      ds == " << ds << endl;

    // Extend the vector by the paramter mu
    if (cflags.arclength && taskid == 0) {
        for (int i = 2; i >= 0; --i) {
            x[i].conservativeResize(N);
            x[i](N - 1) = mu[i];
        }
    }

    const Real guesserrtarget = sqrt(cflags.guessErrMin * cflags.guessErrMax);  // aim between error bounds
    const int Ndsadjust = 6;
    bool prev_search_failed = (1 <= cflags.nlastFailedSteps);
    Real guesserr_prev_search = (cflags.restartMode && cflags.initStep != 0) ? cflags.guesserr_last : guesserrtarget;
    Real guesserr = 0;
    bool reachedTarget = false;
    ds = cflags.restartMode ? ds * pow(0.5, cflags.nlastFailedSteps) : ds;

    // Continuation
    for (int n = cflags.initStep; n < cflags.maxSteps; ++n) {
        if (cflags.arclength)
            AC = ArclengthConstraint(x[0], 0, munorm);

        if (cflags.haveTargetMu && (cflags.targetMu - mu[0]) * (cflags.targetMu - mu[1]) < 0) {
            reachedTarget = true;
            break;
        }

        cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;
        cout << "Continuing previous solutions by quadratic extrapolation in s = (mu,obs)" << endl;
        cout << setw(8) << "i" << setw(W) << "s" << setw(W) << dsiG.printMu() << endl;
        for (int i = 2; i >= 0; --i)
            cout << setw(8) << i << setw(W) << s[i] << setw(W) << mu[i] << endl;

        bool ds_reached_bounds = false;
        bool no_more_adjustments = false;

        Real munew;
        Real snew;
        Real resnew;
        VectorXd xnew;
        string searchdir = "search-" + i2s(n) + "/";
        mkdir(searchdir);
        newton.setOutdir(searchdir);
        ofstream searchlog;
        openfile(searchlog, searchdir + "findsoln.log");
        newton.setLogstream(&searchlog);
        dsiG.setOs(&searchlog);

        // Find a guess with error within error bounds, or under it if previous Newton search has failed
        for (int m = 0; m < Ndsadjust; ++m) {
            cout << "------------------------------------------------------" << endl;
            cout << "Look for decent initial guess, step == " << m << endl;
            snew = s[0] + ds;
            munew = quadraticInterpolate(mu, s, snew);
            dsiG.updateMu(munew);
            xnew = quadraticInterpolate(x, s, snew);

            cout << "calculating error of extrapolated guess for mu = " << munew << " " << flush;
            VectorXd xguess = newton.evalWithAC(xnew, fcount);

            guesserr = L2Norm(xguess, cflags.arclength && (mpirank() == 0));

            cout << endl;
            cout << "dsmin == " << cflags.dsmin << endl;
            cout << "ds    == " << ds << endl;
            cout << "dsmax == " << cflags.dsmax << endl;
            cout << "guesserrmin == " << cflags.guessErrMin << endl;
            cout << "guesserr    == " << guesserr << endl;
            cout << "guesserrmax == " << cflags.guessErrMax << endl;
            cout << "guesserr previous  == " << guesserr_prev_search << endl;
            cout << "prev_search_failed == " << prev_search_failed << endl;
            cout << "ds_reached_bounds  == " << ds_reached_bounds << endl;
            cout << "no_more_adjustment == " << no_more_adjustments << endl;

            // Decide whether to continue adjusting the guess, search on the guess, or give up.
            if (prev_search_failed && ds < cflags.dsmin) {
                cout << "Stopping continuation because continuing would require ds <= dsmin" << endl;
                exit(1);
            } else if (prev_search_failed && ds >= cflags.dsmin) {
                cout << "------------------------------------------------------" << endl;
                cout << "Stopping guess adjustments because previous search failed," << endl;
                cout << "so we're allowing guesserr <= guesserrmin, as long as dsmin <= ds" << endl;
                break;
            }
            // Previous search succeeded.
            if ((guesserr >= cflags.guessErrMin && guesserr <= cflags.guessErrMax)) {
                cout << "Stopping guess adjustments because guess error is in bounds: errmin <= guesserr <= errmax"
                     << endl;
                break;
            } else if (ds_reached_bounds) {
                cout << "Stopping guess adjustments because ds has reached its bounds:" << endl;
                break;
            } else if (no_more_adjustments) {
                cout << "Stopping guess adjustments because a recent search failed and we're being cautious." << endl;
                break;
            } else if (guesserr_prev_search <= cflags.guessErrMin) {
                cout << "Previous search succeeded with guesserr <= guesserrmin. " << endl;
                cout << "Let's try increasing ds, to lesser of 2*ds and value suggested by guesserr = O(ds^3)" << endl;
                cout << "but no more than max allowed value dsmax. And no more guess adjustments after this." << endl;
                ds *= lesser(2.0, pow(cflags.guessErrMin / guesserr, 0.33));
                if (ds > cflags.dsmax) {
                    ds = cflags.dsmax;
                    ds_reached_bounds = true;
                } else
                    no_more_adjustments = true;
                continue;
            } else {
                cout << "Guess not yet within bounds. Try new guess based on model guesserr = O(ds^3)" << endl;
                // Adjust ds on model err = k O(ds^3), ds = 1/k err^1/3
                cout << "guesserrtarget == " << guesserrtarget << endl;
                cout << "guesserr       == " << guesserr << endl;
                cout << "(err/targ)^1/3 == " << pow(guesserrtarget / guesserr, 0.33) << endl;
                ds *= pow(guesserrtarget / guesserr, 0.33);
                if (ds < cflags.dsmin) {
                    ds = cflags.dsmin;
                    ds_reached_bounds = true;
                } else if (ds > cflags.dsmax) {
                    ds = cflags.dsmax;
                    ds_reached_bounds = true;
                }

                continue;
            }
        }

        if (cflags.arclength)
            AC.setDs(AC.arclength(xnew));

        // Done looking for a good guess. Now search on the guess.
        dsiG.saveResults(searchdir);
        save(munew, (searchdir + "mu.asc").c_str());
        cout << "Computing new solution for " << dsiG.printMu() << " " << munew << " in directory " << searchdir
             << "  ..." << endl;
        xnew = newton.solve(dsiG, xnew, resnew);
        Real Dnew = msDSI->observable(xnew);
        Real obsnew = Dnew;

        msDSI->phaseShift(xnew, cflags.arclength);

        if (resnew < newton.epsSearch()) {
            cout << "Found new solution." << endl;

            if (cflags.arclength) {
                munew = AC.muFromVector(xnew);
                save(munew, (searchdir + "mu.asc").c_str());
                ds = AC.arclength(xnew);
            } else
                ds = pythag((obsnew - obs[0]) / obsnorm, (munew - mu[0]) / munorm);
            snew = s[0] + ds;

            push(xnew, x);
            push(munew, mu);
            push(obsnew, obs);
            push(resnew, res);
            push(snew, s);

            cout << setw(8) << "solution" << setw(W) << snew << setw(W) << munew << "   residual == " << resnew << endl;
            stringstream mudirdata;
            mudirdata << setprecision(cflags.digits) << setw(W) << munew << setw(W) << obsnew;
            if (dsiG.Tsearch())
                mudirdata << setw(W) << msDSI->extractT(xnew);
            if (dsiG.XrelSearch())
                mudirdata << setw(W) << msDSI->extractXshift(xnew);
            if (dsiG.ZrelSearch())
                mudirdata << setw(W) << msDSI->extractZshift(xnew);
            mudirdata << setw(W) << snew << setw(W) << guesserr << setw(W) << resnew << "   " << searchdir;
            dos << mudirdata.str() << endl;
            if (cflags.orbOut) {
                pair<string, string> minmax;
                minmax = msDSI->stats_minmax(xnew);
                eos_min << minmax.first << endl;
                eos_max << minmax.second << endl;
            } else {
                eos << msDSI->stats(xnew) << endl;
            }
            prev_search_failed = false;
            guesserr_prev_search = guesserr;
        } else {
            cout << "Failed to find new solution. Halving ds and trying again" << endl;
            cout << setw(8) << "failure" << setw(W) << snew << setw(W) << munew << "   residual == " << resnew << endl;
            ds *= 0.5;
            prev_search_failed = true;
            guesserr_prev_search = guesserr;

            string failuredir("failures");
            mkdir(failuredir);
            rename(searchdir, failuredir + "/search-" + i2s(n));
        }
    }

    if (reachedTarget) {
        VectorXd x_guess = x[0] + (x[1] - x[0]) * (cflags.targetMu - mu[0]) / (mu[1] - mu[0]);
        dsiG.updateMu(cflags.targetMu);
        if (cflags.arclength) {
            x_guess = AC.extractVector(x_guess);
            AC.notUse();
        }

        string searchdir = "Target/";
        cout << endl << "Now searching for new solution in directory " << searchdir << endl;
        mkdir(searchdir);
        save(cflags.targetMu, (searchdir + "mu.asc").c_str());
        ofstream logstream((searchdir + "findsoln.log").c_str());
        newton.setLogstream(&logstream);
        newton.setOutdir(searchdir);
        dsiG.setOs(&logstream);

        Real residual;

        VectorXd xNew = newton.solve(dsiG, x_guess, residual);

        return cflags.targetMu;

    } else {
        return mu[0];
    }
}

}  // namespace chflow
