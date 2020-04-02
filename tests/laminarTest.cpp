/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "channelflow/dns.h"
#include "channelflow/flowfield.h"

// Generate laminar solution for a variety of flow conditions and
// verify that it satisfies Navier-Stokes and flow conditions.

using namespace std;

using namespace chflow;

Real random(Real a = -1, Real b = 1) { return a + (b - a) * drand48(); }
int randint(int a, int b) { return a + (rand() % (b - a + 1)); }               // rand int in [a,b]
Real randscale(int nmin, int nmax) { return pow(10.0, randint(nmin, nmax)); }  // 10^k for rand k in [nmin, nmax]
bool coinflip(Real bias = 0.5) { return (random(0.0, 1.0) < bias) ? true : false; }

int main(int argc, char* argv[]) {
    Real maxerr = 1e-10;
    bool passtest = true;
    int taskid = 0;
    srand48(18947);

#ifdef HAVE_MPI
    CfMPI* cfmpi = NULL;
    cfMPI_Init(&argc, &argv);
    {
        cfmpi = &CfMPI::getInstance();
        taskid = cfmpi->taskid();
#endif

        // bool verbose = false;
        // bool save = false;

        const int Ny = 65;
        const int Ntests = 1000;    // first 40 tests are targeted to
        const Real Hmin = 1.0;      // minimum wall separation
        const Real Hmax = 10.0;     // maximum wall separation
        const Real Rmin = 100.0;    // minimum Reynolds number
        const Real Rmax = 10000.0;  // maximum Reynolds number
        // const int  minscale = -8;  // scale dPdx, Ubulk, Vsuck, randomly bewteen 10^minscale and 10^maxscale
        // const int  maxscale = 2;

        // const Real Uwallmax = 10.0;   // maximum wall separation

        // Options
        // bulk velocty or pressure constraint (coin flip)
        //   Ubulk zero or nonzero  O(1)
        //   dPdx  zero or nonzero -O(nu)
        // Vsuck   zero or nonzero  O(nu)
        // Uwall   zero or nonzero  O(1)
        // theta   zero or nonzero  O(1)

        // random nonzero H         O(1) => a,b
        // random nonzero Reynolds  O(10^n) for n = 1 to 5, nu = 1/Reynolds

        cout << setprecision(3) << left;
        cerr << setprecision(3);

        int W = 12;
        if (taskid == 0) {
            // cerr << "laminarTest: " << flush;
            cout << "\n====================================================" << endl;
            cout << "laminarTest\n\n";
            cout << setw(W) << "constr" << setw(W) << "a" << setw(W) << "b" << setw(W) << "ua" << setw(W) << "ub"
                 << setw(W) << "Re" << setw(W) << "Vsuck" << setw(W) << "dPdx" << setw(W) << "Ubulk" << setw(W)
                 << "Vsuck H/nu" << setw(W) << "erreqn" << setw(W) << "errwall" << setw(W) << "errbulk" << setw(W)
                 << "errtot" << setw(W) << "status\n";
            cout << endl << endl;
        }

        /****************************************
        for (int n = 0; n < 20; ++n)
            cout << randint(-3, 2) << endl;

        for (int n = 0; n < 20; ++n)
            cout << randscale(-3, 2) << endl;

        return 0;
        *************************************/

        for (int ntest = 0; ntest < Ntests; ++ntest) {
            //
            bool canonical_a_b = coinflip(0.5);
            Real a = canonical_a_b ? -1 : Hmax * random(-1, 1);
            Real b = canonical_a_b ? 1 : a + random(Hmin, Hmax);
            Real H = b - a;
            Real H2 = H * H;

            Real Reynolds = coinflip(0.5) ? 1000 : Rmin * pow(10.0, random(0, log10(Rmax)));
            Real nu = 1 / Reynolds;

            MeanConstraint constraint = (ntest < Ntests / 2) ? PressureGradient : BulkVelocity;
            Real dPdx = coinflip(0.5) ? 0.0 : randscale(-2, 2) * random(-1, 1) * nu / H2;
            Real Ubulk = coinflip(0.5) ? 0.0 : randscale(-2, 2) * random(-1, 1);
            Real Vsuck = coinflip(0.5) ? 0.0 : randscale(-10, 1) * random(-1, 1) * nu / H;

            // cout << "dPdx, Ubulk, Vsuck == " << setw(W) << dPdx << setw(W) << Ubulk << setw(W) << Vsuck << endl;

            Real x = random(0, 1);
            Real ua = 0;
            Real ub = 0;
            if (x < 0.33) {
                ua = -1;
                ub = 1;
            } else if (x < 0.67) {
                ua = 4 * random(-1, 1);
                ub = 4 * random(-1, 1);
            }

            // Target first 40 tests on limit Vsuck -> 0. This
            if (ntest < 40) {
                if (ntest % 2 == 0) {
                    dPdx = 1;
                    Vsuck = 0.5 * pow(10.0, -ntest / 2 - 2);
                    Ubulk = 0;
                    Reynolds = 1000;
                    nu = 1 / Reynolds;
                    constraint = PressureGradient;
                } else {
                    dPdx = 0;
                    Vsuck = 0.5 * pow(10.0, -(ntest + 1) / 2 - 2);
                    Ubulk = 1.0;
                    Reynolds = 1000;
                    nu = 1 / Reynolds;
                    constraint = BulkVelocity;
                }
            }
            ChebyCoeff U = laminarProfile(nu, constraint, dPdx, Ubulk, Vsuck, a, b, ua, ub, Ny);

            // For pressure gradient constraint, verify that -Vsuck U'(y) = -dPdx + nu U''(y), i.e.
            // nu U'' + Vsuck U'(y) - dPdx = 0, and

            // For bulk velocity constraint, verify that
            // nu U'' + Vsuck U'(y) = const
            // and that mean(U) = Ubulk

            // Also verify that U(a) = ua and U(b) = ub

            ChebyCoeff Uy = diff(U);
            ChebyCoeff Uyy = diff(Uy);

            Real errtot = 0.0;   // total error
            Real erreqn = 0.0;   // error in momentum balance equation (including pressure gradient)
            Real errwall = 0.0;  // error in wall speeds (boundary conditions)
            Real errbulk = 0.0;  // error in bulk velocity constraint

            ChebyCoeff Ueqn = nu * Uyy + Vsuck * Uy;
            Real normeqn = 1.0 + nu * L2Norm(Uyy) + abs(Vsuck) * L2Norm(Uy);
            // cout << "normeqn == " << normeqn << endl;

            if (constraint == PressureGradient) {
                Ueqn[0] -= dPdx;  // subtract off dPdx so Ueqn ==  nu * Uyy + Vsuck * Uy - dPdx == 0
                normeqn += abs(dPdx);
                // Ubulk = U.mean();  // set Ubulk to actual value
            } else {
                // dPdx = Ueqn[0];  // set dPdx to actual value
                Ueqn[0] = 0.0;  // zero out constant part
                errbulk += abs(U.mean() - Ubulk);
            }

            erreqn = L2Norm(Ueqn) / normeqn;  // don't normalize --some U(y) are zero
            errwall = abs(U.eval_a() - ua) + abs(U.eval_b() - ub);

            errtot = erreqn + errbulk + errwall;
            string constraint_ = (constraint == PressureGradient) ? "gradp" : "bulkv";
            string status = (errtot <= maxerr) ? "pass" : "FAIL";

            if (errtot > maxerr)
                passtest = false;

            if (taskid == 0)
                cout << setw(W) << constraint_ << setw(W) << a << setw(W) << b << setw(W) << ua << setw(W) << ub
                     << setw(W) << Reynolds << setw(W) << Vsuck << setw(W) << dPdx << setw(W) << Ubulk << setw(W)
                     << Vsuck * (b - a) / nu << setw(W) << erreqn << setw(W) << errwall << setw(W) << errbulk << setw(W)
                     << errtot << setw(W) << status << endl;
        }
        if (passtest == true) {
            cerr << "\t   pass   " << endl;
            cout << "\t   pass   " << endl;
        } else {
            cerr << "\t** FAIL **" << endl;
            cout << "\t** FAIL **" << endl;
        }
#ifdef HAVE_MPI
    }
    cfMPI_Finalize();
#endif
    return (passtest == true) ? 0 : 1;
}
