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

// Start a pressure-driven Poisseuille flow with either
// U(y) == 0      and u(x,y,z) == 1-y^2 or
// U(y) == 1-y^20 and u(x,y,z) == 0
// Integrate and verify that nothing changes
//
// Or start U and u from zero and check convergence onto utot = 1-y^2

using namespace std;

using namespace chflow;

class PressureBodyForce : public BodyForce {
   public:
    PressureBodyForce(Real fx, Real fy, Real fz);

    void eval(Real x, Real y, Real z, Real t, Real& fx, Real& fy, Real& fz);
    void eval(Real t, FlowField& f);
    bool nonzero();

   private:
    Real fx_;
    Real fy_;
    Real fz_;
};

PressureBodyForce::PressureBodyForce(Real fx, Real fy, Real fz) : fx_(fx), fy_(fy), fz_(fz) {}

void PressureBodyForce::eval(Real x, Real y, Real z, Real t, Real& fx, Real& fy, Real& fz) {
    fx = fx_;
    fy = fy_;
    fz = fz_;
}

bool PressureBodyForce::nonzero() { return false; }

int main(int argc, char* argv[]) {
    Real err = 0.0;
    Real maxerr = 1e-13;

#ifdef HAVE_MPI
    cfMPI_Init(&argc, &argv);
    {
#endif

        // bool verbose = false;
        bool save = false;

        const int Nx = 8;
        const int Ny = 33;
        const int Nz = 8;

        const int Nd = 3;
        const Real Lx = 2 * pi;
        const Real Lz = 2 * pi;
        const Real a = -1.0;
        const Real b = 1.0;
        const Real Ly = b - a;
        const Real Reynolds = 400.0;
        const Real nu = 1.0 / Reynolds;
        const Real dt = 0.1;
        const int n = 10;
        const Real T0 = 0.0;
        const Real T1 = 10.0;

        bool baseflow = false;

        DNSFlags flags;
        flags.timestepping = SBDF1;
        flags.constraint = BulkVelocity;
        flags.dealiasing = DealiasXZ;
        flags.nonlinearity = Rotational;
        flags.verbosity = Silent;
        flags.baseflow = ZeroBase;
        flags.dt = dt;
        flags.nu = 1.0 / Reynolds;
        flags.dPdx = -2.0 * nu;  // mean streamwise pressure gradient.
        flags.Ubulk = 2.0 / 3.0;
        flags.ulowerwall = 0.0;
        flags.uupperwall = 0.0;

        PressureBodyForce pressure(-2.0 * nu, 0, 0);

        // maxerr is determined by a previous DNS computation
        for (int i = 1; i < argc; ++i) {
            string argument(argv[i]);

            if (argument == "--bulkv")
                flags.constraint = BulkVelocity;

            else if (argument == "--gradp")
                flags.constraint = PressureGradient;

            else if (argument == "--bodyforce") {
                flags.dPdx = 0;
                flags.dPdz = 0;
                flags.bodyforce = &pressure;
            }

            else if (argument == "--cnfe1") {
                flags.timestepping = CNFE1;
            } else if (argument == "--cnab2") {
                flags.timestepping = CNAB2;
            } else if (argument == "--cnrk2") {
                flags.timestepping = CNRK2;
            } else if (argument == "--smrk2") {
                flags.timestepping = SMRK2;
            } else if (argument == "--sbdf2") {
                flags.timestepping = SBDF2;
            } else if (argument == "--sbdf3") {
                flags.timestepping = SBDF3;
            } else if (argument == "--sbdf4") {
                flags.timestepping = SBDF4;
            } else if (argument == "--base") {
                flags.baseflow = LaminarBase;
                baseflow = true;
            } else if (argument == "--fluc") {
                flags.baseflow = ZeroBase;
                baseflow = false;
            }
        }
        cout << "dnsflags == " << flags << endl;

        char s = ' ';
        cerr << "dnsParabolaTest ";
        for (int i = 1; i < argc; ++i) {
            cerr << argv[i];
            int pad = 10 - strlen(argv[i]);
            for (int j = 0; j < pad; ++j)  // crude formatting
                cerr << s;
        }
        cerr << flush;

        cout << "\n====================================================" << endl;
        cout << "dnsParabolaTest\n\n";
        for (int i = 1; i < argc; ++i)
            cout << argv[i] << s;
        cout << endl;
        cout << setprecision(14);
        cout << "Nx Ny Nz Nd == " << Nx << s << Ny << s << Nz << s << Nd << endl;
        cout << "Lx Ly Lz == " << Lx << s << Ly << s << Lz << s << endl;

        Vector y = chebypoints(Ny, a, b);
        ChebyTransform trans(Ny);

        // Build DNS
        ChebyCoeff zero(Ny, a, b, Spectral);
        ChebyCoeff parabola(Ny, a, b, Spectral);
        parabola[0] = 0.5;
        parabola[2] = -0.5;

        vector<FlowField> fields = {FlowField(Nx, Ny, Nz, Nd, Lx, Lz, a, b), FlowField(Nx, Ny, Nz, 1, Lx, Lz, a, b)};

        ChebyCoeff Ubase = (baseflow) ? parabola : zero;
        ChebyCoeff uprof = (baseflow) ? zero : parabola;
        // flags.baseflow   = (baseflow) ? Parabolic : Zero;

        ChebyCoeff Wbase = zero;
        DNS dns(fields, {Ubase, Wbase}, flags);

        fields[0] += uprof;
        FlowField ut = fields[0];

        cout << "Initial data, prior to time stepping" << endl;
        cout << "L2Norm(un)    == " << L2Norm(fields[0]) << endl;
        cout << "divNorm(un)   == " << divNorm(fields[0]) << endl;
        cout << "L2Norm(qn)    == " << L2Norm(fields[1]) << endl;
        cout << "........................................................" << endl;
        cout << endl;

        ofstream ns;
        ofstream uns;
        ofstream uts;
        ofstream ts;

        if (save) {
            ns.open("unorms.asc");
            uts.open("ut.asc");
            uns.open("un.asc");
            ts.open("t.asc");

            ns << setprecision(14);
            uts << setprecision(14);
            uts << "% true solution ut(y,t), (y,t) == (columns, rows)" << endl;
            uns << setprecision(14);
            uns << "% numerical solution un(y,t), (y,t) == (columns, rows)" << endl;
            ts << "% time" << endl;
            Ubase.save("Ubase");
            y.save("y");
        }

        for (Real t = T0; t < T1; t += n * dt) {
            Real l2dist = L2Dist(fields[0], ut);
            Real CFL = dns.CFL(fields[0]);
            Real l2un = L2Norm(fields[0]);

            cout << "t == " << t << endl;
            cout << "CFL == " << CFL << endl;
            cout << "L2Norm(un)    == " << l2un << endl;
            cout << "L2Norm(ut)    == " << L2Norm(ut) << endl;
            cout << "L2Dist(un,ut) == " << l2dist << endl;
            cout << "dPdx  == " << dns.dPdx() << endl;
            cout << "Ubulk == " << dns.Ubulk() << endl;
            err += l2dist;

            ns << L2Dist(fields[0], ut) << ' ' << fabs(flags.dPdx - dns.dPdx()) << ' '
               << fabs(flags.Ubulk - dns.Ubulk()) << ' ' << divNorm(fields[0]) << ' ' << bcNorm(fields[0]) << '\n';

            if (save) {
                ChebyCoeff un00 = Re(fields[0].profile(0, 0, 0));
                ChebyCoeff ut00 = Re(ut.profile(0, 0, 0));

                un00.makePhysical(trans);
                ut00.makePhysical(trans);

                for (int ny = 0; ny < Ny; ++ny)
                    uns << un00(ny) << ' ';
                uns << '\n';
                for (int ny = 0; ny < Ny; ++ny)
                    uts << ut00(ny) << ' ';
                uts << '\n';

                ts << t << '\n';
            }

            /********************8
          if (CFL > 2.0 || l2un > 1) {
            cout << "Problem!" << endl;
            cout << "CFL  == " << CFL << endl;
            cout << "norm == " << l2un << endl;
            cout << "\t** FAIL **" << endl;
            cerr << "\t** FAIL **" << endl;
            return 1;
          }
            **********************/
            dns.advance(fields, n);
        }
        cout << argv[0] << endl;
        cout << flags << endl;
        cout << "FinalError == " << err << " < " << maxerr << endl;
#ifdef HAVE_MPI
    }
    cfMPI_Finalize();
#endif
    if (err < maxerr) {
        cerr << "\t   pass   " << endl;
        cout << "\t   pass   " << endl;
        return 0;
    } else {
        cerr << "\t** FAIL **" << endl;
        cout << "\t** FAIL **" << endl;
        cout << "   err == " << err << endl;
        cout << "maxerr == " << maxerr << endl;
        return 1;
    }
}
