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

// This program simulates a flow with a sinusoidal initial condition
// and compares the DNS solution to the known analytic solution.
// Take base flow U(y) == 0 and constrain the system to have zero
// pressure gradient. For the initial condition
//
// u(x,y,z,0) = eps*sin(2*pi*ky*y[ny]/Ly)
//
// the exact solution of the Navier-Stokes equation is
//
// u(x,y,z,t) = u(x,y,z,0)*exp(-nu*t*square(2*pi*ky/Ly)
//
// In the following output,\n\n";
//
//  un == the numerical solution.\n";
//  ut == the true solution.\n\n";

using namespace std;

using namespace chflow;

int main(int argc, char* argv[]) {
    bool verbose = false;
    bool save = false;

    const int Nx = 8;
    const int Ny = 65;
    const int Nz = 8;

    const int Nd = 3;
    const Real Lx = 2 * pi;
    const Real Lz = 2 * pi;
    const Real a = -1.0;
    const Real b = 1.0;
    const Real Ly = b - a;
    const Real Reynolds = 10.0;
    const Real nu = 1.0 / Reynolds;
    const Real dt = 0.02;
    const Real T0 = 0.0;
    const Real T1 = 1.0;
    const int Ndt = 10;

    // sinusoid perturbation params.
    const Real eps = 1;
    const int ky = 1;

    DNSFlags flags;
    flags.nonlinearity = Rotational;
    flags.verbosity = Silent;
    flags.ulowerwall = 0.0;
    flags.uupperwall = 0.0;
    flags.dt = dt;
    flags.nu = nu;
    Real maxerr = 2e-6;  // default timestepping is SBDF3

    // maxerrsum is determined by a previous DNS computation
    for (int i = 1; i < argc; ++i) {
        string argument(argv[i]);

        if (argument == "--bulkv")
            flags.constraint = BulkVelocity;

        else if (argument == "--gradp")
            flags.constraint = PressureGradient;

        else if (argument == "--cnfe1") {
            flags.timestepping = CNFE1;
            maxerr = 2e-2;
        } else if (argument == "--cnab2") {
            flags.timestepping = CNAB2;
            maxerr = 4e-5;
        } else if (argument == "--cnrk2") {
            flags.timestepping = CNRK2;
            maxerr = 5e-6;
        } else if (argument == "--smrk2") {
            flags.timestepping = SMRK2;
            maxerr = 8e-6;
        } else if (argument == "--sbdf2") {
            flags.timestepping = SBDF2;
            maxerr = 2e-4;
        } else if (argument == "--sbdf3") {
            flags.timestepping = SBDF3;
            maxerr = 2e-6;
        } else if (argument == "--sbdf4") {
            flags.timestepping = SBDF4;
            maxerr = 2e-6;
        } else if (argument == "--parab") {
            flags.baseflow = ParabolicBase;
            flags.dPdx = -2.0 * nu;
            flags.Ubulk = 2.0 / 3.0;
        } else if (argument == "--zero") {
            flags.baseflow = ZeroBase;
            flags.dPdx = 0.0;
            flags.Ubulk = 0.0;
        }
    }

    char s = ' ';
    cerr << "dnsSinusoidTest ";
    for (int i = 1; i < argc; ++i) {
        cerr << argv[i];
        int pad = 10 - strlen(argv[i]);
        for (int j = 0; j < pad; ++j)  // crude formatting
            cerr << s;
    }
    cerr << flush;

    if (verbose) {
        cout << "\n====================================================" << endl;
        cout << "dnsSinusoidTest ";
        for (int i = 1; i < argc; ++i)
            cout << argv[i];
        cout << endl;
        cout << setprecision(14);
    }

    Vector y = chebypoints(Ny, a, b);
    ChebyTransform trans(Ny);

    ChebyCoeff sinusoid(Ny, a, b, Physical);
    for (int ny = 0; ny < Ny; ++ny)
        sinusoid[ny] = eps * sin(2 * pi * ky * y[ny] / Ly);
    sinusoid.makeSpectral(trans);

    vector<FlowField> fields = {FlowField(Nx, Ny, Nz, Nd, Lx, Lz, a, b), FlowField(Nx, Ny, Nz, 1, Lx, Lz, a, b)};
    DNS dns(fields, flags);

    for (int ny = 0; ny < Ny; ++ny)
        fields[0].cmplx(0, ny, 0, 0) = sinusoid[ny];

    FlowField ut = fields[0];

    // FlowField vortn(Nx,Ny,Nz,Nd,Lx,Lz,a,b);
    // FlowField fn1(Nx,Ny,Nz,Nd,Lx,Lz,a,b);
    // FlowField fn(Nx,Ny,Nz,Nd,Lx,Lz,a,b);

    if (verbose) {
        cout << "\nInitial data, prior to time stepping" << endl;
        cout << "L2Norm(un)  == " << L2Norm(fields[0]) << endl;
        cout << "L2Norm(Pn)  == " << L2Norm(fields[1]) << endl;
        cout << "divNorm(un) == " << divNorm(fields[0]) << endl;
        cout << endl;
        cout << "Starting time stepping..." << endl;
    }

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
    }

    Real err = 0.0;
    Real lambda = -nu * square(2 * pi * ky / Ly);
    if (verbose)
        cout << "Exponential decay rate lambda == " << lambda << endl;

    for (Real t = T0; t <= T1; t += Ndt * dt) {
        ut.setToZero();
        ut += sinusoid;
        ut *= exp(lambda * t);

        Real unnorm = L2Norm(fields[0]);
        Real utnorm = L2Norm(ut);
        Real udist = L2Dist(ut, fields[0]);
        Real CFL = dns.CFL(fields[0]);
        err += udist;

        if (verbose) {
            cout << "\n";
            cout << "    t == " << t << endl;
            cout << "  CFL == " << CFL << endl;
            cout << " dPdx == " << dns.dPdx() << endl;
            cout << "Ubulk == " << dns.Ubulk() << endl;
            cout << "L2Norm(un) == " << unnorm << endl;
            cout << "L2Norm(ut) == " << utnorm << endl;
            cout << "L2Norm(un - ut)             == " << udist << endl;
            cout << "(L2Norm(un - ut)/L2Norm(ut) == " << udist / utnorm << endl;

            cout << "err == " << err << endl;
        }

        if (save) {
            ChebyCoeff un00 = Re(fields[0].profile(0, 0, 0));
            ChebyCoeff ut00 = Re(ut.profile(0, 0, 0));
            un00.makePhysical(trans);
            ut00.makePhysical(trans);

            ts << t << '\n';
            ns << unnorm << ' ' << utnorm << ' ' << udist << ' ' << divNorm(fields[0]) << ' ' << bcNorm(fields[0])
               << '\n';
            for (int ny = 0; ny < Ny; ++ny)
                uns << un00(ny) << ' ';
            uns << '\n';
            for (int ny = 0; ny < Ny; ++ny)
                uts << ut00(ny) << ' ';
            uts << '\n';
        }

        if (CFL > 2.0 || unnorm > 1) {
            cout << "Problem!" << endl;
            cout << "CFL  == " << CFL << endl;
            cout << "norm == " << unnorm << endl;
            cout << "\t** FAIL **" << endl;
            cerr << "\t** FAIL **" << endl;
            return 1;
        }

        dns.advance(fields, Ndt);
    }
    cout << argv[0] << s << "FinalError == " << err << " < " << maxerr << '\t' << flags << endl;

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
