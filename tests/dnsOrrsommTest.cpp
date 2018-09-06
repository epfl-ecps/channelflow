/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "cfbasics/complexdefs.h"
#include "channelflow/dns.h"
#include "channelflow/flowfield.h"

// The purpose of this program is to diagnose initialization behavior
// of multistep schemes. It's a stripped-down version of orrsomm.cpp

// This program compares the growth of OrrSommerfeld eigenfuncs integrated
// with
// (1) direct numerical simulation (fully 3d CFD with nonlinearity)
// (2) analytic expression exp(-i omega t) * OS eigenfunction
// (3) an ordinary differential equation intergrated w Runge-Kutta

// The DNS is represented by  FlowField un (u nonlinear)
// The analytic expression by FlowField ul (u linear)
// The ODE is represented by a Complex ark (a runge-kutta)
// Admittedly the names of these variables are poorly chosen.

// To compare the DNS and analytic expression to the ODE integration,
// we compute the inner product of un and ul with the OS eigenfunction:
// Complex an = L2InnerProduct(un, u_oseig);
// Complex al = L2InnerProduct(ul, u_oseig);
// If the three integration methods are identical, an == al == ark.

using namespace std;

using namespace chflow;

int main(int argc, char* argv[]) {
    Real maxerr = 1e-7;
    Real err = 0.0;

#ifdef HAVE_MPI
    cfMPI_Init(&argc, &argv);
    {
#endif

        // fftw_loadwisdom();

        const int Nx = 4;
        const int Ny = 65;
        const int Nz = 4;
        const int Nd = 3;
        const Real Lx = 2 * pi;
        const Real Lz = pi;
        const Real a = -1.0;
        const Real b = 1.0;
        const Real Reynolds = 7500.0;
        const Real nu = 1.0 / Reynolds;
        const Real T0 = 0.0;
        const Real T1 = 13.0;
        const Real dt = 0.02;
        const int N = 50;  // plot results at interval N dt
        const Real scale = 1e-4;

        const bool save = false;

        bool baseflow = true;
        DNSFlags flags;
        flags.baseflow = ZeroBase;
        flags.dealiasing = NoDealiasing;
        flags.nonlinearity = Rotational;
        flags.constraint = BulkVelocity;
        flags.timestepping = SBDF3;
        flags.initstepping = CNRK2;
        flags.verbosity = Silent;
        flags.ulowerwall = 0.0;
        flags.uupperwall = 0.0;
        flags.dt = dt;
        flags.nu = nu;
        flags.dPdx = -2.0 * nu;  // mean streamwise pressure gradient.
        flags.Ubulk = 2.0 / 3.0;

        const char s = ' ';

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
            } else if (argument == "--linU")
                flags.nonlinearity = LinearAboutProfile;
            // else if (argument == "--linu")
            // flags.nonlinearity = LinearAboutField;
            else if (argument == "--rot")
                flags.nonlinearity = Rotational;
            else if (argument == "--skw")
                flags.nonlinearity = SkewSymmetric;
            else if (argument == "--alt")
                flags.nonlinearity = Alternating;
            else if (argument == "--div")
                flags.nonlinearity = Divergence;
            else if (argument == "--cnv")
                flags.nonlinearity = Convection;

            else if (argument == "--parab") {
                flags.baseflow = LaminarBase;
                baseflow = true;
            } else if (argument == "--zero") {
                baseflow = false;
                flags.baseflow = ZeroBase;
            }
        }

        if (flags.nonlinearity == LinearAboutProfile)
            maxerr = 1e-9;

        cerr << "dnsOrrsommTest  ";
        for (int i = 1; i < argc; ++i) {
            cerr << argv[i];
            int pad = 10 - strlen(argv[i]);
            for (int j = 0; j < pad; ++j)  // crude formatting
                cerr << s;
        }
        cerr << flush;

        cout << "\n====================================================" << endl;
        cout << "dnsOrrsommTest ";
        for (int i = 1; i < argc; ++i)
            cout << argv[i] << ' ';
        cout << endl;
        cout << "flags == " << flags << endl;
        cout << setprecision(14);

        ChebyTransform trans(Ny);
        Vector y = chebypoints(Ny, a, b);
        y.save("y");

        // Get Orr-Sommerfeld eigenfunction from file. The data is the
        // y-dependence of the eigenfunction's (u,v) components. x dependence
        // is exp(2 pi i x / Lx). Eigfunction is const in z. Reconstruct
        // w from velocity field, pressure from OS eqns.
        BasisFunc ueig("../data/os_ueig10_65");
        BasisFunc peig("../data/os_peig10_65");
        ueig.makeSpectral(trans);
        peig.makeSpectral(trans);

        BasisFunc ueig_conj = conjugate(ueig);
        BasisFunc peig_conj = conjugate(peig);

        // The temporal variation of the eigenfunction is exp(lambda t)
        Complex omega = 0.24989153647208251 + I * 0.0022349757548207664;
        Complex lambda = -1.0 * I * omega;
        cout << "   omega = " << omega << endl;
        cout << "  lambda = " << lambda << endl;

        ChebyCoeff parabola(Ny, a, b, Physical);
        ChebyCoeff zero(Ny, a, b, Spectral);
        for (int ny = 0; ny < Ny; ++ny)
            parabola[ny] = 1.0 - square(y[ny]);
        parabola.makeSpectral();

        ChebyCoeff Ubase(Ny, a, b, Spectral);
        ChebyCoeff Wbase(Ny, a, b, Spectral);

        // Calculate L2Norm of parabolic flow and perturbation field
        // and rescale so that perturb/parabloic is equal to the given scale.
        FlowField utmp(Nx, Ny, Nz, Nd, Lx, Lz, a, b);
        utmp.setState(Spectral, Spectral);
        utmp += parabola;
        Real uParabNorm = L2Norm(utmp);

        utmp.setToZero();
        utmp += ueig;
        utmp += ueig_conj;

        Real uPerturbNorm = L2Norm(utmp);

        Real c = scale * uParabNorm / uPerturbNorm;
        ueig *= c;
        peig *= c;
        ueig_conj *= c;
        peig_conj *= c;

        // ==============================================================
        // Now set up un, ubase, Ubase, uoffset for these cases
        //
        // For nonlinearity == LinearAboutProfile, check
        // utot == un + Ubase
        //   un == eig,             Ubase == parabola   (baseflow == true)
        //
        // For all other nonlinearities, check
        // utot == un + Ubase
        //   un == eig,             Ubase == parabola   (baseflow == true)
        //   un == eig + parabola   Ubase == zero       (baseflow == false)

        if (baseflow) {
            Ubase = parabola;
            flags.baseflow = ArbitraryBase;
        } else {
            flags.baseflow = ZeroBase;
            Ubase = zero;
        }

        FlowField un(Nx, Ny, Nz, Nd, Lx, Lz, a, b);  // numerical velocity field
        FlowField qn(Nx, Ny, Nz, 1, Lx, Lz, a, b);   // numerical pressure field
        FlowField pn(Nx, Ny, Nz, 1, Lx, Lz, a, b);   // numerical pressure field
        FlowField ul(Nx, Ny, Nz, Nd, Lx, Lz, a, b);  // linear-solution velocity
        FlowField pl(Nx, Ny, Nz, 1, Lx, Lz, a, b);   // linear-solution pressure

        // Set up initial condition
        un += ueig;
        un += ueig_conj;
        if (!baseflow)
            un += parabola;

        vector<FlowField> fields = {un, qn};
        vector<ChebyCoeff> base = {Ubase, Wbase};
        DNS dns(fields, base, flags);

        pn += peig;
        pn += peig_conj;
        dns.up2q(un, pn, fields[1]);

        ul += ueig;
        ul += ueig_conj;
        if (!baseflow)
            ul += parabola;

        pl += peig;
        pl += peig_conj;

        ofstream normn;
        ofstream norml;
        ofstream times;

        if (save) {
            y.save("y");
            normn.open("norms_un2.asc");
            norml.open("norms_ul2.asc");
            times.open("t.asc");
            normn << setprecision(8);
            norml << setprecision(8);
        }

        //         Real err = 0.0;
        for (Real t = T0; t <= T1; t += N * dt) {
            Real udist = L2Dist(un, ul);
            Real pdist = L2Dist(pn, pl);
            Real unnorm = L2Norm(un);
            Real ulnorm = L2Norm(ul);
            Real pnnorm = L2Norm(pn);
            Real plnorm = L2Norm(pl);
            Real CFL = dns.CFL(un);

            err += udist + pdist;

            pl.saveProfile(1, 0, "pl");
            pn.saveProfile(1, 0, "pn");
            ul.saveProfile(1, 0, "ul");
            un.saveProfile(1, 0, "un");

            cout << "\n";
            cout << "    t == " << t << endl;
            cout << "  CFL == " << CFL << endl;
            cout << " dPdx == " << dns.dPdx() << endl;
            cout << "Ubulk == " << dns.Ubulk() << endl;
            cout << "L2Norm(un) == " << unnorm << endl;
            cout << "L2Norm(ul) == " << ulnorm << endl;
            cout << "L2Norm(pn) == " << pnnorm << endl;
            cout << "L2Norm(pl) == " << plnorm << endl;
            cout << "L2Dist(un,ul)             == " << udist << endl;
            cout << "L2Dist(pn,pl)             == " << pdist << endl;
            cout << "L2Dist(un,ul)/L2Norm(ul)  == " << udist / ulnorm << endl;
            cout << "L2Dist(pn,pl)/L2Norm(pl)  == " << pdist / plnorm << endl;
            cout << "err == " << err << endl;

            if (save) {
                normn << t << s << L2Norm(un) << s << L2Norm(pn) << endl;
                norml << t << s << L2Norm(ul) << s << L2Norm(pl) << endl;
                times << t << endl;
            }

            ueig *= exp(lambda * (N * dt));
            peig *= exp(lambda * (N * dt));
            ueig_conj = conjugate(ueig);
            peig_conj = conjugate(peig);

            ul.setToZero();
            ul += ueig;
            ul += ueig_conj;
            if (!baseflow)
                ul += parabola;

            pl.setToZero();
            pl += peig;
            pl += peig_conj;

            if (CFL > 2.0 || unnorm > 1) {
                cout << "Problem!" << endl;
                cout << "CFL  == " << CFL << endl;
                cout << "norm == " << unnorm << endl;
                cout << "\t** FAIL **" << endl;
                cerr << "\t** FAIL **" << endl;
                return 1;
            }

            dns.advance(fields, N);
            un = fields[0];
            qn = fields[1];
            dns.uq2p(un, qn, pn);

            // pn = qn;
        }
        cout << argv[0] << s << "FinalError == " << err << " < " << maxerr << '\t' << flags << endl;
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
