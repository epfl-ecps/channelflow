/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <iomanip>
#include <iostream>
#include "channelflow/dns.h"
#include "channelflow/flowfield.h"
#include "channelflow/utilfuncs.h"

using namespace std;
using namespace chflow;

int main(int argc, char* argv[]) {
    cfMPI_Init(&argc, &argv);
    {
        cout << "================================================================\n";
        cout << "This program integrates a plane Couette flow from a random\n";
        cout << "initial condition. Velocity fields are saved at intervals dT=1.0\n";
        cout << "in a data-couette/ directory, in channelflow's binary data file\n";
        cout << "format." << endl << endl;

        // Define gridsize
        const int Nx = 16;
        const int Ny = 15;
        const int Nz = 16;

        // Define box size
        const Real Lx = 1.7 * pi;
        const Real a = -1.0;
        const Real b = 1.0;
        const Real Lz = 1.2 * pi;

        // Define flow parameters
        const Real Reynolds = 400.0;
        const Real nu = 1.0 / Reynolds;
        const Real dPdx = 0.0;

        // Define integration parameters
        const int n = 32;         // take n steps between printouts
        const Real dt = 1.0 / n;  // integration timestep
        const Real T = 30.0;      // integrate from t=0 to t=T

        fftw_loadwisdom();
        // Define DNS parameters
        DNSFlags flags;
        flags.baseflow = LaminarBase;
        flags.timestepping = SBDF3;
        flags.initstepping = CNRK2;
        flags.nonlinearity = Rotational;
        flags.dealiasing = DealiasXZ;
        // flags.nonlinearity = SkewSymmetric;
        // flags.dealiasing   = NoDealiasing;
        flags.ulowerwall = -1.0;
        flags.uupperwall = 1.0;
        flags.taucorrection = true;
        flags.constraint = PressureGradient;  // enforce constant pressure gradient
        flags.dPdx = dPdx;
        flags.dt = dt;
        flags.nu = nu;

        // Define size and smoothness of initial disturbance
        Real spectralDecay = 0.5;
        Real magnitude = 0.1;
        int kxmax = 3;
        int kzmax = 3;

        // Construct data fields: 3d velocity and 1d pressure
        cout << "building velocity and pressure fields..." << flush;
        vector<FlowField> fields = {FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b), FlowField(Nx, Ny, Nz, 1, Lx, Lz, a, b)};
        cout << "done" << endl;

        // Perturb velocity field
        fields[0].addPerturbations(kxmax, kzmax, 1.0, spectralDecay);
        fields[0] *= magnitude / L2Norm(fields[0]);

        // Construct Navier-Stoke integrator, set integration method
        cout << "building DNS..." << flush;
        DNS dns(fields, flags);
        cout << "done" << endl;

        mkdir("data");
        Real cfl = dns.CFL(fields[0]);
        for (Real t = 0; t <= T; t += n * dt) {
            cout << "         t == " << t << endl;
            cout << "       CFL == " << cfl << endl;
            cout << " L2Norm(u) == " << L2Norm(fields[0]) << endl;
            cout << "divNorm(u) == " << divNorm(fields[0]) << endl;
            cout << "      dPdx == " << dns.dPdx() << endl;
            cout << "     Ubulk == " << dns.Ubulk() << endl;

            // Write velocity and modified pressure fields to disk
            fields[0].save("data/u" + i2s(int(t)));
            fields[1].save("data/q" + i2s(int(t)));

            // Take n steps of length dt
            dns.advance(fields, n);
            cout << endl;
        }
    }
    cfMPI_Finalize();
}
