/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include "channelflow/diffops.h"
#include "channelflow/flowfield.h"
#include "channelflow/symmetry.h"
#include "channelflow/utilfuncs.h"

using namespace std;
using namespace chflow;

Real project(const FlowField& u, const FieldSymmetry& s, int sign);

int main(int argc, char* argv[]) {
    cfMPI_Init(&argc, &argv);
    {
        string purpose("translate a field to maximize (/minimize) its shift-reflect ");

        ArgList args(argc, argv, purpose);

        const bool szopt = args.getflag("-sz", "--z-reflect", "optimize z-phase for mirror symmetry about z=0");
        const bool s1opt = args.getflag("-s1", "--shift-reflect", "optimize x-phase for s1 = sz tx symmetry");
        const bool s2opt = args.getflag("-s2", "--shift-rotate", "optimize x-phase for s2 = sx sy txz symmetry");
        const bool sxyopt = args.getflag("-sxy", "--z-rotate", "optimize x-phase for pi-rotation about z-axis");
        const bool a1opt = args.getflag("-a1", "--anti-shift-reflect", "optimize x-phase for s1 anti-symmetry");
        const bool a2opt = args.getflag("-a2", "--anti-shift-rotate", "optimize z-phase for s2 anti-symmetry");
        const bool force = args.getflag("-f", "--force", "force symmetries by projection after optimization");
        const Real ax = args.getreal("-ax", "--axshift", 0.0, "initial guess for x translation: x -> x+ax*Lx");
        const Real az = args.getreal("-az", "--axshift", 0.0, "initial guess for z translation: z -> z+az*Lz");
        const int Nsteps = args.getint("-N", "--Nsteps", 10, "max # Newton steps");
        const Real eps = args.getreal("-e", "--eps", 1e-14, "stop Newton search when err<eps");
        const Real damp = args.getreal("-d", "--damping", 1.0, "damping factor for Newton steps");
        const bool inplace = args.getflag("-i", "--inplace", "write over input file");
        const bool verbose = args.getflag("-v", "--verbose", "print results during Newton search");
        const string uname = args.getstr((inplace ? 1 : 2), "<infield>", "input field");
        const string oname = inplace ? uname : args.getstr(1, "<outfield>", "output field");

        if ((s1opt && a1opt) || (s2opt && a2opt)) {
            cerr << "Sorry, can't optimize for a symmetry and its antisymmetry.\n";
            cerr << "Please choose only one of -s1,-a1 and one of -s2,-a2" << endl;
            exit(1);
        }
        args.check();
        args.save("./");

        FlowField u(uname);
        FlowField u0(u);
        FlowField su;

        FieldSymmetry sz(1, 1, -1, 0.0, 0);
        FieldSymmetry sxy(-1, -1, 1, 0.0, 0);
        FieldSymmetry s1(1, 1, -1, 0.5, 0);
        FieldSymmetry s2(-1, -1, 1, 0.5, 0.5);
        FieldSymmetry s3(-1, -1, -1, 0.0, 0.5);

        FieldSymmetry a(-1);
        FieldSymmetry a1 = a * s1;
        FieldSymmetry a2 = a * s2;
        // FieldSymmetry a3 = a*s3;

        Real unorm2 = L2Norm2(u);

        cout << endl;
        cout.setf(ios::left);
        cout << "Initial energy decomposition:" << endl;
        cout << "sz  symm: " << setw(12) << project(u, sz, 1) / unorm2;
        cout << "    anti: " << setw(12) << project(u, sz, -1) / unorm2 << endl;
        cout << "sxy symm: " << setw(12) << project(u, sxy, 1) / unorm2;
        cout << "    anti: " << setw(12) << project(u, sxy, -1) / unorm2 << endl;
        cout << "s1  symm: " << setw(12) << project(u, s1, 1) / unorm2;
        cout << "    anti: " << setw(12) << project(u, s1, -1) / unorm2 << endl;
        cout << "s2  symm: " << setw(12) << project(u, s2, 1) / unorm2;
        cout << "    anti: " << setw(12) << project(u, s2, -1) / unorm2 << endl;
        cout << "s3  symm: " << setw(12) << project(u, s3, 1) / unorm2;
        cout << "    anti: " << setw(12) << project(u, s3, -1) / unorm2 << endl;
        cout << endl;
        cout.unsetf(ios::left);
        cout << "\nL2Dist(u,u0) == " << L2Dist(u, u0) << endl;

        FieldSymmetry tau(1, 1, 1, ax, az);
        if (ax != 0.0 || az != 0.0) {
            u *= tau;

            cout << "Applying initial guess at translations: u = tau(u)" << endl;
            cout << endl;
            Real unorm2 = L2Norm2(u);
            cout.setf(ios::left);
            cout << "Post-guess energy decomposition:" << endl;
            cout << "sz  symm: " << setw(12) << project(u, sz, 1) / unorm2;
            cout << "    anti: " << setw(12) << project(u, sz, -1) / unorm2 << endl;
            cout << "sxy symm: " << setw(12) << project(u, sxy, 1) / unorm2;
            cout << "    anti: " << setw(12) << project(u, sxy, -1) / unorm2 << endl;
            cout << "s1  symm: " << setw(12) << project(u, s1, 1) / unorm2;
            cout << "    anti: " << setw(12) << project(u, s1, -1) / unorm2 << endl;
            cout << "s2  symm: " << setw(12) << project(u, s2, 1) / unorm2;
            cout << "    anti: " << setw(12) << project(u, s2, -1) / unorm2 << endl;
            cout << "s3  symm: " << setw(12) << project(u, s3, 1) / unorm2;
            cout << "    anti: " << setw(12) << project(u, s3, -1) / unorm2 << endl;
            cout << endl;
            cout << "\nL2Dist(u,u0) == " << L2Dist(u, u0) << endl;
        }

        FieldSymmetry t;
        FieldSymmetry s;

        if (szopt) {
            t = optimizePhase(u, sz, Nsteps, eps, damp, verbose);
            u = t(u);
            tau *= t;
        }
        if (sxyopt) {
            t = optimizePhase(u, sxy, Nsteps, eps, damp, verbose);
            u = t(u);
            tau *= t;
        }
        if (s1opt) {
            t = optimizePhase(u, s1, Nsteps, eps, damp, verbose);
            u = t(u);
            tau *= t;
        } else if (a1opt) {
            t = optimizePhase(u, a1, Nsteps, eps, damp, verbose);
            u = t(u);
            tau *= t;
        }

        if (s2opt) {
            t = optimizePhase(u, s2, Nsteps, eps, damp, verbose);
            u = t(u);
            tau *= t;
        } else if (a2opt) {
            t = optimizePhase(u, a2, Nsteps, eps, damp, verbose);
            u = t(u);
            tau *= t;
        }

        cout << endl;
        cout.setf(ios::left);
        cout << "Optimized energy decomposition:" << endl;
        cout << "sz  symm: " << setw(12) << project(u, sz, 1) / unorm2;
        cout << "    anti: " << setw(12) << project(u, sz, -1) / unorm2 << endl;
        cout << "sxy symm: " << setw(12) << project(u, sxy, 1) / unorm2;
        cout << "    anti: " << setw(12) << project(u, sxy, -1) / unorm2 << endl;
        cout << "s1  symm: " << setw(12) << project(u, s1, 1) / unorm2;
        cout << "    anti: " << setw(12) << project(u, s1, -1) / unorm2 << endl;
        cout << "s2  symm: " << setw(12) << project(u, s2, 1) / unorm2;
        cout << "    anti: " << setw(12) << project(u, s2, -1) / unorm2 << endl;
        cout << "s3  symm: " << setw(12) << project(u, s3, 1) / unorm2;
        cout << "    anti: " << setw(12) << project(u, s3, -1) / unorm2 << endl;
        cout << endl;
        cout.unsetf(ios::left);
        cout << "Post-translation s1     symm err == " << L2Dist(u, s1(u)) << endl;

        if (force) {
            if (szopt) {
                cout << "\nForcing sz symmetry by projection." << endl;
                (u += sz(u)) *= 0.5;
            }
            if (sxyopt) {
                cout << "\nForcing sxy symmetry by projection." << endl;
                (u += sxy(u)) *= 0.5;
            }
            if (s1opt) {
                cout << "\nForcing s1 symmetry by projection." << endl;
                (u += s1(u)) *= 0.5;
            } else if (a1opt) {
                cout << "\nForcing s1 antisymmetry by projection." << endl;
                (u -= s1(u)) *= 0.5;
            }
            if (s2opt) {
                cout << "Forcing s2  symmetry by projection." << endl;
                (u += s2(u)) *= 0.5;
            } else if (a2opt) {
                cout << "Forcing s2 antisymmetry by projection." << endl;
                (u -= s2(u)) *= 0.5;
            }

            unorm2 = L2Norm2(u);
            cout << endl;
            cout << "Forced energy decomposition:" << endl;
            cout.setf(ios::left);
            cout << "sz  symm: " << setw(12) << project(u, sz, 1) / unorm2;
            cout << "    anti: " << setw(12) << project(u, sz, -1) / unorm2 << endl;
            cout << "sxy symm: " << setw(12) << project(u, sxy, 1) / unorm2;
            cout << "    anti: " << setw(12) << project(u, sxy, -1) / unorm2 << endl;
            cout << "s1  symm: " << setw(12) << project(u, s1, 1) / unorm2;
            cout << "    anti: " << setw(12) << project(u, s1, -1) / unorm2 << endl;
            cout << "s2  symm: " << setw(12) << project(u, s2, 1) / unorm2;
            cout << "    anti: " << setw(12) << project(u, s2, -1) / unorm2 << endl;
            cout << "s3  symm: " << setw(12) << project(u, s3, 1) / unorm2;
            cout << "    anti: " << setw(12) << project(u, s3, -1) / unorm2 << endl;
            cout << endl;
        }
        cout << "\nL2Dist(u,u0)      == " << L2Dist(u, u0) << endl;
        cout << setprecision(16);
        cout << "optimal translation == " << tau << endl;
        u.save(oname);
    }
    cfMPI_Finalize();
}

Real project(const FlowField& u, const FieldSymmetry& s, int sign) {
    FlowField Pu(u);
    if (sign < 0)
        Pu -= s(u);
    else
        Pu += s(u);
    Pu *= 0.5;
    return L2Norm2(Pu);
}
