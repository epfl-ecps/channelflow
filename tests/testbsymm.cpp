/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <iostream>
#include "channelflow/realprofile.h"
#include "channelflow/symmetry.h"

using namespace std;
using namespace chflow;

int main(int argc, char** argv) {
    BasisFlags basisflags;
    basisflags.aBC = Diri;
    basisflags.bBC = Diri;
    basisflags.orthonormalize = false;
    basisflags.zerodivergence = true;

    // Define gridsize
    const int Nx = 48;
    const int Ny = 35;
    const int Nz = 48;
    const Real Lx = 2 * M_PI / 1.14;
    const Real ya = -1;
    const Real yb = 1;
    const Real Lz = 2 * M_PI / 2.5;

    const int kxmax = Nx / 3;
    const int kxmin = -kxmax;
    const int kzmin = 0;
    const int kzmax = Nz / 3;

    cout << "Nx == " << Nx << endl;
    cout << "Ny == " << Ny << endl;
    cout << "Nz == " << Nz << endl;
    cout << "kxmin == " << kxmin << endl;
    cout << "kxmax == " << kxmax << endl;
    cout << "kzmin == " << kzmin << endl;
    cout << "kzmax == " << kzmax << endl;

    cout << "constructing basis set. this takes a long time..." << endl;
    vector<RealProfile> basis = realBasis(Ny, kxmax, kzmax, Lx, Lz, ya, yb, basisflags);
    FieldSymmetry s1(1, 1, -1, 0.5, 0.0);
    FieldSymmetry s2(-1, -1, 1, 0.5, 0.5);
    FieldSymmetry s3(-1, -1, -1, 0.0, 0.5);
    FieldSymmetry e(1, 1, 1, 0.0, 0.0);

    RealProfile tmp;
    int s1ct = 0;
    int s2ct = 0;
    int s3ct = 0;
    int ect = 0;
    for (int i = 0; i < basis.size(); ++i) {
        tmp = basis[i];
        tmp *= s1;
        tmp -= basis[i];
        Real symm1 = L2Norm(tmp) / L2Norm(basis[i]);

        tmp = basis[i];
        tmp *= s2;
        tmp -= basis[i];
        Real symm2 = L2Norm(tmp) / L2Norm(basis[i]);

        tmp = basis[i];
        tmp *= s3;
        tmp -= basis[i];
        Real symm3 = L2Norm(tmp) / L2Norm(basis[i]);

        tmp = basis[i];
        tmp *= e;
        tmp -= basis[i];
        Real symme = L2Norm(tmp) / L2Norm(basis[i]);
        cout << basis[i].psi.kx() << ' ' << basis[i].psi.kz() << endl;
        cout << i << ' ' << symm1 << ' ' << symm2 << ' ' << symm3 << ' ' << symme << endl;

        bool iss1 = (abs(symm1) < 1e-7);
        bool iss2 = (abs(symm2) < 1e-7);
        bool iss3 = (abs(symm3) < 1e-7);
        bool ise = (abs(symme) < 1e-7);

        bool as1 = (abs(symm1 - 2) < 1e-7);
        bool as2 = (abs(symm2 - 2) < 1e-7);
        bool as3 = (abs(symm3 - 2) < 1e-7);

        if (!((as1 ^ iss1) && (as2 ^ iss2) && (as3 ^ iss3))) {
            cerr << as1 << iss1 << ' ' << as2 << iss2 << ' ' << as3 << iss3 << endl;
            cerr << "Non A/S element" << endl;
            exit(1);
        }
        if (iss1)
            s1ct += 1;
        if (iss2)
            s2ct += 1;
        if (iss3)
            s3ct += 1;

        if ((((iss1 ^ iss2) || (as1 ^ as2)) && iss3) || (((iss1 && iss2) || (as1 && as2)) ^ iss3)) {
            cerr << "Failed sanity check!" << endl;
            exit(1);
        }

        if (ise)
            ect += 1;
        else {
            cerr << "Failed identity !" << endl;
            exit(1);
        }
    }
    cout << "Counts: " << s1ct << ' ' << s2ct << ' ' << s3ct << ' ' << ect << endl;
}
