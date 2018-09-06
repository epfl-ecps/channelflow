/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */
#include <iostream>
#include "channelflow/bandedtridiag.h"

using namespace std;
using namespace chflow;

int main() {
    bool save = false;
    bool verbose = true;

    Real err = 0.0;
    Real maxerr = 1e-12;

    const int M = 100;
    BandedTridiag A(M);

    cerr << "tridagTest: " << flush;
    if (verbose) {
        cout << "\n====================================================" << endl;
        cout << "tridagTest\n\n";
    }

    Vector b(M);
    Vector x(M);

    Real decay = 0.8;
    Real mag = 1.0;

    for (int i = 0; i < M; ++i) {
        x[i] = mag * drand48();
        mag *= decay;
        A.band(i) = 1.0 + drand48() / 10;
        A.diag(i) = 1.0 + drand48() / 10;
        if (i > 0)
            A.lodiag(i) = drand48() / 10;
        if (i < M - 2)
            A.updiag(i) = drand48() / 10;
    }

    A.multiply(x, b);

    // cout << "A = " << endl;
    // A.print();

    A.ULdecomp();

    if (verbose)
        cout << "LU(A) =  " << endl;
    // A.print();
    // A.ULprint();

    Vector xx = b;
    A.ULsolve(xx);
    // cout << " b = [ " << b << " ]\n";
    // cout << " x = [ " << x << " ]\n";
    // cout << "xx = [ " << xx << " ]\n";
    int test = 0;

    Real error = L1Norm(x - xx);
    err += error;
    if (verbose) {
        cout << "test " << test << " error = " << error << endl;
        cout << "test " << test << " L1(x) = " << L1Norm(x) << endl;

        cout << "Testing strided solve and multiply" << endl;
        cout << "Offset == 0, stride == 2: " << endl;
    }

    Vector b0(2 * M);
    Vector x0(2 * M);
    Vector xx0(2 * M);

    for (int i = 0; i < M; ++i) {
        b0[2 * i] = b[i];
        x0[2 * i] = x[i];
        x0[2 * i + 1] = b0[2 * i + 1] = drand48();
    }
    A.ULsolveStrided(b0, 0, 2);

    error = L1Norm(x0 - b0);
    err += error;
    if (verbose) {
        cout << "error = " << error << endl;
        cout << "Offset == 1, stride == 2: " << endl;
    }
    for (int i = 0; i < M; ++i) {
        x0[2 * i] = b0[2 * i] = drand48();
        b0[2 * i + 1] = b[i];
        x0[2 * i + 1] = x[i];
    }
    A.ULsolveStrided(b0, 1, 2);

    error = L1Norm(x0 - b0);
    err += error;

    if (verbose) {
        cout << "error = " << error << endl;
    }

    if (save) {
        if (verbose) {
            cout << "Testing IO." << endl;
            cout << "saving A ..." << flush;
        }
        A.save("A");

        if (verbose) {
            cout << "done." << endl;
            cout << "reading B ..." << flush;
        }
        BandedTridiag B("A");

        if (verbose)
            cout << "done." << endl;

        if (!(A == B)) {
            err += 1;
            if (verbose)
                cout << "BandedTridiag IO failed." << endl;
        }
    }

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
