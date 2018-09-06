/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <iostream>
#include "channelflow/chebyshev.h"

using namespace std;

using namespace chflow;

int main() {
    bool save = false;
    bool verbose = true;

    Real error = 0.0;
    Real maxerr = 1e-12;

    Real difffactor = 1e3;  // allows error increase for successive derivatives
    int N = 17;
    Real a = -1.1;
    Real b = 1.4;
    ChebyCoeff f(N, a, b, Physical);
    ChebyCoeff fy(N, a, b, Physical);
    ChebyCoeff fyy(N, a, b, Physical);

    cerr << "chebyTest: " << flush;
    if (verbose) {
        cout << "\n====================================================" << endl;
        cout << "chebyTest\n\n";
    }

    Vector y = chebypoints(N, a, b);
    if (save)
        y.save("y");

    for (int n = 0; n < N; ++n) {
        Real yn = y[n];
        // f[n] = 1-square(yn); //sin(yn);
        // fy[n] = -2*yn; // -2*cos(yn);
        // fyy[n] = -2; // -sin(yn);
        f[n] = 1 + sin(yn);
        fy[n] = cos(yn);
        fyy[n] = -sin(yn);
    }

    for (int n = 0; n < N; ++n) {
        Real yn = y[n];
        // f[n] = 1-square(yn); //sin(yn);
        // fy[n] = -2*yn; // -2*cos(yn);
        // fyy[n] = -2; // -sin(yn);
        f[n] = 1 + sin(yn);
        fy[n] = cos(yn);
        fyy[n] = -sin(yn);
    }
    ChebyTransform t(N);

    ChebyCoeff g(f);
    g.makeSpectral(t);
    g.makePhysical(t);
    if (verbose)
        cout << "L1Dist(f,g) == " << L1Dist(f, g) << endl;
    g.makeSpectral(t);

    Real err = 0;
    for (int n = 0; n < N; ++n)
        err += abs(f(n) - g.eval(y[n]));
    if (verbose)
        cout << "sum abs (f(n)-g.eval(y[n])) == " << err << endl;

    f.makeSpectral();
    ChebyCoeff gy;
    diff(g, gy);

    ChebyCoeff gyy = diff2(g);

    ChebyCoeff Ig = integrate(g);
    ChebyCoeff DIg = diff(Ig);

    err = L1Dist(DIg, g);
    error += err;
    if (verbose)
        cout << "L1Dist(diff(integrate(g)), g) == " << err << endl;

    ChebyCoeff Igy = integrate(gy);
    Igy[0] = g[0];
    err = L1Dist(Igy, g);
    error += err;
    if (verbose)
        cout << "L1Dist(integrate(diff(gy)), g) == " << err << endl;

    g.ichebyfft(t);
    gy.ichebyfft(t);
    gyy.ichebyfft(t);
    Ig.ichebyfft(t);
    Igy.ichebyfft(t);
    DIg.ichebyfft(t);

    f.makePhysical();
    err = L1Norm(f - g);
    error += err;
    if (verbose)
        cout << "trans  error == " << err << endl;

    err = L1Norm(fy - gy);
    error += err / difffactor;
    if (verbose)
        cout << "d/dy   error == " << err << endl;

    err = L1Norm(fyy - gyy);
    error += err / square(difffactor);
    if (verbose)
        cout << "d2/dy2 error == " << err << endl;

    if (save) {
        g.save("g");
        gyy.save("gyy");
        gy.save("gy");
        Igy.save("Igy");
        DIg.save("DIg");
        Ig.save("Ig");
        fyy.save("fyy");
        fy.save("fy");
        f.save("f");
    }
    cout << "test1" << endl;

    ComplexChebyCoeff h(N, a, b, Physical);
    cout << "test2" << endl;
    h.randomize(1.0, 0.6, Free, Free);
    cout << "test3" << endl;

    if (save) {
        h.save("h");

        ComplexChebyCoeff h2("h");
        err = L1Dist(h, h2);
        error += err;

        if (verbose)
            cout << "ComplexChebyCoeff IO error == " << err << endl;
    }
    cout << "test4" << endl;

    if (save) {
        Complex zero = 0.0 + 0.0 * I;
        for (int n = 0; n < N; ++n)
            h.set(n, zero);
        h.setState(Spectral);

        h.set(0, 1.0 + 0.0 * I);
        h.ichebyfft(t);
        h.save("T0");
        h.chebyfft(t);
        h.set(0, zero);

        h.set(1, 1.0 + 0.0 * I);
        h.ichebyfft(t);
        h.save("T1");
        h.chebyfft(t);
        h.set(1, zero);

        h.set(2, 1.0 + 0.0 * I);
        h.ichebyfft(t);
        h.save("T2");
        h.chebyfft(t);
        h.set(1, zero);

        h.set(2, 1.0 + 0.0 * I);
        h.ichebyfft(t);
        h.save("T2");
        h.chebyfft(t);
        h.set(2, zero);

        h.set(3, 1.0 + 0.0 * I);
        h.ichebyfft(t);
        h.save("T3");
        h.chebyfft(t);
        h.set(3, zero);
    }

    if (error < maxerr) {
        cerr << "\t   pass   " << endl;
        cout << "\t   pass   " << endl;
        return 0;
    } else {
        cerr << "\t** FAIL **" << endl;
        cout << "\t** FAIL **" << endl;
        cout << "error  == " << error << endl;
        cout << "maxerr == " << maxerr << endl;
        return 1;
    }
}
