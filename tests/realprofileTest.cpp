/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <sys/stat.h>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "channelflow/diffops.h"
#include "channelflow/dns.h"
#include "channelflow/flowfield.h"
#include "channelflow/realprofile.h"

using namespace std;

using namespace chflow;

Real Linf(const ChebyCoeff& u);

int closest(Real x);

void saveprofiles(const string& filebase, FlowField& f);

void report(const string& testname, FlowField& f, FlowField& g, Real maxerr, bool& failure, bool verbose,
            bool savestuff);

void report(const string& testname, cfarray<Real>& a, cfarray<Real>& b, Real maxerr, bool& failure, bool verbose,
            bool savestuff);

Real L2Norm(cfarray<Real>& a);
Real L2Dist(cfarray<Real>& a, cfarray<Real>& b);

int main(int argc, char* argv[]) {
    // Define gridsize
    const int Nx = 32;
    const int Ny = 17;
    const int Nz = 32;

    const Real Lx = 4 * pi / 5;
    const Real ya = -1.0;
    const Real yb = 1.0;
    const Real Lz = 2 * pi / 3;

    const Real decay = 0.1;
    const Real magn = 1.0;
    const bool verbose = true;
    const bool savestuff = false;

    const Real maxerr = 1e-3;

    BasisFlags flags;
    flags.aBC = Diri;
    flags.bBC = Diri;
    flags.zerodivergence = true;
    flags.orthonormalize = true;

    bool failure = false;

    cerr << "realprofTest: " << flush;
    if (verbose) {
        cout << "\n====================================================" << endl;
        cout << "realprofTest\n\n";
    }

    FlowField u(Nx, Ny, Nz, 3, Lx, Lz, ya, yb);

    u.addPerturbations(4, 4, 1.0, decay);
    u.makePhysical();
    u.makeSpectral();
    u *= 1.0 / L2Norm(u);

    if (savestuff) {
        u.saveSpectrum("uspec");
        u.ygridpts().save("y");
        saveprofiles("u", u);
    }

    if (verbose)
        cout << "constructing basis " << endl;
    vector<RealProfile> e = realBasis(u.kxmax(), u.kzmax(), Ny, Lx, Lz, ya, yb, flags);
    int N = e.size();
    cfarray<Real> a(e.size());

    checkBasis(e, flags, true);
    if (verbose)
        cout << "e size == " << N << endl;

    // Duplicate a number of operations via FlowFields and RealProfiles
    FlowField f;
    FlowField g;
    f.reconfig(u);
    g.reconfig(u);
    RealProfile en;
    RealProfile fn;

    // ===========================================================
    if (verbose)
        cout << "identity test" << endl;
    f = u;
    field2coeff(e, f, a);
    coeff2field(e, a, g);
    report("identity", f, g, maxerr, failure, verbose, savestuff);

    // ===========================================================
    // Laplacian
    if (verbose)
        cout << "lapl test" << endl;
    lapl(u, f);
    g.setToZero();

    for (int n = 0; n < N; ++n) {
        en = e[n];
        Real an = L2InnerProduct(u, en);
        lapl(en, fn);
        fn *= an;
        g += fn;
    }
    report("lapl", f, g, maxerr, failure, verbose, savestuff);

    // ===========================================================
    // curl
    if (verbose)
        cout << "curl test" << endl;
    curl(u, f);
    g.setToZero();

    for (int n = 0; n < N; ++n) {
        en = e[n];
        Real an = L2InnerProduct(u, en);
        curl(en, fn);
        fn *= an;
        g += fn;
    }
    report("curl", f, g, maxerr, failure, verbose, savestuff);

    // ===========================================================
    // xdiff
    if (verbose)
        cout << "xdiff test" << endl;
    xdiff(u, f);
    g.setToZero();

    for (int n = 0; n < N; ++n) {
        en = e[n];
        Real an = L2InnerProduct(u, en);
        xdiff(en, fn);
        fn *= an;
        g += fn;
    }
    report("xdiff", f, g, maxerr, failure, verbose, savestuff);

    // ===========================================================
    // ydiff
    if (verbose)
        cout << "ydiff test" << endl;
    ydiff(u, f);
    g.setToZero();

    for (int n = 0; n < N; ++n) {
        en = e[n];
        Real an = L2InnerProduct(u, en);
        ydiff(en, fn);
        fn *= an;
        g += fn;
    }
    report("ydiff", f, g, maxerr, failure, verbose, savestuff);

    // ===========================================================
    // zdiff
    if (verbose)
        cout << "zdiff test" << endl;
    zdiff(u, f);
    g.setToZero();

    for (int n = 0; n < N; ++n) {
        en = e[n];
        Real an = L2InnerProduct(u, en);
        zdiff(en, fn);
        fn *= an;
        g += fn;
    }
    report("zdiff", f, g, maxerr, failure, verbose, savestuff);

    // ==========================================================
    // Declare some extra objects for binary operations
    FlowField v(Nx, Ny, Nz, 3, Lx, Lz, ya, yb);
    v.addPerturbations(4, 4, 1.0, decay);
    v.makePhysical();
    v.makeSpectral();
    v *= 1.0 / L2Norm(v);

    RealProfile fmn1;
    RealProfile fmn2;

    exit(0);

    // ===========================================================
    // dot

    // A simple test of the RealProfile binary operation, dot.
    if (verbose)
        cout << "simple dot test" << endl;

    for (int n = 0; n < 20; ++n) {
        int kx = (rand() % 4) - 2;
        int kz = rand() % 3;
        int kxp = (rand() % 3) - 1;
        int kzp = rand() % 2;

        if ((kx < 0 && kz == 0) || (kxp < 0 && kzp == 0)) {
            cout << "continue on kx,kz == " << kx << "," << kz << " kxp,kzp == " << kxp << "," << i2s(kzp) << endl;
            continue;
        }

        RealProfile uprof(3, Ny, kx, kz, Lx, Lz, ya, yb, Minus);
        RealProfile vprof(3, Ny, kxp, kzp, Lx, Lz, ya, yb, Minus);
        // uprof[0][0] = 1.0;
        // vprof[0][0] = 1.0;
        uprof.randomize(magn, decay, Free, Free);
        vprof.randomize(magn, decay, Free, Free);
        uprof *= 1.0 / L2Norm(uprof);
        vprof *= 1.0 / L2Norm(vprof);

        u.setToZero();
        v.setToZero();
        u += uprof;
        v += vprof;

        // Duplicate a number of operations via FlowFields and RealProfiles
        FlowField f;
        FlowField g;

        RealProfile fmn1;
        RealProfile fmn2;

        dot(u, v, f);

        g.reconfig(f);  // change to 1d field like f
        g.setToZero();
        dot(uprof, vprof, fmn1, fmn2);
        g += fmn1;
        g += fmn2;
        if (savestuff) {
            fmn1.save("fmn1");
            fmn2.save("fmn2");
            uprof.save("uprof");
            vprof.save("vprof");
        }

        cout << "uprof, vprof signs == " << uprof.sign() << ", " << vprof.sign() << endl;
        string s("simple dot, kx,kz == " + i2s(kx) + "," + i2s(kz) + " kxp,kzp == " + i2s(kxp) + "," + i2s(kzp));
        report(s, f, g, maxerr, failure, verbose, savestuff);
    }

    if (verbose)
        cout << "u.kxmin(), u.kxmax() == " << u.kxmin() << ", " << u.kxmax() << endl;
    if (verbose)
        cout << "u.kzmin(), u.kzmax() == " << u.kzmin() << ", " << u.kzmax() << endl;

    if (verbose)
        cout << "dot test" << endl;
    dot(u, u, f);
    g.reconfig(f);  // change to 1d field like f

    cfarray<Real> b(e.size());
    field2coeff(e, u, a);
    field2coeff(e, v, b);

    if (savestuff)
        a.save("a.asc");

    int kxmax = u.kxmax();
    int kxmin = u.kxmin();
    int kzmin = u.kzmin();
    int kzmax = u.kzmax();

    Real EPSILON = 1e-14;

    // u dot u == (am em) dot (an en) == (em dot en) am an
    for (int m = 0; m < N; ++m) {
        if (verbose)
            cout << m << ' ' << flush;
        const RealProfile& em = e[m];
        Real am = a[m];
        if (abs(am) < EPSILON)
            continue;

        int mkx = em.kx();
        int mkz = em.kz();
        for (int n = 0; n < N; ++n) {
            Real an = a[n];
            if (abs(an * am) < EPSILON)
                continue;
            const RealProfile& en = e[n];

            int rkx = mkx + en.kx();
            int rkz = mkz + en.kz();
            if (rkx < kxmin || rkx > kxmax || rkz < kzmin || rkz > kzmax)
                continue;

            dot(em, e[n], fmn1, fmn2);
            fmn1 *= am * a[n];
            fmn2 *= am * a[n];
            g += fmn1;
            g += fmn2;
        }
    }
    cout << "L2Norm2(u) == " << L2Norm2(u) << endl;
    cout << endl;
    report("dot", f, g, maxerr, failure, verbose, savestuff);

    // ===========================================================
    // grad:

    if (verbose)
        cout << "grad test" << endl;

    grad(u, f);

    g.reconfig(f);
    g.setToZero();
    field2coeff(e, u, a);

    // grad u == grad (an en) == (grad en) an
    RealProfile grad_en;
    for (int n = 0; n < N; ++n) {
        if (verbose)
            cout << n << ' ' << flush;
        Real an = a[n];
        if (abs(an) < EPSILON)
            continue;
        grad(e[n], grad_en);
        grad_en *= an;
        g += grad_en;
    }
    report("grad", f, g, maxerr, failure, verbose, savestuff);

    // ===========================================================
    // dotgrad: compare expansion coeffs rather than fields, because
    // the nonlinearity generates a non-zero-divergence component,
    // which the projection/unprojection on the div-free basis will nullify.

    if (verbose)
        cout << "dotgrad test" << endl;

    // u dotgrad u (calculated with FlowFields)
    FlowField tmp;
    convectionNL(u, f, tmp);

    // u dotgrad u calculated via em grad en am an
    g.reconfig(f);
    g.setToZero();
    field2coeff(e, u, a);

    // u dot grad u == (am em) dot grad (an en) == (em dotgrad en) am an
    // RealProfile grad_en;
    for (int n = 0; n < N; ++n) {
        if (verbose)
            cout << n << ' ' << flush;
        Real an = a[n];
        if (abs(an) < EPSILON)
            continue;
        grad(e[n], grad_en);
        for (int m = 0; m < N; ++m) {
            Real am = a[m];
            if (abs(am * an) < EPSILON)
                continue;

            dot(e[m], grad_en, fmn1, fmn2);
            fmn1 *= am * an;
            fmn2 *= am * an;

            g += fmn1;
            g += fmn2;
        }
    }
    cout << endl;
    report("dot grad", f, g, maxerr, failure, verbose, savestuff);
    /*********************************
    for (int m=0; m<N; ++m) {
      if (verbose) cout << m << ' ' << flush;
      const RealProfile& em = e[m];
      Real am = a[m];
      for (int n=0; n<N; ++n) {
        dotgrad(em, e[n], fmn1, fmn2);
        fmn1 *= am*a[n];
        fmn2 *= am*a[n];
        g += fmn1;
        g += fmn2;
     }
    }
    ****************************/

    if (failure) {
        cerr << "\t** FAIL **" << endl;
        cout << "\t** FAIL **" << endl;
        return 1;
    } else {
        cerr << "\t   pass   " << endl;
        cout << "\t   pass   " << endl;
        return 0;
    }

    if (savestuff) {
        u.ygridpts().save("y");
        u.saveSpectrum("uspec");
        v.saveSpectrum("vspec");
        u.profile(0, 0).save("u00");
        u.profile(1, 0).save("u10");
        u.profile(0, 1).save("u01");
        u.profile(1, 1).save("u11");

        v.profile(0, 0).save("v00");
        v.profile(1, 0).save("v10");
        v.profile(0, 1).save("v01");
        v.profile(1, 1).save("v11");

        u.ygridpts().save("y");

        u -= v;
        u.saveSpectrum("dspec");
    }
}

Real L2Norm(cfarray<Real>& a) {
    Real rtn2 = 0.0;
    for (int n = 0; n < a.N(); ++n)
        rtn2 += square(a[n]);
    return sqrt(rtn2);
}

Real L2Dist(cfarray<Real>& a, cfarray<Real>& b) {
    Real rtn2 = 0.0;
    for (int n = 0; n < a.N(); ++n)
        rtn2 += square(a[n] - b[n]);
    return sqrt(rtn2);
}

void saveprofiles(const string& filebase, FlowField& f) {
    int Kx = lesser(abs(f.kxmin()), 4);
    int Kz = lesser(f.kzmax(), 4);
    for (int kx = -Kx; kx <= Kx; ++kx) {
        int mx = f.mx(kx);
        string sx;
        if (kx < 0)
            sx += "_";
        sx += i2s(abs(kx));

        for (int kz = 0; kz <= Kz; ++kz) {
            int mz = f.mz(kz);
            f.profile(mx, mz).save(filebase + sx + i2s(kz));
        }
    }
}

void report(const string& testname, FlowField& f, FlowField& g, Real maxerr, bool& failure, bool verbose,
            bool savestuff) {
    Real err = L2Dist(f, g);

    if (verbose) {
        cout << "L2Norm(f)   == " << L2Norm(f) << endl;
        cout << "L2Norm(g)   == " << L2Norm(g) << endl;
        cout << "L2Dist(f,g) == " << L2Dist(f, g) << endl;
        cout << "   maxerr   == " << maxerr << endl;
    }
    if (err <= maxerr)
        cout << testname << "  passed " << endl;
    else {
        cout << testname << "  FAILED " << endl;
        failure = true;
    }
    if (err > maxerr && savestuff) {
        f.saveSpectrum("fspec");
        g.saveSpectrum("gspec");
        saveprofiles("f", f);
        saveprofiles("g", g);
        g -= f;
        g.saveSpectrum("dspec");
        saveprofiles("d", g);
        exit(1);
    }
    cout << endl;
}

void report(const string& testname, cfarray<Real>& a, cfarray<Real>& b, Real maxerr, bool& failure, bool verbose,
            bool savestuff) {
    Real err = L2Dist(a, b);

    if (verbose) {
        cout << "L2Norm(a)   == " << L2Norm(a) << endl;
        cout << "L2Norm(b)   == " << L2Norm(b) << endl;
        cout << "L2Dist(a,b) == " << L2Dist(a, b) << endl;
        cout << "   maxerr   == " << maxerr << endl;
    }
    if (err > maxerr) {
        cout << testname << "  FAILED " << endl;
        failure = true;
    }
    if (err > maxerr && savestuff) {
        a.save("a.asc");
        b.save("b.asc");
        exit(1);
    }
    cout << endl;
}
