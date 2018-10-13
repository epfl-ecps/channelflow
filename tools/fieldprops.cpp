/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include "channelflow/diffops.h"
#include "channelflow/dns.h"
#include "channelflow/flowfield.h"
#include "channelflow/periodicfunc.h"
#include "channelflow/symmetry.h"
#include "channelflow/turbstats.h"
#include "channelflow/utilfuncs.h"

using namespace std;

using namespace chflow;

Real energy(FlowField& u, int i, bool normalize = true);

// returns S,A, for symmetric,antisymmetric to eps accuracy
// returns s,a, for symmetric,antisymmetric to sqrt(eps) accuracy
char symmetric(const FieldSymmetry& s, const FlowField& u, Real& serr, Real& aerr, Real eps = 1e-7);

Real project(const FlowField& u, const FieldSymmetry& s, int sign);

int main(int argc, char* argv[]) {
    cfMPI_Init(&argc, &argv);
    {
        string purpose("prints information about a FlowField");

        ArgList args(argc, argv, purpose);

        const bool geom = args.getflag("-g", "--geometry", "show geometrical properties");
        const bool mean = args.getflag("-m", "--mean", "show mean properties of Utot=u+Ubase (baseflow needed -Ub)");
        const bool norm = args.getflag("-n", "--norm", "show norm properties");
        const bool spec = args.getflag("-sp", "--spectral", "show spectral properties");
        const bool symm = args.getflag("-sy", "--symmetry", "show symmetry properties");
        const bool dyn = args.getflag("-d", "--dynamic", "show dynamical properties (DNSFlags needed)");
        const bool mconstr = args.getflag("-mcs", "--mcs", "show dPdx and bulk velocity");
        const bool nrgy = args.getflag("-e", "--energy", "show energy properties");
        const bool wall = args.getflag("-w", "--wall", "show wall-unit properties (baseflow needed -Ub)");
        const bool local = args.getflag("-l", "--local", "show z-localization");
        const bool saveuprof =
            args.getflag("-sup", "--saveUprofile", "save the mean velocity profile to file (baseflow needed -Ub)");
        const bool fieldstat = args.getflag("-fst", "--fieldstats", "Print fieldstats(u)");
        const Real eps = args.getreal("-eps", "--eps", 1e-7, "spectral noise floor");
        const int digits = args.getint("-dg", "--digits", 6, "digits of output for reals");
        const string uname = args.getstr(1, "<flowfield>", "input field");

        DNSFlags flags(args);

        const string Uname = args.getstr(
            "-ub", "--Ubase", "", "input baseflow file of arbitrary U-baseflow (takes precedence over -bf option)");
        const string Wname = args.getstr(
            "-wb", "--Wbase", "", "input baseflow file of arbitrary W-baseflow (takes precedence over -bf option)");
        const Real ubasefac = args.getreal("-Uf", "--Uf", 1, "Multiply baseflow by this factor before adding");

        const Real T = args.getreal("-Tint", "--Tint", flags.dt, "integration time for du/dt estimate");

        args.check();

        bool all =
            (!geom && !mean && !norm && !spec && !symm && !dyn && !nrgy && !wall && !local && !mconstr) ? true : false;

        FlowField u(uname);
        u.makeSpectral();

        const Real unorm = L2Norm(u);
        cout << setprecision(digits);

        vector<ChebyCoeff> Baseflow = baseFlow(u.Ny(), u.a(), u.b(), flags, Uname, Wname);
        for (int i = 0; i < 2; i++) {
            Baseflow[i].makePhysical();
            Baseflow[i] *= ubasefac;
            Baseflow[i].makeSpectral();
        }
        ChebyCoeff Ubase = Baseflow[0];
        ChebyCoeff Wbase = Baseflow[1];
        cout << "Ubase mean == " << Ubase.mean() << endl;
        cout << "Wbase mean == " << Wbase.mean() << endl;

        if (saveuprof) {
            if (u.Nd() == 1) {
                cferror("Not available for scalar fields");
            }

            if (flags.baseflow != ZeroBase) {
                ofstream uProfileOut;
                ofstream wProfileOut;
                openfile(uProfileOut, "uprofile.asc");
                openfile(wProfileOut, "wprofile.asc");

                u.cmplx(0, 0, 0, 1) -= Complex(ubasefac * flags.Vsuck, 0.);
                ChebyCoeff uprof = u.profile(0, 0, 0).re;
                ChebyCoeff wprof = u.profile(0, 0, 2).re;
                uprof += Ubase;
                wprof += Wbase;
                uprof.makePhysical();
                wprof.makePhysical();
                for (int ny = 0; ny < uprof.N(); ++ny) {
                    uProfileOut << setw(14) << uprof[ny];
                    wProfileOut << setw(14) << wprof[ny];
                }
                uProfileOut << endl;
                wProfileOut << endl;
            }

            else
                cferror("to compute mean velocity provide the base flow, either a file or DNS flags to construct it.");
        }

        if (fieldstat) {
            ofstream fout;
            fout.open("fieldstats");

            if (u.Nd() == 1) {
                cferror("Not available for scalar fields");
            }

            if (flags.baseflow != ZeroBase) {
                cout << "computing statistics of Utot = u + Ubase \n" << endl, u += Baseflow;
                u.cmplx(0, 0, 0, 1) -= Complex(ubasefac * flags.Vsuck, 0.);
            }

            fout << fieldstatsheader_t() << endl;
            cout << fieldstatsheader_t() << endl;
            string s = fieldstats_t(u, 0);
            fout << s << endl;
            cout << s << endl;

            return 0;
        }

        if (all || geom) {
            cout << "--------------Geometry------------------" << endl;
            cout << "Nx == " << u.Nx() << endl;
            cout << "Ny == " << u.Ny() << endl;
            cout << "Nz == " << u.Nz() << endl;
            cout << "Nd == " << u.Nd() << endl;
            cout << "Lx == " << u.Lx() << endl;
            cout << "Ly == " << u.Ly() << endl;
            cout << "Lz == " << u.Lz() << endl;
            cout << " a == " << u.a() << endl;
            cout << " b == " << u.b() << endl;
            cout << "lx == " << u.Lx() / (2 * pi) << endl;
            cout << "lz == " << u.Lz() / (2 * pi) << endl;
            cout << "alpha == " << 2 * pi / u.Lx() << endl;
            cout << "gamma == " << 2 * pi / u.Lz() << endl;
            cout << "Lx/Lz == " << u.Lx() / u.Lz() << endl;
            cout << "Lx*Lz == " << u.Lx() * u.Lz() << endl;
            cout << "diag  == " << pythag(u.Lx(), u.Lz()) << " == sqrt(Lx^2 + Lz^2)" << endl;
            cout << "vol   == " << u.Lx() * u.Ly() * u.Lz() << " == Lx*Ly*Lz" << endl;
            //     cout << "nu    == " << u.nu()  << endl;
            cout << "\n" << endl;
        }

        if (all || mean) {
            cout << "--------------Mean---------------------" << endl;

            for (int i = 0; i < u.Nd(); ++i) {
                cout << "mean u[" << i << "] == " << Re(u.profile(0, 0, i)).mean() << endl;
            }
            if (flags.baseflow != ZeroBase) {
                u += Baseflow;
                u.cmplx(0, 0, 0, 1) -= Complex(ubasefac * flags.Vsuck, 0.);

                for (int i = 0; i < u.Nd(); ++i) {
                    cout << "mean u[" << i << "] == " << Re(u.profile(0, 0, i)).mean() << endl;
                }
                u -= Baseflow;
                u.cmplx(0, 0, 0, 1) += Complex(ubasefac * flags.Vsuck, 0.);
            }
        }

        if ((all && u.Nd() > 1) || mconstr) {
            if (u.Nd() == 1) {
                cferror("Not available for scalar fields");
            }

            cout << "----------Pressure Gradient and Bulk velocity------------" << endl;
            cout << "                     at Re = " << 1.0 / flags.nu << "\n" << endl;

            cout << "dPdx(u)  == " << getdPdx(u, flags.nu) << endl;
            cout << "Ubulk(u) == " << getUbulk(u) << endl;

            if (flags.baseflow != ZeroBase) {
                u += Baseflow;
                u.cmplx(0, 0, 0, 1) -= Complex(ubasefac * flags.Vsuck, 0.);
                cout << "dPdx(u+U)  == " << getdPdx(u, flags.nu) << endl;
                cout << "Ubulk(u+U) == " << getUbulk(u) << endl;
                u -= Baseflow;
                u.cmplx(0, 0, 0, 1) += Complex(ubasefac * flags.Vsuck, 0.);
            }
        }

        if (all || symm) {
            cout << "--------------Symmetry-------------------\n";
            FieldSymmetry s1(1, 1, -1, 0.5, 0.0);  // Wally-oriented symmetries, GHC09 S group
            FieldSymmetry s2(-1, -1, 1, 0.5, 0.5);
            FieldSymmetry s3(-1, -1, -1, 0.0, 0.5);

            FieldSymmetry sxtxz(-1, 1, 1, 0.5, 0.5);  // GHC09 Rxz group, conjugate to S.
            FieldSymmetry sztxz(1, 1, -1, 0.5, 0.5);
            FieldSymmetry sxz(-1, -1, -1, 0.0, 0.0);

            FieldSymmetry sx(-1, -1, 1);
            FieldSymmetry sz(1, 1, -1);
            FieldSymmetry tx(1, 1, 1, 0.5, 0.0);
            FieldSymmetry tz(1, 1, 1, 0.0, 0.5);
            FieldSymmetry txz(1, 1, 1, 0.5, 0.5);

            Real unorm2 = L2Norm2(u);

            cout.setf(ios::left);
            cout << "fractions of energy in symmetric/antisymmetric subspaces:" << endl;
            cout << "   s1  symm: " << setw(12) << project(u, s1, 1) / unorm2;
            cout << "       anti: " << setw(12) << project(u, s1, -1) / unorm2 << endl;
            cout << "   s2  symm: " << setw(12) << project(u, s2, 1) / unorm2;
            cout << "       anti: " << setw(12) << project(u, s2, -1) / unorm2 << endl;
            cout << "   s3  symm: " << setw(12) << project(u, s3, 1) / unorm2;
            cout << "       anti: " << setw(12) << project(u, s3, -1) / unorm2 << endl << endl;

            cout << "sxtxz  symm: " << setw(12) << project(u, sxtxz, 1) / unorm2;
            cout << "       anti: " << setw(12) << project(u, sxtxz, -1) / unorm2 << endl;
            cout << "sztxz  symm: " << setw(12) << project(u, sztxz, 1) / unorm2;
            cout << "       anti: " << setw(12) << project(u, sztxz, -1) / unorm2 << endl;
            cout << "sxz    symm: " << setw(12) << project(u, sxz, 1) / unorm2;
            cout << "       anti: " << setw(12) << project(u, sxz, -1) / unorm2 << endl << endl;

            cout << " sx    symm: " << setw(12) << project(u, sx, 1) / unorm2;
            cout << "       anti: " << setw(12) << project(u, sx, -1) / unorm2 << endl;
            cout << " sz    symm: " << setw(12) << project(u, sz, 1) / unorm2;
            cout << "       anti: " << setw(12) << project(u, sz, -1) / unorm2 << endl;
            cout << "   tx  symm: " << setw(12) << project(u, tx, 1) / unorm2;
            cout << "       anti: " << setw(12) << project(u, tx, -1) / unorm2 << endl;
            cout << "   tz  symm: " << setw(12) << project(u, tz, 1) / unorm2;
            cout << "       anti: " << setw(12) << project(u, tz, -1) / unorm2 << endl;
            cout << "   txz symm: " << setw(12) << project(u, txz, 1) / unorm2;
            cout << "       anti: " << setw(12) << project(u, txz, -1) / unorm2 << endl;
            cout << "\n" << endl;
            cout.unsetf(ios::left);
        }

        if (all || norm) {
            cout << "--------------Norms---------------------" << endl;

            cout << " L2Norm(u)    == " << unorm << endl;
            cout << " bcNorm(u)   == " << bcNorm(u) << endl;

            if (u.Nd() > 1) {
                for (int i = 0; i < u.Nd(); ++i)
                    cout << " L2Norm(u[" << i << "]) == " << sqrt(energy(u, i)) << endl;

                cout << "L2Norm3d(u)   == " << L2Norm3d(u) << endl;
                cout << " divNorm(u)   == " << divNorm(u) << endl;
                cout << "\n" << endl;
            }
        }

        if (all || spec) {
            cout << "\n" << endl;
            cout << "------------Spectrum--------------------" << endl;
            int kxmax = 0;
            int kzmax = 0;
            int kymax = 0;
            cout << "u.kxmin() == " << u.kxmin() << endl;
            cout << "u.kxmax() == " << u.kxmax() << endl;
            cout << "u.kzmin() == " << u.kzmin() << endl;
            cout << "u.kzmax() == " << u.kzmax() << endl;
            cout << "u.kxminDealiased() == " << u.kxminDealiased() << endl;
            cout << "u.kxmaxDealiased() == " << u.kxmaxDealiased() << endl;
            cout << "u.kzminDealiased() == " << u.kzminDealiased() << endl;
            cout << "u.kzmaxDealiased() == " << u.kzmaxDealiased() << endl;

            for (int mx = 0; mx < u.Mx(); ++mx) {
                const int kx = u.kx(mx);
                for (int mz = 0; mz < u.Mz(); ++mz) {
                    const int kz = u.kz(mz);
                    BasisFunc prof = u.profile(mx, mz);
                    if (L2Norm(prof) > eps) {
                        kxmax = Greater(kxmax, abs(kx));
                        kzmax = Greater(kzmax, abs(kz));
                    }
                    for (int ky = 0; ky < u.Ny(); ++ky) {
                        Real sum = 0.0;
                        for (int i = 0; i < u.Nd(); ++i)
                            sum += abs(prof[i][ky]);
                        if (sum > eps)
                            kymax = Greater(kymax, ky);
                    }
                }
            }
            cout << "u.padded() == " << (u.padded() ? "true" : "false") << endl;
            cout << "Energy over " << eps << " is confined to : \n";
            cout << " |kx| <= " << kxmax << endl;
            cout << " |ky| <= " << kymax << endl;
            cout << " |kz| <= " << kzmax << endl;

            cout << "Minimum   aliased grid : " << 2 * (kxmax + 1) << " x " << kymax + 1 << " x " << 2 * (kzmax + 1)
                 << endl;

            cout << "Minimum dealiased grid : " << 3 * (kxmax + 1) << " x " << kymax + 1 << " x " << 3 * (kzmax + 1)
                 << endl;

            int kxmin = u.padded() ? u.kxminDealiased() : u.kxmin();
            kxmax = u.padded() ? u.kxmaxDealiased() : u.kxmax();
            kzmax = u.padded() ? u.kzmaxDealiased() : u.kzmax();

            int mzmax = u.mz(kzmax);
            int mymax = u.My() - 1;

            int kxtrunc = 0;
            int kztrunc = 0;
            Real truncation = 0.0;

            for (int kx = kxmin; kx <= kxmax; kx += (kxmax - kxmin)) {
                int mx = u.mx(kx);
                for (int mz = 0; mz <= mzmax; ++mz) {
                    BasisFunc prof = u.profile(mx, mz);
                    Real trunc = L2Norm(prof);
                    if (trunc > truncation) {
                        kxtrunc = kx;
                        kztrunc = u.kz(mz);
                        truncation = trunc;
                    }
                }
            }
            cout << "Max x truncation == " << setw(12) << truncation << " at kx,kz == " << kxtrunc << ',' << kztrunc
                 << endl;

            truncation = 0.0;
            kxtrunc = 0;
            kztrunc = 0;

            for (int i = 0; i < u.Nd(); ++i) {
                for (int kx = kxmin; kx <= kxmax; ++kx) {
                    int mx = u.mx(kx);
                    for (int kz = 0; kz <= kzmax; ++kz) {
                        int mz = u.mz(kz);
                        Real trunc = abs(u.cmplx(mx, mymax, mz, i));
                        if (trunc > truncation) {
                            kxtrunc = kx;
                            kztrunc = kz;
                            truncation = trunc;
                        }
                    }
                }
            }
            cout << "Max y truncation == " << setw(12) << truncation << " at kx,kz == " << kxtrunc << ',' << kzmax
                 << endl;

            truncation = 0.0;
            kxtrunc = 0;
            kztrunc = 0;
            for (int kx = kxmin; kx <= kxmax; ++kx) {
                int mx = u.mx(kx);
                BasisFunc prof = u.profile(mx, mzmax);
                Real trunc = L2Norm(prof);
                if (trunc > truncation) {
                    kxtrunc = kx;
                    truncation = trunc;
                }
            }
            cout << "Max z truncation == " << setw(12) << truncation << " at kx,kz == " << kxtrunc << ',' << kzmax
                 << endl;
            cout << "\n" << endl;
        }

        if ((all && u.Nd() > 1) || nrgy) {
            cout << "--------------Energy---------------------" << endl;

            if (u.Nd() == 1) {
                cferror("Energy properties are not available for scalar fields");
            }

            cout << "dissip (u)     == " << dissipation(u) << endl;
            cout << "wallshear(u)   == " << wallshear(u) << endl;
            // cout << "wallshear(u)   == " <<  0.5*(abs(wallshearUpper(u))+abs(wallshearLower(u))) << endl;
            cout << "energy(u)      == " << 0.5 * L2Norm2(u) << endl;

            if (flags.baseflow != ZeroBase) {
                u += Baseflow;
                u.cmplx(0, 0, 0, 1) -= Complex(ubasefac * flags.Vsuck, 0.);
                cout << "dissip (u+U)   == " << dissipation(u) << endl;
                cout << "wallshear(u+U) == " << wallshear(u) << endl;
                cout << "wallshear(u+U) == " << 0.5 * (abs(wallshearUpper(u)) + abs(wallshearLower(u))) << endl;
                cout << "energy (u+U)   == " << 0.5 * L2Norm2(u) << endl;
                cout << "energy3d(u+u)    == " << 0.5 * L2Norm2_3d(u) << endl;
                u -= Baseflow;
                u.cmplx(0, 0, 0, 1) += Complex(ubasefac * flags.Vsuck, 0.);
            }
        }

        if ((all && u.Nd() > 1) || dyn) {
            if (u.Nd() == 1) {
                cferror("Not available for scalar fields");
            }
            cout << "------------Dynamics--------------------" << endl;

            vector<FlowField> fields = {u, FlowField(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b())};
            DNS dns(fields, flags);

            const int N = iround(T / flags.dt + 1);
            cout << "computing du/dt: " << flush;
            dns.advance(fields, N);
            cout << endl;
            fields[0] -= u;
            fields[0] *= 1.0 / (N * flags.dt);
            cout << "L2Norm(u)     == " << L2Norm(u) << endl;
            cout << "L2Norm(du/dt) == " << L2Norm(fields[0]) << endl;

            FlowField dudx;
            FlowField dudz;
            xdiff(u, dudx);
            zdiff(u, dudz);

            Real dudxnorm = L2Norm(dudx);
            Real dudznorm = L2Norm(dudz);

            cout << "L2Norm(du/dx) == " << dudxnorm << endl;
            cout << "L2Norm(du/dz) == " << dudznorm << endl;

            Real ueps = 1e-12;
            dudx *= (dudxnorm < ueps) ? 0.0 : 1.0 / dudxnorm;
            dudz *= (dudznorm < ueps) ? 0.0 : 1.0 / dudznorm;

            Real ax = L2IP(fields[0], dudx);
            Real az = L2IP(fields[0], dudz);

            FlowField Pu(u);
            Pu.setToZero();

            FlowField utmp;
            utmp = dudx;
            utmp *= ax;
            Pu += utmp;

            utmp = dudz;
            utmp *= az;
            Pu += utmp;

            Real Punorm = L2Norm(Pu);
            Real waviness = Punorm / L2Norm(fields[0]);
            Real theta = acos(L2IP(Pu, fields[0]) / (L2Norm(fields[0]) * Punorm));

            cout << endl;
            cout << " waviness == " << waviness << endl;
            cout << "    theta == " << theta << " == " << theta * 180 / pi << " degrees\n";
            cout << endl;
            cout << "(waviness == fraction of dudt within dudx,dudz tangent plane)\n";
            cout << "(   theta == angle between dudt and dudx,dudz tangent plane)\n";
            cout << endl;
        }

        if ((all && u.Nd() > 1) || wall) {
            if (u.Nd() == 1) {
                cferror("Not available for scalar fields");
            }

            if (flags.baseflow == ZeroBase)
                cferror(
                    "to display wall properties provide the base flow, either a file or DNS flags to construct it.");

            else {
                TurbStats stats(Ubase, flags.nu);
                FlowField tmp(u.Nx(), u.Ny(), u.Nz(), 9, u.Lx(), u.Lz(), u.a(), u.b());
                stats.addData(u, tmp);

                cout << "ustar == " << stats.ustar() << endl;
                cout << "hplus == " << stats.hplus() << endl;
                cout << "saving Reynolds stresses etc to turbstats.asc" << endl;
                stats.msave("turbstats");
            }
        }

        if ((all && u.Nd() > 1) || local) {
            if (u.Nd() == 1) {
                cferror("Not available for scalar fields");
            }

            FlowField e = energy(u);
            e.makeSpectral();

            PeriodicFunc exyavg(e.Nz(), e.Lz(), Spectral);
            for (int mz = 0; mz < e.Mz(); ++mz)
                exyavg.cmplx(mz) = e.cmplx(0, 0, mz, 0);
            exyavg.makePhysical();

            Real emin = exyavg(0);
            Real emax = emin;

            for (int nz = 1; nz < e.Nz(); ++nz) {
                emin = lesser(emin, exyavg(nz));
                emax = Greater(emax, exyavg(nz));
            }
            cout << "sqrt(max(exyavg(z))/min(exyavg(z))) == " << sqrt(emin / emax) << endl;
        }

        return 0;
    }
    cfMPI_Finalize();
}

Real energy(FlowField& u, int i, bool normalize) {
    assert(u.ystate() == Spectral);
    assert(u.xzstate() == Spectral);
    Real sum = 0.0;
    ComplexChebyCoeff prof(u.Ny(), u.a(), u.b(), Spectral);
    for (int mx = 0; mx < u.Mx(); ++mx) {
        Real cz = 1.0;  // cz = 2 for kz>0 to take account of kz<0 ghost modes
        for (int mz = 0; mz < u.Mz(); ++mz) {
            for (int ny = 0; ny < u.Ny(); ++ny)
                prof.set(ny, u.cmplx(mx, ny, mz, i));
            sum += cz * L2Norm2(prof, normalize);
            cz = 2.0;
        }
    }
    if (!normalize)
        sum *= u.Lx() * u.Lz();
    return sum;
}

char symmetric(const FieldSymmetry& s, const FlowField& u, Real& serr, Real& aerr, Real eps) {
    Real sqrteps = sqrt(eps);

    FlowField su = s(u);
    FlowField us, ua;
    ((us = u) += su) *= 0.5;
    ((ua = u) -= su) *= 0.5;
    serr = L2Dist(us, u);
    aerr = L2Dist(ua, u);

    // Check for symmetry
    if (serr < eps)
        return 'S';
    else if (serr < sqrteps)
        return 's';
    else if (aerr < eps)
        return 'A';
    else if (aerr < sqrteps)
        return 'a';
    else
        return ' ';
}

Real project(const FlowField& u, const FieldSymmetry& s, int sign) {
    FlowField Pu(u);
    if (sign < 0)
        Pu -= s(u);
    else
        Pu += s(u);
    // Pu *= 0.5;
    return 0.25 * L2Norm2(Pu);
}
