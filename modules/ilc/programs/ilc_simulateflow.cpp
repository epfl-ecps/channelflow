/**
 * DNS program, like the program for pure shear flows.
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 *
 * Original author: Florian Reetz
 */

#include <sys/stat.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "cfbasics/cfvector.h"
#include "cfbasics/mathdefs.h"
#include "channelflow/chebyshev.h"
#include "channelflow/dns.h"
#include "channelflow/flowfield.h"
#include "channelflow/poissonsolver.h"
#include "channelflow/symmetry.h"
#include "channelflow/tausolver.h"
#include "channelflow/utilfuncs.h"
#include "modules/ilc/ilcdsi.h"

using namespace std;
using namespace chflow;

string printdiagnostics(FlowField& u, const DNS& dns, Real t, const TimeStep& dt, Real nu, Real umin, bool vardt,
                        bool pl2norm, bool pchnorm, bool pdissip, bool pshear, bool pdiverge, bool pUbulk, bool pubulk,
                        bool pdPdx, bool pcfl);

int main(int argc, char* argv[]) {
    cfMPI_Init(&argc, &argv);
    {
        WriteProcessInfo(argc, argv);
        string purpose(
            "integrate inclined layer convection (ILC) in the channel flow domain from a given "
            "initial condition and save velocity and temperature fields to disk.");

        ArgList args(argc, argv, purpose);

        ILCFlags flags(args);
        TimeStep dt(flags);

        args.section("Program options");
        const string outdir = args.getpath("-o", "--outdir", "data/", "output directory");
        const string ulabel = args.getstr("-ul", "--ulabel", "u", "output velocity field prefix");
        const string tlabel = args.getstr("-tl", "--tlabel", "t", "output temperature field prefix");
        const bool savep = args.getflag("-sp", "--savepressure", "save pressure fields");

        const bool pcfl = args.getflag("-cfl", "--cfl", "print CFL number each dT");
        const bool pl2norm = args.getflag("-l2", "--l2norm", "print L2Norm(u) each dT");
        const bool pchnorm = args.getbool("-ch", "--chebyNorm", true, "print chebyNorm(u) each dT");
        const bool pdissip = args.getflag("-D", "--dissipation", "print dissipation each dT");
        const bool pshear = args.getflag("-I", "--input", "print wall shear power input each dT");
        const bool pdiverge = args.getflag("-dv", "--divergence", "print divergence each dT");
        const bool pubulk = args.getflag("-u", "--ubulk", "print ubulk each dT");
        const bool pUbulk = args.getflag("-Up", "--Ubulk-print", "print Ubulk each dT");
        const bool pdPdx = args.getflag("-p", "--pressure", "print pressure gradient each dT");
        const Real umin = args.getreal("-u", "--umin", 0.0, "stop if chebyNorm(u) < umin");

        const Real ecfmin = args.getreal("-e", "--ecfmin", 0.0, "stop if Ecf(u) < ecfmin");
        const int saveint = args.getint("-s", "--saveinterval", 1, "save fields every s dT");

        const int nproc0 =
            args.getint("-np0", "--nproc0", 0, "number of MPI-processes for transpose/number of parallel ffts");
        const int nproc1 = args.getint("-np1", "--nproc1", 0, "number of MPI-processes for one fft");
        const string uname = args.getstr(2, "<flowfield>", "initial condition of velocity field");
        const string tname = args.getstr(1, "<flowfield>", "initial condition of temperature field");

        args.check();
        args.save("./");
        mkdir(outdir);
        args.save(outdir);
        flags.save(outdir);

        CfMPI* cfmpi = &CfMPI::getInstance(nproc0, nproc1);

        printout("Constructing u,q, and optimizing FFTW...");
        FlowField u(uname, cfmpi);
        FlowField temp(tname, cfmpi);

        const int Nx = u.Nx();
        const int Ny = u.Ny();
        const int Nz = u.Nz();
        const Real Lx = u.Lx();
        const Real Lz = u.Lz();
        const Real a = u.a();
        const Real b = u.b();

        FlowField q(Nx, Ny, Nz, 1, Lx, Lz, a, b, cfmpi);
        const bool inttime =
            (abs(saveint * dt.dT() - int(saveint * dt.dT())) < 1e-12) && (abs(flags.t0 - int(flags.t0)) < 1e-12)
                ? true
                : false;

        cout << "Uwall == " << flags.uupperwall << endl;
        cout << "Wwall == " << flags.wupperwall << endl;
        cout << "DeltaT == " << flags.tlowerwall - flags.tupperwall << endl;
        cout << "ilcflags == " << flags << endl;
        cout << "constructing ILC DNS..." << endl;
        ILC dns({u, temp, q}, flags);
        //     u.setnu (flags.nu);

        dns.Ubase().save(outdir + "Ubase");
        dns.Wbase().save(outdir + "Wbase");
        dns.Tbase().save(outdir + "Tbase");

        //     ChebyCoeff Ubase =  laminarProfile (flags, u.a(), u.b(), u.Ny());
        // Ubase.save("Ubase2");

        PressureSolver psolver(u, dns.Ubase(), dns.Wbase(), flags.nu, flags.Vsuck,
                               flags.nonlinearity);  // NOT CORRECT FOR ILC
        psolver.solve(q, u);
        vector<FlowField> fields = {u, temp, q};

        ios::openmode openflag = (flags.t0 > 0) ? ios::app : ios::out;

        ofstream eout, x0out;
        openfile(eout, outdir + "energy.asc", openflag);
        eout << ilcfieldstatsheader_t("t", flags) << endl;

        FlowField u0, du, tmp;

        int i = 0;
        for (Real t = flags.t0; t <= flags.T; t += dt.dT()) {
            string s;
            s = printdiagnostics(fields[0], dns, t, dt, flags.nu, umin, dt.variable(), pl2norm, pchnorm, pdissip,
                                 pshear, pdiverge, pUbulk, pubulk, pdPdx, pcfl);
            if (ecfmin > 0 && Ecf(fields[0]) < ecfmin) {
                cferror("Ecf < ecfmin == " + r2s(ecfmin) + ", exiting");
            }

            cout << s;
            s = ilcfieldstats_t(fields[0], fields[1], t, flags);
            eout << s << endl;

            if (saveint != 0 && i % saveint == 0) {
                fields[0].save(outdir + ulabel + t2s(t, inttime));
                fields[1].save(outdir + tlabel + t2s(t, inttime));
                if (savep)
                    fields[1].save(outdir + "p" + t2s(t, inttime));
            }
            i++;

            dns.advance(fields, dt.n());

            if (dt.variable() &&
                dt.adjust(dns.CFL(fields[0])))  // TODO: dt.variable()==true is checked twice here, remove it.
                dns.reset_dt(dt);
        }
        cout << "done!" << endl;
    }
    cfMPI_Finalize();
}

string printdiagnostics(FlowField& u, const DNS& dns, Real t, const TimeStep& dt, Real nu, Real umin, bool vardt,
                        bool pl2norm, bool pchnorm, bool pdissip, bool pshear, bool pdiverge, bool pUbulk, bool pubulk,
                        bool pdPdx, bool pcfl) {
    // Printing diagnostics
    stringstream sout;
    sout << "           t == " << t << endl;
    if (vardt)
        sout << "          dt == " << Real(dt) << endl;
    if (pl2norm)
        sout << "   L2Norm(u) == " << L2Norm(u) << endl;

    if (pchnorm || umin != 0.0) {
        Real chnorm = chebyNorm(u);
        sout << "chebyNorm(u) == " << chnorm << endl;
        if (chnorm < umin) {
            cout << "Exiting: chebyNorm(u) < umin." << endl;
            exit(0);
        }
    }
    Real h = 0.5 * (u.b() - u.a());
    u += dns.Ubase();
    if (pl2norm)
        sout << "   energy(u+U) == " << 0.5 * L2Norm(u) << endl;
    if (pdissip)
        sout << "   dissip(u+U) == " << dissipation(u) << endl;
    if (pshear)
        sout << "wallshear(u+U) == " << abs(wallshearLower(u)) + abs(wallshearUpper(u)) << endl;
    if (pdiverge)
        sout << "  divNorm(u+U) == " << divNorm(u) << endl;
    if (pUbulk)
        sout << "mean u+U Ubulk == " << dns.Ubulk() << endl;
    u -= dns.Ubase();
    if (u.taskid() == u.task_coeff(0, 0)) {
        if (pubulk)
            sout << "         ubulk == " << Re(u.profile(0, 0, 0)).mean() << endl;
    }
    if (pdPdx)
        sout << "          dPdx == " << dns.dPdx() << endl;
    if (pl2norm)
        sout << "     L2Norm(u) == " << L2Norm(u) << endl;
    if (pl2norm)
        sout << "   L2Norm3d(u) == " << L2Norm3d(u) << endl;

    Real cfl = dns.CFL(u);
    if (u.taskid() == u.task_coeff(0, 0)) {
        ChebyCoeff U = dns.Ubase();
        ChebyCoeff W = dns.Wbase();

        U.makeSpectral();
        U += Re(u.profile(0, 0, 0));
        Real Ucenter = U.eval(0.5 * (u.a() + u.b()));
        Real Uwall = pythag(0.5 * (U.eval_b() - U.eval_a()), 0.5 * (W.eval_b() - W.eval_a()));
        Real Umean = U.mean();
        sout << "        1/nu == " << 1 / nu << endl;
        sout << "  Uwall h/nu == " << Uwall * h / nu << endl;
        sout << "  Ubulk h/nu == " << dns.Ubulk() * h / nu << endl;
        sout << "  Umean h/nu == " << Umean << " * " << h << " / " << nu << endl;
        sout << "  Umean h/nu == " << Umean * h / nu << endl;
        sout << " Uparab h/nu == " << 1.5 * dns.Ubulk() * h / nu << endl;
        sout << "Ucenter h/nu == " << Ucenter * h / nu << endl;
    }
    sout << "         CFL == " << cfl << endl;
    return sout.str();
}
