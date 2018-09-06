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
#include "channelflow/symmetry.h"
#include "channelflow/utilfuncs.h"

using namespace std;

using namespace chflow;

int main(int argc, char* argv[]) {
    cfMPI_Init(&argc, &argv);
    {
        string purpose("apply a differential operation to a given field");

        ArgList args(argc, argv, purpose);

        // const bool channel  = args.getflag("-c", "--channel", "channelflow instead of plane Couette");
        // const bool couette  = args.getflag("-pcf", "--planecouette", "channelflow instead of plane Couette");

        const bool ddx = args.getflag("-ddx", "--ddx", "apply d/dx");
        const bool ddy = args.getflag("-ddy", "--ddy", "apply d/dy");
        const bool ddz = args.getflag("-ddz", "--ddz", "apply d/dz");
        const bool grd = args.getflag("-grad", "--gradient", "apply gradient");
        const bool lpl = args.getflag("-lapl", "--laplacian", "apply laplacian");
        const bool crl = args.getflag("-curl", "--curl", "apply curl");
        const bool dvv = args.getflag("-div", "--divergence", "compute divergence");
        const bool nnl = args.getflag("-nonl", "--nonlinearity", "compute Navier-Stokes nonlinearity");
        const bool nrg = args.getflag("-e", "--energy", "compute energy operator");
        const bool qcr = args.getflag("-Q", "--Qcriterion", "compute Q criterion");
        const bool nrm = args.getflag("-norm", "--norm", "compute pointwise vector norm of field");
        const bool xavg = args.getflag("-xavg", "--xaverage", "compute the streamwise average");
        const string uname = args.getstr(2, "<flowfield>", "input field");
        const string outname = args.getstr(1, "<flowfield>", "filename for output field");

        string Uname, Wname;
        DNSFlags baseflags = setBaseFlowFlags(args, Uname, Wname);
        const Real ubasefac = args.getreal("-Uf", "--Uf", 1, "Multiply baseflow by this factor before adding");
        args.check();

        FlowField u(uname);
        u.makeSpectral();

        vector<ChebyCoeff> base_Flow = baseFlow(u.Ny(), u.a(), u.b(), baseflags, Uname, Wname);
        for (int i = 0; i < 2; i++) {
            base_Flow[i].makePhysical();
            base_Flow[i] *= ubasefac;
            base_Flow[i].makeSpectral();
        }
        u += base_Flow;
        u.cmplx(0, 0, 0, 1) -= Complex(ubasefac * baseflags.Vsuck, 0.);

        FlowField v;

        // Note: could also use v = grad(u) etc, but grad(u,v) is more efficient

        if (ddx)
            xdiff(u, v);
        else if (ddy)
            ydiff(u, v);
        else if (ddz)
            zdiff(u, v);
        else if (grd)
            grad(u, v);
        else if (lpl)
            lapl(u, v);
        else if (crl)
            curl(u, v);
        else if (dvv)
            div(u, v);
        else if (nnl) {
            FlowField tmp(u);
            convectionNL(u, v, tmp);
        } else if (qcr)
            Qcriterion(u, v);
        else if (nrg)
            energy(u, v);
        else if (nrm)
            norm(u, v);
        else if (xavg) {
            FlowField uxavg(1, u.Ny(), u.Nz(), u.Nd(), u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi(), Spectral, Spectral);
            for (int i = 0; i < u.Nd(); ++i)
                for (int my = 0; my < u.My(); ++my)
                    for (int mz = 0; mz < u.Mz(); ++mz)
                        uxavg.cmplx(0, my, mz, i) = u.cmplx(0, my, mz, i);
            uxavg.save(outname + "_xavg");
        } else
            cferror("diffop: please choose a differential operator, rerun with -h option");

        if (!xavg) {
            v.makeSpectral();
            v.save(outname);
        }
    }
    cfMPI_Finalize();
    return 0;
}
