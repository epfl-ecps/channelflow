/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include <iostream>
#include "channelflow/laurettedsi.h"

using namespace std;
using namespace chflow;

Real test_linearization(FlowField& u, DNSFlags flags) {
    cout << "This test checks that the linearized operator eulerdns.advance(u, p, 1, U) which\n"
            "computes N_U (u) = u dotgrad U + U dotgrad u as the nonliner term is actually linear.\n"
            "With the notation that B(u(t)) = u(t+dt) and B_U(u) is the linearized operator,\n"
            "this function checks that\n"
            "(B-I)(u+U) = (B-I)U + (B_U - I)u + [(B-I) u - (B_0 - I)u]\n"
            "where (B-I)u = u(t+dt) - u(t) is computed with one iteration of eulerdns.advance  with a large dt\n\n";

    ChebyCoeff Ubase = laminarProfile(flags.nu, flags.constraint, flags.dPdx, flags.Ubulk, flags.Vsuck, u.a(), u.b(),
                                      flags.ulowerwall, flags.uupperwall, u.Ny());
    ChebyCoeff Wbase = laminarProfile(flags.nu, flags.constraint, flags.dPdz, flags.Wbulk, flags.Vsuck, u.a(), u.b(),
                                      flags.wlowerwall, flags.wupperwall, u.Ny());
    flags.timestepping = FEBE;
    flags.nonlinearity = Rotational;
    EulerDNS alg(u, Ubase, Wbase, flags);
    FlowField p = FlowField(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());

    FlowField U(u);
    U.perturb(0.1, .1);

    cout << "L2Norm(u)           = " << L2Norm(u) << endl;
    cout << "L2Norm(U)           = " << L2Norm(U) << endl;
    cout << "L2Dist(u,U)         = " << L2Dist(u, U) << endl;

    FlowField Uu = U;
    Uu += u;

    FlowField null(u);
    null.setToZero();

    Real Cx = 0.2;
    Real cx = 0.1;

    cout << endl << "Now computing individual terms" << endl;
    FlowField B_I_Uu(Uu);
    alg.advance(B_I_Uu, p, 1, null, false, Cx + cx, 0, false);
    B_I_Uu -= Uu;
    cout << "L2Norm(B_I_Uu)      = " << L2Norm(B_I_Uu) << endl;

    FlowField B_I_U(U);
    alg.advance(B_I_U, p, 1, null, false, Cx, 0, false);
    B_I_U -= U;
    cout << "L2Norm(B_I_U)       = " << L2Norm(B_I_U) << endl;

    FlowField BU_I_u(u);
    alg.advance(BU_I_u, p, 1, U, true, Cx, cx, false);
    BU_I_u -= u;
    cout << "L2Norm(BU_I_u)      = " << L2Norm(BU_I_u) << endl;

    FlowField B_I_u(u);
    alg.advance(B_I_u, p, 1, null, false, cx, 0, false);
    B_I_u -= u;
    cout << "L2Norm(B_I_u)       = " << L2Norm(B_I_u) << endl;

    FlowField L_u(u);
    alg.advance(L_u, p, 1, null, true, 0, 0, false);
    L_u -= u;
    cout << "L2Norm(L_u)         = " << L2Norm(L_u) << endl;

    FlowField lhs(B_I_U);
    lhs += BU_I_u;
    lhs += B_I_u;
    lhs -= L_u;

    cout << endl << "lhs = BU_I_u + B_I_u - L_u" << endl;
    //   sum += dudx;

    //   sum -= B_I_Uu;
    cout << endl << "The final error is" << endl;
    cout << "L2Dist(B_I_Uu, lhs) = " << L2Dist(B_I_Uu, lhs) << endl;

    return L2Dist(B_I_Uu, lhs);
}

int main(int argc, char* argv[]) {
    Real err = 0;
    cfMPI_Init(&argc, &argv);
    {
        CfMPI* cfmpi = new CfMPI();

        const int Nx = 32;
        const int Ny = 33;
        const int Nz = 32;

        const int Nd = 3;
        const Real Lx = 2 * pi;
        const Real Lz = 2 * pi;
        const Real a = -1.0;
        const Real b = 1.0;

        Real Dt = .01;

        FlowField u(Nx, Ny, Nz, Nd, Lx, Lz, a, b, cfmpi);
        FlowField U(Nx, Ny, Nz, Nd, Lx, Lz, a, b, cfmpi);

        DNSFlags flags;
        flags.initstepping = FEBE;
        flags.timestepping = FEBE;
        flags.dt = Dt;
        flags.verbosity = Silent;

        srand48(0);
        u.addPerturbations(.1, .3);

        err = test_linearization(u, flags);

        delete cfmpi;
    }
    cfMPI_Finalize();
    if (err > 1e-14)
        return 1;
    return 0;
}
