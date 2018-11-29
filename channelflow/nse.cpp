/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "channelflow/nse.h"

using namespace std;

namespace chflow {

void navierstokesNL(const FlowField& u_, ChebyCoeff Ubase, ChebyCoeff Wbase, FlowField& f, FlowField& tmp,
                    DNSFlags& flags) {
    FlowField& u = const_cast<FlowField&>(u_);

    fieldstate finalstate = Spectral;
    assert(u.xzstate() == Spectral && u.ystate() == Spectral);
    assert(Ubase.state() == Spectral);
    assert(Wbase.state() == Spectral);
    if (flags.rotation != 0.0) {
        finalstate = Physical;
    }

    if (flags.nonlinearity == LinearAboutProfile)
        linearizedNL(u, Ubase, Wbase, f, finalstate);
    else {
        // u += Ubase;
        for (int ny = 0; ny < u.Ny(); ++ny) {
            if (u.taskid() == u.task_coeff(0, 0))
                u.cmplx(0, ny, 0, 0) += Complex(Ubase(ny), 0.0);
            if (u.taskid() == u.task_coeff(0, 0))
                u.cmplx(0, ny, 0, 2) += Complex(Wbase(ny), 0.0);
        }
        if (u.taskid() == u.task_coeff(0, 0)) {
            u.cmplx(0, 0, 0, 1) -= Complex(flags.Vsuck, 0.);
        }
        switch (flags.nonlinearity) {
            case Rotational:
                rotationalNL(u, f, tmp, finalstate);
                break;
            case Convection:
                convectionNL(u, f, tmp, finalstate);
                break;
            case SkewSymmetric:
                skewsymmetricNL(u, f, tmp, finalstate);
                break;
            case Divergence:
                divergenceNL(u, f, tmp, finalstate);
                break;
            case Alternating:
                divergenceNL(u, f, tmp, finalstate);
                flags.nonlinearity = Alternating_;
                break;
            case Alternating_:
                convectionNL(u, f, tmp, finalstate);
                flags.nonlinearity = Alternating;
                break;
            default:
                cferror("navierstokesNL(method, u,U,f,tmp) : unknown method");
        }

        // add rotation term -(omega x u) = - flags.rotation * flags.nu (y e_y + e_z x u)
        if (flags.rotation != 0.0) {
            u.makePhysical();
            lint Nz = f.Nz();
            lint nxlocmin = f.nxlocmin();
            lint nxlocmax = f.nxlocmin() + f.Nxloc();
            lint nylocmin = f.nylocmin();
            lint nylocmax = f.nylocmax();
            for (lint ny = nylocmin; ny < nylocmax; ++ny)
                for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
                    for (lint nz = 0; nz < Nz; ++nz) {
                        f(nx, ny, nz, 0) -= (flags.rotation) * u(nx, ny, nz, 1);
                        f(nx, ny, nz, 1) += (flags.rotation) * u(nx, ny, nz, 0);
                    }

            u.makeSpectral();
            f.makeSpectral();
        }
        //     u -= Ubase;
        for (int ny = 0; ny < u.Ny(); ++ny) {
            if (u.taskid() == u.task_coeff(0, 0))
                u.cmplx(0, ny, 0, 0) -= Complex(Ubase(ny), 0.0);
            if (u.taskid() == u.task_coeff(0, 0))
                u.cmplx(0, ny, 0, 2) -= Complex(Wbase(ny), 0.0);
        }
        if (u.taskid() == u.task_coeff(0, 0))
            u.cmplx(0, 0, 0, 1) += Complex(flags.Vsuck, 0.);
    }
    u.makeSpectral();
}

NSE::NSE()
    : lambda_t_(0),
      tausolver_(0),
      flags_(),
      taskid_(0),
      // Spatial parameter members
      nxlocmin_(0),
      Nxloc_(0),
      nylocmin_(0),
      nylocmax_(0),
      Nz_(0),
      mxlocmin_(0),
      Mxloc_(0),
      My_(0),
      mzlocmin_(0),
      Mzloc_(0),
      Nyd_(0),
      kxd_max_(0),
      kzd_max_(0),
      Lx_(0),
      Lz_(0),
      a_(0),
      b_(0),
      kxmax_(0),
      kzmax_(0),
      kxloc_(0),
      kzloc_(0),
      // Base flow members
      dPdxRef_(0),
      dPdxAct_(0),
      dPdzRef_(0),
      dPdzAct_(0),
      UbulkRef_(0),
      UbulkAct_(0),
      UbulkBase_(0),
      WbulkRef_(0),
      WbulkAct_(0),
      WbulkBase_(0),
      Ubase_(),
      Ubaseyy_(),
      Wbase_(),
      Wbaseyy_(),
      // Memspace members
      tmp_(),
      uk_(),
      vk_(),
      wk_(),
      Pk_(),
      Pyk_(),
      Ruk_(),
      Rvk_(),
      Rwk_() {}

NSE::NSE(const NSE& nse)
    : lambda_t_(nse.lambda_t_),
      tausolver_(0),
      flags_(nse.flags_),
      taskid_(nse.taskid()),
      // Spatial parameter members
      nxlocmin_(nse.nxlocmin_),
      Nxloc_(nse.Nxloc_),
      nylocmin_(nse.nylocmin_),
      nylocmax_(nse.nylocmax_),
      Nz_(nse.Nz_),
      mxlocmin_(nse.mxlocmin_),
      Mxloc_(nse.Mxloc_),
      My_(nse.My_),
      mzlocmin_(nse.mzlocmin_),
      Mzloc_(nse.Mzloc_),
      Nyd_(nse.Nyd_),
      kxd_max_(nse.kxd_max_),
      kzd_max_(nse.kzd_max_),
      Lx_(nse.Lx_),
      Lz_(nse.Lz_),
      a_(nse.a_),
      b_(nse.b_),
      kxmax_(nse.kxmax_),
      kzmax_(nse.kzmax_),
      kxloc_(nse.kxloc_),
      kzloc_(nse.kzloc_),
      // Base flow members
      dPdxRef_(nse.dPdxRef_),
      dPdxAct_(nse.dPdxAct_),
      dPdzRef_(nse.dPdzRef_),
      dPdzAct_(nse.dPdzAct_),
      UbulkRef_(nse.UbulkRef_),
      UbulkAct_(nse.UbulkAct_),
      UbulkBase_(nse.UbulkBase_),
      WbulkRef_(nse.WbulkRef_),
      WbulkAct_(nse.WbulkAct_),
      WbulkBase_(nse.WbulkBase_),
      Ubase_(nse.Ubase_),
      Ubaseyy_(nse.Ubaseyy_),
      Wbase_(nse.Wbase_),
      Wbaseyy_(nse.Wbaseyy_),
      // Memspace members
      tmp_(nse.tmp_),
      uk_(nse.uk_),
      vk_(nse.vk_),
      wk_(nse.wk_),
      Pk_(nse.Pk_),
      Pyk_(nse.Pyk_),
      Ruk_(nse.Ruk_),
      Rvk_(nse.Rvk_),
      Rwk_(nse.Rwk_) {
    // Allocate memory for [Nsubsteps x Mx_ x Mz_] Tausolver cfarrays
    // and copy tausolvers from nse argument
    int nsub = lambda_t_.size();
    tausolver_ = new TauSolver**[nsub];  // new #1
    for (int j = 0; j < nsub; ++j) {
        tausolver_[j] = new TauSolver*[Mxloc_];  // new #2
        for (int mx = 0; mx < Mxloc_; ++mx) {
            tausolver_[j][mx] = new TauSolver[Mzloc_];  // new #3
            for (int mz = 0; mz < Mzloc_; ++mz)
                tausolver_[j][mx][mz] = nse.tausolver_[j][mx][mz];
        }
    }
}

NSE::NSE(const vector<FlowField>& fields, const DNSFlags& flags)
    : lambda_t_(0),
      tausolver_(0),  // tausolvers are allocated when reset_lambda is called for the first time
      flags_(flags),
      taskid_(fields[0].taskid()),
      // Spatial parameter members
      nxlocmin_(fields[0].nxlocmin()),
      Nxloc_(fields[0].Nxloc()),
      nylocmin_(fields[0].nylocmin()),
      nylocmax_(fields[0].nylocmax()),
      Nz_(fields[0].Nz()),
      mxlocmin_(fields[0].mxlocmin()),
      Mxloc_(fields[0].Mxloc()),
      My_(fields[0].Ny()),
      mzlocmin_(fields[0].mzlocmin()),
      Mzloc_(fields[0].Mzloc()),
      Nyd_(flags.dealias_y() ? 2 * (fields[0].numYmodes() - 1) / 3 + 1 : fields[0].numYmodes()),
      kxd_max_(flags.dealias_xz() ? fields[0].Nx() / 3 - 1 : fields[0].kxmax()),
      kzd_max_(flags.dealias_xz() ? fields[0].Nz() / 3 - 1 : fields[0].kzmax()),
      Lx_(fields[0].Lx()),
      Lz_(fields[0].Lz()),
      a_(fields[0].a()),
      b_(fields[0].b()),
      kxmax_(fields[0].kxmax()),
      kzmax_(fields[0].kzmax()),
      kxloc_(0),
      kzloc_(0),
      // Base flow members
      dPdxRef_(0),
      dPdxAct_(0),
      dPdzRef_(0),
      dPdzAct_(0),
      UbulkRef_(0),
      UbulkAct_(0),
      UbulkBase_(0),
      WbulkRef_(0),
      WbulkAct_(0),
      WbulkBase_(0),
      Ubase_(),
      Ubaseyy_(),
      Wbase_(),
      Wbaseyy_(),
      // Memspace members
      tmp_(),
      uk_(Nyd_, a_, b_, Spectral),
      vk_(Nyd_, a_, b_, Spectral),
      wk_(Nyd_, a_, b_, Spectral),
      Pk_(Nyd_, a_, b_, Spectral),
      Pyk_(Nyd_, a_, b_, Spectral),
      Ruk_(Nyd_, a_, b_, Spectral),
      Rvk_(Nyd_, a_, b_, Spectral),
      Rwk_(Nyd_, a_, b_, Spectral) {
    assert(fields[0].vectorDim() == 3);

    // construct wave number vectors
    kxloc_.resize(Mxloc_);
    kzloc_.resize(Mzloc_);
    for (int mx = 0; mx < Mxloc_; ++mx)
        kxloc_[mx] = fields[0].kx(mx + mxlocmin_);
    for (int mz = 0; mz < Mzloc_; ++mz)
        kzloc_[mz] = fields[0].kz(mz + mzlocmin_);

    // These methods require a 9d (3x3) tmp flowfield
    if (flags_.nonlinearity == Alternating || flags_.nonlinearity == Alternating_ ||
        flags_.nonlinearity == Convection || flags_.nonlinearity == LinearAboutProfile ||
        flags_.nonlinearity == Divergence || flags_.nonlinearity == SkewSymmetric) {
        tmp_.resize(fields[0].Nx(), fields[0].Ny(), fields[0].Nz(), 9, fields[0].Lx(), fields[0].Lz(), fields[0].a(),
                    fields[0].b(), fields[0].cfmpi());
    } else
        tmp_.resize(fields[0].Nx(), fields[0].Ny(), fields[0].Nz(), 3, fields[0].Lx(), fields[0].Lz(), fields[0].a(),
                    fields[0].b(), fields[0].cfmpi());

    // set member variables for base flow
    createCFBaseFlow();

    // set member variables for contraint
    initCFConstraint(fields[0]);
}

NSE::NSE(const vector<FlowField>& fields, const vector<ChebyCoeff>& base, const DNSFlags& flags)
    :  //   fields_(fields),
      lambda_t_(0),
      tausolver_(0),  // tausolvers are allocated when reset_lambda is called for the first time
      flags_(flags),
      taskid_(fields[0].taskid()),
      // Spatial parameter members
      nxlocmin_(fields[0].nxlocmin()),
      Nxloc_(fields[0].Nxloc()),
      nylocmin_(fields[0].nylocmin()),
      nylocmax_(fields[0].nylocmax()),
      Nz_(fields[0].Nz()),
      mxlocmin_(fields[0].mxlocmin()),
      Mxloc_(fields[0].Mxloc()),
      My_(fields[0].Ny()),
      mzlocmin_(fields[0].mzlocmin()),
      Mzloc_(fields[0].Mzloc()),
      Nyd_(flags.dealias_y() ? 2 * (fields[0].numYmodes() - 1) / 3 + 1 : fields[0].numYmodes()),
      kxd_max_(flags.dealias_xz() ? fields[0].Nx() / 3 - 1 : fields[0].kxmax()),
      kzd_max_(flags.dealias_xz() ? fields[0].Nz() / 3 - 1 : fields[0].kzmax()),
      Lx_(fields[0].Lx()),
      Lz_(fields[0].Lz()),
      a_(fields[0].a()),
      b_(fields[0].b()),
      kxmax_(fields[0].kxmax()),
      kzmax_(fields[0].kzmax()),
      kxloc_(0),
      kzloc_(0),
      // Base flow members
      dPdxRef_(0),
      dPdxAct_(0),
      dPdzRef_(0),
      dPdzAct_(0),
      UbulkRef_(0),
      UbulkAct_(0),
      UbulkBase_(0),
      WbulkRef_(0),
      WbulkAct_(0),
      WbulkBase_(0),
      Ubase_(base[0]),
      Ubaseyy_(),
      Wbase_(base[1]),
      Wbaseyy_(),
      // Memspace members
      tmp_(),
      uk_(Nyd_, a_, b_, Spectral),
      vk_(Nyd_, a_, b_, Spectral),
      wk_(Nyd_, a_, b_, Spectral),
      Pk_(Nyd_, a_, b_, Spectral),
      Pyk_(Nyd_, a_, b_, Spectral),
      Ruk_(Nyd_, a_, b_, Spectral),
      Rvk_(Nyd_, a_, b_, Spectral),
      Rwk_(Nyd_, a_, b_, Spectral) {
    assert(fields[0].vectorDim() == 3);

    // construct wave number vectors
    kxloc_.resize(Mxloc_);
    kzloc_.resize(Mzloc_);
    for (int mx = 0; mx < Mxloc_; ++mx)
        kxloc_[mx] = fields[0].kx(mx + mxlocmin_);
    for (int mz = 0; mz < Mzloc_; ++mz)
        kzloc_[mz] = fields[0].kz(mz + mzlocmin_);

    // These methods require a 9d (3x3) tmp flowfield
    if (flags_.nonlinearity == Alternating || flags_.nonlinearity == Alternating_ ||
        flags_.nonlinearity == Convection || flags_.nonlinearity == LinearAboutProfile ||
        flags_.nonlinearity == Divergence || flags_.nonlinearity == SkewSymmetric) {
        tmp_.resize(fields[0].Nx(), fields[0].Ny(), fields[0].Nz(), 9, fields[0].Lx(), fields[0].Lz(), fields[0].a(),
                    fields[0].b(), fields[0].cfmpi());
    } else
        tmp_.resize(fields[0].Nx(), fields[0].Ny(), fields[0].Nz(), 3, fields[0].Lx(), fields[0].Lz(), fields[0].a(),
                    fields[0].b(), fields[0].cfmpi());

    // set member variables for contraint
    initCFConstraint(fields[0]);
}

NSE::~NSE() {
    if (tausolver_) {
        for (uint j = 0; j < lambda_t_.size(); ++j) {
            for (int mx = 0; mx < Mxloc_; ++mx) {
                delete[] tausolver_[j][mx];  // undo new #3
                tausolver_[j][mx] = 0;
            }
            delete[] tausolver_[j];  // undo new #2
            tausolver_[j] = 0;
        }
        delete[] tausolver_;  // undo new #1
        tausolver_ = 0;
    }
}

void NSE::nonlinear(const vector<FlowField>& infields, vector<FlowField>& outfields) {
    // The first entry in vector must be velocity FlowField, only use this entry for NLT calculation.
    // Pressure as second entry in in/outfields is not touched.
    // Dealiasing must be done separately, e.g. by calling nse::solve

    navierstokesNL(infields[0], Ubase_, Wbase_, outfields[0], tmp_, flags_);
    if (flags_.dealias_xz())
        outfields[0].zeroPaddedModes();
}

void NSE::linear(const vector<FlowField>& infields, vector<FlowField>& outfields) {
    // Method takes input fields {u,press} and computes the linear terms for velocity output field {u}

    assert(infields.size() == (outfields.size() + 1));  // Make sure user does not expect a pressure output. Outfields
                                                        // should be created outside NSE with NSE::createRHS()
    const int kxmax = infields[0].kxmax();
    const int kzmax = infields[0].kzmax();

    // Loop over Fourier modes. 2nd derivative and summation of linear term
    // is most sufficient on ComplexChebyCoeff. Therefore, the old loop structure is kept.
    for (lint mx = mxlocmin_; mx < mxlocmin_ + Mxloc_; ++mx) {
        const int kx = infields[0].kx(mx);
        for (lint mz = mzlocmin_; mz < mzlocmin_ + Mzloc_; ++mz) {
            const int kz = infields[0].kz(mz);

            // Skip last and aliased modes
            if ((kx == kxmax || kz == kzmax) || (flags_.dealias_xz() && isAliasedMode(kx, kz)))
                break;

            // Goal is to compute
            // L = nu uj" - kappa2 nu uj - grad qj + C

            // Extract relevant Fourier modes of uj and qj
            for (int ny = 0; ny < Nyd_; ++ny) {
                uk_.set(ny, flags_.nu * infields[0].cmplx(mx, ny, mz, 0));
                vk_.set(ny, flags_.nu * infields[0].cmplx(mx, ny, mz, 1));
                wk_.set(ny, flags_.nu * infields[0].cmplx(mx, ny, mz, 2));
                Pk_.set(ny, infields[1].cmplx(mx, ny, mz, 0));
            }

            // (1) Put nu uj" into in R. (Pyk_ is used as tmp workspace)
            diff2(uk_, Ruk_, Pyk_);
            diff2(vk_, Rvk_, Pyk_);
            diff2(wk_, Rwk_, Pyk_);

            // (2) Put qn' into Pyk (compute y-comp of pressure gradient).
            diff(Pk_, Pyk_);

            // (3) Summation of all derivative terms and assignment to output velocity field.
            const Real kappa2 = 4 * pi * pi * (square(kx / Lx_) + square(kz / Lz_));
            const Complex Dx = infields[0].Dx(mx);
            const Complex Dz = infields[0].Dz(mz);
            for (int ny = 0; ny < Nyd_; ++ny) {
                outfields[0].cmplx(mx, ny, mz, 0) = Ruk_[ny] - kappa2 * uk_[ny] - Dx * Pk_[ny];
                outfields[0].cmplx(mx, ny, mz, 1) = Rvk_[ny] - kappa2 * vk_[ny] - Pyk_[ny];
                outfields[0].cmplx(mx, ny, mz, 2) = Rwk_[ny] - kappa2 * wk_[ny] - Dz * Pk_[ny];
            }

            // (4) Add const. terms
            if (kx == 0 && kz == 0) {
                // L includes const dissipation term of Ubase and Wbase: nu Uyy, nu Wyy
                if (Ubaseyy_.length() > 0)
                    for (int ny = 0; ny < My_; ++ny)
                        outfields[0].cmplx(mx, ny, mz, 0) += Complex(flags_.nu * Ubaseyy_[ny], 0);
                if (Wbaseyy_.length() > 0)
                    for (int ny = 0; ny < My_; ++ny)
                        outfields[0].cmplx(mx, ny, mz, 2) += Complex(flags_.nu * Wbaseyy_[ny], 0);

                // Add base pressure gradient depending on the constraint
                if (flags_.constraint == PressureGradient) {
                    // dPdx is supplied as dPdxRef
                    outfields[0].cmplx(mx, 0, mz, 0) -= Complex(dPdxRef_, 0);
                    outfields[0].cmplx(mx, 0, mz, 2) -= Complex(dPdzRef_, 0);
                } else {  // const bulk velocity
                    // actual dPdx is unknown but defined by constraint of bulk velocity
                    // Determine actual dPdx from Ubase + u.
                    Real Ly = b_ - a_;
                    diff(uk_, Ruk_);
                    diff(wk_, Rwk_);
                    Real dPdxAct = Re(Ruk_.eval_b() - Ruk_.eval_a()) / Ly;
                    Real dPdzAct = Re(Rwk_.eval_b() - Rwk_.eval_a()) / Ly;
                    ChebyCoeff Ubasey = diff(Ubase_);
                    ChebyCoeff Wbasey = diff(Wbase_);
                    if (Ubase_.length() != 0)
                        dPdxAct += flags_.nu * (Ubasey.eval_b() - Ubasey.eval_a()) / Ly;
                    if (Wbase_.length() != 0)
                        dPdzAct += flags_.nu * (Wbasey.eval_b() - Wbasey.eval_a()) / Ly;
                    // add press. gradient to linear term
                    outfields[0].cmplx(mx, 0, mz, 0) -= Complex(dPdxAct, 0);
                    outfields[0].cmplx(mx, 0, mz, 2) -= Complex(dPdzAct, 0);
                }
            }  // End of const. terms
        }
    }  // End of loop over Fourier modes
}

void NSE::solve(vector<FlowField>& outfields, const vector<FlowField>& rhs, const int s) {
    // Method takes a right hand side {u} and solves for output fields {u,press}

    assert(outfields.size() ==
           (rhs.size() +
            1));  // Make sure user provides correct RHS which can be created outside NSE with NSE::createRHS()
    const int kxmax = outfields[0].kxmax();
    const int kzmax = outfields[0].kzmax();

    // Update each Fourier mode with solution of the implicit problem
    for (lint ix = 0; ix < Mxloc_; ++ix) {
        const lint mx = ix + mxlocmin_;
        const int kx = outfields[0].kx(mx);

        for (lint iz = 0; iz < Mzloc_; ++iz) {
            const lint mz = iz + mzlocmin_;
            const int kz = outfields[0].kz(mz);

            // Skip last and aliased modes
            if ((kx == kxmax || kz == kzmax) || (flags_.dealias_xz() && isAliasedMode(kx, kz)))
                break;

            // Construct ComplexChebyCoeff
            for (int ny = 0; ny < Nyd_; ++ny) {
                Ruk_.set(ny, rhs[0].cmplx(mx, ny, mz, 0));
                Rvk_.set(ny, rhs[0].cmplx(mx, ny, mz, 1));
                Rwk_.set(ny, rhs[0].cmplx(mx, ny, mz, 2));
            }

            // Solve the tau equations
            if (kx != 0 || kz != 0)
                tausolver_[s][ix][iz].solve(uk_, vk_, wk_, Pk_, Ruk_, Rvk_, Rwk_);
            // 		solve(ix,iz,uk_,vk_,wk_,Pk_, Ruk_,Rvk_,Rwk_);
            else {  // kx,kz == 0,0
                // LHS includes also the constant terms C which can be added to RHS
                if (Ubaseyy_.length() > 0)
                    for (int ny = 0; ny < My_; ++ny)
                        Ruk_.re[ny] += flags_.nu * Ubaseyy_[ny];  // Rx has addl'l term from Ubase
                if (Wbaseyy_.length() > 0)
                    for (int ny = 0; ny < My_; ++ny)
                        Rwk_.re[ny] += flags_.nu * Wbaseyy_[ny];  // Rz has addl'l term from Wbase

                if (flags_.constraint == PressureGradient) {
                    // pressure is supplied, put on RHS of tau eqn
                    Ruk_.re[0] -= dPdxRef_;
                    Rwk_.re[0] -= dPdzRef_;
                    tausolver_[s][ix][iz].solve(uk_, vk_, wk_, Pk_, Ruk_, Rvk_, Rwk_);
                    // 	  	solve(ix,iz,uk_, vk_, wk_, Pk_, Ruk_,Rvk_,Rwk_);
                    // Bulk vel is free variable determined from soln of tau eqn //TODO: write method that computes
                    // UbulkAct everytime it is needed

                } else {  // const bulk velocity
                    // bulk velocity is supplied, use alternative tau solver

                    // Use tausolver with additional variable and constraint:
                    // free variable: dPdxAct at next time-step,
                    // constraint:    UbulkBase + mean(u) = UbulkRef.
                    tausolver_[s][ix][iz].solve(uk_, vk_, wk_, Pk_, dPdxAct_, dPdzAct_, Ruk_, Rvk_, Rwk_,
                                                UbulkRef_ - UbulkBase_, WbulkRef_ - WbulkBase_);
                    // 		  solve(ix,iz,uk_, vk_, wk_, Pk_, dPdxAct_, dPdzAct_,
                    // 					    Ruk_, Rvk_, Rwk_,
                    // 					    UbulkRef_ - UbulkBase_,
                    // 					    WbulkRef_ - WbulkBase_);

                    assert((UbulkRef_ - UbulkBase_ - uk_.re.mean()) <
                           1e-15);  // test if UbulkRef == UbulkAct = UbulkBase_ + uk_.re.mean()
                    assert((WbulkRef_ - WbulkBase_ - wk_.re.mean()) <
                           1e-15);  // test if WbulkRef == WbulkAct = WbulkBase_ + wk_.re.mean()
                }
            }
            // Load solutions into u and p.
            // Because of FFTW complex symmetries
            // The 0,0 mode must be real.
            // For Nx even, the kxmax,0 mode must be real
            // For Nz even, the 0,kzmax mode must be real
            // For Nx,Nz even, the kxmax,kzmax mode must be real
            if ((kx == 0 && kz == 0) || (outfields[0].Nx() % 2 == 0 && kx == kxmax && kz == 0) ||
                (outfields[0].Nz() % 2 == 0 && kz == kzmax && kx == 0) ||
                (outfields[0].Nx() % 2 == 0 && outfields[0].Nz() % 2 == 0 && kx == kxmax && kz == kzmax)) {
                for (int ny = 0; ny < Nyd_; ++ny) {
                    outfields[0].cmplx(mx, ny, mz, 0) = Complex(Re(uk_[ny]), 0.0);
                    outfields[0].cmplx(mx, ny, mz, 1) = Complex(Re(vk_[ny]), 0.0);
                    outfields[0].cmplx(mx, ny, mz, 2) = Complex(Re(wk_[ny]), 0.0);
                    outfields[1].cmplx(mx, ny, mz, 0) = Complex(Re(Pk_[ny]), 0.0);
                }
            }
            // The normal case, for general kx,kz
            else
                for (int ny = 0; ny < Nyd_; ++ny) {
                    outfields[0].cmplx(mx, ny, mz, 0) = uk_[ny];
                    outfields[0].cmplx(mx, ny, mz, 1) = vk_[ny];
                    outfields[0].cmplx(mx, ny, mz, 2) = wk_[ny];
                    outfields[1].cmplx(mx, ny, mz, 0) = Pk_[ny];
                }
        }
    }
}

vector<FlowField> NSE::createRHS(const vector<FlowField>& fields) const { return {fields[0]}; }

vector<cfarray<FieldSymmetry>> NSE::createSymmVec() const {
    // velocity symmetry
    cfarray<FieldSymmetry> usym = SymmetryList(1);  // unity operation
    if (flags_.symmetries.length() > 0)
        usym = flags_.symmetries;
    // pressure symmetry
    cfarray<FieldSymmetry> psym = SymmetryList(1);  // unity operation
    return {usym, psym};
}

void NSE::createCFBaseFlow() {
    Real ulowerwall = flags_.ulowerwall;
    Real uupperwall = flags_.uupperwall;
    Real wlowerwall = flags_.wlowerwall;
    Real wupperwall = flags_.wupperwall;
    switch (flags_.baseflow) {
        case ZeroBase:
            Ubase_ = ChebyCoeff(My_, a_, b_, Spectral);
            Wbase_ = ChebyCoeff(My_, a_, b_, Spectral);
            break;
        case LinearBase:
            Ubase_ = ChebyCoeff(My_, a_, b_, Spectral);
            Wbase_ = ChebyCoeff(My_, a_, b_, Spectral);
            Ubase_[1] = 1;
            break;
        case ParabolicBase:
            Ubase_ = ChebyCoeff(My_, a_, b_, Spectral);
            Wbase_ = ChebyCoeff(My_, a_, b_, Spectral);
            Ubase_[0] = 0.5;
            Ubase_[2] = -0.5;
            break;
        case SuctionBase:
            Ubase_ = laminarProfile(flags_.nu, PressureGradient, 0, flags_.Ubulk, flags_.Vsuck, a_, b_, -0.5, 0.5, My_);
            Wbase_ = ChebyCoeff(My_, a_, b_, Spectral);
            break;
        case LaminarBase:
            Ubase_ = laminarProfile(flags_.nu, flags_.constraint, flags_.dPdx, flags_.Ubulk, flags_.Vsuck, a_, b_,
                                    ulowerwall, uupperwall, My_);

            Wbase_ = laminarProfile(flags_.nu, flags_.constraint, flags_.dPdz, flags_.Wbulk, flags_.Vsuck, a_, b_,
                                    wlowerwall, wupperwall, My_);
            break;
        case ArbitraryBase:
            cerr << "error in NSE::createBaseFlow :\n";
            cerr << "flags.baseflow is ArbitraryBase.\n";
            cerr << "Please provide {Ubase, Wbase} when constructing DNS.\n";
            cferror("");
        default:
            cerr << "error in NSE::createBaseFlow :\n";
            cerr << "flags.baseflow should be ZeroBase, LinearBase, ParabolicBase, LaminarBase, SuctionBase.\n";
            cerr << "Other cases require use of the DNS::DNS(fields, base, flags) constructor.\n";
            cferror("");
    }
}

void NSE::initCFConstraint(const FlowField& u) {
    // Calculate Ubaseyy_ and related quantities
    UbulkBase_ = Ubase_.mean();
    ChebyCoeff Ubasey = diff(Ubase_);
    Ubaseyy_ = diff(Ubasey);
    WbulkBase_ = Wbase_.mean();
    ChebyCoeff Wbasey = diff(Wbase_);
    Wbaseyy_ = diff(Wbasey);

    // Determine actual Ubulk and dPdx from initial data Ubase + u.

    UbulkAct_ = UbulkBase_ + getUbulk(u);
    WbulkAct_ = WbulkBase_ + getWbulk(u);
    dPdxAct_ = getdPdx(u, flags_.nu);
    dPdzAct_ = getdPdz(u, flags_.nu);

    if (Ubase_.length() != 0) {
        FlowField utmp(u);
        utmp += Ubase_;
        dPdxAct_ = getdPdx(utmp, flags_.nu);
    }

    if (Wbase_.length() != 0) {
        FlowField wtmp(u);
        wtmp += Wbase_;
        dPdzAct_ = getdPdz(wtmp, flags_.nu);
    }

    if (flags_.constraint == BulkVelocity) {
        UbulkRef_ = flags_.Ubulk;
        WbulkRef_ = flags_.Wbulk;
    } else {
        dPdxAct_ = flags_.dPdx;
        dPdxRef_ = flags_.dPdx;
        dPdzAct_ = flags_.dPdz;
        dPdzRef_ = flags_.dPdz;
    }
}

void NSE::reset_lambda(vector<Real> lambda_t) {
    lambda_t_ = lambda_t;

    if (tausolver_ == 0) {  // TauSolver need to be constructed
        // Allocate memory for [Nsubsteps x Mx_ x Mz_] Tausolver cfarray
        tausolver_ = new TauSolver**[lambda_t.size()];  // new #1
        for (uint j = 0; j < lambda_t.size(); ++j) {
            tausolver_[j] = new TauSolver*[Mxloc_];  // new #2
            for (int mx = 0; mx < Mxloc_; ++mx)
                tausolver_[j][mx] = new TauSolver[Mzloc_];  // new #3
        }
    }

    // Configure tausolvers
    //   FlowField u=fields_[0];
    const Real c = 4.0 * square(pi) * flags_.nu;
    //   const int kxmax = u.kxmax();
    //   const int kzmax = u.kzmax();
    for (uint j = 0; j < lambda_t.size(); ++j) {
        for (int mx = 0; mx < Mxloc_; ++mx) {
            int kx = kxloc_[mx];
            for (int mz = 0; mz < Mzloc_; ++mz) {
                int kz = kzloc_[mz];
                Real lambda = lambda_t[j] + c * (square(kx / Lx_) + square(kz / Lz_));

                if ((kx != kxmax_ || kz != kzmax_) && (!flags_.dealias_xz() || !isAliasedMode(kx, kz)))

                    tausolver_[j][mx][mz] =
                        TauSolver(kx, kz, Lx_, Lz_, a_, b_, lambda, flags_.nu, Nyd_, flags_.taucorrection);
            }
        }
    }
}

/***************************************
void devDNSAlgorithm::reset_dPdx(Real dPdx) {
flags_.constraint = PressureGradient;
flags_.dPdx = dPdx;
flags_.Ubulk = 0.0;
dPdxRef_ = dPdx;
UbulkRef_ = 0.0;
}
void devDNSAlgorithm::reset_Ubulk(Real Ubulk) {
flags_.constraint = BulkVelocity;
flags_.Ubulk = Ubulk;
flags_.dPdx = 0.0;
UbulkRef_ = Ubulk;
dPdxRef_ = 0.0;
}
*******************************************/

void NSE::reset_gradp(Real dPdx, Real dPdz) {
    flags_.constraint = PressureGradient;
    flags_.dPdx = dPdx;
    flags_.dPdz = dPdz;
    flags_.Ubulk = 0.0;
    flags_.Wbulk = 0.0;
    dPdxRef_ = dPdx;
    dPdzRef_ = dPdz;
    UbulkRef_ = 0.0;
    WbulkRef_ = 0.0;
}
void NSE::reset_bulkv(Real Ubulk, Real Wbulk) {
    flags_.constraint = BulkVelocity;
    flags_.Ubulk = Ubulk;
    flags_.Wbulk = Wbulk;
    flags_.dPdx = 0.0;
    flags_.dPdz = 0.0;
    UbulkRef_ = Ubulk;
    WbulkRef_ = Wbulk;
    dPdxRef_ = 0.0;
    dPdzRef_ = 0.0;
}
Real NSE::nu() const { return flags_.nu; }
// int devDNSAlgorithm::Nx() const {
//     return Nx_;
// }
int NSE::Ny() const { return My_; }
// int devDNSAlgorithm::Nz() const {
//     return Nz_;
// }
Real NSE::Lx() const { return Lx_; }
Real NSE::Lz() const { return Lz_; }
Real NSE::a() const { return a_; }
Real NSE::b() const { return b_; }
Real NSE::dPdx() const { return dPdxAct_; }
Real NSE::dPdz() const { return dPdzAct_; }
Real NSE::dPdxRef() const { return dPdxRef_; }
Real NSE::dPdzRef() const { return dPdzRef_; }
Real NSE::Ubulk() const { return UbulkAct_; }
Real NSE::Wbulk() const { return WbulkAct_; }
Real NSE::UbulkRef() const { return UbulkRef_; }
Real NSE::WbulkRef() const { return WbulkRef_; }
const ChebyCoeff& NSE::Ubase() const { return Ubase_; }
const ChebyCoeff& NSE::Wbase() const { return Wbase_; }
int NSE::kxmaxDealiased() const { return kxd_max_; }
int NSE::kzmaxDealiased() const { return kzd_max_; }
bool NSE::isAliasedMode(int kx, int kz) const { return (abs(kx) > kxd_max_ || (abs(kz) > kzd_max_)) ? true : false; }

/////////////////////////////////////////////////////////////////////////////////////////////////////////

Real viscosity(Real Reynolds, VelocityScale vscale, MeanConstraint constraint, Real dPdx, Real Ubulk, Real Uwall,
               Real h) {
    Real nu = 0.0;
    /*****************************************************
    cout << "viscosity(Reynolds, vscale, contraint, dPdx, Ubulk, Uwall, h) : " << endl;
    cout << "  Reynolds == " << Reynolds << endl;
    cout << "    vscale == " << vscale << endl;
    cout << "constraint == " << constraint << endl;
    cout << "      dPdx == " << dPdx << endl;
    cout << "     Ubulk == " << Ubulk<< endl;
    cout << "     Uwall == " << Uwall << endl;
    cout << "         h == " << h << endl;
    *****************************************************/
    if (vscale == WallScale) {
        // cout << "Computing WallScale Reynolds" << endl;
        Real U = fabs(Uwall);
        nu = U * h / Reynolds;
    } else {  // vscale == ParabolicScale
        // cout << "Computing Parabolic Reynolds" << endl;
        if (constraint == PressureGradient) {
            // cout << "...from pressure gradient" << endl;
            // Pressure gradient determines Ubulk, so determine Ucenter and nu as follows
            // U(y) == Ucenter (1-(y/h)^2), so 0 == -dPdx + nu U'' gives
            // Ucenter == -h^2/(2nu) dP/dx and Reynolds == Ucenter h/nu then gives
            // nu == sqrt(h^3 |dPdx|/(2 Reynolds))
            nu = sqrt(pow(h, 3) * fabs(dPdx) / (2 * Reynolds));
        }

        else {  // constraint == BulkVelocity
            // cout << "...from bulk velocity" << endl;
            // Ucenter == 3/2 Ubulk, so Reynolds == Ucenter h/nu gives nu == 3/2 Ubulk h/Reynolds
            nu = 1.5 * Ubulk * h / Reynolds;
        }
    }
    // cout << "returning nu == " << nu << endl;
    return nu;
}

// logic
//
// Vsuck == 0 or very small
//   pressure constraint         =>   regular PCF/PPF formula, pressure style
//   bulk velocity constraint    =>   regular PCF/PPF formula, bulk vel style
// Vsuck != 0
//   pressure constraint
//     dPdx == 0                 =>   regular ASBL formula
//     dPdx != 0                 =>   pressure ASBL formula
//   bulk velocity constraint    =>   bulk velocity ASBL formula

ChebyCoeff laminarProfile(Real nu, MeanConstraint constraint, Real dPdx, Real Ubulk, Real Vsuck, Real a, Real b,
                          Real ua, Real ub, int Ny) {
    ChebyCoeff u(Ny, a, b, Spectral);
    Real H = b - a;

    // The laminar solution boundary value problem has two distinct solutions,
    // For Vsuck == 0, we get the quadratic plane Couette / plane Poiseuille solution.
    // For Vsuck != 0, we get an exponential solution for ASBL (dPdx == 0), and
    //                 and an exponential plus quadratic solution for ASBL with dPdx != 0.
    // The Vsuck != 0 solution converges onto the Vsuck == 0 as Vsuck H/nu -> 0.
    // PCF/PPF == pane Couette flow/plane Poiseuille flow, ASBL == asympotic suction boundary layer

    // Vsuck == 0 or very small. (PCF/PPF or ASBL in limit Vsuck nu/H -> 0).
    if (abs(Vsuck * H / nu) < 1e-08) {
        if (constraint == BulkVelocity) {
            u[0] = 0.125 * (ub + ua) + 0.75 * Ubulk;
            u[1] = 0.5 * (ub - ua);
            u[2] = 0.375 * (ub + ua) - 0.75 * Ubulk;
        } else {
            dPdx *= square((b - a) / 2);
            u[0] = 0.5 * (ub + ua) - 0.25 * dPdx / nu;
            u[1] = 0.5 * (ub - ua);
            u[2] = 0.25 * dPdx / nu;
        }
    } else {  // Vsuck != 0 and away from limit Vsuck nu/H -> 0. ASBL or ASBL plus quadratic
        u.setState(Physical);
        Vector y = chebypoints(Ny, a, b);
        Real ub_ua = ub - ua;
        Real Vsuck_nu = Vsuck / nu;
        Real expm1_H_Vsuck_nu = expm1(-H * Vsuck_nu);  // = exp(-H*Vsuck/nu) - 1

        if (constraint == PressureGradient) {
            // Vsuck != 0, dPdx != 0, following jfg 2018-11-19 notes
            // Note that this evaluates to the classic ASBL formula when dPdx == 0.
            Real dPdx_Vsuck = dPdx / Vsuck;
            for (int i = 0; i < Ny; i++) {
                Real y_a = y[i] - a;
                u[i] = ua + ub_ua * expm1(-y_a * Vsuck_nu) / expm1_H_Vsuck_nu +
                       dPdx_Vsuck * (y_a * expm1_H_Vsuck_nu - H * expm1(-y_a * Vsuck_nu)) / expm1(-H * Vsuck_nu);
            }
        } else {
            // Vsuck != 0, bulk velocity constraint, following jfg 2018-11-19 notes
            // Note that in the limit Vsuck H/nu -> 0, k = 1/2 * 1/(1 - Vsuck H/nu)
            // So k is bounded away from 1/2 by enclosing conditional abs(Vsuck*H/nu) > 1e-08.
            // Potential cancellation errors in the numerator of u[i] are similarly bounded to
            // single precision by this condition.
            Real k = -1.0 / expm1_H_Vsuck_nu - nu / (H * Vsuck);
            Real c = (Ubulk - ua - ub_ua * k) / (H * (0.5 - k));  // k is bounded away from 1/2

            for (int i = 0; i < Ny; i++) {
                Real y_a = y[i] - a;
                u[i] = ua + ub_ua * expm1(-y_a * Vsuck_nu) / expm1_H_Vsuck_nu +
                       c * (y_a * expm1_H_Vsuck_nu - H * expm1(-y_a * Vsuck_nu)) / expm1(-H * Vsuck_nu);
            }
        }
        u.makeSpectral();  // all Vsuck != 0 cases
    }
    return u;
}

ChebyCoeff laminarProfile(const DNSFlags& flags, Real a, Real b, int Ny) {
    if (flags.baseflow == SuctionBase)
        return laminarProfile(flags.nu, PressureGradient, 0, flags.Ubulk, flags.Vsuck, a, b, -0.5, 0.5, Ny);
    else
        return laminarProfile(flags.nu, flags.constraint, flags.dPdx, flags.Ubulk, flags.Vsuck, a, b, flags.ulowerwall,
                              flags.uupperwall, Ny);
}

void changeBaseFlow(const ChebyCoeff& ubase0, const FlowField& ufluc0, const FlowField& q0arg, const ChebyCoeff& ubase1,
                    FlowField& u1, FlowField& q1) {
    ChebyCoeff& U0 = (ChebyCoeff&)ubase0;
    fieldstate U0state = U0.state();

    ChebyCoeff& U1 = (ChebyCoeff&)ubase1;
    fieldstate U1state = U1.state();

    FlowField& u0 = (FlowField&)ufluc0;
    fieldstate u0xzstate = u0.xzstate();
    fieldstate u0ystate = u0.ystate();

    FlowField& q0 = (FlowField&)q0arg;
    fieldstate q0xzstate = q0.xzstate();
    fieldstate q0ystate = q0.ystate();

    int Nx = u0.numXgridpts();
    int Ny = u0.numYgridpts();
    int Nz = u0.numZgridpts();

    u1 = u0;  // want u1 FPF
    u1.makeState(Spectral, Physical);
    u0.makePhysical();
    q0.makePhysical();
    q1 = q0;  // want q1 physical

    // At this point
    // u1 == utot - U0
    // q1 == p + 1/2 u0 dot u0

    // Remove 1/2 u0 dot u0 from q1
    for (int ny = 0; ny < Ny; ++ny)
        for (int nx = 0; nx < Nx; ++nx)
            for (int nz = 0; nz < Nz; ++nz)
                q1(nx, ny, nz, 0) -=
                    0.5 * (square(u0(nx, ny, nz, 0)) + square(u0(nx, ny, nz, 1)) + square(u0(nx, ny, nz, 2)));
    // At this point
    // u1 == utot - U0
    // q1 == p

    ChebyTransform t(U0.numModes());
    U0.makePhysical(t);
    U1.makePhysical(t);

    // Add U0-U1 to u1
    ChebyCoeff delta_U(U0);
    delta_U -= U1;
    u1 += delta_U;
    u1.makePhysical();

    // At this point
    // u1 == utot - U1
    // q1 == p

    // Add 1/2 u1 dot u1 to q1
    for (int ny = 0; ny < Ny; ++ny)
        for (int nx = 0; nx < Nx; ++nx)
            for (int nz = 0; nz < Nz; ++nz)
                q1(nx, ny, nz, 0) +=
                    0.5 * (square(u1(nx, ny, nz, 0)) + square(u1(nx, ny, nz, 1)) + square(u1(nx, ny, nz, 2)));
    // At this point
    // u1 == utot - U1
    // q1 == p + 1/2 u1 dot u1
    // et, voila

    U0.makeState(U0state, t);
    U1.makeState(U1state, t);
    u0.makeState(u0xzstate, u0ystate);
    q0.makeState(q0xzstate, q0ystate);
    u1.makeState(u0xzstate, u0ystate);
    q1.makeState(q0xzstate, q0ystate);
}

}  // namespace chflow
