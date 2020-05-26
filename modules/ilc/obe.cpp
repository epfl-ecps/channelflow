/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 *
 * Original author: Florian Reetz
 */

#include "modules/ilc/obe.h"

namespace chflow {

void momentumNL(const FlowField& u, const FlowField& T, ChebyCoeff Ubase, ChebyCoeff Wbase, FlowField& f,
                FlowField& tmp, ILCFlags flags) {
    // compute the nonlinear term of NSE in the usual Channelflow style
    navierstokesNL(u, Ubase, Wbase, f, tmp, flags);

    // substract the linear temperature coupling term
    // f -= sin(gamma)*T e_x + cos(gamma)*T e_y
    Real gammax = flags.gammax;
    Real grav = flags.grav;
#ifdef HAVE_MPI
    for (int mz = f.mzlocmin(); mz < f.mzlocmin() + f.Mzloc(); mz++)
        for (int mx = f.mxlocmin(); mx < f.mxlocmin() + f.Mxloc(); mx++)
            for (int ny = 0; ny < f.Ny(); ny++) {
                f.cmplx(mx, ny, mz, 0) -= grav * sin(gammax) * T.cmplx(mx, ny, mz, 0);
                f.cmplx(mx, ny, mz, 1) -= grav * cos(gammax) * T.cmplx(mx, ny, mz, 0);
            }
#else
    for (int ny = 0; ny < f.Ny(); ny++)
        for (int mx = f.mxlocmin(); mx < f.mxlocmin() + f.Mxloc(); mx++)
            for (int mz = f.mzlocmin(); mz < f.mzlocmin() + f.Mzloc(); mz++) {
                f.cmplx(mx, ny, mz, 0) -= grav * sin(gammax) * T.cmplx(mx, ny, mz, 0);
                f.cmplx(mx, ny, mz, 1) -= grav * cos(gammax) * T.cmplx(mx, ny, mz, 0);
            }
#endif

    // dealiasing modes
    if (flags.dealias_xz())
        f.zeroPaddedModes();
}

void temperatureNL(const FlowField& u_, const FlowField& T_, ChebyCoeff Ubase, ChebyCoeff Wbase, ChebyCoeff Tbase,
                   FlowField& f, FlowField& tmp, ILCFlags flags) {
    FlowField& u = const_cast<FlowField&>(u_);
    FlowField& T = const_cast<FlowField&>(T_);

    // f += Base;
    for (int ny = 0; ny < u.Ny(); ++ny) {
        if (u.taskid() == u.task_coeff(0, 0)) {
            u.cmplx(0, ny, 0, 0) += Complex(Ubase(ny), 0.0);
            u.cmplx(0, ny, 0, 2) += Complex(Wbase(ny), 0.0);
        }
        if (u.taskid() == u.task_coeff(0, 0))
            T.cmplx(0, ny, 0, 0) += Complex(Tbase(ny), 0.0);
    }
    if (u.taskid() == u.task_coeff(0, 0)) {
        u.cmplx(0, 0, 0, 1) -= Complex(flags.Vsuck, 0.);
    }

    // compute the nonlinearity (temperature advection (u*grad)T ) analogous to the convectiveNL and store in f
    dotgradScalar(u, T, f, tmp);

    // f -= Base;
    for (int ny = 0; ny < u.Ny(); ++ny) {
        if (u.taskid() == u.task_coeff(0, 0)) {
            u.cmplx(0, ny, 0, 0) -= Complex(Ubase(ny), 0.0);
            u.cmplx(0, ny, 0, 2) -= Complex(Wbase(ny), 0.0);
        }
        if (u.taskid() == u.task_coeff(0, 0))
            T.cmplx(0, ny, 0, 0) -= Complex(Tbase(ny), 0.0);
    }
    if (u.taskid() == u.task_coeff(0, 0)) {
        u.cmplx(0, 0, 0, 1) += Complex(flags.Vsuck, 0.);
    }

    // dealiasing modes
    if (flags.dealias_xz())
        f.zeroPaddedModes();
}

OBE::OBE(const std::vector<FlowField>& fields, const ILCFlags& flags)
    : NSE(fields, flags),
      heatsolver_(0),  // heatsolvers are allocated when reset_lambda is called for the first time
      flags_(flags),
      gsingx_(0),
      gcosgx_(0),
      Tref_(flags.t_ref),
      Tbase_(),
      Tbaseyy_(),
      Pbasey_(),
      Pbasex_(0),
      Cu_(),
      Cw_(),
      Ct_(),
      nonzCu_(false),
      nonzCw_(false),
      nonzCt_(false),
      Tk_(Nyd_, a_, b_, Spectral),
      Rtk_(Nyd_, a_, b_, Spectral),
      baseflow_(false),
      constraint_(false) {
    gsingx_ = flags.grav * sin(flags_.gammax);
    gcosgx_ = flags.grav * cos(flags_.gammax);

    // set member variables for base flow
    createILCBaseFlow();
    Pbasex_ = hydrostaticPressureGradientX(flags_);
    Pbasey_ = hydrostaticPressureGradientY(Tbase_, flags_);

    // set member variables for contraint
    initILCConstraint(fields[0]);

    // define the constant terms of the OBE
    createConstants();
}

OBE::OBE(const std::vector<FlowField>& fields, const std::vector<ChebyCoeff>& base, const ILCFlags& flags)
    : NSE(fields, base, flags),
      heatsolver_(0),  // heatsolvers are allocated when reset_lambda is called for the first time
      flags_(flags),
      gsingx_(0),
      gcosgx_(0),
      Tref_(flags.t_ref),
      Tbase_(base[2]),
      Tbaseyy_(),
      Pbasey_(),
      Pbasex_(0),
      Cu_(),
      Cw_(),
      Ct_(),
      nonzCu_(false),
      nonzCw_(false),
      nonzCt_(false),
      Tk_(Nyd_, a_, b_, Spectral),
      Rtk_(Nyd_, a_, b_, Spectral),
      baseflow_(false),
      constraint_(false) {
    gsingx_ = flags.grav * sin(flags_.gammax);
    gcosgx_ = flags.grav * cos(flags_.gammax);

    baseflow_ = true;  // base flow is passed to constructor
    Pbasex_ = hydrostaticPressureGradientX(flags_);
    Pbasey_ = hydrostaticPressureGradientY(Tbase_, flags_);

    // set member variables for contraint
    initILCConstraint(fields[0]);

    // define the constant terms of the OBE
    createConstants();
}

OBE::~OBE() {
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

    if (heatsolver_) {
        for (uint j = 0; j < lambda_t_.size(); ++j) {
            for (int mx = 0; mx < Mxloc_; ++mx) {
                delete[] heatsolver_[j][mx];  // undo new #3
                heatsolver_[j][mx] = 0;
            }
            delete[] heatsolver_[j];  // undo new #2
            heatsolver_[j] = 0;
        }
        delete[] heatsolver_;  // undo new #1
        heatsolver_ = 0;
    }
}

void OBE::nonlinear(const std::vector<FlowField>& infields, std::vector<FlowField>& outfields) {
    // The first entry in vector must be velocity FlowField, the second a temperature FlowField.
    // Pressure as third entry in in/outfields is not touched.
    momentumNL(infields[0], infields[1], Ubase_, Wbase_, outfields[0], tmp_, flags_);
    temperatureNL(infields[0], infields[1], Ubase_, Wbase_, Tbase_, outfields[1], tmp_, flags_);
}

void OBE::linear(const std::vector<FlowField>& infields, std::vector<FlowField>& outfields) {
    // Method takes input fields {u,press} and computes the linear terms for velocity output field {u}

    // Make sure user does not expect a pressure output. Outfields should be created outside OBE with OBE::createRHS()
    assert(infields.size() == (outfields.size() + 1));
    const int kxmax = infields[0].kxmax();
    const int kzmax = infields[0].kzmax();
    const Real nu = flags_.nu;
    const Real kappa = flags_.kappa;

    // Loop over Fourier modes. 2nd derivative and summation of linear term
    // is most sufficient on ComplexChebyCoeff. Therefore, the old loop structure is kept.
    for (lint mx = mxlocmin_; mx < mxlocmin_ + Mxloc_; ++mx) {
        const int kx = infields[0].kx(mx);
        for (lint mz = mzlocmin_; mz < mzlocmin_ + Mzloc_; ++mz) {
            const int kz = infields[0].kz(mz);

            // Skip last and aliased modes
            if ((kx == kxmax || kz == kzmax) || (flags_.dealias_xz() && isAliasedMode(kx, kz)))
                break;

            // Compute linear temperature terms
            //================================
            // Goal is to compute
            // L = nu uj" - kappa2 nu uj - grad qj + C

            // Extract relevant Fourier modes of uj and qj
            for (int ny = 0; ny < Nyd_; ++ny) {
                uk_.set(ny, nu * infields[0].cmplx(mx, ny, mz, 0));
                vk_.set(ny, nu * infields[0].cmplx(mx, ny, mz, 1));
                wk_.set(ny, nu * infields[0].cmplx(mx, ny, mz, 2));
                Pk_.set(ny, infields[2].cmplx(mx, ny, mz, 0));  // pressure is 3. entry in fields
            }

            // (1) Put nu uj" into in R. (Pyk_ is used as tmp workspace)
            diff2(uk_, Ruk_, Pyk_);
            diff2(vk_, Rvk_, Pyk_);
            diff2(wk_, Rwk_, Pyk_);

            // (2) Put qn' into Pyk (compute y-comp of pressure gradient).
            diff(Pk_, Pyk_);

            // (3) Summation of all derivative terms and assignment to output velocity field.
            const Real k2 = 4 * pi * pi * (square(kx / Lx_) + square(kz / Lz_));
            const Complex Dx = infields[0].Dx(mx);
            const Complex Dz = infields[0].Dz(mz);
            for (int ny = 0; ny < Nyd_; ++ny) {
                outfields[0].cmplx(mx, ny, mz, 0) = Ruk_[ny] - k2 * uk_[ny] - Dx * Pk_[ny];
                outfields[0].cmplx(mx, ny, mz, 1) = Rvk_[ny] - k2 * vk_[ny] - Pyk_[ny];
                outfields[0].cmplx(mx, ny, mz, 2) = Rwk_[ny] - k2 * wk_[ny] - Dz * Pk_[ny];
            }

            // (4) Add const. terms
            if (kx == 0 && kz == 0) {
                // L includes const dissipation term of Ubase and Wbase: nu Uyy, nu Wyy
                // and bouyancy term of Temperature profile: sin(gamma)(Tbase_-Tref)
                // absolute bouyancy term cancels with Pbasex_ (both constant)
                if (nonzCu_ || nonzCw_) {
                    for (int ny = 0; ny < My_; ++ny) {
                        outfields[0].cmplx(mx, ny, mz, 0) += Cu_[ny];
                        outfields[0].cmplx(mx, ny, mz, 2) += Cw_[ny];
                    }
                }
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
                        dPdxAct += nu * (Ubasey.eval_b() - Ubasey.eval_a()) / Ly;
                    if (Wbase_.length() != 0)
                        dPdzAct += nu * (Wbasey.eval_b() - Wbasey.eval_a()) / Ly;
                    // add press. gradient to linear term
                    outfields[0].cmplx(mx, 0, mz, 0) -= Complex(dPdxAct, 0);
                    outfields[0].cmplx(mx, 0, mz, 2) -= Complex(dPdzAct, 0);
                }
            }  // End of const. terms

            // Compute linear heat equation terms
            //================================
            // Goal is to compute
            // L = kappa T" - k2 kappa T - Ubase dT/dx + C

            // Extract relevant Fourier modes of T: use Pk and Rxk
            for (int ny = 0; ny < Nyd_; ++ny)
                Tk_.set(ny, kappa * infields[1].cmplx(mx, ny, mz, 0));

            // (1) Put kappa T" into in R. (Pyk_ is used as tmp workspace)
            diff2(Tk_, Rtk_, Pyk_);

            // (2+3) Summation of both diffusive terms and linear advection of temperature.
            // k2 and Dx were defined above for the current Fourier mode
            for (int ny = 0; ny < Nyd_; ++ny)
                // from nonlinear:  - Dx*Tk_[ny]*Complex(Ubase_[ny],0)/kappa;
                outfields[1].cmplx(mx, ny, mz, 0) = Rtk_[ny] - k2 * Tk_[ny];

            // (4) Add const. terms
            if (kx == 0 && kz == 0) {
                // L includes const diffusion term of Tbase: kappa Tbase_yy
                if (nonzCt_)
                    for (int ny = 0; ny < My_; ++ny)
                        outfields[1].cmplx(mx, ny, mz, 0) += Ct_[ny];
            }
        }
    }  // End of loop over Fourier modes
}

void OBE::solve(std::vector<FlowField>& outfields, const std::vector<FlowField>& rhs, const int s) {
    // Method takes a right hand side {u} and solves for output fields {u,press}

    // Make sure user provides correct RHS which can be created outside NSE with NSE::createRHS()
    assert(outfields.size() == (rhs.size() + 1));
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
                // negative RHS because HelmholtzSolver solves the negative problem
                Rtk_.set(ny, -rhs[1].cmplx(mx, ny, mz, 0));
            }

            // Solve the tau equations for momentum
            //=============================
            if (kx != 0 || kz != 0)
                tausolver_[s][ix][iz].solve(uk_, vk_, wk_, Pk_, Ruk_, Rvk_, Rwk_);
            // 	solve(ix,iz,uk_,vk_,wk_,Pk_, Ruk_,Rvk_,Rwk_);
            else {  // kx,kz == 0,0
                // LHS includes also the constant terms C which can be added to RHS
                if (nonzCu_ || nonzCw_) {
                    for (int ny = 0; ny < My_; ++ny) {
                        Ruk_.re[ny] += Cu_.re[ny];
                        Rwk_.re[ny] += Cw_.re[ny];
                    }
                }

                if (flags_.constraint == PressureGradient) {
                    // pressure is supplied, put on RHS of tau eqn
                    Ruk_.re[0] -= dPdxRef_;
                    Rwk_.re[0] -= dPdzRef_;
                    tausolver_[s][ix][iz].solve(uk_, vk_, wk_, Pk_, Ruk_, Rvk_, Rwk_);
                    // 	  solve(ix,iz,uk_, vk_, wk_, Pk_, Ruk_,Rvk_,Rwk_);
                    // Bulk vel is free variable determined from soln of tau eqn //TODO: write method that computes
                    // UbulkAct everytime it is needed

                } else {  // const bulk velocity
                    // bulk velocity is supplied, use alternative tau solver

                    // Use tausolver with additional variable and constraint:
                    // free variable: dPdxAct at next time-step,
                    // constraint:    UbulkBase + mean(u) = UbulkRef.
                    tausolver_[s][ix][iz].solve(uk_, vk_, wk_, Pk_, dPdxAct_, dPdzAct_, Ruk_, Rvk_, Rwk_,
                                                UbulkRef_ - UbulkBase_, WbulkRef_ - WbulkBase_);
                    // 	  solve(ix,iz,uk_, vk_, wk_, Pk_, dPdxAct_, dPdzAct_,
                    // 				    Ruk_, Rvk_, Rwk_,
                    // 				    UbulkRef_ - UbulkBase_,
                    // 				    WbulkRef_ - WbulkBase_);

                    // test if UbulkRef == UbulkAct = UbulkBase_ + uk_.re.mean()
                    assert((UbulkRef_ - UbulkBase_ - uk_.re.mean()) < 1e-15);
                    // test if WbulkRef == WbulkAct = WbulkBase_ + wk_.re.mean()
                    assert((WbulkRef_ - WbulkBase_ - wk_.re.mean()) < 1e-15);
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
                    outfields[2].cmplx(mx, ny, mz, 0) = Complex(Re(Pk_[ny]), 0.0);
                }
            }
            // The normal case, for general kx,kz
            else
                for (int ny = 0; ny < Nyd_; ++ny) {
                    outfields[0].cmplx(mx, ny, mz, 0) = uk_[ny];
                    outfields[0].cmplx(mx, ny, mz, 1) = vk_[ny];
                    outfields[0].cmplx(mx, ny, mz, 2) = wk_[ny];
                    outfields[2].cmplx(mx, ny, mz, 0) = Pk_[ny];
                }

            // Solve the helmholtz problem for the heat equation
            //=============================
            if (kx != 0 || kz != 0) {
                heatsolver_[s][ix][iz].solve(Tk_.re, Rtk_.re, 0.0, 0.0);  // BC are considered through the base profile
                heatsolver_[s][ix][iz].solve(Tk_.im, Rtk_.im, 0.0, 0.0);
            } else {  // kx,kz == 0,0
                // LHS includes also the constant term C=kappa Tbase_yy, which can be added to RHS
                if (nonzCt_) {
                    for (int ny = 0; ny < My_; ++ny)
                        Rtk_.re[ny] -= Ct_.re[ny];
                }

                heatsolver_[s][ix][iz].solve(Tk_.re, Rtk_.re, 0.0, 0.0);
                heatsolver_[s][ix][iz].solve(Tk_.im, Rtk_.im, 0.0, 0.0);
            }

            // Load solution into T.
            // Because of FFTW complex symmetries
            // The 0,0 mode must be real.
            // For Nx even, the kxmax,0 mode must be real
            // For Nz even, the 0,kzmax mode must be real
            // For Nx,Nz even, the kxmax,kzmax mode must be real
            if ((kx == 0 && kz == 0) || (outfields[1].Nx() % 2 == 0 && kx == kxmax && kz == 0) ||
                (outfields[1].Nz() % 2 == 0 && kz == kzmax && kx == 0) ||
                (outfields[1].Nx() % 2 == 0 && outfields[1].Nz() % 2 == 0 && kx == kxmax && kz == kzmax)) {
                for (int ny = 0; ny < Nyd_; ++ny)
                    outfields[1].cmplx(mx, ny, mz, 0) = Complex(Re(Tk_[ny]), 0.0);

            }
            // The normal case, for general kx,kz
            else
                for (int ny = 0; ny < Nyd_; ++ny)
                    outfields[1].cmplx(mx, ny, mz, 0) = Tk_[ny];

        }  // End of loop over Fourier modes
    }
}

void OBE::reset_lambda(std::vector<Real> lambda_t) {
    lambda_t_ = lambda_t;

    if ((tausolver_ == 0) && (heatsolver_ == 0)) {  // TauSolver need to be constructed
        // Allocate memory for [Nsubsteps x Mx_ x Mz_] Tausolver cfarray
        tausolver_ = new TauSolver**[lambda_t.size()];  // new #1
        heatsolver_ = new HelmholtzSolver**[lambda_t.size()];
        for (uint j = 0; j < lambda_t.size(); ++j) {
            tausolver_[j] = new TauSolver*[Mxloc_];  // new #2
            heatsolver_[j] = new HelmholtzSolver*[Mxloc_];
            for (int mx = 0; mx < Mxloc_; ++mx) {
                tausolver_[j][mx] = new TauSolver[Mzloc_];  // new #3
                heatsolver_[j][mx] = new HelmholtzSolver[Mzloc_];
            }
        }
    } else if ((tausolver_ == 0) && (heatsolver_ != 0)) {
        tausolver_ = new TauSolver**[lambda_t.size()];  // new #1
        for (uint j = 0; j < lambda_t.size(); ++j) {
            tausolver_[j] = new TauSolver*[Mxloc_];  // new #2
            for (int mx = 0; mx < Mxloc_; ++mx)
                tausolver_[j][mx] = new TauSolver[Mzloc_];  // new #3
        }
    } else if ((tausolver_ != 0) && (heatsolver_ == 0)) {
        heatsolver_ = new HelmholtzSolver**[lambda_t.size()];  // new #1
        for (uint j = 0; j < lambda_t.size(); ++j) {
            heatsolver_[j] = new HelmholtzSolver*[Mxloc_];  // new #2
            for (int mx = 0; mx < Mxloc_; ++mx)
                heatsolver_[j][mx] = new HelmholtzSolver[Mzloc_];  // new #3
        }
    }

    // Configure tausolvers
    //   FlowField u=fields_[0];
    const Real nu = flags_.nu;
    const Real kappa = flags_.kappa;
    const Real c = 4.0 * square(pi);
    //   const int kxmax = u.kxmax();
    //   const int kzmax = u.kzmax();
    for (uint j = 0; j < lambda_t.size(); ++j) {
        for (int mx = 0; mx < Mxloc_; ++mx) {
            int kx = kxloc_[mx];
            for (int mz = 0; mz < Mzloc_; ++mz) {
                int kz = kzloc_[mz];
                Real lambda_tau = lambda_t[j] + nu * c * (square(kx / Lx_) + square(kz / Lz_));
                Real lambda_heat = lambda_t[j] + kappa * c * (square(kx / Lx_) + square(kz / Lz_));

                if ((kx != kxmax_ || kz != kzmax_) && (!flags_.dealias_xz() || !isAliasedMode(kx, kz))) {
                    tausolver_[j][mx][mz] =
                        TauSolver(kx, kz, Lx_, Lz_, a_, b_, lambda_tau, nu, Nyd_, flags_.taucorrection);
                    heatsolver_[j][mx][mz] = HelmholtzSolver(Nyd_, a_, b_, lambda_heat, kappa);
                }
            }
        }
    }
}

const ChebyCoeff& OBE::Tbase() const { return Tbase_; }
const ChebyCoeff& OBE::Ubase() const { return Ubase_; }
const ChebyCoeff& OBE::Wbase() const { return Wbase_; }

std::vector<FlowField> OBE::createRHS(const std::vector<FlowField>& fields) const {
    return {fields[0], fields[1]};  // return only velocity and temperature fields
}

std::vector<cfarray<FieldSymmetry>> OBE::createSymmVec() const {
    // velocity symmetry
    cfarray<FieldSymmetry> usym = SymmetryList(1);  // unity operation
    if (flags_.symmetries.length() > 0)
        usym = flags_.symmetries;
    // temperature symmetry
    cfarray<FieldSymmetry> tsym = SymmetryList(1);  // unity operation
    if (flags_.tempsymmetries.length() > 0)
        tsym = flags_.tempsymmetries;
    // pressure symmetry
    cfarray<FieldSymmetry> psym = SymmetryList(1);  // unity operation
    return {usym, tsym, psym};
}

void OBE::createILCBaseFlow() {
    Real ulowerwall = flags_.ulowerwall;
    Real uupperwall = flags_.uupperwall;
    Real wlowerwall = flags_.wlowerwall;
    Real wupperwall = flags_.wupperwall;

    switch (flags_.baseflow) {
        case ZeroBase:
            Ubase_ = ChebyCoeff(My_, a_, b_, Spectral);
            Wbase_ = ChebyCoeff(My_, a_, b_, Spectral);
            Tbase_ = ChebyCoeff(My_, a_, b_, Spectral);
            break;
        case LinearBase:
            std::cerr << "error in OBE::createBaseFlow :\n";
            std::cerr << "LinearBase is not defined in ILC.\n";
            break;
        case ParabolicBase:
            std::cerr << "error in OBE::createBaseFlow :\n";
            std::cerr << "ParabolicBase is not defined in ILC.\n";
            break;
        case SuctionBase:
            std::cerr << "error in OBE::createBaseFlow :\n";
            std::cerr << "ParabolicBase is not defined in ILC.\n";
            break;
        case LaminarBase:
            Ubase_ = laminarVelocityProfile(flags_.gammax, flags_.dPdx, flags_.Ubulk, ulowerwall, uupperwall, a_, b_,
                                            My_, flags_);

            // choosing angle zero removes buoyancy effect
            Wbase_ =
                laminarVelocityProfile(0.0, flags_.dPdz, flags_.Wbulk, wlowerwall, wupperwall, a_, b_, My_, flags_);

            Tbase_ = linearTemperatureProfile(a_, b_, My_, flags_);

            break;
        case ArbitraryBase:
            std::cerr << "error in NSE::createBaseFlow :\n";
            std::cerr << "flags.baseflow is ArbitraryBase.\n";
            std::cerr << "Please provide {Ubase, Wbase} when constructing DNS.\n";
            cferror("");
        default:
            std::cerr << "error in NSE::createBaseFlow :\n";
            std::cerr << "flags.baseflow should be ZeroBase, LinearBase, ParabolicBase, LaminarBase, SuctionBase.\n";
            std::cerr << "Other cases require use of the DNS::DNS(fields, base, flags) constructor.\n";
            cferror("");
    }

    baseflow_ = true;
}

void OBE::createConstants() {
    if ((!baseflow_) || (!constraint_))
        std::cerr << "OBE::createConstants: Not all base flow members have been initialized." << std::endl;

    ComplexChebyCoeff c(My_, a_, b_, Spectral);
    Real nu = flags_.nu;
    Real kappa = flags_.kappa;
    Real eta = 1.0 / (flags_.alpha * (flags_.tlowerwall - flags_.tupperwall));
    Real t_ref = flags_.t_ref;

    // constant u-term:
    Real hydrostatx = gsingx_ * (Tbase_[0] - eta - t_ref) - Pbasex_;
    if (abs(hydrostatx) > 1e-15)
        *flags_.logstream << "ILC with hydrostatic dPdx." << std::endl;
    if (Ubaseyy_.length() > 0) {
        c[0] = Complex(nu * Ubaseyy_[0] + hydrostatx, 0);
        if ((abs(c[0]) > 1e-15) && !nonzCu_)
            nonzCu_ = true;
        for (int ny = 1; ny < My_; ++ny) {
            c[ny] = Complex(nu * Ubaseyy_[ny] + gsingx_ * Tbase_[ny], 0);
            if ((abs(c[ny]) > 1e-15) && !nonzCu_)
                nonzCu_ = true;
        }
    }
    if (nonzCu_) {
        Cu_ = c;
        *flags_.logstream << "ILC with nonzero U-const." << std::endl;
    }

    // check that constant v-term is zero:
    Real hydrostaty = gcosgx_ * (Tbase_[0] - eta - t_ref) - Pbasey_[0];
    if (abs(hydrostaty) > 1e-15)
        std::cerr << "Wall-normal hydrostatic pressure is unballanced" << std::endl;
    for (int ny = 1; ny < My_; ++ny)
        if (abs(gcosgx_ * Tbase_[ny] - Pbasey_[ny]) > 1e-15)
            std::cerr << "Wall-normal hydrostatic pressure is unballanced" << std::endl;

    // constant w-term:
    if (Wbaseyy_.length() > 0) {
        for (int ny = 0; ny < My_; ++ny) {
            c[ny] = Complex(nu * Wbaseyy_[ny], 0);
            if ((abs(c[ny]) > 1e-15) && !nonzCw_)
                nonzCw_ = true;
        }
    }
    if (nonzCw_) {
        Cw_ = c;
        *flags_.logstream << "ILC with nonzero W-const." << std::endl;
    }

    // constant t-term:
    if (Tbaseyy_.length() > 0) {
        for (int ny = 1; ny < My_; ++ny) {
            c[ny] = Complex(kappa * Tbaseyy_[ny], 0);
            if ((abs(c[ny]) > 1e-15) && !nonzCt_)
                nonzCt_ = true;
        }
    }
    if (nonzCt_) {
        Ct_ = c;
        *flags_.logstream << "ILC with nonzero T-const." << std::endl;
    }
}

void OBE::initILCConstraint(const FlowField& u) {
    if (!baseflow_)
        std::cerr << "OBE::initConstraint: Base flow has not been created." << std::endl;

    // Calculate Ubaseyy_ and related quantities
    UbulkBase_ = Ubase_.mean();
    ChebyCoeff Ubasey = diff(Ubase_);
    Ubaseyy_ = diff(Ubasey);
    WbulkBase_ = Wbase_.mean();
    ChebyCoeff Wbasey = diff(Wbase_);
    Wbaseyy_ = diff(Wbasey);

    ChebyCoeff Tbasey = diff(Tbase_);
    Tbaseyy_ = diff(Tbasey);

    // Determine actual Ubulk and dPdx from initial data Ubase + u.
    Real Ly = b_ - a_;
    ChebyCoeff u00(My_, a_, b_, Spectral);
    Real reucmplx = 0;
    for (int ny = 0; ny < My_; ++ny) {
        if (u.taskid() == 0)
            reucmplx = Re(u.cmplx(0, ny, 0, 0));
#ifdef HAVE_MPI
        MPI_Bcast(&reucmplx, 1, MPI_DOUBLE, u.task_coeff(0, 0), *u.comm_world());
#endif
        u00[ny] = reucmplx;
    }
    ChebyCoeff du00dy = diff(u00);
    UbulkAct_ = UbulkBase_ + u00.mean();
    dPdxAct_ = flags_.nu * (du00dy.eval_b() - du00dy.eval_a()) / Ly;
    ChebyCoeff w00(My_, a_, b_, Spectral);
    for (int ny = 0; ny < My_; ++ny) {
        if (u.taskid() == 0)
            reucmplx = Re(u.cmplx(0, ny, 0, 2));
#ifdef HAVE_MPI
        MPI_Bcast(&reucmplx, 1, MPI_DOUBLE, u.task_coeff(0, 0), *u.comm_world());
#endif
        w00[ny] = reucmplx;
    }
    ChebyCoeff dw00dy = diff(w00);
    WbulkAct_ = WbulkBase_ + w00.mean();
    dPdzAct_ = flags_.nu * (dw00dy.eval_b() - dw00dy.eval_a()) / Ly;

#ifdef HAVE_MPI
    MPI_Bcast(&dPdxAct_, 1, MPI_DOUBLE, u.task_coeff(0, 0), *u.comm_world());
    MPI_Bcast(&UbulkAct_, 1, MPI_DOUBLE, u.task_coeff(0, 0), *u.comm_world());
    MPI_Bcast(&dPdzAct_, 1, MPI_DOUBLE, u.task_coeff(0, 0), *u.comm_world());
    MPI_Bcast(&WbulkAct_, 1, MPI_DOUBLE, u.task_coeff(0, 0), *u.comm_world());
#endif

    if (Ubase_.length() != 0)
        dPdxAct_ += flags_.nu * (Ubasey.eval_b() - Ubasey.eval_a()) / Ly;
    if (Wbase_.length() != 0)
        dPdzAct_ += flags_.nu * (Wbasey.eval_b() - Wbasey.eval_a()) / Ly;

    if (flags_.constraint == BulkVelocity) {
        UbulkRef_ = flags_.Ubulk;
        WbulkRef_ = flags_.Wbulk;
    } else {
        dPdxAct_ = flags_.dPdx;
        dPdxRef_ = flags_.dPdx;
        dPdzAct_ = flags_.dPdz;
        dPdzRef_ = flags_.dPdz;
    }

    constraint_ = true;
}

ChebyCoeff laminarVelocityProfile(Real gamma, Real dPdx, Real Ubulk, Real Ua, Real Ub, Real a, Real b, int Ny,
                                  ILCFlags flags) {
    MeanConstraint constraint = flags.constraint;
    Real nu = flags.nu;
    Real Vsuck = flags.Vsuck;
    Real dT = flags.tlowerwall - flags.tupperwall;
    Real mT = 0.5 * (flags.tlowerwall + flags.tupperwall);
    Real Tref = flags.t_ref;

    ChebyCoeff u(Ny, a, b, Spectral);
    Real grav = flags.grav;
    Real q = grav * sin(gamma) / nu / 16.0;

    if (constraint == BulkVelocity) {
        cferror("Using ILC with constraint BulkVelocity is not implemented yet");
    } else {
        if (Vsuck < 1e-14) {
            if (dPdx < 1e-14) {
                u[0] = q * (mT - Tref) + 0.5 * (Ub + Ua);  // See documentation about base solution
                u[1] = -q / 12.0 * dT + 0.5 * (Ub - Ua);
                u[2] = -q * (mT - Tref);
                u[3] = q / 12.0 * dT;
            } else {
                cferror("Using ILC with nonzero dPdx is not implemented yet");
            }
        } else {
            cferror("Using ILC with SuctionVelocity is not implemented yet");
        }
    }
    return u;
}

ChebyCoeff linearTemperatureProfile(Real a, Real b, int Ny, ILCFlags flags) {
    MeanConstraint constraint = flags.constraint;

    ChebyCoeff T(Ny, a, b, Spectral);
    Real Vsuck = flags.Vsuck;
    Real dPdx = flags.dPdx;
    Real Ta = flags.tlowerwall;
    Real Tb = flags.tupperwall;

    if (constraint == BulkVelocity) {
        cferror("Using ILC with constraint BulkVelocity is not implemented yet");
    } else {
        if (Vsuck < 1e-14) {
            if (dPdx < 1e-14) {
                T[0] = 0.5 * (Tb + Ta);  // the boundary conditions are given in units of Delta_T=Ta-Tb
                T[1] = 0.5 * (Tb - Ta);
            } else {
                cferror("Using ILC with nonzero dPdx is not implemented yet");
            }
        } else {
            cferror("Using ILC with SuctionVelocity is not implemented yet");
        }
    }
    return T;
}

Real hydrostaticPressureGradientX(ILCFlags flags) {
    MeanConstraint constraint = flags.constraint;

    Real Px = 0.0;
    Real Vsuck = flags.Vsuck;
    Real dPdx = flags.dPdx;
    Real sgamma = sin(flags.gammax);
    Real eta = 1.0 / (flags.alpha * (flags.tlowerwall - flags.tupperwall));
    Real grav = flags.grav;

    if (constraint == BulkVelocity) {
        cferror("Using ILC with constraint BulkVelocity is not implemented yet");
    } else {
        if (Vsuck < 1e-14) {
            if (dPdx < 1e-14) {
                Px = -grav * eta * sgamma;
            } else {
                cferror("Using ILC with nonzero dPdx is not implemented yet");
            }
        } else {
            cferror("Using ILC with SuctionVelocity is not implemented yet");
        }
    }
    return Px;
}

ChebyCoeff hydrostaticPressureGradientY(ChebyCoeff Tbase, ILCFlags flags) {
    MeanConstraint constraint = flags.constraint;

    ChebyCoeff Py(Tbase);
    Real Vsuck = flags.Vsuck;
    Real dPdx = flags.dPdx;
    Real cgamma = cos(flags.gammax);
    Real eta = 1.0 / (flags.alpha * (flags.tlowerwall - flags.tupperwall));
    Real grav = flags.grav;

    if (constraint == BulkVelocity) {
        cferror("Using ILC with constraint BulkVelocity is not implemented yet");
    } else {
        if (Vsuck < 1e-14) {
            if (dPdx < 1e-14) {
                Py[0] = grav * cgamma * (Tbase[0] - eta - flags.t_ref);  // TODO: implement general loop
                Py[1] = grav * cgamma * Tbase[1];
            } else {
                cferror("Using ILC with nonzero dPdx is not implemented yet");
            }
        } else {
            cferror("Using ILC with SuctionVelocity is not implemented yet");
        }
    }
    return Py;
}

}  // namespace chflow