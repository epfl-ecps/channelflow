/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "channelflow/poissonsolver.h"

using namespace std;

namespace chflow {

PoissonSolver::PoissonSolver() : Mx_(0), My_(0), Mz_(0), Nd_(0), Lx_(0), Lz_(0), a_(0), b_(0), helmholtz_(0) {}

PoissonSolver::PoissonSolver(int Nx, int Ny, int Nz, int Nd, Real Lx, Real Lz, Real a, Real b, CfMPI* cfmpi)
    : Mx_(Nx), My_(Ny), Mz_(Nz / 2 + 1), Nd_(Nd), Lx_(Lx), Lz_(Lz), a_(a), b_(b), helmholtz_(0) {
    FlowField tmp(Nx, Ny, Nz, 1, Lx, Lz, a, b, cfmpi);
    Mxloc_ = tmp.Mxloc();
    Mzloc_ = tmp.Mzloc();
    mxlocmin_ = tmp.mxlocmin();
    mzlocmin_ = tmp.mzlocmin();

    // Allocate Mx x Mz cfarray of HelmholtzSolvers
    helmholtz_ = new HelmholtzSolver*[Mxloc_];  // new #1
    for (int mx = 0; mx < Mxloc_; ++mx)
        helmholtz_[mx] = new HelmholtzSolver[Mzloc_];  // new #2

    // Assign Helmholtz solvers into cfarray. Each solves u'' - lambda u = f
    for (int mx = 0; mx < Mxloc_; ++mx) {
        //         int kx = (mx <= Nx/2) ? mx : mx - Nx; // same as FlowField::kx(mx)
        int kx = tmp.kx(mx + mxlocmin_);
        for (int mz = 0; mz < Mzloc_; ++mz) {
            //             int kz = mz; // same as FlowField::kz(mz)
            int kz = tmp.kz(mz + mzlocmin_);
            Real lambda = 4.0 * pi * pi * (square(kx / Lx_) + square(kz / Lz_));
            helmholtz_[mx][mz] = HelmholtzSolver(My_, a_, b_, lambda);
        }
    }
}

PoissonSolver::PoissonSolver(const PoissonSolver& ps)
    : Mx_(ps.Mx_),
      My_(ps.My_),
      Mz_(ps.Mz_),
      Mxloc_(ps.Mxloc_),
      Mzloc_(ps.Mzloc_),
      mxlocmin_(ps.mxlocmin_),
      mzlocmin_(ps.mzlocmin_),
      Nd_(ps.Nd_),
      Lx_(ps.Lx_),
      Lz_(ps.Lz_),
      a_(ps.a_),
      b_(ps.b_),
      helmholtz_(0) {
    // Allocate Mx x Mz cfarray of HelmholtzSolvers
    helmholtz_ = new HelmholtzSolver*[Mxloc_];  // new #1
    for (int mx = 0; mx < Mxloc_; ++mx)
        helmholtz_[mx] = new HelmholtzSolver[Mzloc_];  // new #2

    // Assign Helmholtz solvers into cfarray. Each solves p'' - lambda p = f
    for (int mx = 0; mx < Mxloc_; ++mx)
        for (int mz = 0; mz < Mzloc_; ++mz)
            helmholtz_[mx][mz] = ps.helmholtz_[mx][mz];
}

PoissonSolver::PoissonSolver(const FlowField& u)
    : Mx_(u.Mx()),
      My_(u.My()),
      Mz_(u.Mz()),
      Mxloc_(u.Mxloc()),
      Mzloc_(u.Mzloc()),
      mxlocmin_(u.mxlocmin()),
      mzlocmin_(u.mzlocmin()),
      Nd_(u.Nd()),
      Lx_(u.Lx()),
      Lz_(u.Lz()),
      a_(u.a()),
      b_(u.b()),
      helmholtz_(0) {
    // Allocate Mx x Mz cfarray of HelmholtzSolvers
    helmholtz_ = new HelmholtzSolver*[Mxloc_];  // new #1
    for (int mx = 0; mx < Mxloc_; ++mx)
        helmholtz_[mx] = new HelmholtzSolver[Mzloc_];  // new #2

    // Assign Helmholtz solvers into cfarray. Each solves p'' - lambda p = f
    for (int mx = 0; mx < Mxloc_; ++mx) {
        int kx = u.kx(mx + mxlocmin_);
        for (int mz = 0; mz < Mzloc_; ++mz) {
            int kz = u.kz(mz + mzlocmin_);
            Real lambda = 4.0 * pi * pi * (square(kx / Lx_) + square(kz / Lz_));
            helmholtz_[mx][mz] = HelmholtzSolver(My_, a_, b_, lambda);
        }
    }
}

PoissonSolver& PoissonSolver::operator=(const PoissonSolver& ps) {
    if (this == &ps)
        return *this;

    // Delete old helmholtz solvers
    for (int mx = 0; mx < Mxloc_; ++mx)
        delete[] helmholtz_[mx];  // undo new #2
    delete[] helmholtz_;          // undo new #1

    Mx_ = ps.Mx_;
    My_ = ps.My_;
    Mz_ = ps.Mz_;
    Mxloc_ = ps.Mxloc_;
    Mzloc_ = ps.Mzloc_;
    mxlocmin_ = ps.mxlocmin_;
    mzlocmin_ = ps.mzlocmin_;
    Nd_ = ps.Nd_;
    Lx_ = ps.Lx_;
    Lz_ = ps.Lz_;
    a_ = ps.a_;
    b_ = ps.b_;

    // Allocate Mx x Mz cfarray of HelmholtzSolvers
    helmholtz_ = new HelmholtzSolver*[Mxloc_];  // new #1
    for (int mx = 0; mx < Mxloc_; ++mx)
        helmholtz_[mx] = new HelmholtzSolver[Mzloc_];  // new #2

    // Assign Helmholtz solvers into cfarray. Each solves p'' - lambda p = f
    for (int mx = 0; mx < Mxloc_; ++mx)
        for (int mz = 0; mz < Mzloc_; ++mz)
            helmholtz_[mx][mz] = ps.helmholtz_[mx][mz];

    return *this;
}

PoissonSolver::~PoissonSolver() {
    for (int mx = 0; mx < Mxloc_; ++mx)
        delete[] helmholtz_[mx];  // undo new #2
    delete[] helmholtz_;          // undo new #1
}

bool PoissonSolver::geomCongruent(const FlowField& u) const {
    return (u.Mx() == Mx_) && (u.My() == My_) && (u.Mz() == Mz_) && (u.Lx() == Lx_) && (u.Lz() == Lz_) &&
           (u.a() == a_) && (u.b() == b_);
}

bool PoissonSolver::congruent(const FlowField& u) const {
    return (u.Mx() == Mx_) && (u.My() == My_) && (u.Mz() == Mz_) && (u.Nd() == Nd_) && (u.Lx() == Lx_) &&
           (u.Lz() == Lz_) && (u.a() == a_) && (u.b() == b_);
}

void PoissonSolver::solve(FlowField& u, const FlowField& f) const {
    assert(this->congruent(u));
    assert(this->congruent(f));
    f.assertState(Spectral, Spectral);
    if (!(this->congruent(u)))
        u = FlowField(f.Nx(), f.Ny(), f.Nz(), f.Nd(), f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
    else {
        u.setToZero();
        u.setState(Spectral, Spectral);
    }

    ComplexChebyCoeff fk(My_, a_, b_, Spectral);
    ComplexChebyCoeff uk(My_, a_, b_, Spectral);

    Real zero = 0.0;  // dirichlet BC
    for (int i = 0; i < Nd_; ++i)
        for (int mx = 0; mx < Mxloc_; ++mx)
            for (int mz = 0; mz < Mzloc_; ++mz) {
                for (int my = 0; my < My_; ++my)
                    fk.set(my, f.cmplx(mx + mxlocmin_, my, mz + mzlocmin_, i));
                helmholtz_[mx][mz].solve(uk.re, fk.re, zero, zero);
                helmholtz_[mx][mz].solve(uk.im, fk.im, zero, zero);
                for (int my = 0; my < My_; ++my)
                    u.cmplx(mx + mxlocmin_, my, mz + mzlocmin_, i) = uk[my];
            }
}

void PoissonSolver::solve(FlowField& u, const FlowField& f, const FlowField& bc) const {
    assert(this->congruent(f));
    assert(this->congruent(bc));
    assert(bc.xzstate() == Spectral);
    f.assertState(Spectral, Spectral);
    if (!(this->congruent(u)))
        u = FlowField(f.Nx(), f.Ny(), f.Nz(), f.Nd(), f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
    else {
        u.setToZero();
        u.setState(Spectral, Spectral);
    }

    ComplexChebyCoeff fk(My_, a_, b_, Spectral);
    ComplexChebyCoeff bk(My_, a_, b_, Spectral);
    ComplexChebyCoeff uk(My_, a_, b_, Spectral);

    for (int i = 0; i < Nd_; ++i)
        for (int mx = mxlocmin_; mx < Mx_ + mxlocmin_; ++mx)
            for (int mz = mzlocmin_; mz < Mz_ + mzlocmin_; ++mz) {
                for (int my = 0; my < My_; ++my) {
                    fk.set(my, f.cmplx(mx, my, mz, i));
                    bk.set(my, bc.cmplx(mx, my, mz, i));
                }
                helmholtz_[mx - mxlocmin_][mz - mzlocmin_].solve(uk.re, fk.re, bk.re.eval_a(), bk.re.eval_b());
                helmholtz_[mx - mxlocmin_][mz - mzlocmin_].solve(uk.im, fk.im, bk.im.eval_a(), bk.im.eval_b());
                for (int my = 0; my < My_; ++my)
                    u.cmplx(mx, my, mz, i) = uk[my];
            }
}

// verify that lapl u = f and u=0 on BC
Real PoissonSolver::verify(const FlowField& u, const FlowField& f) const {
    assert(this->congruent(u));
    assert(this->congruent(f));
    u.assertState(Spectral, Spectral);
    f.assertState(Spectral, Spectral);

    FlowField lapl_u;
    lapl(u, lapl_u);
    FlowField diff(f);
    diff -= lapl_u;

    /***********************************************
     // Debugging output
    f.saveSpectrum("f");
    lapl_u.saveSpectrum("lapl_u");
    diff.saveSpectrum("diff");

    for (int mx=0; mx<=3; ++mx) {
      for (int mz=0; mz<=3; ++mz) {
        string lbl = i2s(mx)+i2s(mz);
        u.saveProfile(mx,mz, "u"+lbl);
        f.saveProfile(mx,mz, "f"+lbl);
        lapl_u.saveProfile(mx,mz, "lapl_u"+lbl);
      }
    }
    ***********************************************/

    Real l2err = L2Dist(lapl_u, f);
    Real bcerr = bcNorm(u);
    cout << "PoissonSolver::verify(u, f) {\n";
    cout << "  L2Norm(u)         == " << L2Norm(u) << endl;
    cout << "  L2Norm(f)         == " << L2Norm(f) << endl;
    cout << "  L2Norm(lapl u)    == " << L2Norm(lapl_u) << endl;
    cout << "  L2Dist(lapl u, f) == " << l2err << endl;
    cout << "  bcNorm(u)         == " << bcerr << endl;
    cout << "} // PoissonSolver::verify(u, f)\n";
    return l2err + bcerr;
}

Real PoissonSolver::verify(const FlowField& u, const FlowField& f, const FlowField& bc) const {
    assert(this->congruent(u));
    assert(this->congruent(f));
    u.assertState(Spectral, Spectral);
    f.assertState(Spectral, Spectral);

    FlowField lapl_u;
    lapl(u, lapl_u);
    Real l2err = L2Dist(lapl_u, f);
    Real bcerr = bcDist(u, bc);
    cout << "PoissonSolver::verify(u, f) {\n";
    cout << "  L2Norm(u)         == " << L2Norm(u) << endl;
    cout << "  L2Norm(f)         == " << L2Norm(f) << endl;
    cout << "  L2Norm(lapl u)    == " << L2Norm(lapl_u) << endl;
    cout << "  L2Dist(lapl u, f) == " << l2err << endl;
    cout << "  bcNorm(u)         == " << bcNorm(u) << endl;
    cout << "  bcNorm(bc)        == " << bcNorm(bc) << endl;
    cout << "  bcDist(u,bc)      == " << bcerr << endl;
    cout << "} // PoissonSolver::verify(u, f)\n";
    return l2err + bcerr;
}

// ========================================================================

PressureSolver::PressureSolver()
    : PoissonSolver(), U_(), W_(), trans_(1), nonl_(), tmp_(), div_nonl_(), nu_(0), Vsuck_(0) {}

PressureSolver::PressureSolver(int Nx, int Ny, int Nz, Real Lx, Real Lz, Real a, Real b, const ChebyCoeff& U,
                               const ChebyCoeff& W, Real nu, Real Vsuck, NonlinearMethod nonl, CfMPI* cfmpi)
    : PoissonSolver(Nx, Ny, Nz, 1, Lx, Lz, a, b, cfmpi),
      U_(U),
      W_(W),
      trans_(Ny),
      nonl_(Nx, Ny, Nz, 3, Lx, Lz, a, b, cfmpi),
      tmp_(),  // geom will be set in call to navierstokesNL
      div_nonl_(Nx, Ny, Nz, 1, Lx, Lz, a, b, cfmpi),
      nu_(nu),
      Vsuck_(Vsuck),
      nonl_method_(nonl) {
    assert(U_.N() == Ny);
    U_.makeSpectral(trans_);
    assert(W_.N() == Ny);
    W_.makeSpectral(trans_);
}

PressureSolver::PressureSolver(const FlowField& u, Real nu, Real Vsuck, NonlinearMethod nl)
    : PoissonSolver(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi()),
      U_(u.Ny(), u.a(), u.b(), Spectral),
      W_(u.Ny(), u.a(), u.b(), Spectral),
      trans_(u.Ny()),
      nonl_(u.Nx(), u.Ny(), u.Nz(), 3, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi()),
      tmp_(),  // geom will be set in call to navierstokesNL
      div_nonl_(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi()),
      nu_(nu),
      Vsuck_(Vsuck),
      nonl_method_(nl) {}

PressureSolver::PressureSolver(const FlowField& u, const ChebyCoeff& U, const ChebyCoeff& W, Real nu, Real Vsuck,
                               NonlinearMethod nonl_method)
    : PoissonSolver(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi()),
      U_(U),
      W_(W),
      trans_(u.Ny()),
      nonl_(u.Nx(), u.Ny(), u.Nz(), 3, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi()),
      tmp_(),  // geom will be set in call to navierstokesNL
      div_nonl_(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi()),
      nu_(nu),
      Vsuck_(Vsuck),
      nonl_method_(nonl_method) {
    assert(U_.N() == u.Ny());
    assert(W_.N() == u.Ny());
    U_.makeSpectral(trans_);
    W_.makeSpectral(trans_);
}

PressureSolver::~PressureSolver() {}

FlowField PressureSolver::solve(const FlowField& u) {
    FlowField p;
    solve(p, u);
    return p;
}

void PressureSolver::solve(FlowField& p, FlowField u) {
    //     FlowField& u = const_cast<FlowField&>(u_);
    assert(u.xzstate() == Spectral && u.ystate() == Spectral);
    assert(geomCongruent(u));
    if (!congruent(p))
        p = FlowField(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());
    else {
        p.setToZero();
        p.setState(Spectral, Spectral);
    }
    // {Real l2 = L2Norm(u); if (u.taskid() == 0) cout << "1: " << l2 << endl;}
    // I. Get particular solution satisfying
    // lapl p = -div(nonl(u)) BCs p=0
    //        = -div(u grad u) for Convection nonlinearity, for example
    DNSFlags flags;
    flags.nonlinearity = nonl_method_;
    flags.Vsuck = Vsuck_;
    flags.nu = nu_;
    //  {Real l2 = L2Norm(u); if (u.taskid() == 0) cout << "2: " << l2 << endl;}
    navierstokesNL(u, U_, W_, nonl_, tmp_, flags);
    //  {Real l2 = L2Norm(u); if (u.taskid() == 0) cout << "3: " << l2 << endl;}
    div(nonl_, div_nonl_);
    div_nonl_ *= -1.0;

    PoissonSolver::solve(p, div_nonl_);
    // II. Adjust solution to match von Neumann BCs
    //   dp/dy ==  nu d^2 v/dy^2  at y=a and y=b
    // or
    //   dp/dy ==  nu (v_xx + v_zz - u_xy - w_yz) at y=a and y=b

    // Derivation of calculation in 5/19/05 notes, jfg. Idea is that
    //   g = c exp(lambda y) + d exp(-lambda y)
    // satisfies
    //   g" - lambda g = 0 with nonzero BCs.
    // Find the particular values of c and d such that adding g to p leaves
    //   p" - lambda p = f unchanged but sets dp/dy = nu d^v/dy^2 at y= a,b.

    // These tmp variables are for v_yy form of BCs
    ComplexChebyCoeff vk(My_, a_, b_, Spectral);
    ComplexChebyCoeff vkyy(My_, a_, b_, Spectral);

    // These tmp variables are for (v_xx + v_zz - u_xy - w_yz) form of BCs
    // Moderately INEFFICIENT implementation for PPE BCs, initial testing...
    ComplexChebyCoeff ukx(My_, a_, b_, Spectral);
    ComplexChebyCoeff wkz(My_, a_, b_, Spectral);

    ComplexChebyCoeff pk(My_, a_, b_, Spectral);
    ComplexChebyCoeff pky(My_, a_, b_, Spectral);
    ComplexChebyCoeff gk(My_, a_, b_, Physical);

    Vector y = p.ygridpts();
    lint mxlocmin = u.mxlocmin();
    lint mxlocmax = u.mxlocmin() + u.Mxloc();
    lint mzlocmin = u.mzlocmin();
    lint mzlocmax = u.mzlocmin() + u.Mzloc();

    Real H = b_ - a_;

    for (int mx = mxlocmin; mx < mxlocmax; ++mx)
        for (int mz = mzlocmin; mz < mzlocmax; ++mz) {
            // Don't modify the mx=mz=0 (kx=kz=0) solution, dirichlet BCs are fine
            if (mx != 0 || mz != 0) {
                Real lambda = helmholtz_[mx - mxlocmin][mz - mzlocmin].lambda();
                assert(lambda > 0);

                for (int my = 0; my < My_; ++my) {
                    pk.set(my, p.cmplx(mx, my, mz, 0));
                    vk.set(my, u.cmplx(mx, my, mz, 1));
                }
                diff(pk, pky);
                diff2(vk, vkyy);

                Complex alpha = nu_ * vkyy.eval_a() - pky.eval_a();
                Complex beta = nu_ * vkyy.eval_b() - pky.eval_b();

                // 2018-10-31, following jfg notes. Solve boundary value problem
                // g''(y) - mu^2 g(y) = 0, g'(a) = alpha, g'(b) = beta
                // using the general solution
                // g(y) = c exp(mu(y-a)) + d exp(-mu(y-b))
                // Determining constants c,d that match boundary conditions results
                // in the formula below for c,d. Note that this particular formulation
                // of the general solution is well-behaved numerically: both parts have
                // max 1 at one boundary and approach 0 at the other. The prior
                // formulation of the general solution and constants in terms of
                // (g(y) = c exp(mu y) + d exp(-mu y)) was subject to numerical
                // overflow for large mu and a,b.

                // Compute the coefficients c,d that produce g(y) that matches BCs
                Real mu = sqrt(lambda);  // solutions are more easily in terms of mu=sqrt(lambda)

                Real delta = mu * (1 - exp(-2 * mu * H));
                Complex c = (-alpha + beta * exp(-mu * H)) / delta;
                Complex d = (beta - alpha * exp(-mu * H)) / delta;

                // Evaluate g(y) at gridpoint values, transform to spectral, then add to p.
                gk.setState(Physical);
                for (int my = 0; my < My_; ++my)
                    gk.set(my, c * exp(-mu * (y[my] - a_)) + d * exp(mu * (y[my] - b_)));

                gk.makeSpectral(trans_);

                for (int my = 0; my < My_; ++my)
                    p.cmplx(mx, my, mz, 0) += gk[my];
            }
        }
    return;
}

Real PressureSolver::verify(const FlowField& p, const FlowField& u_) {
    FlowField& u = const_cast<FlowField&>(u_);

    assert(u.xzstate() == Spectral && u.ystate() == Spectral);
    assert(congruent(p));
    assert(geomCongruent(u));

    DNSFlags flags;
    flags.nonlinearity = nonl_method_;
    // I. Verify that solution satisfies lapl p = -div(nonl(u))
    navierstokesNL(u, U_, W_, nonl_, tmp_, flags);
    div(nonl_, div_nonl_);
    div_nonl_ *= -1.0;

    PoissonSolver::verify(p, div_nonl_);
    cout << "  L2Norm(u)           == " << L2Norm(u) << endl;

    FlowField lapl_p;
    lapl(p, lapl_p);

    Real l2err = L2Dist(lapl_p, div_nonl_);
    cout << "PressureSolver::verify(p,u) {\n";
    cout << "  L2Norm(u)           == " << L2Norm(u) << endl;
    cout << "  L2Norm(div(nonl(u)) == " << L2Norm(div_nonl_) << endl;
    cout << "  L2Norm(lapl p)      == " << L2Norm(lapl_p) << endl;
    cout << "  L2Dist(lapl p, div(nonl(u))) == " << l2err << endl;
    // I. Verify that solution satisfies dpdy = nu d^2 /dy^2 (u+U) on boundary
    FlowField dpdy;
    ydiff(p, dpdy);
    FlowField v = u[1];
    /************************************
    FlowField v(u.Nx(),u.Ny(),u.Nz(),1,u.Lx(),u.Lz(),u.a(),u.b());
    for (int my=0; my<My_; ++my)
      for (int mx=0; mx<Mx_; ++mx)
        for (int mz=0; mz<Mz_; ++mz)
      v.cmplx(mx,my,mz,0) = u.cmplx(mx,my,mz,1);
    **************************************/
    FlowField nu_vyy;
    ydiff(v, nu_vyy, 2);
    nu_vyy *= nu_;

    Real bcerr = bcDist(dpdy, nu_vyy);
    cout << "  bcNorm(dpdy)         == " << bcNorm(dpdy) << endl;
    cout << "  bcNorm(nu_vyy)       == " << bcNorm(nu_vyy) << endl;
    cout << "  bcDist(dpdy, nu_vyy) == " << bcerr << endl;

    ComplexChebyCoeff vk(My_, a_, b_, Spectral);
    ComplexChebyCoeff vkyy(My_, a_, b_, Spectral);
    ComplexChebyCoeff pk(My_, a_, b_, Spectral);
    ComplexChebyCoeff pky(My_, a_, b_, Spectral);
    cout << "} // PressureSolver::verify(p,u)\n";

    return l2err + bcerr;
}

}  // namespace chflow
