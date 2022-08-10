/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "channelflow/diffops.h"

#include <fstream>
#include <iomanip>

#include "cfbasics/mathdefs.h"
#include "utilfuncs.h"

using namespace std;

namespace chflow {

Real bcNorm2(const FlowField& f, bool normalize) {
    assert(f.xzstate() == Spectral);
    Real bc2 = 0.0;
    const lint mxlocmin = f.mxlocmin();
    const lint mxlocmax = f.mxlocmin() + f.Mxloc();
    const lint My = f.My();
    const lint mzlocmin = f.mzlocmin();
    const lint mzlocmax = f.mzlocmin() + f.Mzloc();
    const int Nd = f.Nd();

    ComplexChebyCoeff prof(f.Ny(), f.a(), f.b(), f.ystate());
    for (lint i = 0; i < Nd; ++i)
        for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
            for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                for (lint my = 0; my < My; ++my)
                    prof.set(my, f.cmplx(mx, my, mz, i));
                bc2 += abs2(prof.eval_a());
                bc2 += abs2(prof.eval_b());
            }
        }
    if (!normalize) {
        bc2 *= f.Lx() * f.Lz();
    }

    // Sum up results from all processes
#ifdef HAVE_MPI
    Real tmp = bc2;
    MPI_Allreduce(&tmp, &bc2, 1, MPI_DOUBLE, MPI_SUM, *f.comm_world());
#endif
    return bc2;
}

Real bcDist2(const FlowField& f, const FlowField& g, bool normalize) {
    assert(f.congruent(g));
    assert(f.xzstate() == Spectral && g.xzstate() == Spectral);
    // assert(f.ystate() == Spectral && g.ystate() == Spectral);

    Real bc2 = 0.0;
    const lint mxlocmin = f.mxlocmin();
    const lint mxlocmax = f.mxlocmin() + f.Mxloc();
    const lint My = f.My();
    const lint mzlocmin = f.mzlocmin();
    const lint mzlocmax = f.mzlocmin() + f.Mzloc();
    const int Nd = f.Nd();
    ComplexChebyCoeff diff(My, f.a(), f.b(), f.ystate());

    for (int i = 0; i < Nd; ++i)
        for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
            for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                for (lint my = 0; my < My; ++my)
                    diff.set(my, f.cmplx(mx, my, mz, i) - g.cmplx(mx, my, mz, i));
                bc2 += abs2(diff.eval_a());
                bc2 += abs2(diff.eval_b());
            }
        }
    if (!normalize) {
        bc2 *= f.Lx() * f.Lz();
    }
    // Sum up results from all processes
#ifdef HAVE_MPI
    Real tmp = bc2;
    MPI_Allreduce(&tmp, &bc2, 1, MPI_DOUBLE, MPI_SUM, *f.comm_world());
#endif

    return bc2;
}

Real bcNorm(const FlowField& f, bool normalize) { return sqrt(bcNorm2(f, normalize)); }

Real bcDist(const FlowField& f, const FlowField& g, bool normalize) { return sqrt(bcDist2(f, g, normalize)); }

Real divNorm(const FlowField& f, bool normalize) { return sqrt(divNorm2(f, normalize)); }

Real divNorm2(const FlowField& f, bool normalize) {
    assert(f.xzstate() == Spectral && f.ystate() == Spectral);
    assert(f.Nd() == 3);

    Real div2 = 0.0;
    //     int Mx = f.Mx();
    //     int Mz = f.Mz();
    const lint mxlocmin = f.mxlocmin();
    const lint mxlocmax = f.mxlocmin() + f.Mxloc();
    const lint mzlocmin = f.mzlocmin();
    const lint mzlocmax = f.mzlocmin() + f.Mzloc();

    BasisFunc prof;
    for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
        for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
            div2 += divNorm2(f.profile(mx, mz), normalize);
        }
    }

    // Sum up results from all processes
#ifdef HAVE_MPI
    Real tmp = div2;
    MPI_Allreduce(&tmp, &div2, 1, MPI_DOUBLE, MPI_SUM, *f.comm_world());
#endif

    return div2;
}

Real divDist2(const FlowField& f, const FlowField& g, bool normalize) {
    assert(f.xzstate() == Spectral && f.ystate() == Spectral);
    assert(f.congruent(g));
    assert(f.Nd() == 3);

    Real div2 = 0.0;
    //     int Mx = f.Mx();
    //     int Mz = f.Mz();
    const lint mxlocmin = f.mxlocmin();
    const lint mxlocmax = f.mxlocmin() + f.Mxloc();
    const lint mzlocmin = f.mzlocmin();
    const lint mzlocmax = f.mzlocmin() + f.Mzloc();
    for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
        for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
            div2 += divDist2(f.profile(mx, mz), g.profile(mx, mz), normalize);
        }
    }

    // Sum up results from all processes
#ifdef HAVE_MPI
    Real tmp = div2;
    MPI_Allreduce(&tmp, &div2, 1, MPI_DOUBLE, MPI_SUM, *f.comm_world());
#endif
    return div2;
}

Real divDist(const FlowField& f, const FlowField& g, bool normalize) { return sqrt(divDist2(f, g, normalize)); }

Real L1Norm(const FlowField& u, bool normalize) {
    fieldstate xstate = u.xzstate();
    fieldstate ystate = u.ystate();
    ((FlowField&)u).makePhysical_xz();

    Real sum = 0.0;

    ChebyCoeff prof;
    ChebyTransform trans(u.Ny());
    for (int i = 0; i < u.vectorDim(); ++i)
        for (int nx = 0; nx < u.Nx(); ++nx)
            for (int nz = 0; nz < u.Nz(); ++nz) {
                prof = Re(u.profile(nx, nz, i));
                prof.makePhysical(trans);
                sum += L1Norm(prof);
            }

    sum /= u.Nx() * u.Nz();

    if (!normalize)
        sum *= u.Lx() * u.Lz();

    ((FlowField&)u).makeState(xstate, ystate);

    return sum;
}

Real L1Dist(const FlowField& f, const FlowField& g, bool normalize) {
    assert(f.congruent(g));
    fieldstate fxstate = f.xzstate();
    fieldstate gxstate = g.xzstate();
    fieldstate fystate = f.ystate();
    fieldstate gystate = g.ystate();

    ((FlowField&)f).makePhysical_xz();
    ((FlowField&)g).makePhysical_xz();

    Real sum = 0.0;

    ChebyCoeff prof;
    ChebyTransform trans(f.Ny());
    for (int i = 0; i < f.vectorDim(); ++i)
        for (int nx = 0; nx < f.Nx(); ++nx)
            for (int nz = 0; nz < f.Nz(); ++nz) {
                prof = Re(f.profile(nx, nz, i));
                prof -= Re(g.profile(nx, nz, i));
                prof.makePhysical(trans);
                sum += L1Norm(prof, normalize);
            }

    sum /= f.Nx() * f.Nz();

    if (!normalize)
        sum *= f.Lx() * f.Lz();

    ((FlowField&)f).makeState(fxstate, fystate);
    ((FlowField&)g).makeState(gxstate, gystate);

    return sum;
}

Real LinfNorm(const FlowField& u) {
    fieldstate xstate = u.xzstate();
    fieldstate ystate = u.ystate();
    ((FlowField&)u).makePhysical_xz();

    Real rtn = 0.0;

    ChebyCoeff prof;
    ChebyTransform trans(u.Ny());
    for (int i = 0; i < u.vectorDim(); ++i)
        for (int nx = 0; nx < u.Nx(); ++nx)
            for (int nz = 0; nz < u.Nz(); ++nz) {
                prof = Re(u.profile(nx, nz, i));
                prof.makePhysical(trans);
                rtn = Greater(rtn, LinfNorm(prof));
            }

    ((FlowField&)u).makeState(xstate, ystate);

    return rtn;
}

Real LinfDist(const FlowField& f, const FlowField& g) {
    assert(f.congruent(g));
    fieldstate fxstate = f.xzstate();
    fieldstate gxstate = g.xzstate();
    fieldstate fystate = f.ystate();
    fieldstate gystate = g.ystate();

    ((FlowField&)f).makePhysical_xz();
    ((FlowField&)g).makePhysical_xz();

    Real rtn = 0.0;

    ChebyCoeff prof;
    ChebyTransform trans(f.Ny());
    for (int i = 0; i < f.vectorDim(); ++i)
        for (int nx = 0; nx < f.Nx(); ++nx)
            for (int nz = 0; nz < f.Nz(); ++nz) {
                prof = Re(f.profile(nx, nz, i));
                prof -= Re(g.profile(nx, nz, i));
                prof.makePhysical();
                rtn = Greater(rtn, LinfNorm(prof));
            }

    ((FlowField&)f).makeState(fxstate, fystate);
    ((FlowField&)g).makeState(gxstate, gystate);

    return rtn;
}

Real chebyNorm(const FlowField& f, bool normalize) { return sqrt(chebyNorm2(f, normalize)); }
Real chebyNorm2(const FlowField& f, bool normalize) {
    Real sum = 0.0;
    const bool padded = f.padded();
    const int kxmin = padded ? -f.kxmaxDealiased() : f.kxmin();
    const int kxmax = padded ? f.kxmaxDealiased() : f.kxmax();
    const int kzmax = padded ? f.kzmaxDealiased() : f.kzmax();

    const lint mxlocmin = f.mxlocmin();
    const lint mxlocmax = f.mxlocmin() + f.Mxloc();
    const lint mzlocmin = f.mzlocmin();
    const lint mzlocmax = f.mzlocmin() + f.Mzloc();

    for (int i = 0; i < f.vectorDim(); ++i)
        for (int ny = f.Ny() - 1; ny >= 0; --ny) {
            const int cy = (ny == 0) ? 2 : 1;
            //             for (int kx=kxmin; kx<=kxmax; ++kx) {
            for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
                const int kx = f.kx(mx);
                if (kx >= kxmin && kx <= kxmax) {
                    //                 const int mx = f.mx(kx);
                    for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                        const int kz = f.kz(mz);
                        if (kz >= 0 && kz <= kzmax) {
                            const int cz = (kz > 0) ? 2 : 1;
                            sum += cy * cz * abs2(f.cmplx(mx, ny, mz, i));
                        }
                    }
                }
            }
        }
    if (!normalize)
        sum *= f.Lx() * f.Ly() * f.Lz();
    sum *= pi / 2;

    // Sum up results from all processes
#ifdef HAVE_MPI
    Real tmp = sum;
    MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, *f.comm_world());
#endif
    return sum;
}

Real chebyDist(const FlowField& f, const FlowField& g, bool normalize) { return sqrt(chebyDist2(f, g, normalize)); }

Real chebyDist2(const FlowField& f, const FlowField& g, bool normalize) {
    Real sum = 0.0;
    assert(f.congruent(g));
    assert(f.xzstate() == Spectral && f.ystate() == Spectral);
    assert(g.xzstate() == Spectral && g.ystate() == Spectral);
    const bool padded = f.padded() && g.padded();
    const int kxmin = padded ? -f.kxmaxDealiased() : f.kxmin();
    const int kxmax = padded ? f.kxmaxDealiased() : f.kxmax();
    const int kzmax = padded ? f.kzmaxDealiased() : f.kzmax();

    const lint mxlocmin = f.mxlocmin();
    const lint mxlocmax = f.mxlocmin() + f.Mxloc();
    const lint mzlocmin = f.mzlocmin();
    const lint mzlocmax = f.mzlocmin() + f.Mzloc();

    for (int i = 0; i < f.vectorDim(); ++i)
        for (int ny = f.Ny() - 1; ny >= 0; --ny) {
            const int cy = (ny == 0) ? 2 : 1;
            //             for (int kx=kxmin; kx<=kxmax; ++kx) {
            //                 const int mx = f.mx(kx);
            for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
                const int kx = f.kx(mx);
                if (kx >= kxmin && kx <= kxmax) {
                    for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                        const int kz = f.kz(mz);
                        if (kz >= 0 && kz <= kzmax) {
                            const int cz = (kz > 0) ? 2 : 1;
                            // 					int cz = 1; // cz = 2 for kz>0 to take account of kz<0
                            // ghost modes 					for (int kz=0; kz<=kzmax; ++kz) {
                            sum += cy * cz * abs2(f.cmplx(mx, ny, mz, i) - g.cmplx(mx, ny, mz, i));
                        }
                    }
                }
            }
        }
    if (!normalize)
        sum *= f.Lx() * f.Ly() * f.Lz();
    sum *= pi / 2;

    // Sum up results from all processes
#ifdef HAVE_MPI
    Real tmp = sum;
    MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, *f.comm_world());
#endif
    return sum;
}

Real L2Dist(const FlowField& u, const FlowField& v, bool normalize) { return sqrt(L2Dist2(u, v, normalize)); }

Real L2Dist2(const FlowField& u, const FlowField& v, bool normalize) {
    Real sum = 0.0;
    assert(u.congruent(v));
    assert(u.xzstate() == Spectral && u.ystate() == Spectral);
    ComplexChebyCoeff u_v(u.Ny(), u.a(), u.b(), Spectral);
    bool padded = u.padded() && v.padded();
    int kxmin = padded ? -u.kxmaxDealiased() : u.kxmin();
    int kxmax = padded ? u.kxmaxDealiased() : u.kxmax();
    int kzmin = 0;
    int kzmax = padded ? u.kzmaxDealiased() : u.kzmax();

    const lint mxlocmin = u.mxlocmin();
    const lint mxlocmax = u.mxlocmin() + u.Mxloc();
    const lint mzlocmin = u.mzlocmin();
    const lint mzlocmax = u.mzlocmin() + u.Mzloc();
#ifdef HAVE_MPI
    int cz = 1;  // cz = 2 for kz>0 to take account of kz<0 ghost modes
    for (int kz = kzmin; kz <= kzmax; ++kz) {
        lint mz = u.mz(kz);
        if (mz >= mzlocmin && mz < mzlocmax) {
            for (int kx = kxmin; kx <= kxmax; ++kx) {
                lint mx = u.mx(kx);
                if (mx >= mxlocmin && mx < mxlocmax)
                    for (int i = 0; i < u.vectorDim(); ++i) {
                        for (lint ny = 0; ny < u.Ny(); ++ny)
                            u_v.set(ny, u.cmplx(mx, ny, mz, i) - v.cmplx(mx, ny, mz, i));
                        sum += cz * L2Norm2(u_v, normalize);
                    }
            }
        }
        cz = 2;
    }
#else
    for (int i = 0; i < u.vectorDim(); ++i)
        for (int kx = kxmin; kx <= kxmax; ++kx) {
            lint mx = u.mx(kx);
            if (mx >= mxlocmin && mx < mxlocmax) {
                int cz = 1;  // cz = 2 for kz>0 to take account of kz<0 ghost modes
                for (int kz = kzmin; kz <= kzmax; ++kz) {
                    lint mz = u.mz(kz);
                    if (mz >= mzlocmin && mz < mzlocmax) {
                        for (lint ny = 0; ny < u.Ny(); ++ny)
                            u_v.set(ny, u.cmplx(mx, ny, mz, i) - v.cmplx(mx, ny, mz, i));
                        sum += cz * L2Norm2(u_v, normalize);
                    }
                    cz = 2;
                }
            }
        }
#endif
    if (!normalize) {
        sum *= u.Lx() * u.Lz();
    }

    // Sum up results from all processes
#ifdef HAVE_MPI
    Real tmp = sum;
    MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, *u.comm_world());
#endif
    return sum;
}

Real L2Norm(const FlowField& u, bool normalize) { return sqrt(L2Norm2(u, normalize)); }

Real L2Norm2(const FlowField& u, bool normalize) {
    // 	if(u.taskid() == 0) cout << "Entering l2Norm2" << endl;
    assert(u.ystate() == Spectral);
    assert(u.xzstate() == Spectral);
    Real sum = 0.0;

    const int kxmin = u.padded() ? -u.kxmaxDealiased() : u.kxmin();
    const int kxmax = u.padded() ? u.kxmaxDealiased() : u.kxmax();
    const int kzmin = 0;
    const int kzmax = u.padded() ? u.kzmaxDealiased() : u.kzmax();

    const lint mxlocmin = u.mxlocmin();
    const lint mxlocmax = u.mxlocmin() + u.Mxloc();
    const lint mzlocmin = u.mzlocmin();
    const lint mzlocmax = u.mzlocmin() + u.Mzloc();
#ifdef HAVE_MPI
    // 	ComplexChebyCoeff prof(u.Ny(), u.a(), u.b(), Spectral);
    int cz = 1;  // cz = 2 for kz>0 to take account of kz<0 ghost modes
    for (int kz = kzmin; kz <= kzmax; ++kz) {
        lint mz = u.mz(kz);
        if (mz >= mzlocmin && mz < mzlocmax) {
            for (int kx = kxmin; kx <= kxmax; ++kx) {
                lint mx = u.mx(kx);
                if (mx >= mxlocmin && mx < mxlocmax) {
                    for (int i = 0; i < u.vectorDim(); ++i) {
                        ComplexChebyCoeff prof(u.Ny(), u.a(), u.b(), Spectral);
                        for (int ny = 0; ny < u.Ny(); ++ny)
                            prof.set(ny, u.cmplx(mx, ny, mz, i));
                        sum += cz * L2Norm2(prof, normalize);
                    }
                }
            }
        }
        cz = 2;
    }
#else
    for (int i = 0; i < u.vectorDim(); ++i) {
        ComplexChebyCoeff prof(u.Ny(), u.a(), u.b(), Spectral);
        for (int kx = kxmin; kx <= kxmax; ++kx) {
            lint mx = u.mx(kx);
            if (mx >= mxlocmin && mx < mxlocmax) {
                int cz = 1;  // cz = 2 for kz>0 to take account of kz<0 ghost modes
                for (int kz = kzmin; kz <= kzmax; ++kz) {
                    lint mz = u.mz(kz);
                    if (mz >= mzlocmin && mz < mzlocmax) {
                        for (int ny = 0; ny < u.Ny(); ++ny)
                            prof.set(ny, u.cmplx(mx, ny, mz, i));
                        sum += cz * L2Norm2(prof, normalize);
                    }
                    cz = 2;
                }
            }
        }
    }
#endif
    if (!normalize)
        sum *= u.Lx() * u.Lz();
//     MPI_Barrier(MPI_COMM_WORLD);

// 	cout << u.taskid() << ": " << sum << endl;
// Sum up results from all processes
#ifdef HAVE_MPI
    Real tmp = sum;
    // 		MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, /*MPI_COMM_WORLD*/);
    MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, *u.comm_world());
#endif
    // 	cout << u.taskid() << ": " << sum << endl;
    // 	if(u.taskid() == 0) cout << "Leaving l2Norm2" << endl << endl;

    return sum;
}

Real L2InnerProduct(const FlowField& f, const FlowField& g, bool normalize) {
    assert(f.ystate() == Spectral);
    assert(g.ystate() == Spectral);
    assert(f.xzstate() == Spectral);
    assert(g.xzstate() == Spectral);
    assert(f.congruent(g));

    Real sum = 0.0;

    bool padded = f.padded() || g.padded();
    int kxmin = padded ? -f.kxmaxDealiased() : f.kxmin();
    int kxmax = padded ? f.kxmaxDealiased() : f.kxmax();
    int kzmin = 0;
    int kzmax = padded ? f.kzmaxDealiased() : f.kzmax();
    const lint mxlocmin = f.mxlocmin();
    const lint mxlocmax = f.mxlocmin() + f.Mxloc();
    const lint mzlocmin = f.mzlocmin();
    const lint mzlocmax = f.mzlocmin() + f.Mzloc();

    for (int i = 0; i < f.vectorDim(); ++i) {
        ComplexChebyCoeff fprof(f.Ny(), f.a(), f.b(), Spectral);
        ComplexChebyCoeff gprof(g.Ny(), g.a(), g.b(), Spectral);
        for (int kx = kxmin; kx <= kxmax; ++kx) {
            lint mx = f.mx(kx);
            if (mx >= mxlocmin && mx < mxlocmax) {
                int cz = 1;  // cz = 2 for kz>0 to take account of kz<0 ghost modes
                for (int kz = kzmin; kz <= kzmax; ++kz) {
                    int mz = f.mz(kz);
                    if (mz >= mzlocmin && mz < mzlocmax) {
                        for (int ny = 0; ny < f.Ny(); ++ny) {
                            fprof.set(ny, f.cmplx(mx, ny, mz, i));
                            gprof.set(ny, g.cmplx(mx, ny, mz, i));
                        }
                        sum += cz * L2InnerProduct(fprof.re, gprof.re, normalize);
                        sum += cz * L2InnerProduct(fprof.im, gprof.im, normalize);
                    }
                    cz = 2;
                }
            }
        }
    }

    if (!normalize) {
        sum *= f.Lx() * f.Lz();
    }

    // Sum up results from all processes
#ifdef HAVE_MPI
    Real tmp = sum;
    MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, *f.comm_world());
#endif
    return sum;
}

Real L2InnerProduct(const FlowField& f, const FlowField& g, int kxmax, int kzmax, bool normalize) {
    assert(f.ystate() == Spectral);
    assert(g.ystate() == Spectral);
    assert(f.xzstate() == Spectral);
    assert(g.xzstate() == Spectral);
    assert(f.congruent(g));

    Real sum = 0.0;

    if (kxmax < 0 || kxmax > f.kxmax() || kxmax > g.kxmax())
        kxmax = lesser(f.padded() ? f.kxmaxDealiased() : f.kxmax(), g.padded() ? g.kxmaxDealiased() : g.kxmax());
    if (kzmax < 0 || kzmax > f.kzmax() || kzmax > g.kzmax())
        kzmax = lesser(f.padded() ? f.kzmaxDealiased() : f.kzmax(), g.padded() ? g.kzmaxDealiased() : g.kzmax());

    int kxmin = -kxmax;
    int kzmin = 0;
    const lint mxlocmin = f.mxlocmin();
    const lint mxlocmax = f.mxlocmin() + f.Mxloc();
    const lint mzlocmin = f.mzlocmin();
    const lint mzlocmax = f.mzlocmin() + f.Mzloc();

    for (int i = 0; i < f.vectorDim(); ++i) {
        ComplexChebyCoeff fprof(f.Ny(), f.a(), f.b(), Spectral);
        ComplexChebyCoeff gprof(g.Ny(), g.a(), g.b(), Spectral);
        for (int kx = kxmin; kx <= kxmax; ++kx) {
            lint mx = f.mx(kx);
            if (mx >= mxlocmin && mx < mxlocmax) {
                int cz = 1;  // cz = 2 for kz>0 to take account of kz<0 ghost modes
                for (int kz = kzmin; kz <= kzmax; ++kz) {
                    lint mz = f.mz(kz);
                    if (mz >= mzlocmin && mz < mzlocmax) {
                        for (int ny = 0; ny < f.Ny(); ++ny) {
                            fprof.set(ny, f.cmplx(mx, ny, mz, i));
                            gprof.set(ny, g.cmplx(mx, ny, mz, i));
                        }
                        sum += cz * L2InnerProduct(fprof.re, gprof.re, normalize);
                        sum += cz * L2InnerProduct(fprof.im, gprof.im, normalize);
                    }
                    cz = 2;
                }
            }
        }
    }
    if (!normalize) {
        sum *= f.Lx() * f.Lz();
    }

    // Sum up results from all processes
#ifdef HAVE_MPI
    Real tmp = sum;
    MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, *f.comm_world());
#endif
    return sum;
}

Real L2Norm2(const FlowField& u, int kxmax, int kzmax, bool normalize) {
    assert(u.ystate() == Spectral);
    assert(u.xzstate() == Spectral);
    Real sum = 0.0;

    if (kxmax < 0 || kxmax > u.kxmax()) {
        kxmax = (u.padded()) ? u.kxmaxDealiased() : u.kxmax();
    }

    if (kzmax < 0 || kzmax > u.kzmax()) {
        kzmax = (u.padded()) ? u.kzmaxDealiased() : u.kzmax();
    }

    const lint mxlocmin = u.mxlocmin();
    const lint mxlocmax = u.mxlocmin() + u.Mxloc();
    const lint mzlocmin = u.mzlocmin();
    const lint mzlocmax = u.mzlocmin() + u.Mzloc();

    // EFFICIENCY: would be better with smarter looping, but prob not worth trouble.

    for (int i = 0; i < u.vectorDim(); ++i) {
        ComplexChebyCoeff prof(u.Ny(), u.a(), u.b(), Spectral);
        for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
            if (abs(u.kx(mx)) > kxmax)
                continue;
            Real cz = 1.0;  // cz = 2 for kz>0 to take account of kz<0 ghost modes
            for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                if (abs(u.kz(mz)) > kzmax)
                    continue;
                for (lint ny = 0; ny < u.Ny(); ++ny)
                    prof.set(ny, u.cmplx(mx, ny, mz, i));
                sum += cz * L2Norm2(prof, normalize);
                cz = 2.0;
            }
        }
    }
    if (!normalize) {
        sum *= u.Lx() * u.Lz();
    }

    // Sum up results from all processes
#ifdef HAVE_MPI
    Real tmp = sum;
    MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, *u.comm_world());
#endif
    return sum;
}

Real L2Norm(const FlowField& u, int kxmax, int kzmax, bool normalize) {
    return sqrt(L2Norm2(u, kxmax, kzmax, normalize));
}

Real L2Dist(const FlowField& u, const FlowField& v, int kxmax, int kzmax, bool normalize) {
    return sqrt(L2Dist2(u, v, kxmax, kzmax, normalize));
}

Real L2Dist2(const FlowField& u, const FlowField& v, int kxmax, int kzmax, bool normalize) {
    Real sum = 0.0;
    assert(u.congruent(v));
    assert(u.xzstate() == Spectral && u.ystate() == Spectral);

    if (kxmax == 0) {
        kxmax = u.padded() ? u.kxmaxDealiased() : u.kxmax();
    }

    if (kzmax == 0) {
        kzmax = u.padded() ? u.kzmaxDealiased() : u.kzmax();
    }

    const lint mxlocmin = u.mxlocmin();
    const lint mxlocmax = u.mxlocmin() + u.Mxloc();
    const lint mzlocmin = u.mzlocmin();
    const lint mzlocmax = u.mzlocmin() + u.Mzloc();

    for (int i = 0; i < u.vectorDim(); ++i) {
        ComplexChebyCoeff u_v(u.Ny(), u.a(), u.b(), Spectral);
        for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
            if (abs(u.kx(mx)) > kxmax)
                continue;
            int cz = 1;
            for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                if (abs(u.kz(mz)) > kzmax)
                    continue;
                for (lint ny = 0; ny < u.Ny(); ++ny)
                    u_v.set(ny, u.cmplx(mx, ny, mz, i) - v.cmplx(mx, ny, mz, i));
                sum += cz * L2Norm2(u_v, normalize);
                cz = 2;
            }
        }
    }
    if (!normalize) {
        sum *= u.Lx() * u.Lz();
    }

    // Sum up results from all processes
#ifdef HAVE_MPI
    Real tmp = sum;
    MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, *u.comm_world());
#endif
    return sum;
}

Real L2Norm2_3d(const FlowField& u, bool normalize) {
    assert(u.ystate() == Spectral);
    assert(u.xzstate() == Spectral);
    Real sum = 0.0;

    const int kxmax = (u.padded()) ? u.kxmaxDealiased() : u.kxmax();
    const int kzmax = (u.padded()) ? u.kzmaxDealiased() : u.kzmax();
    const lint mxlocmin = u.mxlocmin() > 1 ? u.mxlocmin() : 1;  // omit mx=0 modes
    const lint mxlocmax = u.mxlocmin() + u.Mxloc();
    const lint mzlocmin = u.mzlocmin();
    const lint mzlocmax = u.mzlocmin() + u.Mzloc();

    // EFFICIENCY: would be better with smarter looping, but prob not worth trouble.
    ComplexChebyCoeff prof(u.Ny(), u.a(), u.b(), Spectral);
    for (int i = 0; i < u.vectorDim(); ++i)
        for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
            if (abs(u.kx(mx)) > kxmax)
                continue;
            Real cz = 1.0;  // cz = 2 for kz>0 to take account of kz<0 ghost modes
            for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                if (abs(u.kz(mz)) > kzmax)
                    continue;
                for (lint ny = 0; ny < u.Ny(); ++ny)
                    prof.set(ny, u.cmplx(mx, ny, mz, i));
                sum += cz * L2Norm2(prof, normalize);
                cz = 2.0;
            }
        }
    if (!normalize) {
        sum *= u.Lx() * u.Lz();
    }

    // Sum up results from all processes
#ifdef HAVE_MPI
    Real tmp = sum;
    MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, *u.comm_world());
#endif
    return sum;
}

Real L2Norm3d(const FlowField& u, bool normalize) { return sqrt(L2Norm2_3d(u, normalize)); }

Complex L2InnerProduct(const FlowField& u, const BasisFunc& phi, bool normalize) {
    assert(u.xzstate() == Spectral && u.ystate() == Spectral);
    assert(u.congruent(phi));
    BasisFunc psi = u.profile(u.mx(phi.kx()), u.mz(phi.kz()));
    return L2InnerProduct(psi, phi, normalize);
}

Real L2InnerProduct(const FlowField& u, const RealProfile& f, bool normalize) {
    assert(u.xzstate() == Spectral && u.ystate() == Spectral);
    assert(u.congruent(f.psi));
    BasisFunc uprofile = u.profile(u.mx(f.kx()), u.mz(f.kz()));
    Real rtn = 0.0;
    // INEFFICIENT
    Complex ip = L2InnerProduct(uprofile, f.psi, normalize);
    switch (f.sign()) {
        case Minus:
            rtn = 2 * Im(ip);
            break;
        case Plus:
            rtn = 2 * Re(ip);
            break;
    }
    return rtn;
}

Real dissipation(const FlowField& u, bool normalize) {
    Real D = 0.0;
    FlowField ui;
    FlowField grad_ui;
    for (int i = 0; i < u.Nd(); ++i) {
        ui = u[i];
        grad(ui, grad_ui);
        D += L2Norm2(grad_ui, normalize);
    }
    return D;
}

Real wallshear(const FlowField& f, bool normalize) {
    Real dfdy_a = sqrt(f.dudy_a() * f.dudy_a() + f.dwdy_a() * f.dwdy_a());
    Real dfdy_b = sqrt(f.dudy_b() * f.dudy_b() + f.dwdy_b() * f.dwdy_b());
    Real I = 0.5 * (abs(dfdy_a) + abs(dfdy_b));
    if (!normalize)
        I *= 2 * f.Lx() * f.Lz();
    return I;
}

Real wallshearLower(const FlowField& f, bool normalize) {
    Real dfdy_a = sqrt(f.dudy_a() * f.dudy_a() + f.dwdy_a() * f.dwdy_a());
    Real I = 0.5 * dfdy_a;
    if (!normalize)
        I *= 2 * f.Lx() * f.Lz();
    return I;
}

Real wallshearUpper(const FlowField& f, bool normalize) {
    Real dfdy_b = sqrt(f.dudy_b() * f.dudy_b() + f.dwdy_b() * f.dwdy_b());
    Real I = 0.5 * dfdy_b;
    if (!normalize)
        I *= 2 * f.Lx() * f.Lz();
    return I;
}

cfarray<Real> truncerr(const FlowField& u) {
    cfarray<Real> rtn(3);

    int kxmin = u.padded() ? u.kxminDealiased() : u.kxmin();
    int kxmax = u.padded() ? u.kxmaxDealiased() : u.kxmax();
    int kzmax = u.padded() ? u.kzmaxDealiased() : u.kzmax();

    const lint mxlocmin = u.mxlocmin() > 1 ? u.mxlocmin() : 1;  // omit mx=0 modes
    const lint mxlocmax = u.mxlocmin() + u.Mxloc();
    const lint mzlocmin = u.mzlocmin();
    const lint mzlocmax = u.mzlocmin() + u.Mzloc();

    int mzmax = mzlocmax < u.mz(kzmax) ? mzlocmax : u.mz(kzmax);
    int mymax = u.My() - 1;

    Real truncation = 0.0;

    for (int kx = kxmin; kx <= kxmax; kx += (kxmax - kxmin)) {
        int mx = u.mx(kx);
        if (mx >= mxlocmin && mx < mxlocmax) {
            for (int mz = mzlocmin; mz <= mzmax; ++mz) {
                BasisFunc prof = u.profile(mx, mz);
                Real trunc = L2Norm(prof);
                if (trunc > truncation)
                    truncation = trunc;
            }
        }
    }

#ifdef HAVE_MPI
    MPI_Allreduce(&truncation, &rtn[0], 1, MPI_DOUBLE, MPI_MAX, *u.comm_world());
#else
    rtn[0] = truncation;
#endif

    truncation = 0.0;
    for (int i = 0; i < u.Nd(); ++i) {
        for (int kx = kxmin; kx <= kxmax; ++kx) {
            int mx = u.mx(kx);
            if (mx >= mxlocmin && mx < mxlocmax) {
                for (int kz = 0; kz <= kzmax; ++kz) {
                    int mz = u.mz(kz);
                    if (mz >= mzlocmin && mz < mzlocmax) {
                        Real trunc = abs(u.cmplx(mx, mymax, mz, i));
                        if (trunc > truncation)
                            truncation = trunc;
                    }
                }
            }
        }
    }

#ifdef HAVE_MPI
    MPI_Allreduce(&truncation, &rtn[1], 1, MPI_DOUBLE, MPI_MAX, *u.comm_world());
#else
    rtn[1] = truncation;
#endif

    truncation = 0.0;
    if (mzmax >= mzlocmin && mzmax < mzlocmax)
        for (int kx = kxmin; kx <= kxmax; ++kx) {
            int mx = u.mx(kx);
            if (mx >= mxlocmin && mx < mxlocmax) {
                BasisFunc prof = u.profile(mx, mzmax);
                Real trunc = L2Norm(prof);
                if (trunc > truncation)
                    truncation = trunc;
            }
        }

#ifdef HAVE_MPI
    MPI_Allreduce(&truncation, &rtn[2], 1, MPI_DOUBLE, MPI_MAX, *u.comm_world());
#else
    rtn[2] = truncation;
#endif

    return rtn;
}

void field2coeff(const vector<RealProfile>& basis, const FlowField& u, cfarray<Real>& a) {
    int N = basis.size();
    if (a.N() != N)
        a.resize(N);
    for (int n = 0; n < N; ++n)
        a[n] = L2InnerProduct(u, basis[n]);
}

void coeff2field(const vector<RealProfile>& basis, const cfarray<Real>& a, FlowField& u) {
    int N = basis.size();
    assert(a.N() == N);
    u.setToZero();
    RealProfile tmp;

    const Real EPSILON = 1e-16;
    for (int n = 0; n < N; ++n) {
        Real an = a[n];
        if (abs(an) < EPSILON)
            continue;

        tmp = basis[n];
        tmp *= an;

        assert(tmp.kx() <= u.kxmax() && tmp.kx() >= u.kxmin() && tmp.kz() <= u.kzmax() && tmp.kz() >= u.kzmin());

        u += tmp;
    }
}

void field2coeff(const vector<BasisFunc>& basis, const FlowField& u, cfarray<Complex>& a) {
    int N = basis.size();
    if (a.N() != N)
        a.resize(N);
    for (int n = 0; n < N; ++n)
        a[n] = L2InnerProduct(u, basis[n]);
}

void coeff2field(const vector<BasisFunc>& basis, const cfarray<Complex>& a, FlowField& u) {
    int N = basis.size();
    assert(a.N() == N);
    u.setToZero();
    BasisFunc tmp;

    const Real EPSILON = 1e-16;
    for (int n = 0; n < N; ++n) {
        Complex an = a[n];
        if (abs(an) < EPSILON)
            continue;

        tmp = basis[n];
        tmp *= an;

        assert(tmp.kx() <= u.kxmax() && tmp.kx() >= u.kxmin() && tmp.kz() <= u.kzmax() && tmp.kz() >= u.kzmin());

        u += tmp;
    }
}

void field2coeff(const std::vector<RealProfileNG>& basis, const FlowField& u, std::vector<Real>& a) {
    a.clear();
    a.reserve(basis.size());

    for (vector<RealProfileNG>::const_iterator ei = basis.begin(); ei != basis.end(); ++ei)
        a.push_back(L2InnerProduct(*ei, u));
}

void coeff2field(const std::vector<RealProfileNG>& basis, const std::vector<Real>& a, FlowField& u) {
    assert(basis.size() == a.size());

    u.setToZero();

    std::vector<RealProfileNG>::const_iterator ei = basis.begin();
    std::vector<Real>::const_iterator ai = a.begin();
    FlowField tmp(u.Nx(), u.Ny(), u.Nz(), u.Nd(), u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());
    tmp += *ei;
    tmp *= *ai;
    u += tmp;
    ++ai;  // Increment ai only, because of the way the loop is done

    for (; ai != a.end(); ++ai) {
        tmp.setToZero();
        tmp += *(++ei);
        tmp *= *ai;
        u += tmp;
    }
}

const Real EPSILON = 1e-12;
void randomUprofile(ComplexChebyCoeff& u, Real mag, Real decay) {
    // Set a random u(y)
    int N = u.length();
    u.setState(Spectral);
    int n;  // MSVC++ FOR-SCOPE BUG
    for (n = 0; n < N; ++n) {
        u.set(n, mag * randomComplex());
        mag *= decay;
    }
    ChebyTransform trans(N);
    // Adjust u(y) so that u(+-1) == 0
    // cout << "before random u(a) == " << u.eval_a() << endl;
    // cout << "before random u(b) == " << u.eval_b() << endl;
    Complex u0 = (u.eval_b() + u.eval_a()) / 2.0;
    Complex u1 = (u.eval_b() - u.eval_a()) / 2.0;
    u.sub(0, u0);
    u.sub(1, u1);
    // cout << "after random u(a) == " << u.eval_a() << endl;
    // cout << "after random u(b) == " << u.eval_b() << endl;

    assert(abs(u.eval_b()) < EPSILON);
    assert(abs(u.eval_a()) < EPSILON);
}

void randomVprofile(ComplexChebyCoeff& v, Real mag, Real decay) {
    // Put v on [-1,1] temporarily to simplify adjustment of BCs
    Real ya = v.a();
    Real yb = v.b();
    v.setBounds(-1, 1);
    v.setState(Spectral);
    int N = v.length();

    // Assign a random v(y).
    v.set(0, 0.0);
    v.set(1, 0.0);
    v.set(2, 0.0);
    v.set(3, 0.0);
    int n;  // MSVC++ FOR-SCOPE BUG
    for (n = 0; n < 4; ++n)
        v.set(n, 0.0);
    for (n = 4; n < N - 2; ++n) {
        v.set(n, mag * randomComplex());
        mag *= decay;
    }

    for (n = Greater(N - 2, 0); n < N; ++n)
        v.set(n, 0.0);

    // Adjust v so that v(+-1) == v'(+/-1) == 0, by subtracting off
    // s0 T0(x) + s1 T1(x) + s2 T2(x) + s3 T3(x), with s's chosen to
    // have same BCs as v.
    ComplexChebyCoeff vy = diff(v);

    Complex a = v.eval_a();
    Complex b = v.eval_b();
    Complex c = vy.eval_a();
    Complex d = vy.eval_b();

    // The numercial coeffs are inverse of the matrix (values found with Maple)
    // T0(-1)  T1(-1)  T2(-1)  T3(-1)     s0      a
    // T0(1)   T1(1)   T2(1)   T3(1)      s1      b
    // T0'(-1) T1'(-1) T2'(-1) T3'(-1)    s2  ==  c
    // T0'(1)  T1'(1)  T2'(1)  T3'(1)     s3      d

    Complex s0 = 0.5 * (a + b) + 0.125 * (c - d);
    Complex s1 = 0.5625 * (b - a) - 0.0625 * (c + d);
    Complex s2 = 0.125 * (d - c);
    Complex s3 = 0.0625 * (a - b + c + d);

    // Subtract off the coeffs
    v.sub(0, s0);
    v.sub(1, s1);
    v.sub(2, s2);
    v.sub(3, s3);

    v.setBounds(ya, yb);

    if (DEBUG) {
        diff(v, vy);
        assert(abs(v.eval_a()) < EPSILON);
        assert(abs(v.eval_b()) < EPSILON);
        assert(abs(vy.eval_a()) < EPSILON);
        assert(abs(vy.eval_b()) < EPSILON);
    }
}

void chebyUprofile(ComplexChebyCoeff& u, int n, Real decay) {
    // Set a random u(y)
    int N = u.length();
    u.setToZero();
    u.setState(Spectral);
    Real theta = randomReal(0, 2 * pi);
    u.set(n, (cos(theta) + I * sin(theta)) * std::pow(decay, n));

    ChebyTransform trans(N);

    // Adjust u(y) so that u(+-1) == 0
    Complex u0 = (u.eval_b() + u.eval_a()) / 2.0;  // 2.0 is correct for genl a,b
    Complex u1 = (u.eval_b() - u.eval_a()) / 2.0;  // 2.0 is correct for genl a,b
    u.sub(0, u0);
    u.sub(1, u1);
    // cout << "random u(a) == " << u.eval_a() << endl;
    // cout << "random u(b) == " << u.eval_b() << endl;
    assert(abs(u.eval_b()) < EPSILON);
    assert(abs(u.eval_a()) < EPSILON);
}

void chebyVprofile(ComplexChebyCoeff& v, int n, Real decay) {
    v.setToZero();
    v.setState(Spectral);
    Real ya = v.a();
    Real yb = v.b();
    v.setBounds(-1, 1);
    Real theta = randomReal(0, 2 * pi);
    v.set(n, (cos(theta) + I * sin(theta)) * std::pow(decay, n));

    // Adjust v so that v(+-1) == v'(+/-1) == 0, by subtracting off
    // s0 T0(x) + s1 T1(x) + s2 T2(x) + s3 T3(x), with s's chosen to
    // have same BCs as v.
    ComplexChebyCoeff vy = diff(v);

    Complex a = v.eval_a();
    Complex b = v.eval_b();
    Complex c = vy.eval_a();
    Complex d = vy.eval_b();

    // The numercial coeffs are inverse of the matrix (values found with Maple)
    // T0(-1)  T1(-1)  T2(-1)  T3(-1)     s0      a
    // T0(1)   T1(1)   T2(1)   T3(1)      s1      b
    // T0'(-1) T1'(-1) T2'(-1) T3'(-1)    s2  ==  c
    // T0'(1)  T1'(1)  T2'(1)  T3'(1)     s3      d

    // The above matrix is
    // 1  -1   1  -1
    // 1   1   1   1
    // 0   1  -4   9
    // 0   1   4   9

    Complex s0 = 0.5 * (a + b) + 0.125 * (c - d);
    Complex s1 = 0.5625 * (b - a) - 0.0625 * (c + d);
    Complex s2 = 0.125 * (d - c);
    Complex s3 = 0.0625 * (a - b + c + d);

    // ComplexChebyCoeff adj(v.numModes(), v.a(), v.b(), Spectral);
    // adj.set(0, s0);
    // adj.set(1, s1);
    // adj.set(2, s2);
    // adj.set(3, s3);
    // ComplexChebyCoeff adjy = diff(adj);

    // Subtract off the coeffs
    v.sub(0, s0);
    v.sub(1, s1);
    v.sub(2, s2);
    v.sub(3, s3);
    v.setBounds(ya, yb);

    // diff(v,vy);

    // cout << "random v(a)  == " << v.eval_a() << endl;
    // cout << "random v(b)  == " << v.eval_b()  << endl;
    // cout << "random v'(a) == " << vy.eval_a() << endl;
    // cout << "random v'(b) == " << vy.eval_b() << endl;
}

void randomProfile(ComplexChebyCoeff& u, ComplexChebyCoeff& v, ComplexChebyCoeff& w, int kx, int kz, Real Lx, Real Lz,
                   Real mag, Real decay) {
    int N = u.length();
    ChebyTransform trans(N);
    u.setState(Spectral);
    v.setState(Spectral);
    w.setState(Spectral);
    // u.setToZero();
    // v.setToZero();
    // w.setToZero();
    if (kx == 0 && kz == 0) {
        // Assign an odd perturbation to u, so as not to change mean(U).
        // Just set even modes to zero.

        randomUprofile(w, mag, decay);
        w.im.setToZero();

        randomUprofile(u, mag, decay);
        u.im.setToZero();

        v.setToZero();

    } else {
        // Other kx,kz cases are based on a random v(y).

        randomVprofile(v, mag, decay);
        ComplexChebyCoeff vy = diff(v);

        if (kx == 0) {
            randomUprofile(u, mag, decay);
            u.im.setToZero();
            w = vy;
            w *= -Lz / ((2 * pi * kz) * I);
        } else if (kz == 0) {
            randomUprofile(w, mag, decay);
            w.im.setToZero();
            u = vy;
            u *= -Lx / ((2 * pi * kx) * I);
        } else {
            ComplexChebyCoeff v0(v);
            ComplexChebyCoeff v1(v);
            randomVprofile(v0, mag, decay);
            randomVprofile(v1, mag, decay);

            ComplexChebyCoeff v0y = diff(v0);
            ComplexChebyCoeff v1y = diff(v1);

            // Finally, the general case, where kx, kz != 0 and u,v,w are nonzero
            // Set a random u(y)
            ComplexChebyCoeff u0(v.numModes(), v.a(), v.b(), Spectral);
            ComplexChebyCoeff w0(v.numModes(), v.a(), v.b(), Spectral);
            ComplexChebyCoeff u1(v.numModes(), v.a(), v.b(), Spectral);
            ComplexChebyCoeff w1(v.numModes(), v.a(), v.b(), Spectral);

            randomUprofile(u0, mag, decay);

            // Calculate w0 from div u0 == u0x + v0y + w0z == 0.
            ComplexChebyCoeff u0x(u0);
            u0x *= (2 * pi * kx / Lx) * I;
            w0 = v0y;
            w0 += u0x;
            w0 *= -Lz / ((2 * pi * kz) * I);  // Set w = -Lz/(2*pi*I*kz) * (ux + vy);

            // randomUprofile(w1, mag, decay);

            // Calculate u0 from div u0 == u0x + v0y + w0z == 0.
            ComplexChebyCoeff w1z(w1);
            w1z *= (2 * pi * kz / Lz) * I;
            u1 = v1y;
            u1 += w1z;
            u1 *= -Lx / ((2 * pi * kx) * I);  // Set w = -Lz/(2*pi*I*kz) * (ux + vy);

            u = u0;
            v = v0;
            w = w0;
            u += u1;
            v += v1;
            w += w1;
        }
    }

    if (DEBUG) {
        // Check divergence
        ComplexChebyCoeff ux(u);
        ux *= (2 * pi * kx / Lx) * I;
        ComplexChebyCoeff wz(w);
        wz *= (2 * pi * kz / Lz) * I;
        ComplexChebyCoeff vy = diff(v);

        ComplexChebyCoeff div(ux);
        div += vy;
        div += wz;

        if (abs(u.eval_a()) + abs(u.eval_b()) > EPSILON || abs(v.eval_a()) + abs(v.eval_b()) > EPSILON ||
            abs(w.eval_a()) + abs(w.eval_b()) > EPSILON) {
            cout << "u.eval_a()) == " << u.eval_a() << endl;
            cout << "u.eval_b()) == " << u.eval_b() << endl;
            cout << "v.eval_a()) == " << v.eval_a() << endl;
            cout << "v.eval_b()) == " << v.eval_b() << endl;
            cout << "w.eval_a()) == " << w.eval_a() << endl;
            cout << "w.eval_b()) == " << w.eval_b() << endl;
        }
        assert(L2Norm(div) < EPSILON);
        assert(abs(u.eval_a()) + abs(u.eval_b()) < EPSILON);
        assert(abs(v.eval_a()) + abs(v.eval_b()) < EPSILON);
        assert(abs(w.eval_a()) + abs(w.eval_b()) < EPSILON);
    }
}

void chebyProfile(ComplexChebyCoeff& u, ComplexChebyCoeff& v, ComplexChebyCoeff& w, int un, int vn, int kx, int kz,
                  Real Lx, Real Lz, Real decay) {
    int N = u.length();
    ChebyTransform trans(N);
    u.setState(Spectral);
    v.setState(Spectral);
    w.setState(Spectral);
    // u.setToZero();
    // v.setToZero();
    // w.setToZero();
    if (kx == 0 && kz == 0) {
        chebyUprofile(u, un, decay);
        chebyUprofile(w, vn, decay);  // yes, vn
        u.im.setToZero();
        w.im.setToZero();
        v.setToZero();
    } else {
        // Other kx,kz cases are based on a random v(y).

        chebyVprofile(v, vn, decay);
        ComplexChebyCoeff vy = diff(v);

        if (kx == 0) {
            chebyUprofile(u, un, decay);
            u.im.setToZero();
            w = vy;
            w *= -Lz / ((2 * pi * kz) * I);
        } else if (kz == 0) {
            chebyUprofile(w, un, decay);  // yes, un
            w.im.setToZero();
            u = vy;
            u *= -Lx / ((2 * pi * kx) * I);
        } else {
            ComplexChebyCoeff v0(v);
            ComplexChebyCoeff v1(v);
            chebyVprofile(v0, vn, decay);
            chebyVprofile(v1, vn, decay);

            ComplexChebyCoeff v0y = diff(v0);
            ComplexChebyCoeff v1y = diff(v1);

            // Finally, the general case, where kx, kz != 0 and u,v,w are nonzero
            // Set a random u(y)
            ComplexChebyCoeff u0(v.numModes(), v.a(), v.b(), Spectral);
            ComplexChebyCoeff w0(v.numModes(), v.a(), v.b(), Spectral);
            ComplexChebyCoeff u1(v.numModes(), v.a(), v.b(), Spectral);
            ComplexChebyCoeff w1(v.numModes(), v.a(), v.b(), Spectral);

            chebyUprofile(u0, un, decay);

            // Calculate w0 from div u0 == u0x + v0y + w0z == 0.
            ComplexChebyCoeff u0x(u0);
            u0x *= (2 * pi * kx / Lx) * I;
            w0 = v0y;
            w0 += u0x;
            w0 *= -Lz / ((2 * pi * kz) * I);  // Set w = -Lz/(2*pi*I*kz) * (ux + vy);

            // randomUprofile(w1, mag, decay);

            // Calculate u0 from div u0 == u0x + v0y + w0z == 0.
            ComplexChebyCoeff w1z(w1);
            w1z *= (2 * pi * kz / Lz) * I;
            u1 = v1y;
            u1 += w1z;
            u1 *= -Lx / ((2 * pi * kx) * I);  // Set w = -Lz/(2*pi*I*kz) * (ux + vy);

            u = u0;
            v = v0;
            w = w0;
            u += u1;
            v += v1;
            w += w1;
        }
    }

    // Check divergence
    ComplexChebyCoeff ux(u);
    ux *= (2 * pi * kx / Lx) * I;
    ComplexChebyCoeff wz(w);
    wz *= (2 * pi * kz / Lz) * I;
    ComplexChebyCoeff vy = diff(v);

    ComplexChebyCoeff div(ux);
    div += vy;
    div += wz;

    Real divNorm = L2Norm(div);
    Real ubcNorm = abs(u.eval_a()) + abs(u.eval_b());
    Real vbcNorm = abs(v.eval_a()) + abs(v.eval_b());
    Real wbcNorm = abs(w.eval_a()) + abs(w.eval_b());
    assert(divNorm < EPSILON);
    assert(ubcNorm < EPSILON);
    assert(vbcNorm < EPSILON);
    assert(wbcNorm < EPSILON);
    // supress unused-variable compiler warnings...
    wbcNorm += divNorm + ubcNorm + vbcNorm;
}

void assignOrrSommField(FlowField& u, FlowField& P, Real t, Real Reynolds, Complex omega, const ComplexChebyCoeff& ueig,
                        const ComplexChebyCoeff& veig, const ComplexChebyCoeff& peig) {
    int Ny = u.Ny();

    // Reconstruct velocity field (Poisseuille plus OS perturbation) from
    // y-profile of (kx,kz) == (1,0) Spectral mode of pertubation (ueig & veig).
    u.setState(Spectral, Physical);
    u.setToZero();
    Complex c = exp((-t * omega) * I);
    int n = u.Mx() - 1;
    int ny;  // MSVC++ FOR-SCOPE BUG
    for (ny = 0; ny < Ny; ++ny) {
        Complex uc = c * ueig[ny];
        Complex vc = c * veig[ny];
        if (u.taskid() == u.task_coeff(0, 0)) {
            u.cmplx(0, ny, 0, 0) = Complex(1.0 - square(u.y(ny)));
        }
        if (u.taskid() == u.task_coeff(1, 0)) {
            u.cmplx(1, ny, 0, 0) = uc;
            u.cmplx(1, ny, 0, 1) = vc;
        }
        if (u.taskid() == u.task_coeff(n, 0)) {
            u.cmplx(n, ny, 0, 0) = conj(uc);
            u.cmplx(n, ny, 0, 1) = conj(vc);
        }
    }

    // Assign pressure perturbation p to P field.
    P.setState(Spectral, Physical);
    P.setToZero();
    for (ny = 0; ny < Ny; ++ny) {
        Complex pc = c * peig[ny];
        if (P.taskid() == P.task_coeff(1, 0))
            P.cmplx(1, ny, 0, 0) = pc;
        if (P.taskid() == P.task_coeff(1, 0))
            P.cmplx(n, ny, 0, 0) = conj(pc);
    }

    // Add velocity contrib to get modified pressure P = p + 1/2 |u|^2
    u.makePhysical();
    P.makePhysical();

    for (ny = u.nylocmin(); ny < u.nylocmax(); ++ny)
        for (int nx = u.nxlocmin(); nx < u.nxlocmin() + u.Nxloc(); ++nx)
            for (int nz = 0; nz < u.Nz(); ++nz)
                P(nx, ny, nz, 0) +=
                    0.5 * (square(u(nx, ny, nz, 0)) + square(u(nx, ny, nz, 1)) + square(u(nx, ny, nz, 2)));

    u.makeSpectral();
    P.makeSpectral();
}

void xdiff(const FlowField& f_, FlowField& dfdx, int n) {
    if (n == 0)
        return;

    FlowField& f = (FlowField&)f_;
    fieldstate sxz = f.xzstate();
    fieldstate sy = f.ystate();
    f.makeSpectral_xz();

    const Real Lx = f.Lx();

    if (!f.congruent(dfdx))
        dfdx.resize(f.Nx(), f.Ny(), f.Nz(), f.Nd(), f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
    else
        dfdx.setToZero();
    dfdx.setState(Spectral, sy);

    Complex rot(0.0, 0.0);
    switch (n % 4) {
        case 0:
            rot = Complex(1.0, 0.0);
            break;
        case 1:
            rot = Complex(0.0, 1.0);
            break;
        case 2:
            rot = Complex(-1.0, 0.0);
            break;
        case 3:
            rot = Complex(0.0, -1.0);
            break;
        default:
            cferror("xdiff(f, dfdx, n) : impossible: n % 4 > 4 !!");
            break;
    }

    const int Nd = f.Nd();
    //     int Mx = f.Mx();
    const int My = f.My();
    //     int Mz = f.Mz();
    const int kxmax = f.kxmax();

    for (int i = 0; i < Nd; ++i) {
        for (int my = 0; my < My; ++my) {
            for (int mx = f.mxlocmin(); mx < f.mxlocmin() + f.Mxloc(); ++mx) {
                int kx = f.kx(mx);
                Complex cx = rot * (std::pow(2 * pi * kx / Lx, n) * zero_last_mode(kx, kxmax, n));
                for (int mz = f.mzlocmin(); mz < f.mzlocmin() + f.Mzloc(); ++mz)
                    dfdx.cmplx(mx, my, mz, i) = cx * f.cmplx(mx, my, mz, i);
            }
        }
    }
    f.makeState(sxz, sy);
    dfdx.makeSpectral();
}

void ydiff(const FlowField& f, FlowField& dfdy, int n) {
    if (n < 0)
        cferror("ydiff(f,dfdy,n) : n must be >= 0");
    if (n == 0)
        dfdy = f;
    else if (n == 1)
        ydiffOnce(f, dfdy);
    else {
        FlowField tmp(f);
        for (int k = 0; k < n; ++k) {
            ydiffOnce(tmp, dfdy);
            if (k < n - 1)
                tmp = dfdy;
        }
    }
    return;
}

void ydiffOnce(const FlowField& f_, FlowField& dfdy) {
    FlowField& f = (FlowField&)f_;
    fieldstate sxz = f.xzstate();
    fieldstate sy = f.ystate();
    f.makeSpectral_y();

    if (!f.congruent(dfdy))
        dfdy.resize(f.Nx(), f.Ny(), f.Nz(), f.Nd(), f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
    else
        dfdy.setToZero();
    dfdy.setState(sxz, Spectral);

    int Nd = f.Nd();
    if (sxz == Spectral) {
        //         int Mx = f.Mx();
        int Ny = f.Ny();
        //         int Mz = f.Mz();
        int Nyb = Ny - 1;

        const lint mxlocmin = f.mxlocmin();
        const lint mxlocmax = f.mxlocmin() + f.Mxloc();
        const lint mzlocmin = f.mzlocmin();
        const lint mzlocmax = f.mzlocmin() + f.Mzloc();
        Complex zero(0.0, 0.0);
        Real scale = 4.0 / f.Ly();

        for (int i = 0; i < Nd; ++i) {
            for (lint mx = mxlocmin; mx < mxlocmax; ++mx)
                for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                    dfdy.cmplx(mx, Nyb, mz, i) = zero;
                    dfdy.cmplx(mx, Nyb - 1, mz, i) = scale * Nyb * f.cmplx(mx, Nyb, mz, i);
                }
            for (int ny = Nyb - 2; ny >= 0; --ny) {
                for (int mx = mxlocmin; mx < mxlocmax; ++mx)
                    for (int mz = mzlocmin; mz < mzlocmax; ++mz)
                        dfdy.cmplx(mx, ny, mz, i) =
                            dfdy.cmplx(mx, ny + 2, mz, i) + scale * (ny + 1) * f.cmplx(mx, ny + 1, mz, i);
            }
            for (int mx = mxlocmin; mx < mxlocmax; ++mx)
                for (int mz = mzlocmin; mz < mzlocmax; ++mz)
                    dfdy.cmplx(mx, 0, mz, i) *= 0.5;
        }
    } else {
#ifdef HAVE_MPI
        cferror("State Physical,Spectral requried. This state does not exist in MPI-version");
#endif
        int Nx = f.Nx();
        int Ny = f.Ny();
        int Nz = f.Nz();
        int Nyb = Ny - 1;
        Real scale = 4.0 / f.Ly();

        for (int i = 0; i < Nd; ++i) {
            for (int nx = 0; nx < Nx; ++nx)
                for (int nz = 0; nz < Nz; ++nz) {
                    dfdy(nx, Nyb, nz, i) = 0.0;
                    dfdy(nx, Nyb - 1, nz, i) = scale * Nyb * f(nx, Nyb, nz, i);
                }
            for (int ny = Nyb - 2; ny >= 0; --ny) {
                for (int nx = 0; nx < Nx; ++nx)
                    for (int nz = 0; nz < Nz; ++nz)
                        dfdy(nx, ny, nz, i) = dfdy(nx, ny + 2, nz, i) + scale * (ny + 1) * f(nx, ny + 1, nz, i);
            }
            for (int nx = 0; nx < Nx; ++nx)
                for (int nz = 0; nz < Nz; ++nz)
                    dfdy(nx, 0, nz, i) *= 0.5;
        }
    }

    f.makeState(sxz, sy);
    dfdy.makeSpectral();
}

// MPI-Question: is this function still used? Not parellelized!
void ydiffOld(const FlowField& f_, FlowField& dfdy, int n) {
    if (n == 0)
        return;

    FlowField& f = (FlowField&)f_;
    fieldstate sxz = f.xzstate();
    fieldstate sy = f.ystate();
    f.makeSpectral_y();

    if (!f.congruent(dfdy))
        dfdy.resize(f.Nx(), f.Ny(), f.Nz(), f.Nd(), f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
    else
        dfdy.setToZero();
    dfdy.setState(sxz, Spectral);

    int Nd = f.Nd();
    if (sxz == Spectral) {
        int Mx = f.Mx();
        int My = f.My();
        int Mz = f.Mz();

        ComplexChebyCoeff g(f.Ny(), f.a(), f.b(), Spectral);
        ComplexChebyCoeff gy(f.Ny(), f.a(), f.b(), Spectral);

        for (int i = 0; i < Nd; ++i)
            for (int mx = 0; mx < Mx; ++mx)
                for (int mz = 0; mz < Mz; ++mz) {
                    for (int my = 0; my < My; ++my)
                        g.set(my, f.cmplx(mx, my, mz, i));

                    // differentiate n times and leave derivative in *g
                    for (int k = 0; k < n; ++k) {
                        diff(g, gy);
                        swap(g, gy);
                    }
                    for (int my = 0; my < My; ++my)
                        dfdy.cmplx(mx, my, mz, i) = g[my];
                }
    } else {
        int Nx = f.Nx();
        int Ny = f.Ny();
        int Nz = f.Nz();
        ChebyCoeff g(Ny, f.a(), f.b(), Spectral);
        ChebyCoeff gy(Ny, f.a(), f.b(), Spectral);

        for (int i = 0; i < Nd; ++i)
            for (int nx = 0; nx < Nx; ++nx)
                for (int nz = 0; nz < Nz; ++nz) {
                    for (int ny = 0; ny < Ny; ++ny)
                        g[ny] = f(nx, ny, nz, i);

                    // differentiate n times and leave derivative in g
                    for (int k = 0; k < n; ++k) {
                        diff(g, gy);
                        swap(g, gy);
                    }
                    for (int ny = 0; ny < Ny; ++ny)
                        dfdy(nx, ny, nz, i) = g[ny];
                }
    }

    f.makeState(sxz, sy);
    dfdy.makeSpectral();
}

void zdiff(const FlowField& f_, FlowField& dfdz, int n) {
    if (n == 0)
        return;

    FlowField& f = (FlowField&)f_;
    fieldstate sxz = f.xzstate();
    fieldstate sy = f.ystate();
    f.makeSpectral_xz();

    // compute gradf(nx,ny,nz,i) = df(nx,ny,nz,0)/dx_i for scalar-valued f
    if (!f.congruent(dfdz))
        dfdz.resize(f.Nx(), f.Ny(), f.Nz(), f.Nd(), f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
    else
        dfdz.setToZero();
    dfdz.setState(Spectral, sy);

    Complex rot(0.0, 0.0);
    switch (n % 4) {
        case 0:
            rot = Complex(1.0, 0.0);
            break;
        case 1:
            rot = Complex(0.0, 1.0);
            break;
        case 2:
            rot = Complex(-1.0, 0.0);
            break;
        case 3:
            rot = Complex(0.0, -1.0);
            break;
        default:
            cferror("zdiff(f, dfdz, n) : impossible: n % 4 > 4 !!");
            break;
    }

    int Nd = f.Nd();
    //     int Mx = f.Mx();
    int My = f.My();
    //     int Mz = f.Mz();
    const lint mxlocmin = f.mxlocmin();
    const lint mxlocmax = f.mxlocmin() + f.Mxloc();
    const lint mzlocmin = f.mzlocmin();
    const lint mzlocmax = f.mzlocmin() + f.Mzloc();
    int kzmax = f.kzmax();
    Real Lz = f.Lz();

    Real az = 2 * pi / Lz;
    for (int i = 0; i < Nd; ++i)
        for (lint my = 0; my < My; ++my)
            for (lint mx = mxlocmin; mx < mxlocmax; ++mx)
                for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                    int kz = f.kz(mz);
                    Complex cz = rot * (std::pow(az * kz, n) * zero_last_mode(kz, kzmax, n));
                    dfdz.cmplx(mx, my, mz, i) = cz * f.cmplx(mx, my, mz, i);
                }

    f.makeState(sxz, sy);
    dfdz.makeSpectral();
    return;
}

void diff(const FlowField& f, FlowField& df, int i, int n) {
    assert(i >= 0 && i < 3);
    if (i == 0)
        xdiff(f, df, n);
    else if (i == 1)
        ydiff(f, df, n);
    else
        zdiff(f, df, n);
}

void diff(const FlowField& f_, FlowField& df, int nx, int ny, int nz) {
    FlowField& f = (FlowField&)f_;
    fieldstate sxz = f.xzstate();
    fieldstate sy = f.ystate();
    f.makeSpectral();

    Real Lx = f.Lx();
    Real Lz = f.Lz();

    if (!f.congruent(df))
        df.resize(f.Nx(), f.Ny(), f.Nz(), f.Nd(), f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
    else
        df.setToZero();
    df.setState(Spectral, Spectral);

    Complex xrot(0.0, 0.0);
    switch (nx % 4) {
        case 0:
            xrot = Complex(1.0, 0.0);
            break;
        case 1:
            xrot = Complex(0.0, 1.0);
            break;
        case 2:
            xrot = Complex(-1.0, 0.0);
            break;
        case 3:
            xrot = Complex(0.0, -1.0);
            break;
        default:
            cferror("diff(f, df,nx,ny,nz) : impossible: nx % 4 > 4 !!");
            break;
    }
    Complex zrot(0.0, 0.0);
    switch (nz % 4) {
        case 0:
            zrot = Complex(1.0, 0.0);
            break;
        case 1:
            zrot = Complex(0.0, 1.0);
            break;
        case 2:
            zrot = Complex(-1.0, 0.0);
            break;
        case 3:
            zrot = Complex(0.0, -1.0);
            break;
        default:
            cferror("diff(f, df,nx,ny,nz) : impossible: nz % 4 > 4 !!");
            break;
    }

    int Nd = f.Nd();
    //     int Mx = f.Mx();
    int My = f.My();
    //     int Mz = f.Mz();
    const lint mxlocmin = f.mxlocmin();
    const lint mxlocmax = f.mxlocmin() + f.Mxloc();
    const lint mzlocmin = f.mzlocmin();
    const lint mzlocmax = f.mzlocmin() + f.Mzloc();
    int kxmax = f.kxmax();
    int kzmax = f.kzmax();

    // Do the x and z differentiation
    Real ax = 2 * pi / Lx;
    Real az = 2 * pi / Lz;
    for (int i = 0; i < Nd; ++i)
        for (lint my = 0; my < My; ++my)
            for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
                int kx = f.kx(mx);
                Complex cx = xrot * (std::pow(ax * kx, nx) * zero_last_mode(kx, kxmax, nx));
                for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                    int kz = f.kz(mz);
                    Complex cz = zrot * (std::pow(az * kz, nz) * zero_last_mode(kz, kzmax, nz));
                    df.cmplx(mx, my, mz, i) = cx * cz * f.cmplx(mx, my, mz, i);
                }
            }

    // Do the y differentiation
    ComplexChebyCoeff g(f.Ny(), f.a(), f.b(), Spectral);
    ComplexChebyCoeff gy(f.Ny(), f.a(), f.b(), Spectral);

    for (int i = 0; i < Nd; ++i)
        for (lint mx = mxlocmin; mx < mxlocmax; ++mx)
            for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                for (lint my = 0; my < My; ++my)
                    g.set(my, df.cmplx(mx, my, mz, i));

                // differentiate n times and leave derivative in *g
                for (int k = 0; k < ny; ++k) {
                    diff(g, gy);
                    swap(g, gy);
                }
                for (int my = 0; my < My; ++my)
                    df.cmplx(mx, my, mz, i) = g[my];
            }

    f.makeState(sxz, sy);
    df.makeSpectral();
    return;
}

//  gradf(nx,ny,nz,i,j) = df(nx,ny,nz,i)/dx_j for vector-valued f
//  gradf(nx,ny,nz,i) = df(nx,ny,nz,0)/dx_i for scalar-valued f

void grad(const FlowField& f_, FlowField& gradf) {
    FlowField& f = (FlowField&)f_;
    fieldstate sxz = f.xzstate();
    fieldstate sy = f.ystate();
    f.makeSpectral();

    Real Lx = f.Lx();
    Real Lz = f.Lz();

    // compute gradf(nx,ny,nz,i) = df(nx,ny,nz,0)/dx_i for scalar-valued f
    if (f.Nd() == 1) {
        if (!f.geomCongruent(gradf) || gradf.vectorDim() != 3)
            gradf.resize(f.Nx(), f.Ny(), f.Nz(), 3, f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
        else
            gradf.setToZero();
        gradf.setState(Spectral, Spectral);

        //         int Mx = f.Mx();
        int My = f.My();
        //         int Mz = f.Mz();
        const lint mxlocmin = f.mxlocmin();
        const lint mxlocmax = f.mxlocmin() + f.Mxloc();
        const lint mzlocmin = f.mzlocmin();
        const lint mzlocmax = f.mzlocmin() + f.Mzloc();
        int kxmax = f.kxmax();
        int kzmax = f.kzmax();

        // Assign df/dx to Df(i) and df/dz to Df(i)
#ifdef HAVE_MPI
        for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
            int kz = f.kz(mz);
            Complex czkz(0.0, 2.0 * pi * kz / Lz * zero_last_mode(kz, kzmax, 1));
            for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
                int kx = f.kx(mx);
                Complex cxkx(0.0, 2.0 * pi * kx / Lx * zero_last_mode(kx, kxmax, 1));
                for (lint my = 0; my < My; ++my) {
                    Complex fval = f.cmplx(mx, my, mz, 0);
                    gradf.cmplx(mx, my, mz, 0) = fval * cxkx;
                    gradf.cmplx(mx, my, mz, 2) = fval * czkz;
                }
            }
        }
#else
        for (lint my = 0; my < My; ++my)
            for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
                int kx = f.kx(mx);
                Complex cxkx(0.0, 2.0 * pi * kx / Lx * zero_last_mode(kx, kxmax, 1));
                for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                    int kz = f.kz(mz);
                    Complex czkz(0.0, 2.0 * pi * kz / Lz * zero_last_mode(kz, kzmax, 1));
                    Complex fval = f.cmplx(mx, my, mz, 0);
                    gradf.cmplx(mx, my, mz, 0) = fval * cxkx;
                    gradf.cmplx(mx, my, mz, 2) = fval * czkz;
                }
            }
#endif

        // Assign df/dy to Df(i,1)
        ComplexChebyCoeff g(f.Ny(), f.a(), f.b(), Spectral);
        ComplexChebyCoeff gy(f.Ny(), f.a(), f.b(), Spectral);
#ifdef HAVE_MPI
        for (lint mz = mzlocmin; mz < mzlocmax; ++mz)
            for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
                for (lint my = 0; my < My; ++my)
                    g.set(my, f.cmplx(mx, my, mz, 0));
                diff(g, gy);
                for (lint my = 0; my < My; ++my)
                    gradf.cmplx(mx, my, mz, 1) = gy[my];
            }
#else
        for (lint mx = mxlocmin; mx < mxlocmax; ++mx)
            for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                for (lint my = 0; my < My; ++my)
                    g.set(my, f.cmplx(mx, my, mz, 0));
                diff(g, gy);
                for (lint my = 0; my < My; ++my)
                    gradf.cmplx(mx, my, mz, 1) = gy[my];
            }
#endif
    } else if (f.Nd() == 3) {
        if (!f.geomCongruent(gradf) || gradf.vectorDim() != 9)
            gradf.resize(f.Nx(), f.Ny(), f.Nz(), 9, f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
        else
            gradf.setToZero();
        gradf.setState(Spectral, Spectral);

        //         int Mx = f.Mx();
        int My = f.My();
        //         int Mz = f.Mz();
        int Nd = 3;
        const lint mxlocmin = f.mxlocmin();
        const lint mxlocmax = f.mxlocmin() + f.Mxloc();
        const lint mzlocmin = f.mzlocmin();
        const lint mzlocmax = f.mzlocmin() + f.Mzloc();
        int kxmax = f.kxmax();
        int kzmax = f.kzmax();
        // Assign df_i/dx to Df(i,0) and df_i/dz to Df(i,2)
#ifdef HAVE_MPI
        for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
            int kz = f.kz(mz);
            Complex czkz(0.0, 2.0 * pi * kz / Lz * zero_last_mode(kz, kzmax, 1));
            for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
                int kx = f.kx(mx);
                Complex cxkx(0.0, 2.0 * pi * kx / Lx * zero_last_mode(kx, kxmax, 1));
                for (lint my = 0; my < My; ++my)
                    for (int i = 0; i < Nd; ++i) {
                        Complex fval = f.cmplx(mx, my, mz, i);
                        gradf.cmplx(mx, my, mz, i3j(i, 0)) = fval * cxkx;
                        gradf.cmplx(mx, my, mz, i3j(i, 2)) = fval * czkz;
                    }
            }
        }
#else
        for (int i = 0; i < Nd; ++i)
            for (lint my = 0; my < My; ++my)
                for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
                    int kx = f.kx(mx);
                    Complex cxkx(0.0, 2.0 * pi * kx / Lx * zero_last_mode(kx, kxmax, 1));
                    for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                        int kz = f.kz(mz);
                        Complex czkz(0.0, 2.0 * pi * kz / Lz * zero_last_mode(kz, kzmax, 1));
                        Complex fval = f.cmplx(mx, my, mz, i);
                        gradf.cmplx(mx, my, mz, i3j(i, 0)) = fval * cxkx;
                        gradf.cmplx(mx, my, mz, i3j(i, 2)) = fval * czkz;
                    }
                }
#endif
        // Assign df_i/dy to Df(i,1)
        ComplexChebyCoeff g(f.Ny(), f.a(), f.b(), Spectral);
        ComplexChebyCoeff gy(f.Ny(), f.a(), f.b(), Spectral);
#ifdef HAVE_MPI
        for (lint mz = mzlocmin; mz < mzlocmax; ++mz)
            for (lint mx = mxlocmin; mx < mxlocmax; ++mx)
                for (int i = 0; i < Nd; ++i) {
                    for (lint my = 0; my < My; ++my)
                        g.set(my, f.cmplx(mx, my, mz, i));
                    diff(g, gy);
                    for (lint my = 0; my < My; ++my)
                        gradf.cmplx(mx, my, mz, i3j(i, 1)) = gy[my];
                }
#else
        for (int i = 0; i < Nd; ++i)
            for (lint mx = mxlocmin; mx < mxlocmax; ++mx)
                for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                    for (lint my = 0; my < My; ++my)
                        g.set(my, f.cmplx(mx, my, mz, i));
                    diff(g, gy);
                    for (lint my = 0; my < My; ++my)
                        gradf.cmplx(mx, my, mz, i3j(i, 1)) = gy[my];
                }
#endif
    } else
        cferror("grad(FlowField f, FlowField grad_f) : f must be 1d or 3d");

    f.makeState(sxz, sy);

    return;
}

// // Begin SlicesHack
// void d_dx(const FlowField& f_, FlowField& ddxf) {
//     FlowField& f = (FlowField&) f_;
//     fieldstate sxz = f.xzstate();
//     fieldstate sy  = f.ystate();
//     f.makeSpectral();
//
//     Real Lx = f.Lx();
// //     Real Lz = f.Lz();
//
// 	if (f.Nd() == 3) {
//         if (!f.geomCongruent(ddxf) || ddxf.vectorDim() != 3)
//             ddxf.resize(f.Nx(), f.Ny(), f.Nz(), 3, f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
//         else
//             ddxf.setToZero();
//         ddxf.setState(Spectral, Spectral);
//
// //         int Mx = f.Mx();
//         int My = f.My();
// //         int Mz = f.Mz();
//         int Nd = 3;
// 		const lint mxlocmin = f.mxlocmin();
// 		const lint mxlocmax = f.mxlocmin() + f.Mxloc();
// 		const lint mzlocmin = f.mzlocmin();
// 		const lint mzlocmax = f.mzlocmin() + f.Mzloc();
//         int kxmax = f.kxmax();
// //         int kzmax = f.kzmax();
//
//         // Assign df_i/dx to Df(i,0) and df_i/dz to Df(i,2)
//         for (int i=0; i<Nd; ++i)
//             for (lint my=0; my<My; ++my)
//                 for (lint mx=mxlocmin; mx<mxlocmax; ++mx) {
//                     int kx = f.kx(mx);
//                     Complex cxkx(0.0, 2.0*pi*kx/Lx*zero_last_mode(kx,kxmax,1));
//                     for (lint mz=mzlocmin; mz<mzlocmax; ++mz) {
// //                         int kz = f.kz(mz);
// //                         Complex czkz(0.0, 2.0*pi*kz/Lz*zero_last_mode(kz,kzmax,1));
//                         Complex fval = f.cmplx(mx,my,mz,i);
//                         ddxf.cmplx(mx,my,mz,i) = fval*cxkx;
// //                         gradf.cmplx(mx,my,mz,i3j(i,2)) = fval*czkz;
//                     }
//                 }
//
//     }
//     else
//         cferror("ddzf(FlowField f, FlowField ddz_f) : f must be 3d");
//
//     f.makeState(sxz, sy);
//     return;
// }
//
// void d_dz(const FlowField& f_, FlowField& ddzf) {
//     FlowField& f = (FlowField&) f_;
//     fieldstate sxz = f.xzstate();
//     fieldstate sy  = f.ystate();
//     f.makeSpectral();
//
// //     Real Lx = f.Lx();
//     Real Lz = f.Lz();
//
// 	if (f.Nd() == 3) {
//         if (!f.geomCongruent(ddzf) || ddzf.vectorDim() != 3)
//             ddzf.resize(f.Nx(), f.Ny(), f.Nz(), 3, f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
//         else
//             ddzf.setToZero();
//         ddzf.setState(Spectral, Spectral);
//
// //         int Mx = f.Mx();
//         int My = f.My();
// //         int Mz = f.Mz();
//         int Nd = 3;
// 		const lint mxlocmin = f.mxlocmin();
// 		const lint mxlocmax = f.mxlocmin() + f.Mxloc();
// 		const lint mzlocmin = f.mzlocmin();
// 		const lint mzlocmax = f.mzlocmin() + f.Mzloc();
// //         int kxmax = f.kxmax();
//         int kzmax = f.kzmax();
//
//         for (int i=0; i<Nd; ++i)
//             for (lint my=0; my<My; ++my)
//                 for (lint mx=mxlocmin; mx<mxlocmax; ++mx) {
// //                     int kx = f.kx(mx);
//                     for (lint mz=mzlocmin; mz<mzlocmax; ++mz) {
//                         int kz = f.kz(mz);
//                         Complex czkz(0.0, 2.0*pi*kz/Lz*zero_last_mode(kz,kzmax,1));
//                         Complex fval = f.cmplx(mx,my,mz,i);
//                         ddzf.cmplx(mx,my,mz,i) = fval*czkz;
//                     }
//                 }
//     }
//     else
//         cferror("ddzf(FlowField f, FlowField ddz_f) : f must be 3d");
//
//     f.makeState(sxz, sy);
//     return;
// }
//
// // end SlicesHack

void lapl(const FlowField& f_, FlowField& laplf) {
    FlowField& f = (FlowField&)f_;
    fieldstate sxz = f.xzstate();
    fieldstate sy = f.ystate();
    f.makeSpectral();

    if (!f.congruent(laplf))
        laplf.resize(f.Nx(), f.Ny(), f.Nz(), f.Nd(), f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
    else
        laplf.setToZero();
    laplf.setState(Spectral, Spectral);

    // compute gradf(nx,ny,nz,i) = df(nx,ny,nz,0)/dx_i for scalar-valued f
    Real Lx = f.Lx();
    Real Lz = f.Lz();
    //     int Mx = f.Mx();
    int My = f.My();
    //     int Mz = f.Mz();
    int Nd = f.Nd();
    const lint mxlocmin = f.mxlocmin();
    const lint mxlocmax = f.mxlocmin() + f.Mxloc();
    const lint mzlocmin = f.mzlocmin();
    const lint mzlocmax = f.mzlocmin() + f.Mzloc();

    ComplexChebyCoeff g(f.Ny(), f.a(), f.b(), Spectral);
    ComplexChebyCoeff gyy(f.Ny(), f.a(), f.b(), Spectral);

    Real az = -1.0 * square(2.0 * pi / Lz);
    for (int i = 0; i < Nd; ++i) {
        // Assign lapl f_i = (d^2/dx^2 + d^2/dz^2) f_i
        for (lint my = 0; my < My; ++my)
            for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
                int kx = f.kx(mx);
                Complex cx = -1.0 * square(2.0 * pi * kx / Lx);
                for (lint mz = mzlocmin; mz < mzlocmax; ++mz)
                    laplf.cmplx(mx, my, mz, i) = (cx + az * square(f.kz(mz))) * f.cmplx(mx, my, mz, i);
            }

        // Add d^2/dy^2 f_i on to previous result
        for (lint mx = mxlocmin; mx < mxlocmax; ++mx)
            for (int mz = mzlocmin; mz < mzlocmax; ++mz) {
                for (lint my = 0; my < My; ++my)
                    g.set(my, f.cmplx(mx, my, mz, i));
                diff2(g, gyy);
                for (lint my = 0; my < My; ++my)
                    laplf.cmplx(mx, my, mz, i) += gyy[my];
            }
    }
    f.makeState(sxz, sy);
}

void norm2(const FlowField& f_, FlowField& norm2f) {
    FlowField& f = (FlowField&)f_;
    fieldstate sxz = f.xzstate();
    fieldstate sy = f.ystate();
    f.makeState(Spectral, Spectral);

    if (!f.geomCongruent(norm2f) || norm2f.vectorDim() != 1)
        norm2f.resize(f.Nx(), f.Ny(), f.Nz(), 1, f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
    else
        norm2f.setToZero();
    norm2f.setState(Physical, Physical);

    f.makePhysical();

    //     int Nx = f.Nx();
    //     int Ny = f.Ny();
    int Nz = f.Nz();
    int Nd = f.Nd();
    lint nxlocmin = f.nxlocmin();
    lint nxlocmax = f.nxlocmin() + f.Nxloc();
    lint nylocmin = f.nylocmin();
    lint nylocmax = f.nylocmax();

    for (int i = 0; i < Nd; ++i)
        for (lint ny = nylocmin; ny < nylocmax; ++ny)
            for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
                for (int nz = 0; nz < Nz; ++nz)
                    norm2f(nx, ny, nz, 0) += square(f(nx, ny, nz, i));

    f.makeState(sxz, sy);
    norm2f.makeSpectral();
    return;
}

void norm(const FlowField& f_, FlowField& normf) {
    FlowField& f = (FlowField&)f_;
    fieldstate sxz = f.xzstate();
    fieldstate sy = f.ystate();
    f.makePhysical();

    if (!f.geomCongruent(normf) || normf.vectorDim() != 1)
        normf.resize(f.Nx(), f.Ny(), f.Nz(), 1, f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
    else
        normf.setToZero();
    normf.setState(Physical, Physical);

    // compute gradf(nx,ny,nz,i) = df(nx,ny,nz,0)/dx_i for scalar-valued f
    //     int Nx = f.Nx();
    //     int Ny = f.Ny();
    int Nz = f.Nz();
    int Nd = f.Nd();
    lint nxlocmin = f.nxlocmin();
    lint nxlocmax = f.nxlocmin() + f.Nxloc();
    lint nylocmin = f.nylocmin();
    lint nylocmax = f.nylocmax();

    for (int ny = nylocmin; ny < nylocmax; ++ny)
        for (int nx = nxlocmin; nx < nxlocmax; ++nx)
            for (int nz = 0; nz < Nz; ++nz) {
                Real nrm2 = 0.0;
                for (int i = 0; i < Nd; ++i)
                    nrm2 += square(f(nx, ny, nz, i));
                normf(nx, ny, nz, 0) = sqrt(nrm2);
            }

    normf.makeSpectral();
    f.makeState(sxz, sy);
    return;
}

/*******************************************************
// Superceded by faster version (following) with different loop ordering for y-derivs.
void curl(const FlowField& f_, FlowField& curlf) {
  FlowField& f = (FlowField&) f_;
  assert(f.Nd() == 3);
  fieldstate sxz = f.xzstate();
  fieldstate sy  = f.ystate();
  f.makeSpectral();

  if (!f.geomCongruent(curlf) || curlf.vectorDim() != 3)
    curlf.resize(f.Nx(), f.Ny(), f.Nz(), 3, f.Lx(), f.Lz(), f.a(), f.b());
  //else
  //curlf.setToZero();
  curlf.setState(Spectral, Spectral);

  int Mx = f.Mx();
  int My = f.My();
  int Mz = f.Mz();
  int kxmax = f.kxmax();
  int kzmax = f.kzmax();
  Real Lx = f.Lx();
  Real Lz = f.Lz();

  // curl_x = w_y - v_z
  // curl_y = u_z - w_x
  // curl_z = v_x - u_y

  // Assign d/dx and d/dz terms to curl
  for (int my=0; my<My; ++my)
    for (int mx=0; mx<Mx; ++mx) {
      int kx = f.kx(mx);
      Complex cx(0.0, 2.0*pi*kx/Lx*zero_last_mode(kx,kxmax,1));
      for (int mz=0; mz<Mz; ++mz) {
        int kz = f.kz(mz);
        Complex cz(0.0, 2.0*pi*kz/Lz*zero_last_mode(kz,kzmax,1));
        Complex u = f.cmplx(mx,my,mz,0);
        Complex v = f.cmplx(mx,my,mz,1);
        Complex w = f.cmplx(mx,my,mz,2);
        curlf.cmplx(mx,my,mz,0) = -cz*v;
        curlf.cmplx(mx,my,mz,1) = cz*u - cx*w;
        curlf.cmplx(mx,my,mz,2) = cx*v;
      }
    }

  // Assign df_i/dy to Df(i,1)
  ComplexChebyCoeff w(f.Ny(),  f.a(), f.b(), Spectral);
  ComplexChebyCoeff wy(f.Ny(), f.a(), f.b(), Spectral);
  ComplexChebyCoeff u(f.Ny(),  f.a(), f.b(), Spectral);
  ComplexChebyCoeff uy(f.Ny(), f.a(), f.b(), Spectral);
  for (int mx=0; mx<Mx; ++mx)
    for (int mz=0; mz<Mz; ++mz) {
      for (int my=0; my<My; ++my) {
        u.set(my, f.cmplx(mx,my,mz,0));
        w.set(my, f.cmplx(mx,my,mz,2));
      }
      diff(u,uy);
      diff(w,wy);
      for (int my=0; my<My; ++my) {
        curlf.cmplx(mx,my,mz,0) += wy[my];
        curlf.cmplx(mx,my,mz,2) -= uy[my];
      }
    }
  f.makeState(sxz, sy);
}
*********************/

void curl(const FlowField& f_, FlowField& curlf) {
    FlowField& f = (FlowField&)f_;
    assert(f.Nd() == 3);
    fieldstate sxz = f.xzstate();
    fieldstate sy = f.ystate();
    f.makeSpectral();

    if (!f.geomCongruent(curlf) || curlf.vectorDim() != 3)
        curlf.resize(f.Nx(), f.Ny(), f.Nz(), 3, f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
    // else
    // curlf.setToZero();
    curlf.setState(Spectral, Spectral);

    //     int Mx = f.Mx();
    int My = f.My();
    //     int Mz = f.Mz();
    lint mxlocmin = f.mxlocmin();
    lint mxlocmax = f.mxlocmin() + f.Mxloc();
    lint mzlocmin = f.mzlocmin();
    lint mzlocmax = f.mzlocmin() + f.Mzloc();
    int kxmax = f.kxmax();
    int kzmax = f.kzmax();
    Real Lx = f.Lx();
    Real Lz = f.Lz();
    int Myb = My - 1;
    Complex zero(0.0, 0.0);
    Real scale = 4.0 / f.Ly();

    // curlf[0] = df[2]/dy - df[1]/dz
    // curlf[1] = df[0]/dz - df[2]/dx
    // curlf[2] = df[1]/dx - df[0]/dy

#ifdef HAVE_MPI  // parallel data distributions requires different looping order for optimal performance

    // Seperate loops (see serial below) do not make sense for the parallel data distribution (18/05/2018, FR)
    for (int mz = mzlocmin; mz < mzlocmax; ++mz) {
        for (int mx = mxlocmin; mx < mxlocmax; ++mx) {
            curlf.cmplx(mx, Myb, mz, 0) = zero;
            curlf.cmplx(mx, Myb - 1, mz, 0) = Myb * scale * f.cmplx(mx, Myb, mz, 2);
            for (int my = Myb - 2; my >= 0; --my)
                curlf.cmplx(mx, my, mz, 0) =
                    curlf.cmplx(mx, my + 2, mz, 0) + (my + 1) * scale * f.cmplx(mx, my + 1, mz, 2);
            curlf.cmplx(mx, 0, mz, 0) *= 0.5;
            curlf.cmplx(mx, Myb, mz, 2) = zero;
            curlf.cmplx(mx, Myb - 1, mz, 2) = -Myb * scale * f.cmplx(mx, Myb, mz, 0);
            for (int my = Myb - 2; my >= 0; --my)
                curlf.cmplx(mx, my, mz, 2) =
                    curlf.cmplx(mx, my + 2, mz, 2) - (my + 1) * scale * f.cmplx(mx, my + 1, mz, 0);
            curlf.cmplx(mx, 0, mz, 2) *= 0.5;
        }
    }
#else
    // It is fastest to do the y-derivs in sepratae loops. I've benchmarked. 2008-004-09 jfg
    // assign curlf[0] =  df[2]/dy;
    for (int mx = mxlocmin; mx < mxlocmax; ++mx) {
        for (int mz = mzlocmin; mz < mzlocmax; ++mz) {
            curlf.cmplx(mx, Myb, mz, 0) = zero;
            curlf.cmplx(mx, Myb - 1, mz, 0) = Myb * scale * f.cmplx(mx, Myb, mz, 2);
        }
    }
    for (int my = Myb - 2; my >= 0; --my) {
        for (int mx = mxlocmin; mx < mxlocmax; ++mx)
            for (int mz = mzlocmin; mz < mzlocmax; ++mz)
                curlf.cmplx(mx, my, mz, 0) =
                    curlf.cmplx(mx, my + 2, mz, 0) + (my + 1) * scale * f.cmplx(mx, my + 1, mz, 2);
    }
    for (int mx = mxlocmin; mx < mxlocmax; ++mx)
        for (int mz = mzlocmin; mz < mzlocmax; ++mz)
            curlf.cmplx(mx, 0, mz, 0) *= 0.5;

    // assign curlf[2] =  -df[2]/dy;
    for (int mx = mxlocmin; mx < mxlocmax; ++mx) {
        for (int mz = mzlocmin; mz < mzlocmax; ++mz) {
            curlf.cmplx(mx, Myb, mz, 2) = zero;
            curlf.cmplx(mx, Myb - 1, mz, 2) = -Myb * scale * f.cmplx(mx, Myb, mz, 0);
        }
    }
    for (int my = Myb - 2; my >= 0; --my) {
        for (int mx = mxlocmin; mx < mxlocmax; ++mx)
            for (int mz = mzlocmin; mz < mzlocmax; ++mz)
                curlf.cmplx(mx, my, mz, 2) =
                    curlf.cmplx(mx, my + 2, mz, 2) - (my + 1) * scale * f.cmplx(mx, my + 1, mz, 0);
    }
    for (int mx = mxlocmin; mx < mxlocmax; ++mx)
        for (int mz = mzlocmin; mz < mzlocmax; ++mz)
            curlf.cmplx(mx, 0, mz, 2) *= 0.5;
#endif

    // Assign d/dx and d/dz terms to curl. It's fastest to do these together. (surprising)
    for (int my = 0; my < My; ++my)
        for (int mx = mxlocmin; mx < mxlocmax; ++mx) {
            int kx = f.kx(mx);
            Complex d_dx(0.0, 2.0 * pi * kx / Lx * zero_last_mode(kx, kxmax, 1));
            for (int mz = mzlocmin; mz < mzlocmax; ++mz) {
                int kz = f.kz(mz);
                Complex d_dz(0.0, 2.0 * pi * kz / Lz * zero_last_mode(kz, kzmax, 1));
                Complex f0 = f.cmplx(mx, my, mz, 0);
                Complex f1 = f.cmplx(mx, my, mz, 1);
                Complex f2 = f.cmplx(mx, my, mz, 2);
                curlf.cmplx(mx, my, mz, 0) -= d_dz * f1;
                curlf.cmplx(mx, my, mz, 1) = d_dz * f0 - d_dx * f2;
                curlf.cmplx(mx, my, mz, 2) += d_dx * f1;
            }
        }
    f.makeState(sxz, sy);
}

void outer(const FlowField& f_, const FlowField& g_, FlowField& fg) {
    FlowField& f = (FlowField&)f_;
    FlowField& g = (FlowField&)g_;
    fieldstate fxz = f.xzstate();
    fieldstate fy = f.ystate();
    fieldstate gxz = g.xzstate();
    fieldstate gy = g.ystate();

    assert(f.geomCongruent(g));
    f.makePhysical();
    g.makePhysical();

    if (!f.geomCongruent(fg) || fg.Nd() != f.Nd() * g.Nd())
        fg.resize(f.Nx(), f.Ny(), f.Nz(), f.Nd() * g.Nd(), f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
    else
        fg.setToZero();
    fg.setState(Physical, Physical);

    //     int Nx = f.Nx();
    //     int Ny = f.Ny();
    int Nz = f.Nz();
    int fd = f.Nd();
    int gd = g.Nd();
    lint nxlocmin = f.nxlocmin();
    lint nxlocmax = f.nxlocmin() + f.Nxloc();
    lint nylocmin = f.nylocmin();
    lint nylocmax = f.nylocmax();

#ifdef HAVE_MPI
    for (int nx = nxlocmin; nx < nxlocmax; ++nx)
        for (int nz = 0; nz < Nz; ++nz)
            for (int ny = nylocmin; ny < nylocmax; ++ny)
                for (int i = 0; i < fd; ++i)
                    for (int j = 0; j < gd; ++j) {
                        int ij = i * gd + j;  // generalized form of i3j == i*3 + j
                        fg(nx, ny, nz, ij) = f(nx, ny, nz, i) * g(nx, ny, nz, j);
                    }
#else
    for (int i = 0; i < fd; ++i)
        for (int j = 0; j < gd; ++j) {
            int ij = i * gd + j;  // generalized form of i3j == i*3 + j
            for (int ny = nylocmin; ny < nylocmax; ++ny)
                for (int nx = nxlocmin; nx < nxlocmax; ++nx)
                    for (int nz = 0; nz < Nz; ++nz)
                        fg(nx, ny, nz, ij) = f(nx, ny, nz, i) * g(nx, ny, nz, j);
        }
#endif

    f.makeState(fxz, fy);
    g.makeState(gxz, gy);
    fg.makeSpectral();
    return;
}

void dot(const FlowField& f_, const FlowField& g_, FlowField& fdotg) {
    FlowField& f = (FlowField&)f_;
    FlowField& g = (FlowField&)g_;
    fieldstate fxz = f.xzstate();
    fieldstate fy = f.ystate();
    fieldstate gxz = g.xzstate();
    fieldstate gy = g.ystate();

    f.makePhysical();
    g.makePhysical();
    if (f.congruent(g)) {
        if (!f.geomCongruent(fdotg) || fdotg.Nd() != 1)
            fdotg.resize(f.Nx(), f.Ny(), f.Nz(), 1, f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
        else
            fdotg.setToZero();
        fdotg.setState(Physical, Physical);

        lint Nz = f.Nz();
        lint Nd = f.Nd();
        lint nxlocmin = f.nxlocmin();
        lint nxlocmax = f.nxlocmin() + f.Nxloc();
        lint nylocmin = f.nylocmin();
        lint nylocmax = f.nylocmax();

#ifdef HAVE_MPI
        for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
            for (lint nz = 0; nz < Nz; ++nz)
                for (lint ny = nylocmin; ny < nylocmax; ++ny)
                    for (lint i = 0; i < Nd; ++i)
                        fdotg(nx, ny, nz, 0) += f(nx, ny, nz, i) * g(nx, ny, nz, i);
#else
        for (lint i = 0; i < Nd; ++i)
            for (lint ny = nylocmin; ny < nylocmax; ++ny)
                for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
                    for (lint nz = 0; nz < Nz; ++nz)
                        fdotg(nx, ny, nz, 0) += f(nx, ny, nz, i) * g(nx, ny, nz, i);
#endif

    } else if (f.geomCongruent(g) && f.Nd() == 3 && g.Nd() == 9) {
        if (!f.congruent(fdotg) || fdotg.Nd() != 3)
            fdotg.resize(f.Nx(), f.Ny(), f.Nz(), 3, f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
        else
            fdotg.setToZero();
        fdotg.setState(Physical, Physical);

        lint Nz = f.Nz();
        lint nxlocmin = f.nxlocmin();
        lint nxlocmax = f.nxlocmin() + f.Nxloc();
        lint nylocmin = f.nylocmin();
        lint nylocmax = f.nylocmax();
#ifdef HAVE_MPI
        for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
            for (lint nz = 0; nz < Nz; ++nz)
                for (lint ny = nylocmin; ny < nylocmax; ++ny)
                    for (lint i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j) {
                            int ij = i3j(i, j);
                            fdotg(nx, ny, nz, 0) += f(nx, ny, nz, i) * g(nx, ny, nz, ij);
                        }
#else
        for (lint i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                int ij = i3j(i, j);
                for (lint ny = nylocmin; ny < nylocmax; ++ny)
                    for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
                        for (lint nz = 0; nz < Nz; ++nz)
                            fdotg(nx, ny, nz, 0) += f(nx, ny, nz, i) * g(nx, ny, nz, ij);
            }
#endif
    } else {
        cerr << "error in dot(f, g, fdotg) : incompatible f and g. Exiting." << endl;
        exit(1);
    }

    f.makeState(fxz, fy);
    g.makeState(gxz, gy);
    fdotg.makeSpectral();
    return;
}

void div(const FlowField& f_, FlowField& divf, const fieldstate finalstate) {
    FlowField& f = (FlowField&)f_;
    fieldstate sxz = f.xzstate();
    fieldstate sy = f.ystate();
    f.makeSpectral();

    //     int Mx = f.Mx();
    lint My = f.My();
    //     int Mz = f.Mz();
    lint mxlocmin = f.mxlocmin();
    lint mxlocmax = f.mxlocmin() + f.Mxloc();
    lint mzlocmin = f.mzlocmin();
    lint mzlocmax = f.mzlocmin() + f.Mzloc();
    int kxmax = f.kxmax();
    int kzmax = f.kzmax();
    Real Lx = f.Lx();
    Real Lz = f.Lz();

    if (f.Nd() == 3) {
        if (!f.geomCongruent(divf) || divf.Nd() != 1)
            divf.resize(f.Nx(), f.Ny(), f.Nz(), 1, f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
        else
            divf.setToZero();
        divf.setState(Spectral, Spectral);

        // Add df0/dx + df2/dz to divf
        for (lint my = 0; my < My; ++my)
            for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
                int kx = f.kx(mx);
                Complex cxkx(0.0, 2.0 * pi * kx / Lx * zero_last_mode(kx, kxmax, 1));
                for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                    int kz = f.kz(mz);
                    Complex czkz(0.0, 2.0 * pi * kz / Lz * zero_last_mode(kz, kzmax, 1));
                    divf.cmplx(mx, my, mz, 0) = f.cmplx(mx, my, mz, 0) * cxkx + f.cmplx(mx, my, mz, 2) * czkz;
                }
            }

        // Add df1/dy to to divf
        ComplexChebyCoeff g(f.Ny(), f.a(), f.b(), Spectral);
        ComplexChebyCoeff gy(f.Ny(), f.a(), f.b(), Spectral);
        for (lint mx = mxlocmin; mx < mxlocmax; ++mx)
            for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                for (lint my = 0; my < My; ++my)
                    g.set(my, f.cmplx(mx, my, mz, 1));
                diff(g, gy);
                for (lint my = 0; my < My; ++my)
                    divf.cmplx(mx, my, mz, 0) += gy[my];
            }
    } else if (f.Nd() == 9) {
        if (!f.geomCongruent(divf) || divf.Nd() != 3)
            divf.resize(f.Nx(), f.Ny(), f.Nz(), 3, f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
        else
            divf.setToZero();
        divf.setState(Spectral, Spectral);

        for (int j = 0; j < 3; ++j) {
            // Add df0j/dx + df2j/dz to divfj
            for (lint my = 0; my < My; ++my)
                for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
                    int kx = f.kx(mx);
                    Complex cxkx(0.0, 2.0 * pi * f.kx(mx) / Lx * zero_last_mode(kx, kxmax, 1));
                    for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                        int kz = f.kz(mz);
                        Complex czkz(0.0, 2.0 * pi * kz / Lz * zero_last_mode(kz, kzmax, 1));
                        divf.cmplx(mx, my, mz, j) +=
                            f.cmplx(mx, my, mz, i3j(0, j)) * cxkx + f.cmplx(mx, my, mz, i3j(2, j)) * czkz;
                    }
                }

            // Add df1j/dy to to divf
            ComplexChebyCoeff g(f.Ny(), f.a(), f.b(), Spectral);
            ComplexChebyCoeff gy(f.Ny(), f.a(), f.b(), Spectral);
            for (lint mx = mxlocmin; mx < mxlocmax; ++mx)
                for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                    for (lint my = 0; my < My; ++my)
                        g.set(my, f.cmplx(mx, my, mz, i3j(1, j)));
                    diff(g, gy);
                    for (lint my = 0; my < My; ++my)
                        divf.cmplx(mx, my, mz, j) += gy[my];
                }
        }
    } else
        cferror("div(FlowField f, FlowField divf): f must be 3d or 9d");

    f.makeState(sxz, sy);
    if (finalstate == Spectral)
        divf.makeSpectral();
    return;
}

void cross(const FlowField& f_, const FlowField& g_, FlowField& fcg, fieldstate finalstate) {
    FlowField& f = (FlowField&)f_;
    FlowField& g = (FlowField&)g_;
    fieldstate fxz = f.xzstate();
    fieldstate fy = f.ystate();
    fieldstate gxz = g.xzstate();
    fieldstate gy = g.ystate();
    assert(g.congruent(f));
    assert(f.Nd() == 3 && g.Nd() == 3);

    f.makePhysical();
    g.makePhysical();

    if (!f.geomCongruent(fcg) || fcg.Nd() != 3)
        fcg.resize(f.Nx(), f.Ny(), f.Nz(), 3, f.Lx(), f.Lz(), f.a(), f.b(), f.cfmpi());
    // else
    // fcg.setToZero();

    fcg.setState(Physical, Physical);

    //     int Nx = f.Nx();
    //     int Ny = f.Ny();
    lint Nz = f.Nz();
    lint Nd = f.Nd();
    lint nxlocmin = f.nxlocmin();
    lint nxlocmax = f.nxlocmin() + f.Nxloc();
    lint nylocmin = f.nylocmin();
    lint nylocmax = f.nylocmax();
#ifdef HAVE_MPI
    for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
        for (lint nz = 0; nz < Nz; ++nz)
            for (lint ny = nylocmin; ny < nylocmax; ++ny)
                for (lint i = 0; i < Nd; ++i) {
                    const int j = (i + 1) % 3;
                    const int k = (i + 2) % 3;
                    fcg(nx, ny, nz, i) = f(nx, ny, nz, j) * g(nx, ny, nz, k) - f(nx, ny, nz, k) * g(nx, ny, nz, j);
                }
#else
    for (lint i = 0; i < Nd; ++i) {
        const int j = (i + 1) % 3;
        const int k = (i + 2) % 3;
        for (lint ny = nylocmin; ny < nylocmax; ++ny)
            for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
                for (lint nz = 0; nz < Nz; ++nz)
                    fcg(nx, ny, nz, i) = f(nx, ny, nz, j) * g(nx, ny, nz, k) - f(nx, ny, nz, k) * g(nx, ny, nz, j);
    }
#endif
    f.makeState(fxz, fy);
    g.makeState(gxz, gy);
    if (finalstate == Spectral)
        fcg.makeSpectral();
}

void energy(const FlowField& u_, FlowField& e) {
    FlowField& u = (FlowField&)u_;
    fieldstate sxz = u.xzstate();
    fieldstate sy = u.ystate();
    u.makePhysical();

    if (!u.geomCongruent(e) || e.Nd() != 1)
        e.resize(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());
    else
        e.setToZero();
    e.setState(Physical, Physical);

    //     int Nx = u.Nx();
    //     int Ny = u.Ny();
    lint Nz = u.Nz();
    lint Nd = u.Nd();
    lint nxlocmin = u.nxlocmin();
    lint nxlocmax = u.nxlocmin() + u.Nxloc();
    lint nylocmin = u.nylocmin();
    lint nylocmax = u.nylocmax();

    for (lint i = 0; i < Nd; ++i)
        for (lint ny = nylocmin; ny < nylocmax; ++ny)
            for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
                for (int nz = 0; nz < Nz; ++nz)
                    e(nx, ny, nz, 0) += 0.5 * square(u(nx, ny, nz, i));

    e.makeSpectral();
    u.makeState(sxz, sy);
    return;
}

void energy(const FlowField& u_, const ChebyCoeff& U_, FlowField& e) {
    FlowField& u = (FlowField&)u_;
    ChebyCoeff& U = (ChebyCoeff&)U_;
    fieldstate uxzstate = u.xzstate();
    fieldstate uystate = u.ystate();
    fieldstate Ustate = U.state();
    assert(U.numModes() == u.Ny());

    if (!u.geomCongruent(e) || e.Nd() != 1)
        e.resize(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());
    else
        e.setToZero();
    e.setState(Physical, Physical);

    u.makePhysical();
    U.makePhysical();

    e.setToZero();
    e.setState(Physical, Physical);

    //     int Nx = u.Nx();
    //     int Ny = u.Ny();
    lint Nz = u.Nz();
    lint Nd = u.Nd();
    lint nxlocmin = u.nxlocmin();
    lint nxlocmax = u.nxlocmin() + u.Nxloc();
    lint nylocmin = u.nylocmin();
    lint nylocmax = u.nylocmax();

    for (lint ny = nylocmin; ny < nylocmax; ++ny)
        for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
            for (lint nz = 0; nz < Nz; ++nz)
                e(nx, ny, nz, 0) += 0.5 * square(u(nx, ny, nz, 0) + U(ny));

    for (lint i = 1; i < Nd; ++i)
        for (lint ny = nylocmin; ny < nylocmax; ++ny)
            for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
                for (lint nz = 0; nz < Nz; ++nz)
                    e(nx, ny, nz, 0) += 0.5 * square(u(nx, ny, nz, i));

    e.makeSpectral();
    u.makeState(uxzstate, uystate);
    U.makeState(Ustate);

    return;
}

// New function to average stuff in x-y slices.
FlowField xyavg(FlowField& u) {
    // const int Nx = u.Nx();
    // const int Ny = u.Ny();
    const int Nz = u.Nz();
    const int Nd = u.Nd();
    // const Real Lx = u.Lx();
    // const Real Lz = u.Lz();
    // const Real a = u.a();
    // const Real b = u.b();

    FlowField uxyavg(4, 1, Nz, Nd, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());
    u.makeSpectral();

    for (int i = 0; i < Nd; ++i)
        for (int mz = u.mzlocmin(); mz < u.mzlocmin() + u.Mzloc(); ++mz)
            uxyavg.cmplx(0, 0, mz, i) = u.cmplx(0, 0, mz, i);

    return uxyavg;
}

FlowField Qcriterion(const FlowField& u) {
    FlowField Q;
    Qcriterion(u, Q);
    return Q;
}

void Qcriterion(const FlowField& u_, FlowField& Q) {
    FlowField& u = (FlowField&)u_;
    assert(u.Nd() == 3);
    fieldstate xzstate = u.xzstate();
    fieldstate ystate = u.ystate();

    u.makeSpectral();
    FlowField gradu;
    grad(u, gradu);
    gradu.makePhysical();

    const int Nx = u.Nx();
    const int Ny = u.Ny();
    const int Nz = u.Nz();
    lint nxlocmin = u.nxlocmin();
    lint nxlocmax = u.nxlocmin() + u.Nxloc();
    lint nylocmin = u.nylocmin();
    lint nylocmax = u.nylocmax();

    Q.resize(Nx, Ny, Nz, 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());
    Q.setToZero();
    Q.makePhysical();

    Real A[9];  // Aij = A[3i+j], A = antisymm part of gradu
    Real S[9];  // Sij = S[3i+j], S = symmetric part of gradu

    for (int nx = nxlocmin; nx < nxlocmax; ++nx) {
        for (int nz = 0; nz < Nz; ++nz)
            for (int ny = nylocmin; ny < nylocmax; ++ny) {
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        A[3 * i + j] = gradu(nx, ny, nz, i3j(i, j)) - gradu(nx, ny, nz, i3j(j, i));
                        S[3 * i + j] = gradu(nx, ny, nz, i3j(i, j)) + gradu(nx, ny, nz, i3j(j, i));
                    }
                Real q = 0;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        q += square(A[3 * i + j]) - square(S[3 * i + j]);

                Q(nx, ny, nz, 0) = 0.5 * q;
            }
    }
    Q.makeSpectral();
    u.makeState(xzstate, ystate);
}

FlowField xdiff(const FlowField& f, int n) {
    FlowField g;
    xdiff(f, g, n);
    return g;
}

FlowField ydiff(const FlowField& f, int n) {
    FlowField g;
    ydiff(f, g, n);
    return g;
}

FlowField zdiff(const FlowField& f, int n) {
    FlowField g;
    zdiff(f, g, n);
    return g;
}

FlowField diff(const FlowField& f, int i, int n) {
    FlowField g;
    diff(f, g, i, n);
    return g;
}

FlowField diff(const FlowField& f, int nx, int ny, int nz) {
    FlowField g;
    diff(f, g, nx, ny, nz);
    return g;
}

FlowField grad(const FlowField& f) {
    FlowField g;
    grad(f, g);
    return g;
}
FlowField lapl(const FlowField& f) {
    FlowField g;
    lapl(f, g);
    return g;
}
FlowField curl(const FlowField& f) {
    FlowField g;
    curl(f, g);
    return g;
}
FlowField norm(const FlowField& f) {
    FlowField g;
    norm(f, g);
    return g;
}
FlowField norm2(const FlowField& f) {
    FlowField g;
    norm2(f, g);
    return g;
}
FlowField div(const FlowField& f) {
    FlowField g;
    div(f, g);
    return g;
}

FlowField outer(const FlowField& f, const FlowField& g) {
    FlowField fg;
    outer(f, g, fg);
    return fg;
}
FlowField cross(const FlowField& f, const FlowField& g) {
    FlowField fxg;
    cross(f, g, fxg);
    return fxg;
}
FlowField dot(const FlowField& f, const FlowField& g) {
    FlowField fdotg;
    dot(f, g, fdotg);
    return fdotg;
}
FlowField energy(const FlowField& u) {
    FlowField e;
    energy(u, e);
    return e;
}
FlowField energy(const FlowField& u, ChebyCoeff& U) {
    FlowField e;
    energy(u, U, e);
    return e;
}

void rotationalNL(const FlowField& u_, FlowField& f, FlowField& tmp, const fieldstate finalstate) {
    FlowField& u = (FlowField&)u_;
    FlowField& vort = tmp;

    assert(u.Nd() == 3);
    fieldstate uxzstate = u.xzstate();
    fieldstate uystate = u.ystate();

    if (!u.geomCongruent(f) || f.Nd() != 3)
        f.resize(u.Nx(), u.Ny(), u.Nz(), 3, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());
    f.setState(Physical, Physical);

    if (!u.geomCongruent(vort) || vort.Nd() < 3)
        vort.resize(u.Nx(), u.Ny(), u.Nz(), 3, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());

    u.makeSpectral();
    curl(u, vort);

    u.makePhysical();
    vort.makePhysical();
    cross(vort, u, f, finalstate);

    if (finalstate == Spectral) {
        f.makeSpectral();
    }
    // Possible extra transforms if entry state was Physical
    u.makeState(uxzstate, uystate);

    return;
}

void convectionNL(const FlowField& u_, FlowField& f, FlowField& tmp, const fieldstate finalstate) {
    dotgrad(u_, u_, f, tmp);
}

void adjointTerms(const FlowField& u_, FlowField& u_dir_, FlowField& f, FlowField& tmp, FlowField& tmpadj,
                  const fieldstate finalstate) {
    FlowField& v = (FlowField&)u_;
    FlowField& u = (FlowField&)u_dir_;
    FlowField& grad_v = tmp;
    FlowField& grad_uT = tmpadj;

    assert(u.Nd() == 3 && v.Nd() == 3);
    assert(u.geomCongruent(v));

    if (!u.geomCongruent(f) || f.Nd() != 3)
        f.resize(u.Nx(), u.Ny(), u.Nz(), 3, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());
    else
        f.setToZero();

    if (!u.geomCongruent(grad_v) || grad_v.Nd() < 9)
        grad_v.resize(u.Nx(), u.Ny(), u.Nz(), 9, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());

    if (!u.geomCongruent(grad_uT) || grad_uT.Nd() < 9)
        grad_uT.resize(u.Nx(), u.Ny(), u.Nz(), 9, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());

    fieldstate uxzstate = u.xzstate();
    fieldstate uystate = u.ystate();
    fieldstate vxzstate = v.xzstate();
    fieldstate vystate = v.ystate();

    // u_dir dot grad u
    // u     dot grad v
    v.makeSpectral();
    grad(v, grad_v);
    grad_v.makePhysical();
    v.makePhysical();

    // u dot gradT u_dir
    // v dot gradT u
    u.makeSpectral();
    grad(u, grad_uT);
    grad_uT.makePhysical();
    u.makePhysical();
    f.setState(Physical, Physical);

    //     int Nx = u.Nx();
    //     lint Ny = u.Ny();
    lint Nz = u.Nz();
    lint nxlocmin = u.nxlocmin();
    lint nxlocmax = u.nxlocmin() + u.Nxloc();
    lint nylocmin = u.nylocmin();
    lint nylocmax = u.nylocmax();
#ifdef HAVE_MPI
    for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
        for (lint nz = 0; nz < Nz; ++nz)
            for (lint ny = nylocmin; ny < nylocmax; ++ny)
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        f(nx, ny, nz, i) -= u(nx, ny, nz, j) * grad_v(nx, ny, nz, i3j(i, j));
                        // index change i j -> j i for transpose of grad
                        f(nx, ny, nz, i) += v(nx, ny, nz, j) * grad_uT(nx, ny, nz, i3j(j, i));
                    }
#else
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (lint ny = nylocmin; ny < nylocmax; ++ny)
                for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
                    for (lint nz = 0; nz < Nz; ++nz) {
                        f(nx, ny, nz, i) -= u(nx, ny, nz, j) * grad_v(nx, ny, nz, i3j(i, j));
                        // index change i j -> j i for transpose of grad
                        f(nx, ny, nz, i) += v(nx, ny, nz, j) * grad_uT(nx, ny, nz, i3j(j, i));
                    }
#endif
    f.makeSpectral();
    u.makeState(uxzstate, uystate);
    v.makeState(vxzstate, vystate);
}

void perturbationTermsNLin(const FlowField& u_, FlowField& u_nlin_, FlowField& f, FlowField& tmp, FlowField& tmppert,
                           const fieldstate finalstate) {
    FlowField& v = (FlowField&)u_;
    FlowField& u = (FlowField&)u_nlin_;
    FlowField& grad_v = tmp;
    FlowField& grad_u = tmppert;

    assert(u.Nd() == 3 && v.Nd() == 3);
    assert(u.geomCongruent(v));

    if (!u.geomCongruent(f) || f.Nd() != 3)
        f.resize(u.Nx(), u.Ny(), u.Nz(), 3, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());
    else
        f.setToZero();

    if (!u.geomCongruent(grad_v) || grad_v.Nd() < 9)
        grad_v.resize(u.Nx(), u.Ny(), u.Nz(), 9, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());

    if (!u.geomCongruent(grad_u) || grad_u.Nd() < 9)
        grad_u.resize(u.Nx(), u.Ny(), u.Nz(), 9, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());

    fieldstate uxzstate = u.xzstate();
    fieldstate uystate = u.ystate();
    fieldstate vxzstate = v.xzstate();
    fieldstate vystate = v.ystate();

    // u_ecs dot grad u
    // u     dot grad v
    v.makeSpectral();
    grad(v, grad_v);
    grad_v.makePhysical();
    v.makePhysical();

    // u dot grad u_ecs
    // v dot grad u
    u.makeSpectral();
    grad(u, grad_u);
    grad_u.makePhysical();
    u.makePhysical();
    f.setState(Physical, Physical);

    //     int Nx = u.Nx();
    //     lint Ny = u.Ny();
    lint Nz = u.Nz();
    lint nxlocmin = u.nxlocmin();
    lint nxlocmax = u.nxlocmin() + u.Nxloc();
    lint nylocmin = u.nylocmin();
    lint nylocmax = u.nylocmax();
#ifdef HAVE_MPI
    for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
        for (lint nz = 0; nz < Nz; ++nz)
            for (lint ny = nylocmin; ny < nylocmax; ++ny)
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        f(nx, ny, nz, i) += u(nx, ny, nz, j) * grad_v(nx, ny, nz, i3j(i, j));
                        f(nx, ny, nz, i) += v(nx, ny, nz, j) * grad_u(nx, ny, nz, i3j(i, j));
                        f(nx, ny, nz, i) += v(nx, ny, nz, j) * grad_v(nx, ny, nz, i3j(i, j));  // non linear term
                    }
#else
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (lint ny = nylocmin; ny < nylocmax; ++ny)
                for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
                    for (lint nz = 0; nz < Nz; ++nz) {
                        f(nx, ny, nz, i) += u(nx, ny, nz, j) * grad_v(nx, ny, nz, i3j(i, j));
                        f(nx, ny, nz, i) += v(nx, ny, nz, j) * grad_u(nx, ny, nz, i3j(i, j));
                        f(nx, ny, nz, i) += v(nx, ny, nz, j) * grad_v(nx, ny, nz, i3j(i, j));  // non linear term
                    }
#endif

    f.makeSpectral();
    u.makeState(uxzstate, uystate);
    v.makeState(vxzstate, vystate);
}

void perturbationTermsLin(const FlowField& u_, FlowField& u_nlin_, FlowField& f, FlowField& tmp, FlowField& tmppert,
                          const fieldstate finalstate) {
    FlowField& v = (FlowField&)u_;
    FlowField& u = (FlowField&)u_nlin_;
    FlowField& grad_v = tmp;
    FlowField& grad_u = tmppert;

    assert(u.Nd() == 3 && v.Nd() == 3);
    assert(u.geomCongruent(v));

    if (!u.geomCongruent(f) || f.Nd() != 3)
        f.resize(u.Nx(), u.Ny(), u.Nz(), 3, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());
    else
        f.setToZero();

    if (!u.geomCongruent(grad_v) || grad_v.Nd() < 9)
        grad_v.resize(u.Nx(), u.Ny(), u.Nz(), 9, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());

    if (!u.geomCongruent(grad_u) || grad_u.Nd() < 9)
        grad_u.resize(u.Nx(), u.Ny(), u.Nz(), 9, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());

    fieldstate uxzstate = u.xzstate();
    fieldstate uystate = u.ystate();
    fieldstate vxzstate = v.xzstate();
    fieldstate vystate = v.ystate();

    // u_ecs dot grad u
    // u     dot grad v
    v.makeSpectral();
    grad(v, grad_v);
    grad_v.makePhysical();
    v.makePhysical();

    // u dot grad u_ecs
    // v dot grad u
    u.makeSpectral();
    grad(u, grad_u);
    grad_u.makePhysical();
    u.makePhysical();
    f.setState(Physical, Physical);

    //     int Nx = u.Nx();
    //     lint Ny = u.Ny();
    lint Nz = u.Nz();
    lint nxlocmin = u.nxlocmin();
    lint nxlocmax = u.nxlocmin() + u.Nxloc();
    lint nylocmin = u.nylocmin();
    lint nylocmax = u.nylocmax();
#ifdef HAVE_MPI
    for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
        for (lint nz = 0; nz < Nz; ++nz)
            for (lint ny = nylocmin; ny < nylocmax; ++ny)
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        f(nx, ny, nz, i) += u(nx, ny, nz, j) * grad_v(nx, ny, nz, i3j(i, j));
                        f(nx, ny, nz, i) += v(nx, ny, nz, j) * grad_u(nx, ny, nz, i3j(i, j));
                        // f(nx,ny,nz,i) += v(nx,ny,nz,j)*grad_v(nx,ny,nz,i3j(i,j)); // non linear term
                    }
#else
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (lint ny = nylocmin; ny < nylocmax; ++ny)
                for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
                    for (lint nz = 0; nz < Nz; ++nz) {
                        f(nx, ny, nz, i) += u(nx, ny, nz, j) * grad_v(nx, ny, nz, i3j(i, j));
                        f(nx, ny, nz, i) += v(nx, ny, nz, j) * grad_u(nx, ny, nz, i3j(i, j));
                        // f(nx,ny,nz,i) += v(nx,ny,nz,j)*grad_v(nx,ny,nz,i3j(i,j)); // non linear term
                    }
#endif
    f.makeSpectral();
    u.makeState(uxzstate, uystate);
    v.makeState(vxzstate, vystate);
}

void divergenceNL(const FlowField& u_, FlowField& f, FlowField& tmp, const fieldstate finalstate) {
    FlowField& u = (FlowField&)u_;
    FlowField& uu = tmp;

    assert(u.Nd() == 3);

    fieldstate uxzstate = u.xzstate();
    fieldstate uystate = u.ystate();

    if (!u.geomCongruent(f) || f.Nd() != 3)
        f.resize(u.Nx(), u.Ny(), u.Nz(), 3, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());

    if (!u.geomCongruent(uu) || uu.Nd() < 9)
        uu.resize(u.Nx(), u.Ny(), u.Nz(), 9, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());

    u.makePhysical();
    outer(u, u, uu);

    uu.makeSpectral();
    div(uu, f, finalstate);

    // Possible extra transforms if entry states are Physical
    u.makeState(uxzstate, uystate);
    return;
}

// This function spells out the computation in low-level operations rather
// than making calls to calling grad and outer, because the sequence of
// computations for latter would require a few extra transforms on u.

// Thesis notes 4/22/01, 12/01/03
// Compute nonlinearity as 1/2 [u dot grad v + div (uv)]
void skewsymmetricNL(const FlowField& u_, FlowField& f, FlowField& tmp, const fieldstate finalstate) {
    FlowField& u = (FlowField&)u_;

    assert(u.Nd() == 3);

    fieldstate uxzstate = u.xzstate();
    fieldstate uystate = u.ystate();

    if (!u.geomCongruent(f) || f.Nd() != 3)
        f.resize(u.Nx(), u.Ny(), u.Nz(), 3, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());
    else
        f.setToZero();  // UNNECESSARY?
    f.setState(Physical, Physical);

    if (!u.geomCongruent(tmp) || tmp.Nd() < 9)
        tmp.resize(u.Nx(), u.Ny(), u.Nz(), 9, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());
    else
        tmp.setToZero();

    // Possible extra transforms if entry states are Physical
    u.makeSpectral();

    // ====================================================================
    // Compute 1/2 u dotgrad v.

    FlowField& grad_u = tmp;
    grad(u, grad_u);

    grad_u.makePhysical();
    u.makePhysical();

    //     int Nx = u.Nx();
    //     int Ny = u.Ny();
    int Nz = u.Nz();
    lint nxlocmin = u.nxlocmin();
    lint nxlocmax = u.nxlocmin() + u.Nxloc();
    lint nylocmin = u.nylocmin();
    lint nylocmax = u.nylocmax();

// Accumulate 1/2 u_j du_i/dx_j in f_i
#ifdef HAVE_MPI
    for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
        for (int nz = 0; nz < Nz; ++nz)
            for (lint ny = nylocmin; ny < nylocmax; ++ny)
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        int ij = i3j(i, j);
                        f(nx, ny, nz, i) += 0.5 * u(nx, ny, nz, j) * grad_u(nx, ny, nz, ij);
                    }
#else
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            int ij = i3j(i, j);
            for (lint ny = nylocmin; ny < nylocmax; ++ny)
                for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
                    for (int nz = 0; nz < Nz; ++nz)
                        f(nx, ny, nz, i) += 0.5 * u(nx, ny, nz, j) * grad_u(nx, ny, nz, ij);
        }
#endif

    // ================================================================
    // II. Add grad dot (u v) to f. Spell out loops because outer(u,v,f)
    // and div(uv, f) would overwrite results already in f (and changing
    // order of div and convec calculations would require an extra transform)

    FlowField& uu = tmp;
    outer(u, u, uu);

    uu.makeSpectral();
    f.makeSpectral();

    //     int Mx = u.Mx();
    int My = u.My();
    //     int Mz = u.Mz();
    lint mxlocmin = u.mxlocmin();
    lint mxlocmax = u.mxlocmin() + u.Mxloc();
    lint mzlocmin = u.mzlocmin();
    lint mzlocmax = u.mzlocmin() + u.Mzloc();
    int kxmax = u.kxmax();
    int kzmax = u.kzmax();
    Real Lx = u.Lx();
    Real Lz = u.Lz();

    ComplexChebyCoeff tmpProfile(My, u.a(), u.b(), Spectral);
    ComplexChebyCoeff tmpProfile_y(My, u.a(), u.b(), Spectral);

    // Now set f_i += d/dx_j (u_i u_j)
#ifdef HAVE_MPI
    for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
        int kx = u.kx(mx);
        Complex Dx(0.0, 2 * pi * kx / Lx * zero_last_mode(kx, kxmax, 1));
        for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
            int kz = u.kz(mz);
            Complex Dz(0.0, 2 * pi * kz / Lz * zero_last_mode(kz, kzmax, 1));
            for (int i = 0; i < 3; ++i) {
                int i0 = i3j(i, 0);
                int i1 = i3j(i, 1);
                int i2 = i3j(i, 2);
                for (lint my = 0; my < My; ++my) {
                    tmpProfile.set(my, uu.cmplx(mx, my, mz, i1));
                }
                diff(tmpProfile, tmpProfile_y);
                for (lint my = 0; my < My; ++my) {
                    f.cmplx(mx, my, mz, i)
                        // Add in u_j du_i/dx and u_j du_i/dz, that is, d/dx_j (u_i u_j) for j=0,2
                        += 0.5 * (Dx * uu.cmplx(mx, my, mz, i0) + Dz * uu.cmplx(mx, my, mz, i2)) +
                           0.5 * tmpProfile_y[my];  // Add in du_i/dy, that is d/dx_j (u_i v_j) for j=1
                }
            }
        }
    }
#else
    for (int i = 0; i < 3; ++i) {
        int i0 = i3j(i, 0);
        int i1 = i3j(i, 1);
        int i2 = i3j(i, 2);

        // Add in u_j du_i/dx and u_j du_i/dz, that is, d/dx_j (u_i u_j) for j=0,2
        for (lint my = 0; my < My; ++my)
            for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
                int kx = u.kx(mx);
                Complex Dx(0.0, 2 * pi * kx / Lx * zero_last_mode(kx, kxmax, 1));
                for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                    int kz = u.kz(mz);
                    Complex Dz(0.0, 2 * pi * kz / Lz * zero_last_mode(kz, kzmax, 1));
                    f.cmplx(mx, my, mz, i) += 0.5 * (Dx * uu.cmplx(mx, my, mz, i0) + Dz * uu.cmplx(mx, my, mz, i2));
                }
            }
        // Add in du_i/dy, that is d/dx_j (u_i v_j) for j=1
        for (lint mx = mxlocmin; mx < mxlocmax; ++mx)
            for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                for (lint my = 0; my < My; ++my)
                    tmpProfile.set(my, uu.cmplx(mx, my, mz, i1));
                diff(tmpProfile, tmpProfile_y);
                for (lint my = 0; my < My; ++my)
                    f.cmplx(mx, my, mz, i) += 0.5 * tmpProfile_y[my];  // j=1
            }
    }
#endif
    if (finalstate == Physical)
        f.makePhysical();
    // Possible extra transforms if entry states are Physical
    u.makeState(uxzstate, uystate);
    return;
}

void linearizedNL(const FlowField& u_, const ChebyCoeff& U_, const ChebyCoeff& W_, FlowField& f,
                  const fieldstate finalstate) {
    FlowField& u = (FlowField&)u_;
    ChebyCoeff& U = (ChebyCoeff&)U_;
    ChebyCoeff& W = (ChebyCoeff&)W_;

    assert(u.Nd() == 3);
    assert(U.N() == u.Ny());
    assert(W.N() == u.Ny());
    assert(U.a() == u.a() && U.b() == u.b());
    assert(W.a() == u.a() && W.b() == u.b());

    fieldstate uxzstate = u.xzstate();
    fieldstate uystate = u.ystate();
    fieldstate Ustate = U.state();
    fieldstate Wstate = W.state();

    if (!u.geomCongruent(f) || f.Nd() != 3)
        f.resize(u.Nx(), u.Ny(), u.Nz(), 3, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());
    else
        f.setToZero();
    f.setState(Spectral, Physical);

    ChebyTransform trans(U.N());
    U.makeSpectral(trans);
    W.makeSpectral(trans);
    ChebyCoeff Uy, Wy;
    diff(U, Uy);
    diff(W, Wy);
    U.makePhysical(trans);
    Uy.makePhysical(trans);
    W.makePhysical(trans);
    Wy.makePhysical(trans);

    Complex cu;
    Complex cv;
    Complex cw;

    //     int Mx = u.Mx();
    lint Ny = u.Ny();
    //     int Mz = u.Mz();
    lint mxlocmin = u.mxlocmin();
    lint mxlocmax = u.mxlocmin() + u.Mxloc();
    lint mzlocmin = u.mzlocmin();
    lint mzlocmax = u.mzlocmin() + u.Mzloc();
    Real Lx = u.Lx();
    Real Lz = u.Lz();
    int kxmax = u.kxmax();
    int kzmax = u.kzmax();

    u.makeState(Spectral, Physical);
    for (lint ny = 0; ny < Ny; ++ny)
        for (lint mx = mxlocmin; mx < mxlocmax; ++mx) {
            int kx = u.kx(mx);
            Complex d_dx(0.0, 2 * pi * kx / Lx * zero_last_mode(kx, kxmax, 1));
            for (lint mz = mzlocmin; mz < mzlocmax; ++mz) {
                int kz = u.kz(mz);
                Complex d_dz(0.0, 2 * pi * kz / Lz * zero_last_mode(kz, kzmax, 1));
                Complex Uddx_Wddz = U[ny] * d_dx + W[ny] * d_dz;

                cu = u.cmplx(mx, ny, mz, 0);
                cv = u.cmplx(mx, ny, mz, 1);
                cw = u.cmplx(mx, ny, mz, 2);

                f.cmplx(mx, ny, mz, 0) = Uddx_Wddz * cu + cv * Uy[ny];
                f.cmplx(mx, ny, mz, 1) = Uddx_Wddz * cv;
                f.cmplx(mx, ny, mz, 2) = Uddx_Wddz * cw + cv * Wy[ny];  // correct sign?
            }
        }
    if (finalstate == Spectral)
        f.makeSpectral();
    else
        f.makePhysical();
    u.makeState(uxzstate, uystate);
    U.makeState(Ustate);
    W.makeState(Wstate);
    return;
}

void linearAboutFieldNL(const FlowField& u, const FlowField& ubase, const FlowField& ubtot, const FlowField& grad_ubtot,
                        FlowField& f, FlowField& tmp, const fieldstate finalstate) {
    // f = utot dotgrad utot, where utot = u + Ubase.
    //
    // Linearize f about ubtot = ubase + Ubase. Let du = u - ubase
    //
    // then utot = du + ubtot, and f = (du + ubtot) dotgrad (du + ubtot)
    //
    // lin(f) = du dotgrad ubtot + ubtot dotgrad du + ubtot dotgrad ubtot
    //

    /*****************************************************************
    // This gives L2Norm(du/dt)  == 5.3628322372532557e-06 at T=1
    // in 39s wSMRK2, 18s wSBDF3
    FlowField utot(du);
    utot += ubtot;

    FlowField du(u);
    du -= ubase;


    FlowField ftmp;
    convectionNL(utot,  f, tmp);
    convectionNL(du, ftmp, tmp);
    f -= ftmp;
    *******************************************************************/
    FlowField& utot = const_cast<FlowField&>(ubtot);
    utot += u;
    utot -= ubase;

    FlowField& du = const_cast<FlowField&>(u);
    du -= ubase;

    FlowField ftmp;
    rotationalNL(utot, f, tmp, finalstate);
    rotationalNL(du, ftmp, tmp, finalstate);
    f -= ftmp;

    du += ubase;
    utot += ubase;
    utot -= u;
}

void linearAboutFieldNL(const FlowField& u, const FlowField& ubase, const ChebyCoeff& Ubase, FlowField& f,
                        FlowField& tmp, FlowField& ftmp, const fieldstate finalstate) {
    // Gives L2Norm(du/dt)  == 5.3628325247665347e-06 for T=1
    FlowField& utot = const_cast<FlowField&>(u);
    utot += Ubase;
    convectionNL(utot, f, tmp, finalstate);
    utot -= Ubase;

    FlowField& du = utot;
    du -= ubase;

    convectionNL(du, ftmp, tmp, finalstate);
    du += ubase;

    f -= ftmp;

    /************************************************
    // Gives L2Norm(du/dt)  == 5.3628325247665347e-06 for T=1
    FlowField utot(u);
    utot += Ubase;
    convectionNL(utot, f, tmp);
    utot -= Ubase;

    FlowField du(u);
    du -= ubase;

    FlowField ftmp;
    convectionNL(du, ftmp, tmp);

    f -= ftmp;
    ************************************************/

    /**************************************
     // Gives L2Norm(du/dt)  == 0.27293579138778834  for T=1
    FlowField& du    = const_cast<FlowField&>(u);
    FlowField& ubtot = const_cast<FlowField&>(ubase);
    du    -= ubase;
    ubtot += Ubase;

    FlowField ftmp;
    dotgrad(du,ubtot,ftmp);

    f = ftmp;
    dotgrad(ubtot,du,ftmp);
    f += ftmp;

    ubtot -= Ubase;
    du    += ubase;

    dotgrad(ubase,ubase,ftmp);
    f += ftmp;
    **************************************/

    /**************************************
    FlowField fdu;
    FlowField du(u);


    FlowField ftmp;
    dotgrad(du,ubtot,ftmp);
    f = ftmp;

    dotgrad(ubtot,du,ftmp);
    f += ftmp;

    dotgrad(ubase,ubase,ftmp);

    f += ftmp;
    *****************************************/
}

void linearizedNL(const FlowField& u_, const FlowField& ubase, const ChebyCoeff& Ubase, FlowField& f, FlowField& tmp,
                  const fieldstate finalstate) {
    /*****************************************
     // Gives L2Norm(du/dt)  == 5.3628325247665347e-06 for T=1
    FlowField& u = const_cast<FlowField&>(u_);
    u += Ubase;
    convectionNL(u, f, tmp);
    u -= Ubase;
    *******************************************/

    FlowField& u = const_cast<FlowField&>(u_);
    u += Ubase;
    convectionNL(u, f, tmp, finalstate);
    u -= Ubase;

    FlowField du(u_);
    du -= ubase;

    FlowField ftmp;
    convectionNL(du, ftmp, tmp, finalstate);

    f -= ftmp;

    /***********************

    FlowField fdu;
    FlowField du(u_);

    FlowField ftmp;
    dotgrad(du,ubtot,ftmp);
    f = ftmp;

    dotgrad(ubtot,du,ftmp);
    f += ftmp;

    dotgrad(ubase,ubase,ftmp);

    f += ftmp;
    *************************/
}

void linearizedNL(const FlowField& u_, const FlowField& ubtot, const FlowField& grad_ubtot, FlowField& f,
                  FlowField& tmp, const fieldstate finalstate) {
    FlowField& u = const_cast<FlowField&>(u_);
    ubtot.assertState(Physical, Physical);
    grad_ubtot.assertState(Physical, Physical);

    cout << "LinearizedNL(u,ubase,vortbase,f,tmp) { " << endl;

    assert(u.Nd() == 3 && ubtot.Nd() == 3);
    assert(u.geomCongruent(ubtot));

    if (!u.geomCongruent(f) || f.Nd() != 3)
        f.resize(u.Nx(), u.Ny(), u.Nz(), 3, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());
    else
        f.setToZero();

    if (!u.geomCongruent(tmp) || tmp.Nd() < 9)
        tmp.resize(u.Nx(), u.Ny(), u.Nz(), 9, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());

    fieldstate uxzstate = u.xzstate();
    fieldstate uystate = u.ystate();

    cout << "L2Norm(u)     == " << L2Norm(u) << endl;
    // cout << "L2Norm(ubtot) == " << L2Norm(ubtot) << endl;

    FlowField& grad_u = tmp;
    u.makeSpectral();
    grad(u, grad_u);
    grad_u.makePhysical();

    f.setState(Physical, Physical);

    //     int Nx = u.Nx();
    //     int Ny = u.Ny();
    lint Nz = u.Nz();
    lint nxlocmin = u.nxlocmin();
    lint nxlocmax = u.nxlocmin() + u.Nxloc();
    lint nylocmin = u.nylocmin();
    lint nylocmax = u.nylocmax();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            int ij = i3j(i, j);
            for (lint ny = nylocmin; ny < nylocmax; ++ny)
                for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
                    for (lint nz = 0; nz < Nz; ++nz)
                        f(nx, ny, nz, i) += ubtot(nx, ny, nz, j) * grad_u(nx, ny, nz, ij);
        }

    u.makePhysical();

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            int ij = i3j(i, j);
            for (lint ny = nylocmin; ny < nylocmax; ++ny)
                for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
                    for (lint nz = 0; nz < Nz; ++nz)
                        f(nx, ny, nz, i) += u(nx, ny, nz, j) * grad_ubtot(nx, ny, nz, ij);
        }
    if (finalstate == Spectral)
        f.makeSpectral();
    u.makeState(uxzstate, uystate);
    cout << "L2Norm(f)    == " << L2Norm(f) << endl;
}

void dotgrad(const FlowField& u_, const FlowField& v_, FlowField& f, FlowField& tmp) {
    FlowField& u = (FlowField&)u_;
    FlowField& v = (FlowField&)v_;
    FlowField& grad_v = tmp;

    assert(u.Nd() == 3 && v.Nd() == 3);
    assert(u.geomCongruent(v));

    if (!u.geomCongruent(f) || f.Nd() != 3)
        f.resize(u.Nx(), u.Ny(), u.Nz(), 3, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());
    else
        f.setToZero();

    if (!u.geomCongruent(grad_v) || grad_v.Nd() < 9)
        grad_v.resize(u.Nx(), u.Ny(), u.Nz(), 9, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());

    fieldstate uxzstate = u.xzstate();
    fieldstate uystate = u.ystate();
    fieldstate vxzstate = v.xzstate();
    fieldstate vystate = v.ystate();

    v.makeSpectral();
    grad(v, grad_v);
    grad_v.makePhysical();
    u.makePhysical();

    f.setState(Physical, Physical);

    //     int Nx = u.Nx();
    //     lint Ny = u.Ny();
    lint Nz = u.Nz();
    lint nxlocmin = u.nxlocmin();
    lint nxlocmax = u.nxlocmin() + u.Nxloc();
    lint nylocmin = u.nylocmin();
    lint nylocmax = u.nylocmax();
#ifdef HAVE_MPI
    for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
        for (lint nz = 0; nz < Nz; ++nz)
            for (lint ny = nylocmin; ny < nylocmax; ++ny)
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        int ij = i3j(i, j);
                        f(nx, ny, nz, i) += u(nx, ny, nz, j) * grad_v(nx, ny, nz, ij);
                    }
#else
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            int ij = i3j(i, j);
            for (lint ny = nylocmin; ny < nylocmax; ++ny)
                for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
                    for (lint nz = 0; nz < Nz; ++nz)
                        f(nx, ny, nz, i) += u(nx, ny, nz, j) * grad_v(nx, ny, nz, ij);
        }
#endif
    f.makeSpectral();
    u.makeState(uxzstate, uystate);
    v.makeState(vxzstate, vystate);
}

void dotgradScalar(const FlowField& u_, const FlowField& s_, FlowField& f, FlowField& tmp) {
    FlowField& u = (FlowField&)u_;
    FlowField& s = (FlowField&)s_;
    FlowField& grad_s = tmp;

    assert(u.Nd() == 3 && s.Nd() == 1);
    assert(u.geomCongruent(s));

    if (!u.geomCongruent(f) || f.Nd() != 1)
        f.resize(u.Nx(), u.Ny(), u.Nz(), 1, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());
    else
        f.setToZero();

    if (!u.geomCongruent(grad_s) || grad_s.Nd() < 3)
        grad_s.resize(u.Nx(), u.Ny(), u.Nz(), 3, u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());

    fieldstate uxzstate = u.xzstate();
    fieldstate uystate = u.ystate();
    fieldstate sxzstate = s.xzstate();
    fieldstate systate = s.ystate();

    s.makeSpectral();
    grad(s, grad_s);
    grad_s.makePhysical();
    u.makePhysical();

    f.setState(Physical, Physical);

    //     int Nx = u.Nx();
    //     lint Ny = u.Ny();
    lint Nz = u.Nz();
    lint nxlocmin = u.nxlocmin();
    lint nxlocmax = u.nxlocmin() + u.Nxloc();
    lint nylocmin = u.nylocmin();
    lint nylocmax = u.nylocmax();

#ifdef HAVE_MPI
    for (int j = 0; j < 3; ++j)
        for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
            for (lint nz = 0; nz < Nz; ++nz)
                for (lint ny = nylocmin; ny < nylocmax; ++ny)
                    f(nx, ny, nz, 0) += u(nx, ny, nz, j) * grad_s(nx, ny, nz, j);
#else
    for (int j = 0; j < 3; ++j)
        for (lint ny = nylocmin; ny < nylocmax; ++ny)
            for (lint nx = nxlocmin; nx < nxlocmax; ++nx)
                for (lint nz = 0; nz < Nz; ++nz)
                    f(nx, ny, nz, 0) += u(nx, ny, nz, j) * grad_s(nx, ny, nz, j);
#endif

    f.makeSpectral();
    u.makeState(uxzstate, uystate);
    s.makeState(sxzstate, systate);
}

FlowField dotgrad(const FlowField& u, const FlowField& v, FlowField& tmp) {
    FlowField u_dg_v;
    dotgrad(u, v, u_dg_v, tmp);
    return u_dg_v;
}

Real L2InnerProduct(const RealProfileNG& e, const FlowField& f, bool normalize) {
    assert(e.state() == Spectral);
    assert(f.ystate() == Spectral);
    assert(f.xzstate() == Spectral);
    assert(f.congruent(e));

    Real sum = 0.0;

    ComplexChebyCoeff fprof_p(f.Ny(), f.a(), f.b(), Spectral);
    ComplexChebyCoeff fprof_m(f.Ny(), f.a(), f.b(), Spectral);

    bool padded = f.padded();
    int kxmax = padded ? f.kxmaxDealiased() : f.kxmax();
    int kzmax = padded ? f.kzmaxDealiased() : f.kzmax();

    if (abs(e.jx()) > kxmax || abs(e.jz()) > kzmax)
        return 0;

    int mxp = f.mx(abs(e.jx()));
    int mxm = f.mx(-abs(e.jx()));
    int mz = f.mz(abs(e.jz()));
    int cz;  // cz = 2 for kz>0 to take account of kz<0 ghost modes
    e.jz() == 0 ? cz = 1 : cz = 2;
    for (int i = 0; i < f.vectorDim(); ++i) {
        const Complex norm_p = e.normalization_p(i);
        const Complex norm_m = e.normalization_m(i);
        const ChebyCoeff& ei = e.u_[i];
        for (int ny = 0; ny < f.Ny(); ++ny) {
            fprof_p.set(ny, f.cmplx(mxp, ny, mz, i));
        }
        if (e.jx() != 0) {
            for (int ny = 0; ny < f.Ny(); ++ny) {
                fprof_m.set(ny, f.cmplx(mxm, ny, mz, i));
            }
        }
        if (norm_p.real() != 0)
            sum += cz * norm_p.real() * L2InnerProduct(fprof_p.re, ei, normalize);
        if (norm_p.imag() != 0)
            sum += cz * norm_p.imag() * L2InnerProduct(fprof_p.im, ei, normalize);
        if (norm_m.real() != 0)
            sum += cz * norm_m.real() * L2InnerProduct(fprof_m.re, ei, normalize);
        if (norm_m.imag() != 0)
            sum += cz * norm_m.imag() * L2InnerProduct(fprof_m.im, ei, normalize);
    }
    if (!normalize)
        sum *= f.Lx() * f.Lz();
    return sum;
}

/**************************************************************************
void calc_dedt(const FlowField& u, const ChebyCoeff& U, const FlowField& p,
               Real dPdx, Real nu, FlowField& dedt, FlowField& tmp3d_a,
               FlowField& tmp3d_b, FlowField& tmp3d_c, FlowField& tmp9d) {
  assert(u.Nd() == 3);
  assert(U.state() == Spectral);
  assert(u.xzstate() == Spectral && u.ystate() == Spectral);

  FlowField& utot   = tmp3d_a;
  FlowField& grad_p = tmp3d_b;
  FlowField& lapl_u = tmp3d_c;
  FlowField& grad_u = tmp9d;

  // Set utot = u + U. This will reduce number of terms to calculate on RHS
  utot = u;
  utot += U;

  // Calculate derivatives of u and p
  grad(utot,grad_u);
  grad(p,   grad_p);
  lapl(utot,lapl_u);

  utot.makePhysical();
  grad_p.makePhysical();
  grad_u.makePhysical();
  lapl_u.makePhysical();

  dedt.setToZero();
  dedt.setState(Physical, Physical);

  int Nx = utot.Nx();
  int Ny = utot.Ny();
  int Nz = utot.Nz();
  int Nd = utot.Nd();

  // Calculate terms with two sums on i,j
  for (int j=0; j<Nd; ++j)
    for (int i=0; i<Nd; ++i)
      for (int ny=0; ny<Ny; ++ny)
        for (int nx=0; nx<Nx; ++nx)
          for (int nz=0; nz<Nz; ++nz)
            dedt(nx,ny,nz,0) =
              nu*utot(nx,ny,nz,i)*lapl_u(nx,ny,nz,i)
              - utot(nx,ny,nz,i)*utot(nx,ny,nz,j)*grad_u(nx,ny,nz,i3j(i,j));

  // Calculate terms with one sum on i
  for (int i=0; i<Nd; ++i)
    for (int ny=0; ny<Ny; ++ny)
      for (int nx=0; nx<Nx; ++nx)
        for (int nz=0; nz<Nz; ++nz)
          dedt(nx,ny,nz,0) -= utot(nx,ny,nz,i)*grad_p(nx,ny,nz,i);


  // Calculate terms with no sum
  for (int ny=0; ny<Ny; ++ny)
    for (int nx=0; nx<Nx; ++nx)
      for (int nz=0; nz<Nz; ++nz)
        dedt(nx,ny,nz,0) -= utot(nx,ny,nz,0)*dPdx;

 dedt.makeSpectral();
}
***********************************************************************/
FlowField extractRolls(const FlowField& u) {
    assert(u.xzstate() == Spectral && u.Nd() == 3);
    FlowField rolls(u.Nx(), u.Ny(), u.Nz(), u.Nd(), u.Lx(), u.Lz(), u.a(), u.b(), u.cfmpi());
    rolls.makeSpectral();
    rolls.setToZero();
    const lint mxlocmin = u.mxlocmin();
    const lint mxlocmax = u.mxlocmin() + u.Mxloc();
    const lint mzlocmin = u.mzlocmin();
    const lint mzlocmax = u.mzlocmin() + u.Mzloc();
    for (int ny = 0; ny < u.Ny(); ++ny)
        for (int kz = u.kzmin(); kz < u.kzmax(); ++kz) {
            const lint mx = u.mx(0);
            const lint mz = u.mz(kz);
            if (mx >= mxlocmin && mx < mxlocmax && mz >= mzlocmin && mz < mzlocmax) {
                rolls.cmplx(mx, ny, mz, 1) = u.cmplx(mx, ny, mz, 1);
                rolls.cmplx(mx, ny, mz, 2) = u.cmplx(mx, ny, mz, 2);
            }
        }
    return rolls;
}

// spatially averaged (in x and z) pressure gradient
Real getdPdx(const FlowField& u, Real nu) {
    Real dPdx;
    Real Ly = u.b() - u.a();
    dPdx = nu * (u.dudy_b() - u.dudy_a()) / Ly;
    return dPdx;
}

Real getdPdz(const FlowField& u, Real nu) {
    Real dPdz;
    Real Ly = u.b() - u.a();
    dPdz = nu * (u.dwdy_b() - u.dwdy_a()) / Ly;
    return dPdz;
}

Real getUbulk(const FlowField& u) {
    Real ubulk;

#ifdef HAVE_MPI
    if (u.taskid() == u.task_coeff(0, 0))
        ubulk = Re(u.profile(0, 0, 0)).mean();
    MPI_Bcast(&ubulk, 1, MPI_DOUBLE, u.task_coeff(0, 0), *u.comm_world());
#else
    ubulk = Re(u.profile(0, 0, 0)).mean();
#endif

    if (abs(ubulk) < 1e-15)
        ubulk = 0.0;
    return ubulk;
}

Real getWbulk(const FlowField& u) {
    Real wbulk;

#ifdef HAVE_MPI
    if (u.taskid() == u.task_coeff(0, 0))
        wbulk = Re(u.profile(0, 0, 2)).mean();
    MPI_Bcast(&wbulk, 1, MPI_DOUBLE, u.task_coeff(0, 0), *u.comm_world());
#else
    wbulk = Re(u.profile(0, 0, 2)).mean();
#endif

    if (abs(wbulk) < 1e-15)
        wbulk = 0.0;
    return wbulk;
}

Real L2Norm_uvw(const FlowField& u, const bool ux, const bool uy, const bool uz) {
    assert(u.ystate() == Spectral);
    assert(u.xzstate() == Spectral);
    Real sum = 0.0;

    const lint mxlocmin = u.mxlocmin();
    const lint mxlocmax = u.mxlocmin() + u.Mxloc();
    const lint mzlocmin = u.mzlocmin();
    const lint mzlocmax = u.mzlocmin() + u.Mzloc();

    int kxmin = u.padded() ? -u.kxmaxDealiased() : u.kxmin();
    int kxmax = u.padded() ? u.kxmaxDealiased() : u.kxmax();
    int kzmin = 0;
    int kzmax = u.padded() ? u.kzmaxDealiased() : u.kzmax();
    ComplexChebyCoeff prof(u.Ny(), u.a(), u.b(), Spectral);
    for (int i = 0; i < 3; ++i) {
        if ((i == 0 && !ux) || (i == 1 && !uy) || (i == 2 && !uz)) {
            continue;
        }

        for (int kx = kxmin; kx <= kxmax; ++kx) {
            lint mx = u.mx(kx);
            if (mx >= mxlocmin && mx < mxlocmax) {
                int cz = 1;  // cz = 2 for kz>0 to take account of kz<0 ghost modes
                for (int kz = kzmin; kz <= kzmax; ++kz) {
                    lint mz = u.mz(kz);
                    if (mz >= mzlocmin && mz < mzlocmax) {
                        for (int ny = 0; ny < u.Ny(); ++ny) {
                            prof.set(ny, u.cmplx(mx, ny, mz, i));
                        }
                        sum += cz * L2Norm2(prof, false);
                    }
                    cz = 2;
                }
            }
        }
    }

    // Sum up results from all processes
#ifdef HAVE_MPI
    Real tmp = sum;
    MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, *u.comm_world());
#endif
    return sqrt(sum);
}

Real Ecf(const FlowField& u) { return pow(L2Norm_uvw(u, false, true, true), 2); }

string fieldstatsheader() {
    stringstream header;
    header << setw(14) << "L2" << setw(14) << "u2" << setw(14) << "v2" << setw(14) << "w2" << setw(14) << "e3d"
           << setw(14) << "ecf" << setw(14) << "ubulk" << setw(14) << "wbulk" << setw(14) << "wallshear" << setw(14)
           << "wallshear_a" << setw(14) << "wallshear_b" << setw(14) << "dissipation";
    return header.str();
}

string fieldstatsheader_t(const string tname) {
    stringstream header;
    header << setw(8) << "#(" << tname << ")" << fieldstatsheader();
    return header.str();
}

string fieldstats(const FlowField& u) {
    stringstream s;
    double l2n = L2Norm(u);
    if (std::isnan(l2n)) {
        cferror("L2Norm(u) is nan");
    }
    s << setw(14) << L2Norm(u) << setw(14) << L2Norm_uvw(u, true, false, false) << setw(14)
      << L2Norm_uvw(u, false, true, false) << setw(14) << L2Norm_uvw(u, false, false, true) << setw(14) << L2Norm3d(u)
      << setw(14) << Ecf(u) << setw(14) << getUbulk(u) << setw(14) << getWbulk(u) << setw(14) << wallshear(u)
      << setw(14) << wallshearLower(u) << setw(14) << -1 * wallshearUpper(u) << setw(14) << dissipation(u);
    return s.str();
}

// Return some statistics about energy and velocity
string fieldstats_t(const FlowField& u, Real t) {
    stringstream s;
    s << setw(8);
    s << t;
    s << fieldstats(u);
    return s.str();
}

Real min_x_L2Dist(const FlowField& u0, const FlowField& u1, Real tol) {
    Real ax = optPhaseShiftx(u0, u1, -0.5, 0.5, tol);
    FieldSymmetry s(ax, 0);
    FlowField v = s * u1;
    return L2Dist(u0, v);
}

}  // namespace chflow
