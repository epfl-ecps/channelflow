/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */
#include "nsolver/config.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <iostream>
#include <stdexcept>
#include "nsolver/eigenvals.h"

using namespace std;

using namespace Eigen;

// Arnoldi iteration estimates the eigenvalues of a matrix A by iteratively
// constructing a QR decomposition of a matrix whose columns are
// Ab, A^2 b, etc., where b is an arbitrary starting vector.

// For the present problem, A is a discretized form of a nonlinear PDE,
// linearized about the equilibrium solutions, and b is the discretized
// form of an arbitrary initial condition.

// In continuous terms, the matrix A corresponds to a differential operator
// L and b to a perturbation du from an invariant solution u* of the PDE.
//
// Let f^T be the time-T map of the PDE, such that
//   f^T : u(t) -> u(t+T),
//
// Let u*,T,sigma be an equilibrium solution of f (e.g. a relative periodic orbit), such that
//   u* = sigma f^T(u*).
//
// Define (approximately) the linear operator L as
//  L du = sigma f^T(u* + du) - sigma f^T(u*) =  sigma f^T(u* + du) - u*
//
// This program computes L du, L^2 du , L^3 du, etc. with a DNS algorithm
// and translates between fields of the PDE du^n = L^n du and the corresponding vectors
// v^n = A^n b by using a dynamical system interface (DSI).

namespace chflow {

EigenvalsFlags::EigenvalsFlags() {}

EigenvalsFlags::EigenvalsFlags(ArgList& args) {
    args.section("Arnoldi options");
    isnormal =
        args.getflag("-isnorm", "--isnormal",
                     "define whether the system is normal, i.e, self-adjoint (use Lanczos) or not (use Arnoldi)");
    Narnoldi = args.getint("-N", "--Narnoldi", 100, "number of Arnoldi iterations");
    Nstable = args.getint("-Ns", "--Nstable", 5,
                          "save this many stable eigenfuncs in addition to all unstable and marginal ones");
    fixedNs = args.getflag("-fNs", "--fixedNsave", "save fixed number of eigfuncs: the Ns most unstable ones");
    EPS_kry = args.getreal("-ek", "--epsKrylov", 1e-10, "min. condition # of Krylov vectors");
    centdiff = args.getflag("-cd", "--centerdiff", "centered differencing to estimate Jacobian");
    orthochk = args.getflag("-oc", "--orthocheck", "save Q' * Q into QtQ.asc at every iteration");
    outdir = args.getpath("-o", "--outdir", "./", "output directory");
    EPS_stab = args.getreal("-est", "--epsStability", 1e-6, "threshold for marginal eigenvalues");
}

void EigenvalsFlags::save(const string& outdir) const {
    if (mpirank() == 0) {
        string filename = appendSuffix(outdir, "eigenvalsflags.txt");
        ofstream os(filename.c_str());
        if (!os.good())
            cferror("EigenvalsFlags::save(outdir) :  can't open file " + filename);
        os.precision(16);
        os.setf(ios::left);
        os << setw(REAL_IOWIDTH) << isnormal << "  %isnormal\n"
           << setw(REAL_IOWIDTH) << Narnoldi << "  %Narnoldi\n"
           << setw(REAL_IOWIDTH) << Nstable << "  %Nstable\n"
           << setw(REAL_IOWIDTH) << fixedNs << "  %fixedNs\n"
           << setw(REAL_IOWIDTH) << EPS_kry << "  %EPS_kry\n"
           << setw(REAL_IOWIDTH) << centdiff << "  %centdiff\n"
           << setw(REAL_IOWIDTH) << orthochk << "  %orthochk\n"
           << setw(REAL_IOWIDTH) << EPS_stab << "  %EPS_stab\n";
        os.unsetf(ios::left);
    }
}

void EigenvalsFlags::load(int taskid, const string indir) {
    ifstream is;
    if (taskid == 0) {
        is.open(indir + "eigenvalsflags.txt");
        if (!is.good())
            cferror(" EigenvalsFlags::load(taskid, indir):  can't open file " + indir + "EigenvalsFlags.txt");
    }
    isnormal = getIntfromLine(taskid, is);
    Narnoldi = getIntfromLine(taskid, is);
    Nstable = getIntfromLine(taskid, is);
    fixedNs = getIntfromLine(taskid, is);
    EPS_kry = getRealfromLine(taskid, is);
    centdiff = getIntfromLine(taskid, is);
    orthochk = getIntfromLine(taskid, is);
    EPS_stab = getRealfromLine(taskid, is);
}

Eigenvals::Eigenvals(EigenvalsFlags eigenflags_) : eigenflags(eigenflags_) {}

Eigenvals::Eigenvals(ArgList& args) : Eigenvals(EigenvalsFlags(args)) {}

void Eigenvals::checkConjugacy(const VectorXcd& u, const VectorXcd& v) {
    //   VectorXd f = u.real();
    //   VectorXd g = v.real();
    //   Real rerr = L2Dist(f,g);
    //   f = u.imag();
    //   g = v.imag();
    //   g = g * -1;
    //   Real ierr = L2Dist(f,g);
    //   if (rerr + ierr > 1e-14*(L2Norm(u)+L2Norm(v))) {
    //      cout << "error : nonconjugate u,v, saving these vectors" << endl;
    //      save(u, "uerr");
    //      save(v, "verr");
    //   }
}

void Eigenvals::solve(DSI& dsi, const VectorXd& x, VectorXd& dx, Real T, Real eps) {
    const int prec = 16;

    int taskid = 0;
#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
#endif

    mkdir(eigenflags.outdir);

    if (taskid == 0)
        cout << "Arnoldi flags = " << eigenflags << endl;

    eigenflags.save(eigenflags.outdir);

    dx /= L2Norm(dx);

    unique_ptr<Arnoldi> arnoldi;
    // If the system is normal (self-adjoint), use Lanczos algorithm instead of Arnoldi one.
    if (eigenflags.isnormal) {
        arnoldi = unique_ptr<Arnoldi>(new Lanczos(dx, eigenflags.Narnoldi, eigenflags.EPS_kry));
    } else {
        arnoldi = unique_ptr<Arnoldi>(new Arnoldi(dx, eigenflags.Narnoldi, eigenflags.EPS_kry));
    }

    // Initialize output streams
    mkdir(eigenflags.outdir);
    VectorXd Gx;
    for (int n = 0; n < eigenflags.Narnoldi; ++n) {
        if (taskid == 0) {
            cout << "======================================================" << endl;
            switch (n % 10) {
                case 1:
                    cout << n << "st iteration:" << endl;
                    break;
                case 2:
                    cout << n << "nd iteration:" << endl;
                    break;
                default:
                    cout << n << "th iteration:" << endl;
            }
        }
        // Compute v = Aq in Arnoldi iteration terms, where q is a test vector.
        // In PDE terms, this is
        //      L du == f^T(u* + du) - f^T(u*)
        // where |du| << |u*|, so that RHS expression is approximately linear
        // and where f^T is the forward time (T) map of the PDE.

        const VectorXd& q = arnoldi->testVector();

        int fcount = 0;
        cout << "L2Norm(q) = " << setprecision(prec) << dsi.DSIL2Norm(q) << endl;
        dx = q;

        // Gx is set to -eps*dx so that
        // Jacobian returns (f^T(x + eps*dx) - f^T(x))/eps
        Gx = -eps * dx;

        VectorXd Aq = dsi.Jacobian(x, dx, Gx, eps, eigenflags.centdiff, fcount);

        // If centdiff is true
        // Jacobian returns (f^T(x + eps/2*dx) - f^T(x-eps/2*dx))/eps - dx
        // So adding dx to Aq will return the desired result.
        if (eigenflags.centdiff)
            Aq += dx;

        cout << "\nL2Norm(Aq) = " << setprecision(prec) << dsi.DSIL2Norm(Aq) << endl;

        // Send the product Aq back to Arnoldi
        if (taskid == 0)
            cout << "Computing new orthogonal Krylov subspace basis vector..." << endl;
        arnoldi->iterate(Aq);

        if (taskid == 0)
            cout << "results..." << endl;
        // Print outputs: Lambda and the eigenvalues of map f^T
        const VectorXcd& Lambda = arnoldi->ew();
        const VectorXd& Residu = arnoldi->rd();
        VectorXcd lambda(Lambda.rows());
        int Nnonstable = 0;
        for (int j = 0; j <= n; ++j) {
            lambda(j) = (1.0 / T) * log(Lambda(j));
            if (Re(lambda(j)) >= -eigenflags.EPS_stab)
                ++Nnonstable;
        }
        save(Lambda, eigenflags.outdir + "Lambda");
        save(lambda, eigenflags.outdir + "lambda");
        save(Residu, eigenflags.outdir + "Residu");

        // Save either Nstable number of stable modes or all unstable, marginal modes etc
        int Nsave = lesser(Lambda.rows(), (eigenflags.fixedNs) ? eigenflags.Nstable : Nnonstable + eigenflags.Nstable);

        if (taskid == 0) {
            cout << setw(prec + 7) << "abs(Lambda)" << setw(prec + 7) << "arg(Lambda)" << endl;
            for (int j = 0; j < Nsave; ++j)
                cout << setw(prec + 7) << abs(Lambda(j)) << setw(prec + 7) << arg(Lambda(j)) << endl;

            cout << "\n"
                 << setw(prec + 7) << "Re(1/T log(Lambda))" << setw(prec + 7) << "Im(1/T log(Lambda))" << setw(prec + 9)
                 << "Residual" << endl;
            for (int j = 0; j < Nsave; ++j)
                cout << setw(prec + 7) << Re(lambda(j)) << setw(prec + 7) << Im(lambda(j)) << setw(prec + 9)
                     << Residu(j) << endl;
        }
    }
    // Done with Arnoldi iteration.

    const MatrixXcd& Vn = arnoldi->ev();
    const VectorXcd& Lambda = arnoldi->ew();
    VectorXcd lambda(Lambda.rows());

    int Nnonstable = 0;
    for (int j = 0; j < lambda.rows(); ++j) {
        lambda(j) = (1.0 / T) * log(Lambda(j));

        // Count the nonstable (marginal or unstable) eigvals
        if (Re(lambda(j)) >= -eigenflags.EPS_stab)
            ++Nnonstable;
    }

    int Nsave = lesser(Lambda.rows(), (eigenflags.fixedNs) ? eigenflags.Nstable : Nnonstable + eigenflags.Nstable);

    // Reconstruct and save leading eigenfunctions
    for (int j = 0; j < Nsave; ++j) {
        string sj1 = i2s(j + 1);
        string sj2 = i2s(j + 2);
        Real imlambdaj = Im(Lambda(j));
        int lambdarows = Lambda.rows();
#ifdef HAVE_MPI
        MPI_Bcast(&imlambdaj, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&lambdarows, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
        // Real-valued eigenvalue
        if (imlambdaj == 0.0) {
            VectorXd ev = Vn.col(j).real();
            dsi.saveEigenvec(ev, sj1, eigenflags.outdir);
        }
        // Complex eigenvalue pair
        else if (j + 1 < lambdarows) {
            Complex LambdaA = Lambda(j);
            Complex LambdaB = Lambda(j + 1);
            if (LambdaA != conj(LambdaB)) {
                cerr << "Warning! Non-conjugate complex eigenvalues!" << endl;
                cerr << "abs, arg Lambda(" << sj1 << ") == " << setw(prec + 7) << abs(LambdaA) << ' ' << setw(prec + 7)
                     << arg(LambdaA) << endl;
                cerr << "abs, arg Lambda(" << sj2 << ") == " << setw(prec + 7) << abs(LambdaB) << setw(prec + 7)
                     << arg(LambdaB) << endl;
            }
            checkConjugacy(Vn.col(j), Vn.col(j + 1));  // This function is empty.

            VectorXd evA = Vn.col(j).real();
            VectorXd evB = Vn.col(j).imag();
            dsi.saveEigenvec(evA, evB, sj1, sj2, eigenflags.outdir);

            ++j;  // j is increased inside the loop since (j+1) is already computed and saved
        }
    }
    if (eigenflags.orthochk)
        arnoldi->orthocheck();
}

ostream& operator<<(ostream& os, const EigenvalsFlags& flags) {
    string s(", ");
    const int p = os.precision();
    os.precision(16);
    os << "isnormal= " << flags.isnormal << s << "Narnoldi= " << flags.Narnoldi << s << "Nstable= " << flags.Nstable
       << s << "fixedNs= " << flags.fixedNs << s << "EPS_kry= " << flags.EPS_kry << s << "centdiff= " << flags.centdiff
       << s << "orthochk= " << flags.orthochk << s << "outdir= " << flags.outdir << s << "EPS_stab= " << flags.EPS_stab
       << s;
    os.precision(p);
    return os;
}

}  // namespace chflow
