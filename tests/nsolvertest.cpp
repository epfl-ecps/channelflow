/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include <Eigen/IterativeLinearSolvers>
#include "nsolver/nsolver.h"

using namespace std;
using namespace Eigen;
using namespace chflow;

VectorXd Ax(const VectorXd& x, const MatrixXd& A) { return A * x; }

/* Initialize a matrix A and vector b and solve Ax=b for x with methods
 * from Eigen (reference) and nsolver, using GMRES and BICGStab. Assert
 * that the solution is accurate.
 */

int main(int argc, char** argv) {
    Real err = 0;
    const Real maxErr = 1e-7;
    const Real tol = 1e-8;
    int nIter = 0;

    int N = 100;

    MatrixXd A = MatrixXd::Zero(N, N);
    VectorXd b(N);

    for (int i = 0; i < N; ++i) {
        b(i) = N / 2 - i;

        A(i, i) = i;
        int ip1 = (i - 1) % N;
        if (ip1 < 0)
            ip1 += N;
        A(i, ip1) = 1;
        A(i, (i + 1) % N) = 1;
    }

    VectorXd x_Eigen(VectorXd::Zero(N));
    VectorXd x_BiCGStab(VectorXd::Zero(N));
    VectorXd x_GMRES(VectorXd::Zero(N));

    const int Lmax = 10;

    x_Eigen = A.fullPivLu().solve(b);
    Real err_Eigen = (A * x_Eigen - b).norm() / b.norm();

    cout << "Eigen error: " << err_Eigen;
    if (err_Eigen < tol)
        cout << " (system is solvable)" << endl;
    else
        cout << " (system is unsolvable, don't worry if test fails)" << endl;

    GMRES gmres(b, N);
    for (nIter = 0; nIter < N; ++nIter) {
        VectorXd q = gmres.testVector();
        VectorXd Aq = A * q;
        gmres.iterate(Aq);
        //     cout << n << ", " << gmres.residual() << endl;

        if (gmres.residual() < tol) {
            break;
        }
    }
    x_GMRES = gmres.solution();

    cout << "GMRES #iterations: " << nIter << endl;
    cout << "GMRES residual(): " << gmres.residual() << endl;
    cout << "GMRES error: " << (A * x_GMRES - b).norm() / b.norm() << endl;
    cout << endl;
    err += (A * x_GMRES - b).norm() / b.norm();

    BiCGStab bicgstab(b);
    for (nIter = 0; nIter < N; nIter++) {
        VectorXd p = bicgstab.step1();
        VectorXd Ap = A * p;
        VectorXd s = bicgstab.step2(Ap);
        VectorXd As = A * s;
        bicgstab.step3(As);
        //     cout << i << ", " << bicgstab.residual() << endl;
        if (bicgstab.residual() < tol) {
            break;
        }
    }
    x_BiCGStab = bicgstab.solution();

    cout << "BiCGStab #iterations: " << nIter << endl;
    cout << "BiCGStab residual(): " << bicgstab.residual() << endl;
    cout << "BiCGStab error: " << (A * x_BiCGStab - b).norm() / b.norm() << endl;
    cout << endl;

    err += (A * x_BiCGStab - b).norm() / b.norm();

    for (int l = 0; l < Lmax; ++l) {
        Rn2Rnfunc Ax2 = [A](const VectorXd& x) { return Ax(x, A); };

        BiCGStabL<VectorXd> bl(Ax2, b, l, N);
        nIter = 0;
        while (bl.residual() > tol && nIter <= N) {
            nIter++;
            bl.iterate();
        }
        x_BiCGStab = bl.solution();

        cout << "BiCGStab(" << l << ") #iterations: " << nIter << endl;
        cout << "BiCGStab(" << l << ") residual(): " << bl.residual() << endl;
        cout << "BiCGStab(" << l << ") error: " << (A * x_BiCGStab - b).norm() / b.norm() << endl;
        cout << endl;

        err += (A * x_BiCGStab - b).norm() / b.norm();
    }

    if (err < maxErr) {
        cerr << "\t   pass   " << endl;
        cout << "\t   pass   " << endl;
        return 0;
    } else {
        cerr << "\t** FAIL **" << endl;
        cout << "\t** FAIL **" << endl;
        cout << "   err == " << err << endl;
        cout << "maxerr == " << maxErr << endl;
        return 1;
    }
}
