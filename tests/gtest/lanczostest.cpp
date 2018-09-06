/**
 * Unit test for the nsolver class Lanczos.
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "cfbasics/cfbasics.h"
#include "gtest/gtest.h"
#include "nsolver/nsolver.h"

using namespace std;
using namespace Eigen;
using namespace ::testing;

namespace chflow {
namespace test {
// L Lanczos object with some reasonable settings
class LanczosTest : public ::testing::Test {
   protected:
    LanczosTest() {
        b = VectorXd(size);
        for (int i = 0; i < size; ++i) {
            b(i) = log(i + 1) * sin(i);
        }
        b /= L2Norm(b);
        L = Lanczos(b, 100, 1e-14);
        MatrixXd tmp(size, size);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                tmp(i, j) = ((i + 1) % (j + 2) - (j + 2) % (i + 1)) / (log(i + j + 2));
            }
        }
        M = MatrixXd(size, size);
        M = tmp + tmp.transpose();
    }

    Lanczos L;
    int size = 100;
    int iter = 99;
    VectorXd b;
    MatrixXd M;
    const double eps = 1e-11;
};

// Test if the values from the constructor are set correctly
TEST_F(LanczosTest, Init) {
    EXPECT_EQ(0, L.n());
    EXPECT_EQ(100, L.Niter());
}

// Test L testVector()
TEST_F(LanczosTest, testVector) {
    VectorXd tmp = L.testVector();
    Real diff = L2Dist(tmp, b);
    EXPECT_NEAR(diff, 0.0, eps);
}

// Test L iterate()
TEST_F(LanczosTest, iterate) {
    // Compute eigenvalues and eigenvectors with EigenSolver
    SelfAdjointEigenSolver<MatrixXd> eignHn(M);
    VectorXd ew = eignHn.eigenvalues();
    MatrixXd W = eignHn.eigenvectors();
    sort_by_abs(ew, W);
    // Compute eigenvalues and eigenvectors with Lanczos iteration
    for (int n = 0; n < iter; ++n) {
        const VectorXd& q = L.testVector();
        VectorXd Lq = M * q;
        L.iterate(Lq);
    }
    const VectorXcd& Lambda = L.ew();
    const MatrixXcd& Vn = L.ev();
    VectorXcd lambda(Lambda);
    // Check the first 10 eigenvalues.
    Map<VectorXd, 0, InnerStride<1>> tmp1(ew.data(), 10);
    Map<VectorXcd, 0, InnerStride<1>> tmp2(lambda.data(), 10);
    VectorXd t1 = tmp1.cwiseAbs();
    VectorXd t2 = tmp2.cwiseAbs();
    Real diff = L2Dist(t1, t2);
    EXPECT_NEAR(diff, 0.0, eps);
    // Check the first eigenvector.
    t1 = W.col(0).cwiseAbs();
    t2 = Vn.col(0).cwiseAbs();
    diff = L2Dist(t1, t2);
    EXPECT_NEAR(diff, 0.0, eps);
}
}  // namespace test
}  // namespace chflow
