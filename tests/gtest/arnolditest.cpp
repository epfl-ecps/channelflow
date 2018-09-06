/**
 * Unit test for the nsolver class Arnoldi.
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
// A Arnoldi object with some reasonable settings
class ArnoldiTest : public ::testing::Test {
   protected:
    ArnoldiTest() {
        b = VectorXd(size);
        for (int i = 0; i < size; ++i) {
            b(i) = log(i + 1) * sin(i);
        }
        b /= L2Norm(b);
        A = Arnoldi(b, 100, 1e-14);
        M = MatrixXd(size, size);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                M(i, j) = ((i + 1) % (j + 2) - (j + 2) % (i + 1)) / (log(i + j + 2));
            }
        }
        // M = MatrixXd::Random(size,size);
    }

    Arnoldi A;
    int size = 100;
    int iter = 99;
    VectorXd b;
    MatrixXd M;
    const double eps = 1e-11;
};

// Test if the values from the constructor are set correctly
TEST_F(ArnoldiTest, Init) {
    EXPECT_EQ(0, A.n());
    EXPECT_EQ(100, A.Niter());
}

// Test A testVector()
TEST_F(ArnoldiTest, testVector) {
    VectorXd tmp = A.testVector();
    Real diff = L2Dist(tmp, b);
    EXPECT_NEAR(diff, 0.0, eps);
}

// Test A iterate()
TEST_F(ArnoldiTest, iterate) {
    // Compute eigenvalues and eigenvectors with EigenSolver
    EigenSolver<MatrixXd> eignHn(M, /*compute_eigenvectors=*/true);
    VectorXcd ew = eignHn.eigenvalues();
    MatrixXcd W = eignHn.eigenvectors();
    sort_by_abs(ew, W);
    // Compute eigenvalues and eigenvectors with Arnoldi iteration
    for (int n = 0; n < iter; ++n) {
        const VectorXd& q = A.testVector();
        VectorXd Aq = M * q;
        A.iterate(Aq);
    }
    const VectorXcd& Lambda = A.ew();
    const MatrixXcd& Vn = A.ev();
    VectorXcd lambda(Lambda);
    // Check the first 10 eigenvalues.
    Map<VectorXcd, 0, InnerStride<1>> tmp1(ew.data(), 10);
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
