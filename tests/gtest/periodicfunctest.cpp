/**
 * Unit test for the channelflow class PeriodicFunc.
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "channelflow/periodicfunc.h"
#include "gtest/gtest.h"

using namespace std;

namespace chflow {
namespace test {

/// A physical periodic function with 10 points and p[i]==i
class PeriodicFuncPhysicalTest : public ::testing::Test {
   protected:
    PeriodicFuncPhysicalTest() : p(10, 2.0, Physical) {}

    virtual void SetUp() {
        for (uint i = 0; i < 10; ++i) {
            p[i] = i;
        }
    }

    PeriodicFunc p;
};

TEST_F(PeriodicFuncPhysicalTest, Init) {
    EXPECT_EQ(p.state(), Physical);
    EXPECT_EQ(p.N(), 10);
    EXPECT_NEAR(p.L(), 2.0, 1e-16);
}

TEST_F(PeriodicFuncPhysicalTest, SetGet) {
    // Test the two setter/getters [] and ()
    for (uint i = 0; i < 10; ++i) {
        p(i) = i + 1;
        EXPECT_NEAR(p[i], i + 1, 1e-16);
        p[i] = i + 2;
        EXPECT_NEAR(p(i), i + 2, 1e-16);
    }

    // Assertions are only run in debug mode
#ifndef NDEBUG
    // Spectral data should not be accessible
    // if function is in Physical state
    ASSERT_DEATH(p.cmplx(1), "");

    // Out-of-range access should fail
    ASSERT_DEATH(p[15], "");
    ASSERT_DEATH(p(15), "");
#endif
}

TEST_F(PeriodicFuncPhysicalTest, SetZero) {
    for (uint i = 1; i < 10; ++i) {
        EXPECT_NE(p[i], 0);
    }
    p.setToZero();
    for (uint i = 0; i < 10; ++i) {
        EXPECT_EQ(p[i], 0);
    }
}

/// A physical periodic function with 128 points
class PeriodicFuncSineTest : public ::testing::Test {
   protected:
    PeriodicFuncSineTest() : p(128, 2.0, Physical) {}

    virtual void SetUp() {}

    void setSine(Real prefactor, int wavenumber) {
        p.makePhysical();
        double K = 2 * pi / p.L() * wavenumber;
        for (uint n = 0; n < p.N(); ++n) {
            p(n) = prefactor * sin(K * p.x(n));
        }
    }

    void setCosine(Real prefactor, int wavenumber) {
        p.makePhysical();
        double K = 2 * pi / p.L() * wavenumber;
        for (uint n = 0; n < p.N(); ++n) {
            p(n) = prefactor * cos(K * p.x(n));
        }
    }

    PeriodicFunc p;
};

// Test that transform works
TEST_F(PeriodicFuncSineTest, SpectralMode) {
    const Real eps = 1e-10;

    const vector<Real> As = {1.0, 2.0, 3.0};
    const vector<uint> ks = {0, 5, 5};
    const vector<bool> cosines = {true, false, true};
    // For k==0, we only test cosine, because sin(0)==0

    for (int i = 0; i < As.size(); ++i) {
        Real A = As[i];
        uint k = ks[i];

        Complex fk;
        if (cosines[i]) {
            fk = Complex(A, 0);
            setCosine(A, k);
        } else {
            fk = Complex(0, -A);
            setSine(A, k);
        }
        // Nonzero modes are split to positive and negative
        if (k != 0) {
            fk *= 0.5;
        }

        p.makeSpectral();

        for (uint j = 0; j <= p.kmax(); ++j) {
            if (j == k) {
                EXPECT_NEAR(fk.real(), p.cmplx(j).real(), eps);
                EXPECT_NEAR(fk.imag(), p.cmplx(j).imag(), eps);
            } else {
                EXPECT_NEAR(0.0, p.cmplx(j).real(), eps);
                EXPECT_NEAR(0.0, p.cmplx(j).imag(), eps);
            }
        }
    }
}

// Test that transform maintains the norm
TEST_F(PeriodicFuncSineTest, SpectralNorm) {
    const Real eps = 1e-10;

    const vector<Real> As = {1.0, 1.0, 2.0, 3.0, 4.0};
    const vector<uint> ks = {0, 5, 5, 5, 5};
    const vector<bool> cosines = {true, false, true, true, true};
    // For k==0, we only test cosine, because sin(0)==0

    for (int i = 0; i < As.size(); ++i) {
        Real A = As[i];
        uint k = ks[i];

        Complex fk;
        if (cosines[i]) {
            fk = Complex(A, 0);
            setCosine(A, k);
        } else {
            fk = Complex(0, -A);
            setSine(A, k);
        }

        p.makeSpectral();

        Real norm = L2Norm(p);
        if (k == 0) {
            EXPECT_NEAR(A, norm, eps);
        } else {
            EXPECT_NEAR(A / sqrt(2), norm, eps);
        }
    }
}

// Test that forward-backward-transform goes back to initial shape
TEST_F(PeriodicFuncSineTest, SpectralTransform) {
    const Real eps = 1e-10;

    setSine(3.0, 7);
    const PeriodicFunc p0(p);

    p.makeSpectral();
    p.makePhysical();

    for (uint i = 0; i < p.N(); ++i) {
        EXPECT_NEAR(p0[i], p[i], eps);
    }
}
}  // namespace test
}  // namespace chflow
