/**
 * Unit test for the channelflow class TimeStep.
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "channelflow/dnsflags.h"
#include "gtest/gtest.h"

using namespace std;

namespace chflow {
namespace test {
/// A TimeStep object with some reasonable settings
class TimeStepTest : public ::testing::Test {
   protected:
    TimeStepTest() : ts(0.1, 1e-2, 1.0, 10.0, 0.4, 0.6, true) {}

    TimeStep ts;
    const double eps = 1e-16;
};

// Test if the values from the constructor are set correctly
TEST_F(TimeStepTest, Init) {
    EXPECT_NEAR(0.1, ts.dt(), eps);
    EXPECT_NEAR(1e-2, ts.dtmin(), eps);
    EXPECT_NEAR(1, ts.dtmax(), eps);
    EXPECT_NEAR(10.0, ts.dT(), eps);
    EXPECT_NEAR(0.4, ts.CFLmin(), eps);
    EXPECT_NEAR(0.6, ts.CFLmax(), eps);
    EXPECT_NEAR(0.1, Real(ts), eps);
    EXPECT_TRUE(ts.variable());
}

// Test if adjusting the total time works
TEST_F(TimeStepTest, adjustForT) {
    ts.adjust_for_T(90.0, false);

    EXPECT_NEAR(90.0, ts.T(), eps);
    EXPECT_NEAR(90.0, ts.N() * ts.dT(), eps);
    EXPECT_NEAR(90.0 / ts.N(), ts.n() * ts.dt(), eps);
    EXPECT_GE(ts.CFL(), 0.4);
    EXPECT_LE(ts.CFL(), 0.6);

    ts.adjust_for_T(68.0, false);

    EXPECT_NEAR(68.0, ts.T(), eps);
    EXPECT_NEAR(68.0, ts.N() * ts.dT(), eps);
    EXPECT_NEAR(68.0 / ts.N(), ts.n() * ts.dt(), eps);
    EXPECT_GE(ts.CFL(), 0.4);
    EXPECT_LE(ts.CFL(), 0.6);
}

// Test if adjusting the timestep works
TEST_F(TimeStepTest, adjustCFL) {
    ts.adjust_for_T(90.0, false);
    bool ret = ts.adjust(0.35, false);
    EXPECT_TRUE(ret);

    EXPECT_NEAR(90.0, ts.T(), eps);
    EXPECT_NEAR(90.0, ts.N() * ts.dT(), eps);
    EXPECT_NEAR(90.0 / ts.N(), ts.n() * ts.dt(), eps);
    EXPECT_GE(ts.CFL(), 0.4);  // not guaranteed
    EXPECT_LE(ts.CFL(), 0.6);  // not guaranteed
    EXPECT_GE(ts.dt(), 1e-2);
    EXPECT_LE(ts.dt(), 1.0);
}

// Run a fake simulation with varying CFL number
// Structure copied from cfdsi::f
TEST_F(TimeStepTest, totalTime) {
    ts.adjust_for_T(360.0, false);

    double local_dt = ts.dt();
    double totalTime = 0;
    int totalTimeSteps = 0;

    for (int s = 1; s <= ts.N(); ++s) {
        // Fake CFL between 0.01 and 0.99 at dt=1e-2
        Real CFL = (0.5 + 0.49 * sin(totalTime)) * 1e2 * ts.dt();

        for (int i = 0; i < ts.n(); ++i) {
            totalTime += local_dt;
        }

        if (ts.variable() && ts.adjust(CFL, false)) {
            local_dt = ts.dt();
        }
        totalTimeSteps = s;
    }

    EXPECT_EQ(totalTimeSteps, 36);
    EXPECT_NEAR(360.0, totalTime, 1e-9);  // Attention: Machine precision is not reached!
}
}  // namespace test
}  // namespace chflow
