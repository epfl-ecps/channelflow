/**
 * Unit test for the class Multishooting.
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "gtest/gtest.h"
#include "nsolver/nsolver.h"

using namespace std;
using namespace Eigen;
using namespace ::testing;

namespace chflow {
namespace test {
class exampleDSI : public DSI {
   public:
    exampleDSI();

    VectorXd eval(const VectorXd& x) override;

    VectorXd eval(const VectorXd& x1, const VectorXd& x2, bool sig) override;

    VectorXd xdiff(const VectorXd& x) override;

    VectorXd zdiff(const VectorXd& x) override;

    VectorXd tdiff(const VectorXd& x, Real epsDt) override;

    Real extractT(const VectorXd& x) override;

    Real extractXshift(const VectorXd& x) override;

    Real extractZshift(const VectorXd& x) override;

    Real observable(VectorXd& x) override;

    void phaseShift(MatrixXd& y) override;

    Real tph_observable(VectorXd& x) override;

    Real DSIL2Norm(const VectorXd& x) override;

    void save(const VectorXd& x, const std::string filebase, const std::string outdir, const bool fieldsonly) override;

    VectorXd getsaveData();

    string stats(const VectorXd& x) override;

    pair<string, string> stats_minmax(const VectorXd& x) override;

    VectorXd saveData_;  // to test save; the first elements of a given vector if fieldsonly otherwise the last element
};

exampleDSI::exampleDSI() { saveData_.resize(0); }

VectorXd exampleDSI::eval(const VectorXd& x) { return x; }

VectorXd exampleDSI::eval(const VectorXd& x1, const VectorXd& x2, bool sig) {
    Real a = 2.0;
    VectorXd f = a * x1 - x2;
    return f;
}

VectorXd exampleDSI::xdiff(const VectorXd& x) { return x; }

VectorXd exampleDSI::zdiff(const VectorXd& x) { return x; }

VectorXd exampleDSI::tdiff(const VectorXd& x, Real epsDt) { return x; }

Real exampleDSI::extractT(const VectorXd& x) { return x(x.size() - 3); }

Real exampleDSI::extractXshift(const VectorXd& x) { return x(x.size() - 2); }

Real exampleDSI::extractZshift(const VectorXd& x) { return x(x.size() - 1); }

Real exampleDSI::observable(VectorXd& x) {
    Real obs = 0;
    for (int i = 0; i < x.size(); i++)
        obs += x(i);
    return obs / x.size();
}

void exampleDSI::phaseShift(MatrixXd& y) {
    Real tmp = y(y.rows() - 4, 0);
    for (int i = 0; i < y.rows() - 3; i++)
        for (int j = 0; j < y.cols(); j++)
            y(i, j) += tmp;
}

Real exampleDSI::tph_observable(VectorXd& x) {
    Real obs = x(x.size() - 4);
    return obs;
}

Real exampleDSI::DSIL2Norm(const VectorXd& x) {
    Real obs = x(x.size() - 5);
    return obs;
}

void exampleDSI::save(const VectorXd& x, const string filebase, const string outdir, const bool fieldsonly) {
    int size = saveData_.size();
    size++;
    saveData_.conservativeResize(size);
    if (!fieldsonly)
        saveData_(size - 1) = x(0);
    else
        saveData_(size - 1) = x(x.size() - 4);
}

VectorXd exampleDSI::getsaveData() { return saveData_; }

string exampleDSI::stats(const VectorXd& x) {
    string str = "x:";
    for (int i = 0; i < x.size(); i++)
        str += r2s(x(i));
    return str;
}

pair<string, string> exampleDSI::stats_minmax(const VectorXd& x) {
    pair<string, string> minmax;
    minmax = make_pair(stats(x), stats(x));
    return minmax;
}

#ifdef HAVE_MPI
class MPIEnvironment : public ::testing::Environment {
   public:
    virtual void SetUp() {
        char** argv;
        int argc = 0;
        int mpiError = MPI_Init(&argc, &argv);
        ASSERT_FALSE(mpiError);
    }
    virtual void TearDown() {
        int mpiError = MPI_Finalize();
        ASSERT_FALSE(mpiError);
    }
    virtual ~MPIEnvironment() {}
};
Environment* const foo_env = ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
#endif

/// A Multishooting object with some reasonable settings
class msDSITest : public ::testing::Test {
   protected:
    msDSITest() : msDSI(2, true, true, true, 10.0, 0.5, 0.5, false) {
        msDSI.setDSI(tmpDSI, 10, false);

        // x(i)=i; size_x = 17; ((10-3)*2+3)
        x.resize(17);
        for (int i = 0; i < 17; i++)
            x(i) = Real(i);

        // First column:  0 1 2 3  4  5  6  70  0  0
        // Second column: 7 8 9 10 11 12 13 70 7.5 8 -> T = 140 (TRef = 10) , ax = 7.5 (axRef = 0.5) , az = 8 (azRef =
        // 0.5)
        y.resize(10, 2);
        for (int i = 0; i < 7; i++) {
            y(i, 0) = i;
            y(i, 1) = i + 7;
        }
        y(7, 0) = y(7, 1) = 70;
        y(8, 0) = y(9, 0) = 0;
        y(8, 1) = 7.5;
        y(9, 1) = 8.0;
    }

    ~msDSITest() {}

    MultishootingDSI msDSI;
    exampleDSI tmpDSI;
    VectorXd x;
    MatrixXd y;
    const double eps = 1e-14;
};

// Test if the values from the constructor are set correctly
TEST_F(msDSITest, Init) {
    EXPECT_EQ(2, msDSI.nShot());
    EXPECT_FALSE(msDSI.tph());
}

// Test if dsi is correctly set
TEST_F(msDSITest, setDSI) { EXPECT_TRUE(msDSI.isDSIset()); }

// Test if the size of a vector is equal to MS vector size
TEST_F(msDSITest, isVecMS) {
    EXPECT_TRUE(msDSI.isDSIset());
    EXPECT_TRUE(msDSI.isVecMS(17, false));
    EXPECT_TRUE(msDSI.isVecMS(18, true));
    EXPECT_FALSE(msDSI.isVecMS(13, false));
    msDSI.setDSI(tmpDSI, 29, true);
    EXPECT_TRUE(msDSI.isVecMS(29, false));
    EXPECT_TRUE(msDSI.isVecMS(30, true));
    EXPECT_FALSE(msDSI.isVecMS(13, false));
}

// Test if MS can update mu in dsi
TEST_F(msDSITest, updatemu) {
    msDSI.updateMu(0.01);
    EXPECT_NEAR(tmpDSI.mu(), 0.01, eps);
    msDSI.updateMu(100.0);
    EXPECT_NEAR(tmpDSI.mu(), 100.0, eps);
}

// Test MS eval
TEST_F(msDSITest, MS_eval) {
    VectorXd Gx = msDSI.eval(x);
    EXPECT_NEAR(Gx(0), -7.0, eps);
    EXPECT_NEAR(Gx(1), -6.0, eps);
    EXPECT_NEAR(Gx(2), -5.0, eps);
    EXPECT_NEAR(Gx(3), -4.0, eps);
    EXPECT_NEAR(Gx(4), -3.0, eps);
    EXPECT_NEAR(Gx(5), -2.0, eps);
    EXPECT_NEAR(Gx(6), -1.0, eps);
    EXPECT_NEAR(Gx(7), 14.0, eps);
    EXPECT_NEAR(Gx(8), 15.0, eps);
    EXPECT_NEAR(Gx(9), 16.0, eps);
    EXPECT_NEAR(Gx(10), 17.0, eps);
    EXPECT_NEAR(Gx(11), 18.0, eps);
    EXPECT_NEAR(Gx(12), 19.0, eps);
    EXPECT_NEAR(Gx(13), 20.0, eps);
    EXPECT_NEAR(Gx(14), 14.0, eps);
    EXPECT_NEAR(Gx(15), 30.0, eps);
    EXPECT_NEAR(Gx(16), 32.0, eps);
}

// Test MS Jacobian
// our example eval is linear so the jacobain * dx should be equal to G(dx)
TEST_F(msDSITest, MS_Jacobian) {
    VectorXd Gx = msDSI.eval(x);
    VectorXd dx(17);
    for (int i = 0; i < 17; i++)
        dx(i) = (Real)rand() / (Real)RAND_MAX;

    int fcount = 1;
    VectorXd Jdx = msDSI.Jacobian(x, dx, Gx, 4.3, false, fcount);

    VectorXd Gdx = msDSI.eval(dx);

    for (int i = 0; i < 17; i++)
        EXPECT_NEAR(Gdx(i), Jdx(i), eps);
}

// Test MS tovector
TEST_F(msDSITest, MS_toVector) {
    MatrixXd y2v = msDSI.toVector(y);
    for (int i = 0; i < 10; i++)
        EXPECT_NEAR(y2v(i), x(i), eps);
}

// Test MS extractvector
TEST_F(msDSITest, MS_extractV) {
    MatrixXd x2m = msDSI.extractVectors(x);
    for (int i = 0; i < 10; i++) {
        EXPECT_NEAR(x2m(i, 0), y(i, 0), eps);
        EXPECT_NEAR(x2m(i, 1), y(i, 1), eps);
    }
}

// Test MS makeMSvector
TEST_F(msDSITest, MS_makeV) {
    VectorXd y0(10);
    for (int i = 0; i < 10; i++)
        y0(i) = Real(i);

    VectorXd x0 = msDSI.makeMSVector(y0);
    EXPECT_EQ(x0.size(), 17);

    for (int i = 0; i < 7; i++)
        EXPECT_NEAR(x0(i), y0(i), eps);
    exampleDSI dsi;
    VectorXd Gx0 = dsi.eval(y0, VectorXd::Zero(10), false);
    for (int i = 7; i < 14; i++)
        EXPECT_NEAR(x0(i), Gx0(i - 7), eps);
    EXPECT_NEAR(x0(14), 0.7, eps);
    EXPECT_NEAR(x0(15), 16, eps);
    EXPECT_NEAR(x0(16), 18, eps);
}

// Test MS xdiff
TEST_F(msDSITest, MS_xdiff) {
    VectorXd x(17);
    for (int i = 0; i < 17; i++)
        x(i) = (Real)rand() / (Real)RAND_MAX;

    VectorXd xdiff = msDSI.xdiff(x);

    for (int i = 0; i < 7; i++)
        EXPECT_NEAR(xdiff(i), x(i), eps);
    for (int i = 7; i < 17; i++)
        EXPECT_NEAR(xdiff(i), 0, eps);
}

// Test MS zdiff
TEST_F(msDSITest, MS_zdiff) {
    VectorXd x(17);
    for (int i = 0; i < 17; i++)
        x(i) = (Real)rand() / (Real)RAND_MAX;

    VectorXd zdiff = msDSI.zdiff(x);

    for (int i = 0; i < 7; i++)
        EXPECT_NEAR(zdiff(i), x(i), eps);
    for (int i = 7; i < 17; i++)
        EXPECT_NEAR(zdiff(i), 0, eps);
}

// Test MS tdiff
TEST_F(msDSITest, MS_tdiff) {
    VectorXd x(17);
    for (int i = 0; i < 17; i++)
        x(i) = (Real)rand() / (Real)RAND_MAX;

    VectorXd tdiff = msDSI.tdiff(x, 0.01);

    for (int i = 0; i < 7; i++)
        EXPECT_NEAR(tdiff(i), x(i), eps);
    for (int i = 7; i < 17; i++)
        EXPECT_NEAR(tdiff(i), 0, eps);
}

// Test MS extractT
TEST_F(msDSITest, MS_T) {
    Real T = msDSI.extractT(x);
    EXPECT_NEAR(T, 140.0, eps);
}

// Test MS extractXshift
TEST_F(msDSITest, MS_ax) {
    Real ax = msDSI.extractXshift(x);
    EXPECT_NEAR(ax, 7.5, eps);
}

// Test MS extractZshift
TEST_F(msDSITest, MS_az) {
    Real az = msDSI.extractZshift(x);
    EXPECT_NEAR(az, 8.0, eps);
}

// Test MS observable
TEST_F(msDSITest, MS_obs) {
    Real obs = msDSI.observable(x);
    EXPECT_NEAR(obs, 12.325, eps);
}

// Test MS phaseshift
TEST_F(msDSITest, MS_phaseshift) {
    VectorXd x0 = x;
    msDSI.phaseShift(x, false);
    for (int i = 0; i < 14; i++)
        EXPECT_NEAR(x(i), x0(i) + 6.0, eps);
    EXPECT_EQ(x.size(), 17);
    EXPECT_EQ(x0.size(), 17);
    EXPECT_NEAR(x(14), x0(14), eps);
    EXPECT_NEAR(x(15), x0(15), eps);
    EXPECT_NEAR(x(16), x0(16), eps);

    x = x0;
    x.conservativeResize(18);  // with AC
    x(17) = 329;
    x0 = x;
    msDSI.phaseShift(x, true);
    for (int i = 0; i < 14; i++)
        EXPECT_NEAR(x(i), x0(i) + 6.0, eps);
    EXPECT_EQ(x.size(), 18);
    EXPECT_EQ(x0.size(), 18);
    EXPECT_NEAR(x(14), x0(14), eps);
    EXPECT_NEAR(x(15), x0(15), eps);
    EXPECT_NEAR(x(16), x0(16), eps);
    EXPECT_NEAR(x(17), x0(17), eps);
}

// Test MS fixtphase
TEST_F(msDSITest, MS_fixtph) {
    Real diffobs = msDSI.fixtphase(x);
    EXPECT_NEAR(diffobs, 6.0 / (140 * 1e-5), eps);
}

// Test MS DSIL2Norm
TEST_F(msDSITest, MS_DSIL2) {
    Real l2 = msDSI.DSIL2Norm(x);
    EXPECT_NEAR(l2, 5.0, eps);
}

// Test MS save
TEST_F(msDSITest, MS_save) {
    msDSI.save(x, "", "");
    VectorXd saveData = tmpDSI.getsaveData();
    EXPECT_EQ(saveData.size(), msDSI.nShot());
    EXPECT_NEAR(saveData(0), 0, eps);
    EXPECT_NEAR(saveData(1), 13, eps);
}

// Test MS stats
TEST_F(msDSITest, MS_stats) {
    string str;
    str = msDSI.stats(x);
    EXPECT_EQ(str, "x:01234567000");
}

// Test MS stats_minmax
TEST_F(msDSITest, MS_statsminmax) {
    string str;
    str = msDSI.stats_minmax(x).first;
    EXPECT_EQ(str, "x:012345614000");
}
}  // namespace test
}  // namespace chflow
