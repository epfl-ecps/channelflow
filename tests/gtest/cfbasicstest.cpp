/**
 * Unit test for the functions in cfbasics.h
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include <cfbasics/cfbasics.h>
#include <gtest/gtest.h>

using namespace Eigen;

namespace chflow {
namespace test {
class SortByAbsTest : public ::testing::Test {
   protected:
    SortByAbsTest() : lambda_d(4), expected_d(4), V_d(4, 4) {
        // Vector of eigenvalues
        lambda_d << 1.0, -1.1, 0.9, -0.25;
        // Expected result
        expected_d << -1.1, 1.0, 0.9, -0.25;

        // Matrix of eigenvectors
        V_d << 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0;
    }

    VectorXd lambda_d;
    VectorXd expected_d;

    MatrixXd V_d;

    const double eps = 1e-16;
};

template <class Vector>
::testing::AssertionResult ExpectNearVector(const char* expr1, const char* expr2, const char* abs_error_expr,
                                            const Vector& v, const Vector& expected, const double& eps) {
    auto is_near = [eps](VectorXd::Scalar x, VectorXd::Scalar y) {
        const auto diff = fabs(x - y);
        return diff <= eps;
    };

    // Check that the sizes of the two vectors are equal
    if (v.size() != expected.size()) {
        return ::testing::AssertionFailure()
               << "The size of " << expr1 << " and " << expr2 << " differs, as " << expr1 << " has size " << v.size()
               << ", " << expr2 << " has size " << expected.size() << ".";
    }

    // Check the abs of each element
    auto failed_indexes = std::vector<unsigned long>();
    auto failed_abs_errors = std::vector<double>();
    for (auto idx = 0ul; idx < v.size(); ++idx) {
        // Record any mismatch that has been found
        if (!is_near(v(idx), expected(idx))) {
            const auto diff = fabs(v(idx) - expected(idx));
            failed_indexes.push_back(idx);
            failed_abs_errors.push_back(diff);
        }
    }

    if (!failed_indexes.empty()) {
        std::stringstream details;

        for (auto index_id = 0ul; index_id < failed_indexes.size(); ++index_id) {
            const auto idx = failed_indexes[index_id];
            details << "\nAt index " << idx << ":\n"
                    << "\t" << expr1 << "(" << idx << ") evaluates to " << v(idx) << ",\n"
                    << "\t" << expr2 << "(" << idx << ") evaluates to " << expected(idx) << ",\n"
                    << "\t" << abs_error_expr << " evaluates to " << failed_abs_errors[index_id] << ".\n";
        }

        return ::testing::AssertionFailure()
               << "Found elements in " << expr1 << " and " << expr2 << " that have abs differences which exceed "
               << abs_error_expr << " (" << eps << ").\n"
               << details.str();
    }
    return ::testing::AssertionSuccess();
}

#define EXPECT_NEAR_VECTOR(val1, val2, abs_error) EXPECT_PRED_FORMAT3(ExpectNearVector, val1, val2, abs_error)

TEST_F(SortByAbsTest, HandleDoubleArgs) {
    chflow::sort_by_abs(lambda_d, V_d);
    EXPECT_NEAR_VECTOR(lambda_d, expected_d, eps);

    VectorXd c0 = V_d.col(0);
    VectorXd expected_c0 = VectorXd::Constant(4, 2.0);
    EXPECT_NEAR_VECTOR(c0, expected_c0, eps);

    VectorXd c1 = V_d.col(1);
    VectorXd expected_c1 = VectorXd::Constant(4, 1.0);
    EXPECT_NEAR_VECTOR(c1, expected_c1, eps);

    VectorXd c2 = V_d.col(2);
    VectorXd expected_c2 = VectorXd::Constant(4, 3.0);
    EXPECT_NEAR_VECTOR(c2, expected_c2, eps);

    VectorXd c3 = V_d.col(3);
    VectorXd expected_c3 = VectorXd::Constant(4, 4.0);
    EXPECT_NEAR_VECTOR(c3, expected_c3, eps);
}
}  // namespace test
}  // namespace chflow
