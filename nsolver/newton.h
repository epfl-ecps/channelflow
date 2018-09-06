/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */

#ifndef NSOLVER_NEWTON_H
#define NSOLVER_NEWTON_H

#include "cfbasics/arglist.h"
#include "cfbasics/cfbasics.h"
#include "nsolver/dsi.h"
#include "nsolver/multiShootingDSI.h"

/** \file newton.h
 * This file provides an abstract base class (ABC) for Newton algorithms.
 * Additionally, it provides a class ArclengthConstraint, that stores information
 * about the last known solution for continuation algorithms.
 * For this purpose, functions evalWithAC and jacobianWithAC are provided, that
 * extend the user-specified eval and Jacobian by putting the arclength in the last
 * vector component.
 */

namespace chflow {

class ArclengthConstraint {
   public:
    /** \class ArclengthConstraint
     *
     * For notation: vector y = (x, mu), with x the (physical) vector and mu
     * the parameter. Eval function then reads f(x,mu).
     */

    /// Empty constructor - sets use() to false
    ArclengthConstraint();

    /** \brief Constructor, activates Arclength constraint (use() = true)
     * \param[in] yLast the last known solution (including mu at last position)
     * \param[in] muRef reference value of mu, mu is scaled by this when computing arclength
     */
    ArclengthConstraint(const Eigen::VectorXd& yLast, Real ds = 0, Real muRef = 1.);

    /** \brief compute pseudo arclength squared
     * \param[in] y current guess vector, including mu at last position
     * \return pseudo arclength: L2Dist2(x-xLast) + (mu-muLast)^2
     */
    Real arclength2(const Eigen::VectorXd& y);

    /** \brief compute pseudo arclength
     * \param[in] y current guess vector, including mu at last position
     * \return pseudo arclength: sqrt(L2Dist2(x-xLast) + (mu-muLast)^2)
     */
    Real arclength(const Eigen::VectorXd& y);

    /** \brief compute difference between pseudo arclength and desired value
     * \param[in] y current guess vector
     * \return arclength - ds^2
     */
    Real arclengthDiff(const Eigen::VectorXd& y);

    /// Getter function for target arclength distance
    Real ds();
    /// Setter for target arclength distance
    void setDs(Real newDs);

    void setYLast(const Eigen::VectorXd& yLast);

    Eigen::VectorXd yLast();

    /// Is an arclength constraint set?
    bool use();

    // Set use_ to false
    void notUse();

    /// Extract parameter from vector
    Real muFromVector(const Eigen::VectorXd& y);

    /// Combine vector x and parameter mu into one larger vector
    Eigen::VectorXd makeVector(const Eigen::VectorXd& x, Real mu);

    Eigen::VectorXd extractVector(const Eigen::VectorXd& y);

   private:
    bool use_ = false;
    Eigen::VectorXd yLast_;
    Real muRef_;
    Real ds_;
};

/** Interface class for Newton algorithms. The solve function returns a VectorXd that
 * is a root of dsi.eval().
 */
class Newton {
   public:
    Newton(std::ostream* logstream_ = &std::cout, std::string outdir_ = "./", Real epsSearch = 1e-12);

    virtual ~Newton() {
        if (!isACset)
            delete AC;
        delete msDSI_;
    }

    /** \brief Find a root of dsi.eval()
     * \param[in] dsi specifies (via eval()) the function for which a root is searched
     * \param[in] x initial guess for root
     * \param[out] residual the final search residual
     * \return the root (or the last try)
     */
    virtual Eigen::VectorXd solve(DSI& dsi, const Eigen::VectorXd& x, Real& residual) = 0;
    virtual void setLogstream(std::ostream* os) { logstream = os; }
    virtual void setOutdir(std::string od) { outdir = od; }
    virtual std::string getOutdir() { return outdir; }
    virtual void setEpsSearch(Real es) { epsSearch_ = es; }
    virtual Real epsSearch() { return epsSearch_; }
    virtual std::ostream* getLogstream() { return logstream; }
    bool getConvergence() { return success; }

    Eigen::VectorXd evalWithAC(const Eigen::VectorXd& y, int& fcount);
    Eigen::VectorXd jacobianWithAC(const Eigen::VectorXd& y, const Eigen::VectorXd& dy, const Eigen::VectorXd& Gy,
                                   const Real& epsDx, bool centdiff, int& fcount);

    void setArclengthConstraint(ArclengthConstraint* newAC);
    MultishootingDSI* getMultishootingDSI();

   protected:
    std::ostream* logstream;  ///< Replaces cout
    std::string outdir;       ///< save files and further information here
    Real epsSearch_;          ///< solve tries to find x such that L2(f(x)) < epsSearch
    bool success;
    bool isACset = false;  // to know if setArclengthConstraint is called and the pointer is pointing to an object whcih
                           // is not allocated in this class

    ArclengthConstraint* AC;
    MultishootingDSI* msDSI_;
};

}  // namespace chflow
#endif
