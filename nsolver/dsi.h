/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */

#ifndef NSOLVER_DSI_H
#define NSOLVER_DSI_H

#include "cfbasics/arglist.h"
#include "cfbasics/cfbasics.h"

namespace chflow {

/*==================================================================================*/
/*               Base class for vector-valued functions                             */
/*==================================================================================*/

class DSI {
    /** DSI (Dynamical Systems Interface) is an interface class for specifying dynamical systems.
     * The equations of the DS to be solved, saving results, computing norms and updating parameters
     * are specified here.
     */
   public:
    DSI();
    DSI(std::ostream* os);

    /** \brief The infamous virtual destructor */
    virtual ~DSI() = default;

    /** \brief The function to be solved in the Newton search. Searching for a fixed point of dx/dt = f(x),
     * this function should return x(T) - x(0).
     * \param[in] x the position at which f is evaluated
     * \return f(x)
     *
     * When implementing eval, the vector x may be longer then the expected input (e.g. the arclength may
     * be added to x). It is the implementations responsibility to ignore the components which are too much
     * and return a vector of the same size as the input.
     */

    virtual Eigen::VectorXd eval(const Eigen::VectorXd& x) = 0;

    virtual Eigen::VectorXd eval(const Eigen::VectorXd& x0, const Eigen::VectorXd& x1, bool symopt = false);

    /** Save vector x in "outdir/filebase" */
    virtual void save(const Eigen::VectorXd& x, const std::string filebase, const std::string outdir = "./",
                      const bool fieldsonly = false);

    /** Save eigenvectors x in "outdir */
    virtual void saveEigenvec(const Eigen::VectorXd& x, const std::string label,
                              const std::string outdir);  // Save real eigenvectors

    // Save complex conjugate eigenvectors pair
    virtual void saveEigenvec(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const std::string label1,
                              const std::string label2, const std::string outdir);

    virtual Eigen::VectorXd quadraticInterpolate_vector(const cfarray<Eigen::VectorXd>& xn, const cfarray<Real>& s,
                                                        Real snew);
    /** Norm(x) */
    virtual Real DSIL2Norm(const Eigen::VectorXd& x);

    virtual Real extractT(const Eigen::VectorXd& x);
    virtual Real extractXshift(const Eigen::VectorXd& x);
    virtual Real extractZshift(const Eigen::VectorXd& x);

    /** Give a string that contains various (physical) properties computed from x, E.g. different norms.
     * This string may be used by programs (such as continuation) to save information the state to
     * the disk. Standard delimiter should be tab ("\t")
     */
    virtual std::string stats(const Eigen::VectorXd& x);
    virtual std::pair<std::string, std::string> stats_minmax(const Eigen::VectorXd& x);

    /** Return an appropriate file header for the string returned by stats. */
    virtual std::string statsHeader();

    /** The continuation algorithm treats a system of dx/dt = f(x,mu) with a parameter mu. The
     * handling of parameter updates is done in this function. Implementations should know
     * what the parameter mu means and update the ODE accordingly.
     */
    virtual void updateMu(Real mu) { mu_ = mu; }
    virtual void saveParameters(std::string searchdir) {}
    virtual void saveResults(std::string searchdir) {}
    virtual Real mu() const { return mu_; }
    virtual std::string printMu() { return ""; }
    virtual Real observable(Eigen::VectorXd& x) { return 0; }
    virtual void phaseShift(Eigen::MatrixXd& y) {}
    virtual void phaseShift(Eigen::VectorXd& x) {}

    /** Number of entries at the end of the vector x that do not correspond to physical coefficients, e.g.
     * entries containing translation distances in directions of continuous symmetries.
     * The residual in the hookstep search is computed as the L2Norm of the components 0 to N-uunk() of
     * the vector x.
     */
    //   virtual int uunk() const {return 0;}

    /** \brief Finite difference approximation of the action of the Jacobian J on a test vector
     * \param[in] x the position at which the Jacobian is evaluated
     * \param[in] dx the direction in which the derivative is computed
     * \param[in] Gx eval(x)
     * \param[in] epsDx epsilon used for finite difference approximation of derivative
     * \param[in] centdiff use centered finite differences instead of forward fd
     * \param[out] fcount number of calls to eval()
     * \return J_x dot dx
     */
    virtual Eigen::VectorXd Jacobian(const Eigen::VectorXd& x, const Eigen::VectorXd& dx, const Eigen::VectorXd& Gx,
                                     const Real& epsDx, bool centdiff, int& fcount);

    /** \brief compute derivative of vector along x axis.
     * Used in enforcing orthogonality when searching for travelling waves drifting in x.
     * This is optional since not every DSI implementation is requried to allow for travelling waves.
     * \param[in] a compute da/dx
     * \return da/dx
     */
    virtual Eigen::VectorXd xdiff(const Eigen::VectorXd& a);

    /** \brief Optional: compute derivative of vector along z axis.
     * Used in enforcing orthogonality when searching for travelling waves drifting in z
     * This is optional since not every DSI implementation is requried to allow for travelling waves.
     * \param[in] a compute da/dz
     * \return da/dz
     */
    virtual Eigen::VectorXd zdiff(const Eigen::VectorXd& a);

    /** \brief Optional: compute derivative of vector along time evolution
     * Used in enforcing orthogonality when searching for periodic orbits.
     * This is optional since not every DSI implementation is requried to allow for periodic orbits.
     * \param[in] a compute da/dt
     * \return da/dt
     */
    virtual Eigen::VectorXd tdiff(const Eigen::VectorXd& a, Real epsDt);

    virtual Real tph_observable(Eigen::VectorXd& x);

    void setOs(std::ostream* newos) { os_ = newos; }

    /** \brief Optional: getter functions to be used in NSolver
     */
    virtual bool XrelSearch() const { return false; };

    virtual bool ZrelSearch() const { return false; };

    virtual bool Tsearch() const { return false; };

   private:
    DSI(DSI& D);  // DSI objects may not be copied!

   protected:
    Real mu_;

    std::ostream* os_;
};

}  // namespace chflow
#endif
