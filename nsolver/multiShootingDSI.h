/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */
#ifndef MultiShootingDSI_H
#define MultiShootingDSI_H

#include "cfbasics/cfbasics.h"
#include "nsolver/dsi.h"

namespace chflow {

class MultishootingDSI {
   public:
    MultishootingDSI();
    MultishootingDSI(int nShot, bool TSearch, bool xrelative, bool zrelative, Real Tfac, Real Xfac, Real Zfac,
                     bool fix_tphase);

    ~MultishootingDSI() {}

    void setDSI(DSI& dsi, int vec_size, bool isVecLong = false);  // to set dsi and vector sizes and the position of T,
                                                                  // ax and az in the short and long vector formats
    bool isDSIset();                                              // to check if dsi is previously set
    bool isVecMS(int MSVec_size, bool isAC);  // to find out the format of a vector which is passed to Newton
    void updateMu(Real mu);                   // to call updatemu of DSI
    Eigen::VectorXd eval(
        const Eigen::VectorXd& x);  // to integrate shots in time by separately calling DSI::eval on them
    Eigen::VectorXd Jacobian(const Eigen::VectorXd& x, const Eigen::VectorXd& dx, const Eigen::VectorXd& Gx,
                             const Real& epsDx, bool centdiff,
                             int& fcount);                     // to calculate jacobian
    Eigen::MatrixXd extractVectors(const Eigen::VectorXd& x);  // to transform a long vector to a matrix of short
                                                               // vectors; ax and az are stored in the
    // last vector but T is devided and stored in all of them
    Eigen::VectorXd makeMSVector(
        const Eigen::VectorXd& yvec);  // to form an initial long vector by integrating an initial field
    Eigen::VectorXd toVector(const Eigen::MatrixXd& y);  // to transform a matrix of short vectors to a long vector
    Eigen::VectorXd xdiff(
        const Eigen::VectorXd& x);  // to call DSI::xdiff on the first shot and put it in the long vector format
    Eigen::VectorXd zdiff(
        const Eigen::VectorXd& x);  // to call DSI::zdiff on the first shot and put it in the long vector format
    Eigen::VectorXd tdiff(const Eigen::VectorXd& x,
                          Real epsDt);  // to call DSI::tdiff on the first shot and put it in the long vector format
    Real extractT(const Eigen::VectorXd& x);  // to extract T from the vector
    Real extractXshift(const Eigen::VectorXd& x);
    Real extractZshift(const Eigen::VectorXd& x);
    Real observable(Eigen::VectorXd& x);  // to call DSI::observable on the first shot
    void phaseShift(Eigen::VectorXd& x,
                    bool isAC);  // to call DSI::phaseshift on the first shot, but apply the result on all of the shots
    Real fixtphase(const Eigen::VectorXd& x);  // to fix the first shot at a position of the orbit with zero variation
                                               // of an observable called by DSI::tph_observable.
    bool tph();  // to know if t phase is fixed at a point with zero variation of the observable
    Real DSIL2Norm(const Eigen::VectorXd& x);     // to call DSI::L2Norm on the first shot
    std::string stats(const Eigen::VectorXd& x);  // to return stats of the first shot
    std::pair<std::string, std::string> stats_minmax(
        const Eigen::VectorXd& x);  // to return DSI::stats_minmax on the first shot
    void save(const Eigen::VectorXd& x, const std::string filebase, const std::string outdir);  // to save the results
    int nShot();  // to return number of the shots

   private:
    DSI* dsi_;
    bool Tsearch_;
    bool xrelative_;
    bool zrelative_;
    bool fixtphase_;
    int nShot_ = 1;
    Real axRef_ = 1;
    Real azRef_ = 1;
    Real TRef_ = 1;

    int Nxtot_;  // size of the long vector
    int Nx_;     // size of the long vector without T, ax ,az
    int Ny_;     // size of the short vector without T, ax, az
    int Nytot_;  // size of the short vector

    int NT_x_;   // position of T in the long vector
    int Nax_x_;  // position of ax in the long vector
    int Naz_x_;  // position of az in the long vector

    int NT_y_;   // position of T in the short vector
    int Nax_y_;  // position of ax in the short vector
    int Naz_y_;  // position of az in the short vector

    bool isDSIset_ = false;
};

}  // namespace chflow
#endif
