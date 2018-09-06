/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */
#include "nsolver/config.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "nsolver/dsi.h"

using namespace std;

using namespace Eigen;

namespace chflow {

DSI::DSI() : os_(&cout) {}

DSI::DSI(DSI& D) {}

DSI::DSI(ostream* os) : os_(os) {}

VectorXd DSI::eval(const VectorXd& x0, const VectorXd& x1, bool symopt) {
    throw runtime_error(
        "You are trying to call DSI::eval(x0,x1), which is not present for the DSI implementation you are "
        "using. DSI::eval(x0,x1) is required for multiShooting.");
}

void DSI::save(const VectorXd& x, const string filebase, const string outdir, const bool fieldsonly) {
    //   cout << "DSI::save()" << endl;
}

void DSI::saveEigenvec(const VectorXd& x, const string label, const string outdir) {
    //   cout << "DSI::saveEigenvec()" << endl;
}

void DSI::saveEigenvec(const VectorXd& x1, const VectorXd& x2, const string label1, const string label2,
                       const string outdir) {
    //   cout << "DSI::saveEigenvec()" << endl;
}

VectorXd DSI::quadraticInterpolate_vector(const cfarray<VectorXd>& xn, const cfarray<Real>& s, Real snew) {
    VectorXd xnew(xn[0].rows());
    xnew.setZero();
    return xnew;
}

Real DSI::DSIL2Norm(const VectorXd& x) { return L2Norm(x); }

Real DSI::extractT(const VectorXd& x) { return 0.0; }

Real DSI::extractXshift(const VectorXd& x) { return 0.0; }

Real DSI::extractZshift(const VectorXd& x) { return 0.0; }

string DSI::stats(const VectorXd& x) { return ""; }

pair<string, string> DSI::stats_minmax(const VectorXd& x) {
    pair<string, string> Pair = make_pair("", "");
    return Pair;
}

string DSI::statsHeader() { return ""; }

VectorXd DSI::Jacobian(const VectorXd& x, const VectorXd& dx, const VectorXd& Gx, const Real& epsDx, bool centdiff,
                       int& fcount) {
    Real step_magn = L2Norm(dx);
    Real eps = (step_magn < epsDx) ? 1 : epsDx / step_magn;
    VectorXd DG_dx;

    if (centdiff) {
        Real eps2 = 0.5 * eps;
        VectorXd x_epsdx = x + eps2 * dx;
        VectorXd Gx_epsdxplus = eval(x_epsdx);
        ++fcount;
        x_epsdx = x - eps2 * dx;
        VectorXd Gx_epsdxminus = eval(x_epsdx);
        ++fcount;
        DG_dx = 1 / eps * (Gx_epsdxplus - Gx_epsdxminus);
    } else {
        VectorXd x_epsdx = x + eps * dx;
        VectorXd Gx_epsdx = eval(x_epsdx);
        ++fcount;
        DG_dx = 1 / eps * (Gx_epsdx - Gx);
    }
    return DG_dx;
}

VectorXd DSI::xdiff(const VectorXd& a) {
    throw runtime_error(
        "You are trying to call DSI::xdiff(), which is not present for the DSI implementation you are using.");
}
VectorXd DSI::zdiff(const VectorXd& a) {
    throw runtime_error(
        "You are trying to call DSI::zdiff(), which is not present for the DSI implementation you are using.");
}
VectorXd DSI::tdiff(const VectorXd& a, Real epsDt) {
    throw runtime_error(
        "You are trying to call DSI::tdiff(), which is not present for the DSI implementation you are using.");
}
Real DSI::tph_observable(VectorXd& x) {
    throw runtime_error(
        "You are trying to call DSI::tph_observable(), which is not present for the DSI implementation you "
        "are using (required to be used for tphasehack).");
}

}  // namespace chflow
